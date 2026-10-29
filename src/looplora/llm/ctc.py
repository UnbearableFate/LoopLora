"""Study CTC (docs/prereg/CTC.md): a classify-then-code writer into the frozen P1-L r = 128 readers.

Linear heads classify each fact's head entity, relation and tail from frozen Qwen3 hidden states over the support text.
The writer then writes the oracle's own codes (`OracleRandomWriter`, seed 17), looked up from the fixed tables, with the
oracle's decay-first delta rule. The readers are never trained or changed. This module only adds code: no existing
LLM-pipeline module is edited, and the memory enters through `PreparedBatch` replacement as in `evaluate.py`'s swap.
Everything that loads the backbone runs on Miyabi-G compute only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import socket
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looplora.core.memory import delta_update, zeros_state
from looplora.data.relations import generate
from looplora.data.render import load_name_pool
from looplora.infra.seeds import derive_seed, stable_hash
from .batching import tokenize_batch

ROOT = Path(__file__).resolve().parents[3]
V, N_REL, SEEN, R_MEM, LAYER = 1024, 4, 768, 128, 14
SITES = ("o14", "h14", "h21", "h28")
POSITIONS = ("head_last", "rel_last", "tail", "final")
TARGETS = ("head", "relation", "tail")
FORMS = ("full", "rank64")
SITE_SPEC = {"o14": ("o_proj", 14), "h14": ("hidden", 14), "h21": ("hidden", 21), "h28": ("hidden", 28)}
CLASSES = {"head": V, "relation": N_REL, "tail": V}
TIE = 0.005  # Stage A: candidates within 0.5 pp of the best accuracy are tied
READER_ADAPTER_PARAMETERS = 475138
READER_CONFIG = ROOT / "configs" / "p1l" / "main-r128" / "M1-L4-s17.json"
TRAINING_MANIFEST = ROOT / "reports" / "p1l" / "r128" / "training-manifest.json"
READER_HASHES = {"M1-L4": "c90d23ff1a27d509ed61ae43d5185832", "B4": "acfa784892a1dd2fd2e31a7d04ff86ae",
                 "B3-L4": "37020c367f482656bd2d14bb97e85b8f"}
READER_DEPTHS = {"M1-L4": tuple(range(1, 9)), "B4": (1,), "B3-L4": (4,)}
ORACLE_HASH = "db99a779aebd647bc72caf4aabba2c48"
EVAL_SEED = 4242
STAGE_A_TRAIN = dict(split="train", start=0, episodes=8192)
STAGE_A_SELECT = dict(split="val", start=1000, episodes=1000)
FINAL_EVAL = dict(split="val", start=0, episodes=1000)
PROBE_STEPS = 1200
STAGE_B_STEPS = 8000


def digest(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_parameters(dim: int, classes: int, form: str) -> int:
    """Parameters of one head at its registered (1,024-way for entities) size."""
    if form == "full":
        return dim * classes + classes
    if form == "rank64":
        return dim * 64 + 64 * classes + classes
    raise ValueError(f"unknown head form {form!r}")


def reader_config() -> dict[str, Any]:
    return json.loads(READER_CONFIG.read_text())


def reader_path(label: str) -> Path:
    manifest = json.loads(TRAINING_MANIFEST.read_text())
    return Path(manifest["runs"][f"{label}-s17"]) / "checkpoint.pt"


def load_reader(label: str, model, tokenizer):
    """A frozen P1-L reader (seed 17, r = 128), after re-checking the checkpoint hash and the oracle writer hash."""
    from .train import load_checkpoint

    path = reader_path(label)
    if digest(path) != READER_HASHES[label]:
        raise ValueError(f"reader checkpoint {label} differs from the registered hash")
    experiment = load_checkpoint(path, model=model, tokenizer=tokenizer)
    try:
        if oracle_hash(experiment.writer) != ORACLE_HASH:
            raise ValueError("oracle writer hash differs from P1-L's r = 128 oracle")
        freeze_reader(experiment)
    except BaseException:
        experiment.close()
        raise
    return experiment


def oracle_hash(writer) -> str:
    from looplora.sym.evaluate import tensor_hash

    return stable_hash({name: tensor_hash(value) for name, value in writer.state_dict().items()})


def freeze_reader(experiment) -> None:
    experiment.model.eval()
    experiment.model.requires_grad_(False)
    if experiment.adapter is not None:
        experiment.adapter.requires_grad_(False)


def reader_state_hash(experiment) -> str:
    from looplora.sym.evaluate import tensor_hash

    state = {name: tensor_hash(value) for name, value in experiment.adapter.state_dict().items()}
    return stable_hash(state)


# ---------------------------------------------------------------------------------------------------------------
# Fact positions and hidden-state features
# ---------------------------------------------------------------------------------------------------------------
class NameTokens:
    """Token ids of every name as it starts a fact line (no leading space), for the head-last position."""

    def __init__(self, tokenizer, pool=None, count: int = V):
        pool = load_name_pool() if pool is None else pool
        ids = [tokenizer.encode(name, add_special_tokens=False) for name in pool.names[:count]]
        self.length = torch.tensor([len(i) for i in ids])
        self.first = torch.tensor([i[0] for i in ids])
        self.last = torch.tensor([i[-1] for i in ids])


def fact_positions4(arrays, tokens, names: NameTokens) -> Tensor:
    """[B, E, 4] token indices (head_last, rel_last, tail, final) of every fact on CPU support tokens.

    final, rel_last (the token before the tail) and tail come from the WP-L3 helper; the head's last token is the line
    start plus the head name's token count. Token identities are checked against the name table.
    """
    from .writer_fix import token_fact_positions

    base = token_fact_positions(arrays, tokens)  # (final, pre_tail, tail)
    final, rel_last, tail = base[..., 0], base[..., 1], base[..., 2]
    previous = torch.cat((torch.full_like(final[:, :1], -1), final[:, :-1]), dim=1)
    start = previous + 1
    heads = torch.as_tensor(arrays.facts[:, :, 0], dtype=torch.long)
    head_last = start + names.length[heads] - 1
    ids = tokens.support_ids
    if not (bool((ids.gather(1, start) == names.first[heads]).all())
            and bool((ids.gather(1, head_last) == names.last[heads]).all()) and bool((head_last < rel_last).all())):
        raise AssertionError("head name tokens do not match the name table at their fact line")
    return torch.stack((head_last, rel_last, tail, final), dim=-1)


@contextmanager
def bare_backbone(model):
    """The unmodified backbone: temporarily put every adapter's base projection back (features must not see adapters)."""
    bank = getattr(model, "looplora_adapters", None)
    if bank is None:
        yield
        return
    installed = {layer: model.model.layers[layer].self_attn.o_proj for layer in bank.modules}
    for layer, module in bank.modules.items():
        model.model.layers[layer].self_attn.o_proj = module.base
    try:
        yield
    finally:
        for layer, projection in installed.items():
            model.model.layers[layer].self_attn.o_proj = projection


@torch.no_grad()
def extract_features(model, support_ids: Tensor, support_mask: Tensor, positions: Tensor,
                     site_spec: Mapping[str, tuple[str, int]] = SITE_SPEC) -> dict[str, Tensor]:
    """{site: [B, E, P, d]} bf16 features of the frozen bare backbone at the given support positions.

    `o14` is the input of the layer's `o_proj` (attention output); `hN` is `hidden_states[N]` (the residual stream
    entering block N; the last index is after the final norm).
    """
    captured, handles = {}, []
    for layer in {layer for kind, layer in site_spec.values() if kind == "o_proj"}:
        projection = model.model.layers[layer].self_attn.o_proj
        if not isinstance(projection, nn.Linear):
            raise ValueError("backbone features need the bare backbone; use bare_backbone()")
        handles.append(projection.register_forward_pre_hook(
            lambda _, args, layer=layer: captured.__setitem__(layer, args[0])))
    try:
        output = model.model(input_ids=support_ids, attention_mask=support_mask, use_cache=False,
                             output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()
    B, E, P = positions.shape
    index = positions.reshape(B, E * P).to(support_ids.device)
    features = {}
    for site, (kind, layer) in site_spec.items():
        x = captured[layer] if kind == "o_proj" else output.hidden_states[layer]
        features[site] = x.gather(1, index[..., None].expand(-1, -1, x.shape[-1])).reshape(B, E, P, x.shape[-1])
    return features


def cpu_tokens(arrays, tokenizer):
    return tokenize_batch(arrays, tokenizer, device="cpu", include_support=True)


# ---------------------------------------------------------------------------------------------------------------
# The writer: linear heads, then a fixed code table
# ---------------------------------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, dim: int, classes: int, form: str, mean: Tensor, std: Tensor, seed: int):
        super().__init__()
        self.classes, self.form = classes, form
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("scale", std.detach().float().clamp_min(1e-4))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.map = (nn.Linear(dim, classes) if form == "full" else
                        nn.Sequential(nn.Linear(dim, 64, bias=False), nn.Linear(64, classes)))
        unseen = torch.zeros(classes, dtype=torch.bool)
        if classes == V:
            unseen[SEEN:] = True  # closed vocabulary of the 768 seen names, in training and evaluation
        self.register_buffer("unseen", unseen)
        self.masked = classes == V

    def forward(self, x: Tensor) -> Tensor:
        logits = self.map((x.float() - self.mean) / self.scale)
        return logits.masked_fill(self.unseen, float("-inf")) if self.masked else logits


class CTCWriter(nn.Module):
    """Three heads, each on its own (site, position, form); the oracle tables are held outside the module state."""

    def __init__(self, plan: Mapping[str, Mapping[str, str]], stats: Mapping[str, Mapping[str, Tensor]], oracle):
        super().__init__()
        self.plan = {target: dict(plan[target]) for target in TARGETS}
        self.heads = nn.ModuleDict()
        for target in TARGETS:
            cell = self.plan[target]
            mean, std = stats[f"{cell['site']}/{cell['position']}"]["mean"], stats[f"{cell['site']}/{cell['position']}"]["std"]
            self.heads[target] = Head(mean.numel(), CLASSES[target], cell["form"], mean, std,
                                      derive_seed(17, "ctc", "head-init", target))
        self._oracle = [oracle]

    @property
    def oracle(self):
        return self._oracle[0]

    def logits(self, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
        out = {}
        for target in TARGETS:
            cell = self.plan[target]
            out[target] = self.heads[target](features[cell["site"]][:, :, POSITIONS.index(cell["position"])])
        return out

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.heads.parameters())


def write_memory(keys: Tensor, values: Tensor, valid: Tensor) -> Tensor:
    """The oracle's scan (`OracleWriter.forward`): decay-first delta rule, lambda = beta = 1, facts in list order."""
    state = zeros_state(len(keys), keys.shape[-1], keys.dtype, keys.device)
    for t in range(keys.shape[1]):
        state = delta_update(state, keys[:, t], values[:, t], valid[:, t], lam=1.0, beta=1.0)
    return state


def hard_indices(logits: Mapping[str, Tensor]) -> Tensor:
    """[B, E, 3] argmax (head, relation, tail)."""
    return torch.stack([logits[target].argmax(-1) for target in TARGETS], dim=-1)


def hard_memory(oracle, indices: Tensor, valid: Tensor) -> Tensor:
    """Exact table lookup: with correct classifications this is OracleRandomWriter's memory, bitwise."""
    indices = indices.masked_fill(~valid[..., None], 0)
    pair = oracle.pair_codes.flatten(0, 1)
    keys = pair[indices[..., 0] * N_REL + indices[..., 1]]
    values = oracle.global_codes[indices[..., 2]]
    return write_memory(keys, values, valid)


def soft_codes(oracle, logits: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Probability-weighted key (renormalized to unit length, as the delta rule assumes) and value codes."""
    ph, pr, pt = (F.softmax(logits[target].float(), dim=-1) for target in TARGETS)
    B, E = ph.shape[:2]
    pair = oracle.pair_codes.reshape(V, -1)
    mixed = (ph @ pair).reshape(B, E, N_REL, -1)
    key = F.normalize((pr.unsqueeze(-1) * mixed).sum(2), dim=-1)
    value = pt @ oracle.global_codes[:V]
    return key, value


def soft_memory(oracle, logits: Mapping[str, Tensor], valid: Tensor) -> Tensor:
    key, value = soft_codes(oracle, logits)
    return write_memory(key, value, valid)


def classification_loss(logits: Mapping[str, Tensor], labels: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Mean over facts of the head, relation and tail cross-entropies."""
    parts = {target: F.cross_entropy(logits[target].float().flatten(0, 1), labels[..., column].flatten())
             for column, target in enumerate(TARGETS)}
    return sum(parts.values()), parts


def with_memory(prepared, state: Tensor, layer: int = LAYER):
    """The prepared oracle batch with its memory replaced by the writer's (as evaluate.py's swap condition)."""
    return replace(prepared, snapshot=replace(prepared.snapshot, states={layer: state}))


# ---------------------------------------------------------------------------------------------------------------
# Stage A: probes
# ---------------------------------------------------------------------------------------------------------------
def collect(model, tokenizer, spec, split: str, start: int, episodes: int, seed: int, names: NameTokens, *,
            batch_size: int = 64, site_spec=SITE_SPEC) -> dict[str, Any]:
    """Features of every fact at every (site, position): {'x': {(site, position): bf16 [N, d]}, labels, fact_index}."""
    from .evaluate import _slice

    arrays = generate(spec, split, start, episodes, seed)
    chunks: dict[tuple[str, str], list[Tensor]] = {(s, p): [] for s in site_spec for p in POSITIONS}
    labels = []
    with bare_backbone(model):
        for begin in range(0, episodes, batch_size):
            part = _slice(arrays, slice(begin, min(begin + batch_size, episodes)))
            tokens = cpu_tokens(part, tokenizer)
            positions = fact_positions4(part, tokens, names)
            features = extract_features(model, tokens.support_ids.to(model.device),
                                        tokens.support_mask.to(model.device), positions, site_spec)
            for site, x in features.items():
                for column, position in enumerate(POSITIONS):
                    chunks[site, position].append(x[:, :, column].flatten(0, 1).cpu())
            labels.append(torch.as_tensor(part.facts, dtype=torch.long).flatten(0, 1))
    E = arrays.facts.shape[1]
    labels = torch.cat(labels)
    return dict(x={key: torch.cat(value) for key, value in chunks.items()}, labels=labels,
                fact_index=torch.arange(len(labels)) % E, episodes=episodes, split=split, start=start, seed=seed)


def select_cells(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per target: the best selection accuracy; cells within TIE of it are tied and the tie goes to fewer parameters,
    then the lower site index, then the earlier position (docs/prereg/CTC.md, Stage A)."""
    chosen = {}
    for target in TARGETS:
        candidates = [row for row in rows if row["target"] == target]
        best = max(row["accuracy"] for row in candidates)
        tied = [row for row in candidates if best - row["accuracy"] <= TIE + 1e-12]
        chosen[target] = min(tied, key=lambda row: (row["parameters"], SITES.index(row["site"]),
                                                    POSITIONS.index(row["position"])))
    return chosen


def stage_a_lock(rows, predictions: Mapping[tuple, np.ndarray], select: dict[str, Any], dims: Mapping[str, int],
                 E: int) -> dict[str, Any]:
    chosen = select_cells(rows)
    correct = {}
    for target, column in zip(TARGETS, range(3)):
        row = chosen[target]
        prediction = predictions[target, row["site"], row["position"], row["form"]]
        correct[target] = prediction == select["labels"][:, column].numpy()
    joint = correct["head"] & correct["relation"] & correct["tail"]
    key = correct["head"] & correct["relation"]
    index = select["fact_index"].numpy()
    plan = {target: dict(site=row["site"], position=row["position"], form=row["form"]) for target, row in chosen.items()}
    parameters = {target: head_parameters(dims[row["site"]], CLASSES[target], row["form"]) for target, row in chosen.items()}
    return dict(
        selection={target: dict(**plan[target], accuracy=chosen[target]["accuracy"], train_accuracy=chosen[target]["train_accuracy"],
                                parameters=parameters[target], n=chosen[target]["n"]) for target in TARGETS},
        plan=plan, writer_parameters=sum(parameters.values()), reader_adapter_parameters=READER_ADAPTER_PARAMETERS,
        selection_joint_accuracy=float(joint.mean()), selection_key_accuracy=float(key.mean()),
        selection_joint_by_fact=[float(joint[index == j].mean()) for j in range(E)],
        selection_facts=int(len(joint)),
        entity_at_own_token=dict(head=chosen["head"]["position"] == "head_last", tail=chosen["tail"]["position"] == "tail"))


def run_stage_a(out: Path, *, train=STAGE_A_TRAIN, select=STAGE_A_SELECT, steps: int = PROBE_STEPS,
                smoke: bool = False) -> dict[str, Any]:
    """Extract features and fit every probe; write probes.csv, predictions.npz, feature-stats.pt and lock.json to `out`.

    The registered ranges are train [0, 8192) (seed derive_seed(17, "ctc", "stage-a-train")) and val [1000, 2000)
    (seed 4242). A smoke run passes other ranges but never one that touches val [0, 2000).
    """
    from .diagnosis_probes import fit_probe
    from .inject import load_model, load_tokenizer
    from .run import _write_json, source_metadata
    from .train import make_task_spec

    if not smoke and (dict(train) != STAGE_A_TRAIN or dict(select) != STAGE_A_SELECT or steps != PROBE_STEPS):
        raise ValueError("a registered Stage A run uses the registered ranges and steps")
    if smoke and select["split"] == "val" and select["start"] < 2000:
        raise ValueError("smoke runs must not read val [0, 2000)")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    spec = make_task_spec(reader_config())
    model, tokenizer = load_model(), load_tokenizer()
    names = NameTokens(tokenizer)
    stamps = dict(load=time.perf_counter() - started)
    tr = collect(model, tokenizer, spec, train["split"], train["start"], train["episodes"],
                 derive_seed(17, "ctc", "stage-a-train"), names)
    sel = collect(model, tokenizer, spec, select["split"], select["start"], select["episodes"], EVAL_SEED, names)
    stamps["extract"] = time.perf_counter() - started - stamps["load"]
    del model
    torch.cuda.empty_cache()
    stats = {}
    dims = {site: tr["x"][site, POSITIONS[0]].shape[1] for site in SITES}
    for (site, position), x in tr["x"].items():
        xf = x.float().cuda()
        stats[f"{site}/{position}"] = dict(mean=xf.mean(0).cpu(), std=xf.std(0).cpu())
        del xf
    torch.save(stats, out / "feature-stats.pt")
    rows, predictions = [], {}
    fields = ["target", "site", "position", "form", "parameters", "classes", "dimensions", "train_n", "train_accuracy",
              "n", "correct", "accuracy", "seconds"]
    with (out / "probes.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for site in SITES:
            for position in POSITIONS:
                for target, column in zip(TARGETS, range(3)):
                    for form in (FORMS if target != "relation" else ("full",)):
                        began = time.perf_counter()
                        # Seen names only: a 768-way head is the 1,024-way head with the unseen logits masked.
                        classes = SEEN if target != "relation" else N_REL
                        row, prediction = fit_probe(
                            tr["x"][site, position], tr["labels"][:, column], sel["x"][site, position],
                            sel["labels"][:, column], classes=classes, rank=None if form == "full" else 64,
                            purpose=f"ctc/{site}/{position}/{target}", steps=steps)
                        row.update(target=target, site=site, position=position, form=form,
                                   parameters=head_parameters(dims[site], CLASSES[target], form),
                                   seconds=time.perf_counter() - began)
                        rows.append(row)
                        predictions[target, site, position, form] = prediction
                        writer.writerow({key: row[key] for key in fields})
                        file.flush()
                        print(json.dumps({key: row[key] for key in fields}), flush=True)
    np.savez_compressed(out / "predictions.npz", **{"/".join(key): value for key, value in predictions.items()})
    lock = stage_a_lock(rows, predictions, sel, dims, spec.E)
    lock.update(study="CTC", stage="A", status="lock", train_range=dict(train, seed=tr["seed"]),
                selection_range=dict(select, seed=sel["seed"]), probe_steps=steps, fits=len(rows),
                tie_margin=TIE, probes_csv_hash=digest(out / "probes.csv"),
                predictions_hash=digest(out / "predictions.npz"),
                feature_stats=dict(path=str((out / "feature-stats.pt").resolve()), hash=digest(out / "feature-stats.pt")),
                dimensions=dims, smoke=smoke, seconds=time.perf_counter() - started, stage_seconds=stamps,
                host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"), **source_metadata())
    _write_json(out / "lock.json", lock)
    return lock


def load_lock(path: Path) -> dict[str, Any]:
    lock = json.loads(Path(path).read_text())
    if lock.get("status") != "lock" or lock.get("stage") != "A":
        raise ValueError("not a Stage A lock")
    stats = lock["feature_stats"]
    if digest(Path(stats["path"])) != stats["hash"]:
        raise ValueError("Stage A feature statistics differ from the hash in the lock")
    return lock


# ---------------------------------------------------------------------------------------------------------------
# Stage B: train the heads
# ---------------------------------------------------------------------------------------------------------------
def build_writer(lock: Mapping[str, Any], oracle, device) -> CTCWriter:
    stats = torch.load(lock["feature_stats"]["path"], map_location="cpu", weights_only=True)
    return CTCWriter(lock["plan"], stats, oracle).to(device)


def make_oracle(device, *, rank: int = R_MEM, V_: int = V):
    from .oracle import OracleRandomWriter

    return OracleRandomWriter(V_, N_REL, 17, r_mem=rank).to(device)


def train_writer(writer: CTCWriter, arm: str, *, model, tokenizer, names: NameTokens, spec, out: Path,
                 experiment=None, steps: int = STAGE_B_STEPS, batch_size: int = 64, lr: float = .01,
                 warmup: int = 20, clip: float = 1.0, answer_weight: float = 1.0, depth: int = 4, workers: int = 2,
                 log_every: int = 100, site_spec=SITE_SPEC) -> dict[str, Any]:
    """Arm "ce": head/relation/tail cross-entropy. Arm "ceans": the same plus `answer_weight` times the frozen
    reader's answer cross-entropy at depth 4 through the soft memory. Reader and backbone stay frozen."""
    from .prefetch import BatchStream

    if arm not in {"ce", "ceans"}:
        raise ValueError("arm must be ce or ceans")
    if (arm == "ceans") != (experiment is not None):
        raise ValueError("arm ceans needs the frozen reader experiment and arm ce must not use one")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "writer.pt").exists() or (out / "train_log.jsonl").exists():
        raise FileExistsError(f"training output already exists: {out}")
    device = next(writer.parameters()).device
    before = None if experiment is None else reader_state_hash(experiment)
    params = [p for p in writer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.)
    data_seed = derive_seed(17, "ctc", "stage-b-train")
    durations, final = [], {}
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    sync()
    started = time.perf_counter()
    with (out / "train_log.jsonl").open("w") as log, BatchStream(
            spec, "train", batch_size, data_seed, steps, tokenizer=tokenizer, workers=workers,
            include_support=True) as batches:
        for step in range(steps):
            sync()
            begin = time.perf_counter()
            arrays, tokens = next(batches)
            if tokens is None:
                tokens = cpu_tokens(arrays, tokenizer)
            positions = fact_positions4(arrays, tokens, names)
            ids, mask = tokens.support_ids.to(device), tokens.support_mask.to(device)
            labels = torch.as_tensor(arrays.facts, dtype=torch.long, device=device)
            valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device=device)
            with bare_backbone(model):
                features = extract_features(model, ids, mask, positions, site_spec)
            rate = lr * min(1., (step + 1) / max(1, warmup)) * (.1 + .9 * (1 + math.cos(math.pi * step / steps)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            logits = writer.logits(features)
            loss, parts = classification_loss(logits, labels)
            row = {}
            if arm == "ceans":
                prepared = experiment.prepare(arrays, tokens=tokens)
                prepared = with_memory(prepared, soft_memory(writer.oracle, logits, valid), experiment.layer)
                answer_logits, _ = experiment.score(prepared, depth)
                answer = F.cross_entropy(answer_logits.float().flatten(0, 1), prepared.answer_ids.flatten())
                loss = loss + answer_weight * answer
                row.update(answer_loss=float(answer.detach()), answer_accuracy=float(
                    (answer_logits.argmax(-1) == prepared.answer_ids).float().mean()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, clip, error_if_nonfinite=True)
            optimizer.step()
            sync()
            durations.append(time.perf_counter() - begin)
            if step == 0 or (step + 1) % log_every == 0 or step + 1 == steps:
                with torch.no_grad():
                    hard = hard_indices(logits)
                    accuracy = {target: float((hard[..., c] == labels[..., c]).float().mean())
                                for c, target in enumerate(TARGETS)}
                    joint = float((hard == labels).all(-1).float().mean())
                final = dict(step=step + 1, loss=float(loss.detach()), lr=rate, grad_norm=float(norm),
                             step_seconds=durations[-1], joint_accuracy=joint,
                             **{f"loss_{t}": float(v.detach()) for t, v in parts.items()},
                             **{f"accuracy_{t}": a for t, a in accuracy.items()}, **row)
                log.write(json.dumps(final, allow_nan=False) + "\n")
                log.flush()
    sync()
    if experiment is not None and reader_state_hash(experiment) != before:
        raise RuntimeError("the frozen reader changed during writer training")
    summary = dict(arm=arm, steps=steps, batch_size=batch_size, lr=lr, warmup_steps=warmup, grad_clip=clip,
                   answer_weight=answer_weight if arm == "ceans" else 0., depth=depth if arm == "ceans" else None,
                   data_seed=data_seed, train_seconds=time.perf_counter() - started,
                   train_step_seconds=float(np.mean(durations)), train_step_p95_seconds=float(np.quantile(durations, .95)),
                   final=final, writer_parameters=writer.parameter_count(), plan=writer.plan,
                   reader_state_hash=before, host=socket.gethostname(), job_id=os.environ.get("PBS_JOBID"))
    torch.save(dict(format_version=1, plan=writer.plan, heads=writer.heads.state_dict(), summary=summary),
               out / "writer.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def load_writer(path: Path, lock: Mapping[str, Any], oracle, device) -> CTCWriter:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record["plan"] != lock["plan"]:
        raise ValueError("writer plan differs from the Stage A lock")
    writer = build_writer(lock, oracle, device)
    writer.heads.load_state_dict(record["heads"], strict=True)
    return writer.eval()


# ---------------------------------------------------------------------------------------------------------------
# Evaluation and decision
# ---------------------------------------------------------------------------------------------------------------
def roll_within_batches(states: Tensor, size: int = 16) -> Tensor:
    """Shuffled-memory control: episode i reads the memory of episode i + 1 in its evaluation batch (cyclic)."""
    return torch.cat([part.roll(-1, 0) for part in states.split(size)])


@torch.no_grad()
def writer_outputs(writer: CTCWriter, model, tokenizer, names: NameTokens, arrays, *, batch_size: int = 16,
                   site_spec=SITE_SPEC) -> dict[str, Any]:
    """Hard and soft memories and the classification record of one writer on an evaluation set."""
    from .evaluate import _slice

    hard_states, soft_states, preds, truth, mass = [], [], [], [], []
    with bare_backbone(model):
        for begin in range(0, len(arrays.facts), batch_size):
            part = _slice(arrays, slice(begin, begin + batch_size))
            tokens = cpu_tokens(part, tokenizer)
            positions = fact_positions4(part, tokens, names)
            features = extract_features(model, tokens.support_ids.to(model.device),
                                        tokens.support_mask.to(model.device), positions, site_spec)
            logits = writer.logits(features)
            valid = torch.as_tensor(part.fact_valid, dtype=torch.bool, device=model.device)
            labels = torch.as_tensor(part.facts, dtype=torch.long, device=model.device)
            indices = hard_indices(logits)
            hard_states.append(hard_memory(writer.oracle, indices, valid))
            soft_states.append(soft_memory(writer.oracle, logits, valid))
            preds.append(indices.cpu())
            truth.append(labels.cpu())
            mass.append(torch.stack([F.softmax(logits[t].float(), -1).gather(-1, labels[..., c, None])[..., 0]
                                     for c, t in enumerate(TARGETS)], dim=-1).cpu())
    preds, truth, mass = torch.cat(preds), torch.cat(truth), torch.cat(mass)
    return dict(hard=torch.cat(hard_states), soft=torch.cat(soft_states), predictions=preds, truth=truth, mass=mass)


def code_accuracy(record: Mapping[str, Tensor]) -> dict[str, Any]:
    right = record["predictions"] == record["truth"]
    joint = right.all(-1)
    key = right[..., 0] & right[..., 1]
    return dict(head=float(right[..., 0].float().mean()), relation=float(right[..., 1].float().mean()),
                tail=float(right[..., 2].float().mean()), key=float(key.float().mean()),
                joint=float(joint.float().mean()), joint_by_fact=[float(joint[:, j].float().mean())
                                                                  for j in range(joint.shape[1])],
                episodes_all_facts_correct=float(joint.all(1).float().mean()),
                soft_mass_head=float(record["mass"][..., 0].mean()), soft_mass_relation=float(record["mass"][..., 1].mean()),
                soft_mass_tail=float(record["mass"][..., 2].mean()),
                soft_mass_joint=float(record["mass"].prod(-1).mean()))


def bootstrap(values: np.ndarray, *, resamples: int = 2000, seed: int = 2718) -> dict[str, float]:
    """Mean and 95 % percentile interval over episodes of a per-episode vector (paired contrasts pass differences)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = values[rng.integers(0, n, size=(resamples, n))].mean(1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(draws, .025)), hi=float(np.quantile(draws, .975)),
                se=float(values.std(ddof=1) / math.sqrt(n)))


def decide(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """The registered decision (docs/prereg/CTC.md). `metrics` holds, per arm, hard-lookup M1-L4@4 hop-1 EM, the
    pooled hops 2-3 gain over B4@1, the zero/shuffled hop-1 EM and the per-fact joint accuracy."""
    arms = metrics["arms"]
    ce, ans = arms["ce"]["hop1_em"], arms["ceans"]["hop1_em"]
    selected = "ceans" if ans - ce >= .01 else "ce"
    verdicts = {}
    for arm, m in arms.items():
        integrity = m["zero_hop1_em"] < .05 and m["shuffled_hop1_em"] < .05
        verdicts[arm] = dict(
            hop1_em=m["hop1_em"], gain_pp=100 * m["gain"], hop1_ok=m["hop1_em"] >= .40, gain_ok=m["gain"] >= .10,
            integrity_ok=integrity,
            candidate=bool(m["hop1_em"] >= .40 and m["gain"] >= .10 and integrity),
            falsifier_F1=bool(m["hop1_em"] < .35 and m["joint"] >= .95), falsifier_F2=bool(m["gain"] < .05),
            borderline=bool(abs(m["hop1_em"] - .40) < m["hop1_se"] or abs(m["gain"] - .10) < m["gain_se"]))
    return dict(selected_arm=selected, arm_rule="higher hop-1 EM of M1-L4@4 (hard); a difference below 1 pp goes to ce",
                outcome="candidate" if verdicts[selected]["candidate"] else "stop", arms=verdicts)


@torch.no_grad()
def evaluate_readers(label: str, experiment, arrays, conditions: Mapping[str, Tensor], *, batch_size: int = 16,
                     depths=None) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Full-vocabulary EM and candidate accuracy of one frozen reader under each memory condition.

    Returns {condition: {L: {"prediction": [N, Q], "correct": [N, Q], "candidate_correct": [N, Q]}}}.
    """
    from .evaluate import _slice

    depths = READER_DEPTHS[label] if depths is None else depths
    out = {c: {d: {k: [] for k in ("prediction", "correct", "candidate_correct")} for d in depths} for c in conditions}
    n = len(arrays.facts)
    for begin in range(0, n, batch_size):
        part = _slice(arrays, slice(begin, min(begin + batch_size, n)))
        prepared = experiment.prepare(part)
        candidates, valid = prepared.candidate_ids, prepared.candidate_mask
        for name, states in conditions.items():
            cell = with_memory(prepared, states[begin:begin + len(part.facts)], experiment.layer)
            for depth in depths:
                logits, _ = experiment.score(cell, depth)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("non-finite answer logits")
                prediction = logits.argmax(-1)
                restricted = logits.gather(-1, candidates[:, None, :].expand(-1, logits.shape[1], -1))
                choice = restricted.masked_fill(~valid[:, None, :], -torch.inf).argmax(-1)
                record = out[name][depth]
                record["prediction"].append(prediction.cpu().numpy())
                record["correct"].append((prediction == prepared.answer_ids).cpu().numpy())
                record["candidate_correct"].append((candidates.gather(1, choice) == prepared.answer_ids).cpu().numpy())
    return {c: {d: {k: np.concatenate(v) for k, v in record.items()} for d, record in by_depth.items()}
            for c, by_depth in out.items()}


def hop_matrix(arrays) -> np.ndarray:
    """[N, Q] true hop of every evaluation query; stratified evaluation has hop j + 1 in column j."""
    hops = np.asarray(arrays.q_hop_true)
    if not np.array_equal(hops, np.broadcast_to(np.arange(1, hops.shape[1] + 1), hops.shape)):
        raise ValueError("evaluation queries are not one per hop in order")
    return hops


def contrast(results: Mapping[str, Any], condition: str) -> dict[str, float]:
    """Hop-1 EM of M1-L4 at L = 4 and its hops 2-3 gain over B4 at L = 1 (paired by episode), under one memory."""
    m1 = results["M1-L4"][condition][4]["correct"].astype(float)
    b4 = results["B4"][condition][1]["correct"].astype(float)
    gain = bootstrap((m1[:, 1:3] - b4[:, 1:3]).mean(1))
    p = float(m1[:, 0].mean())
    return dict(hop1_em=p, hop1_se=math.sqrt(p * (1 - p) / len(m1)), gain=gain["mean"], gain_lo=gain["lo"],
                gain_hi=gain["hi"], gain_se=gain["se"])


def arm_metrics(results: Mapping[str, Any], arm: str, joint: float) -> dict[str, float]:
    """The numbers `decide` needs for one arm (hard lookup): the contrast plus the zero/shuffled hop-1 EM."""
    return dict(**contrast(results, f"{arm}/hard"), joint=joint,
                zero_hop1_em=float(results["M1-L4"]["zero"][4]["correct"][:, 0].mean()),
                shuffled_hop1_em=float(results["M1-L4"][f"{arm}/shuffled"][4]["correct"][:, 0].mean()))


def em_rows(results: Mapping[str, Any], hops: np.ndarray) -> list[dict[str, Any]]:
    """Full-vocabulary EM and candidate accuracy by reader, condition, depth and hop."""
    rows = []
    for reader, conditions in results.items():
        for condition, depths in conditions.items():
            for depth, record in depths.items():
                for hop in range(1, hops.shape[1] + 1):
                    mask = hops == hop
                    rows.append(dict(reader=reader, condition=condition, L=int(depth), hop=hop, n=int(mask.sum()),
                                     em=float(record["correct"][mask].mean()),
                                     candidate_em=float(record["candidate_correct"][mask].mean())))
    return rows
