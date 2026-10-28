"""Synthetic P1-L / P1L-S artifacts without a model: finished training directories, confirmation samples, manifests.

check_training and finalize only read files. This module writes the fields they read (and nothing else), so a test
can damage one field at a time. tests/test_p1ls_defaults.py gives the baseline tree and the current tree the same
files; tests/test_p1ls_pipeline.py uses the P1L-S builders. Accuracies are fixed, so G1-L, G2-L and the B7 memory-use
check pass unless `accuracy` overrides say otherwise. Compute node only (torch, looplora).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from looplora.llm.evaluate import aggregate_samples
from looplora.llm.oracle import OracleSharedWriter

SEEDS = (17, 29, 43)
EPISODES = 2000
LABELS = ("B8", "M1-L4", "B4", "B5-L4", "B3-L4", "B1", "B2")
LRS = {"B8": .001, "M1-L4": .003, "B4": .001, "B5-L4": .001, "B3-L4": .001, "B1": .001, "B2": .001}
ACCURACY = {"B8": .65, "M1-L4": .62, "B4": .30, "B5-L4": .40, "B3-L4": .12, "B1": .20, "B2": .20, "B0": .05}
DEPTHS = {"M1-L4": tuple(range(1, 9)), "B4": (1,), "B1": (1,), "B2": (1,), "B0": (1,)}
RETRAINED = LABELS[:4]  # P1L-S retrains these; B3-L4 is P1-L's run
SHARED_ROWS = 1025 * 32  # oracle_shared's table C: (V + 1) x r_mem


def digest(path) -> str:
    return hashlib.blake2b(Path(path).read_bytes(), digest_size=16).hexdigest()


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


@contextlib.contextmanager
def edited(path, edit):
    """Rewrite a JSON file with `edit` applied to its value, and restore its exact text afterwards."""
    path = Path(path)
    original = path.read_text()
    value = json.loads(original)
    edit(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    try:
        yield value
    finally:
        path.write_text(original)


def shared_state(name: str) -> dict:
    """A writer state as oracle_shared leaves it: the seeded fixed parts and a table C that moved by run."""
    writer = OracleSharedWriter(V=1024, n_rel=4, seed=17, r_mem=32)
    state = {key: value.detach().clone() for key, value in writer.state_dict().items()}
    noise = torch.randn(state["C"].shape, generator=torch.Generator().manual_seed(zlib.crc32(name.encode())))
    state["C"] = state["C"] + .01 * noise
    return state


def make_run(directory: Path, label: str, seed: int, rank: int, writer: str, steps: int, lr: float,
             *, state: dict | None = None, extra: dict | None = None) -> str:
    """One finished training directory (record.json, config.json, checkpoint.pt); returns its path."""
    cfg = dict(name=f"{label}-s{seed}", seed=seed, writer=writer, oracle_seed=17, smoke=False,
               model=dict(r=rank), train=dict(steps=steps, lr=lr), eval=dict(allow_confirmation=False, splits=["val"]),
               task=dict(V=1024, n_rel=4))
    write_json(directory / "config.json", cfg)
    torch.save({"writer": state or {}}, directory / "checkpoint.pt")
    record = dict(status="complete", config=cfg, checkpoint_hash=digest(directory / "checkpoint.pt"),
                  writer_hash=f"{writer}-code")
    record.update(extra or {})
    write_json(directory / "record.json", record)
    return str(directory)


def make_samples(label: str, seed: int, setting: str, depths, accuracy) -> dict:
    """EPISODES episodes x 8 hops per depth; every run shares the episode/query/hop/answer labels."""
    rng = np.random.default_rng(zlib.crc32(f"{label}/{seed}/{setting}".encode()))
    episode, query = np.divmod(np.arange(EPISODES * 8), 8)
    hop = query + 1
    answer = (episode * 7 + hop * 13) % 1024
    parts = []
    for depth in depths:
        correct = rng.random(len(hop)) < accuracy(label, depth)
        prediction = np.where(correct, answer, (answer + 1) % 1024)
        parts.append(dict(episode_id=episode, query_id=query, hop=hop, L=np.full(len(hop), depth), answer=answer,
                          correct=correct, prediction=prediction, full_prediction=prediction, full_correct=correct,
                          candidate_prediction=prediction, candidate_correct=correct | (rng.random(len(hop)) < .2),
                          candidate_size=np.full(len(hop), 8)))
    samples = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    count = len(samples["hop"])
    return dict(samples, split=np.full(count, "confirm"), condition=np.full(count, "memory_only/original"),
                protocol=np.full(count, "memory_only"))


def write_swap_and_readout(out: Path, label: str, seed: int) -> None:
    rng = np.random.default_rng(zlib.crc32(f"swap/{label}/{seed}".encode()))
    parts, rows = [], []
    for depth in ((1, 4, 8) if label == "M1-L4" else (4,)):
        for hop in (1, 2):
            answer = (np.arange(EPISODES) * 5 + hop) % 1024
            prediction = np.where(rng.random(EPISODES) < .5, answer, answer + 1)
            original = np.where(rng.random(EPISODES) < .2, answer, answer + 1)
            parts.append(dict(hop=np.full(EPISODES, hop), L=np.full(EPISODES, depth), prediction=prediction,
                              answer=answer, original_prediction=original))
            rows.append(dict(hop=hop, L=depth, n=EPISODES, follows_new=float((prediction == answer).mean()),
                             original_predicts_new=float((original == answer).mean())))
    np.savez_compressed(out / "swap-successors-samples.npz",
                        **{key: np.concatenate([part[key] for part in parts]) for key in parts[0]})
    write_json(out / "swap-successors.json", rows)
    write_json(out / "readout.json", dict(
        table=[dict(method="logit_lens", hop=1, reader_step=1, entity_step=1, n=EPISODES, rank_sum=3000., mean_rank=1.5,
                    top1_count=1000, top1_rate=.5, tied_top1_count=0, tied_top1_rate=0.)],
        faithfulness=[dict(method="logit_lens", hop=1, n=EPISODES, matches=1500, rate=.75)]))


def eval_run(root: Path, name: str, train_dir: str, authorization: Path, *, memory: str, window: str | None,
             start: int, accuracy, writer_parameters: int = 0, rank: int = 32) -> str:
    """One finished confirmation directory for `name`, as evaluate_p1l_main.main leaves it; returns its path."""
    label, seed = name.rsplit("-s", 1)
    seed = int(seed)
    out = root / "confirmation" / label / name
    write_json(out / "record.json", dict(status="complete", name=name, authorization_hash=digest(authorization),
                                         checkpoint_hash=digest(Path(train_dir) / "checkpoint.pt")))
    depths = DEPTHS.get(label, (4,))
    timings = [dict(protocol="memory_only", L=depths[0], batch_episodes=16, batch_questions=128, warmups=5, repeats=20,
                    question_shape=[128, 9], mean_seconds_per_question=.001, median_seconds_per_question=.001,
                    batch_seconds=[.1] * 20)]
    write_json(out / "inference.json", dict(
        rank=rank, adapter_parameters=1000, trainable_adapter_parameters=900, timings=timings,
        **({"writer_parameters": writer_parameters} if writer_parameters else {})))
    for setting in ("primary", "typed"):
        directory = out / setting
        directory.mkdir(parents=True)
        samples = make_samples(label, seed, setting, depths, accuracy)
        np.savez_compressed(directory / "samples.npz", **samples)
        write_json(directory / "metrics.json", dict(
            table=aggregate_samples(samples), gate0_passed=True,
            episode_ranges={"confirm": dict(split="test", start=start, count=EPISODES)},
            memory_hashes=[dict(split="confirm", protocol="memory_only", condition="memory_only/original",
                                batch_start=batch, memory=f"{memory}-{batch}", write_count=f"write-count-{batch}")
                           for batch in (0, 16, 32)]))
        write_json(directory / "config.json", dict(eval=dict(
            allow_confirmation=True, episodes=EPISODES, seed=4242, splits=["confirm"],
            **({"confirmation": window} if window else {}))))
    if label in {"M1-L4", "B8"}:
        write_swap_and_readout(out, label, seed)
    return str(out)


def accuracy_function(overrides: dict | None = None):
    table = {**ACCURACY, **(overrides or {})}
    return lambda label, depth: .25 if label == "M1-L4" and depth != 4 else table[label]


def build_p1l(root: Path, *, overrides: dict | None = None) -> dict:
    """P1-L's two conditions: r = 32 (21 runs and B0) with its confirmation artifacts, r = 128 (15 runs), and the
    global authorization (two gated conditions). Returns the paths the tests need."""
    accuracy = accuracy_function(overrides)
    runs32 = {f"{label}-s{seed}": make_run(root / "p1l" / "r32" / f"{label}-s{seed}", label, seed, 32,
                                         "oracle_random" if label in LABELS[:5] else "learned", 17000, LRS[label])
              for label in LABELS for seed in SEEDS}
    runs32["B0-s17"] = make_run(root / "p1l" / "r32" / "B0-s17", "B0", 17, 32, "learned", 0, .001)
    runs128 = {f"{label}-s{seed}": make_run(root / "p1l" / "r128" / f"{label}-s{seed}", label, seed, 128,
                                          "oracle_random", 17000, LRS[label])
               for label in LABELS[:5] for seed in SEEDS}
    manifest = {32: dict(condition="oracle-r32", rank=32, S=17000, lrs=LRS, runs=runs32, lock_commit="c0ffee",
                         lock_file_hash="f" * 32),
                128: dict(condition="oracle-r128", rank=128, S=17000, lrs=LRS, runs=runs128, lock_commit="c0ffee",
                          lock_file_hash="f" * 32)}
    paths = {rank: write_json(root / "p1l" / f"training-manifest-r{rank}.json", value)
             for rank, value in manifest.items()}
    authorization = write_json(root / "p1l" / "confirmation-authorization.json", dict(
        status="allowed", orchestrator_decision="fixture", gated_conditions=2, confidence=.975,
        condition_manifests={"oracle-r32": str(paths[32]), "oracle-r128": str(paths[128])}))
    evaluations = {name: eval_run(root / "p1l", name, train, authorization, memory="memory-oracle_random", window=None,
                                  start=100000, accuracy=accuracy) for name, train in runs32.items()}
    return dict(manifest32=paths[32], manifest128=paths[128], authorization=authorization, runs32=runs32,
                evaluations=write_json(root / "p1l" / "evaluation-manifest-r32.json", dict(runs=evaluations)))


def build_p1ls(root: Path, p1l: dict, writer: str, decision_root: Path, *, overrides: dict | None = None,
               status: str = "selected") -> dict:
    """P1L-S: 12 runs retrained under `writer` plus P1-L's three B3-L4 runs (from `p1l`), the committed WP-L6 decision
    under `decision_root`, the manifest, the authorization (one gated condition) and the confirmation artifacts."""
    accuracy = accuracy_function(overrides)
    lrs = {label: LRS[label] for label in RETRAINED}
    decision = write_json(decision_root / "reports/wp-l6/decision.json", dict(
        status=status, selected=writer if status == "selected" else None))
    runs, shared = {}, writer == "oracle_shared"
    for label in RETRAINED:
        for seed in SEEDS:
            name = f"{label}-s{seed}"
            # A fixed code has one writer hash for every run; oracle_shared's hash includes each run's table C.
            extra = dict(writer_trainable_parameters=SHARED_ROWS, writer_hash=f"{writer}-{name}",
                         code_drift=dict(seen=dict(rows=768, mean_cosine_to_init=.99), unseen=dict(rows=256))
                         ) if shared else dict(writer_trainable_parameters=0)
            runs[name] = make_run(root / "p1ls" / name, label, seed, 32, writer, 17000, lrs[label],
                                  state=shared_state(name) if shared else None, extra=extra)
    reused = {}
    for seed in SEEDS:
        name = f"B3-L4-s{seed}"
        runs[name] = p1l["runs32"][name]
        reused[name] = dict(source=runs[name], manifest=str(p1l["manifest32"]), manifest_hash=digest(p1l["manifest32"]),
                            checkpoint_hash=digest(Path(runs[name]) / "checkpoint.pt"))
    manifest = write_json(root / "p1ls" / "training-manifest.json", dict(
        study="p1l-s", condition="p1l-s-r32", rank=32, writer=writer, S=17000, lrs=lrs, runs=runs, reused=reused,
        lock_commit="c0ffee", lock_file_hash="f" * 32))
    authorization = write_json(root / "p1ls" / "confirmation-authorization.json", dict(
        study="p1l-s", status="allowed", orchestrator_decision="fixture", gated_conditions=1,
        confirmation_window="p1l-s", decision=dict(path="reports/wp-l6/decision.json", hash=digest(decision)),
        condition_manifests={"p1l-s-r32": str(manifest)}))
    evaluations = {}
    for name, train in runs.items():
        fixed = name.startswith("B3-L4") or not shared
        evaluations[name] = eval_run(
            root / "p1ls", name, train, authorization, window="p1l-s", start=110000, accuracy=accuracy,
            memory=f"memory-{'oracle_random' if name.startswith('B3-L4') else writer}" + ("" if fixed else f"-{name}"),
            writer_parameters=SHARED_ROWS if shared and not name.startswith("B3-L4") else 0)
    return dict(manifest=manifest, authorization=authorization, decision=decision, runs=runs, reused=reused,
                evaluations=write_json(root / "p1ls" / "evaluation-manifest.json", dict(runs=evaluations)))
