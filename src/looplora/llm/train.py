"""Single-position Qwen adapter experiments and full-vocabulary answer training."""
from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig, validate_steps
from looplora.data.relations import TaskSpec
from looplora.infra.seeds import derive_seed
from .batching import PreparedBatch, repeat_snapshot, tokenize_batch
from .inject import (MODEL_ID, MODEL_REVISION, AdapterBank, attach_adapters,
                     configure_numerics, detach_adapters, load_model, load_tokenizer)
from .protocols import SupportSnapshot, answer_scores, encode_support
from .prefetch import BatchStream
from .oracle import OracleRandomWriter

WRITER_NAMES = ("A", "K", "V", "a_lambda", "a_beta", "w_lambda", "w_beta")


def make_task_spec(config: dict[str, Any], evaluation: bool = False) -> TaskSpec:
    fields = dict(config["task"])
    if config["stage"] == "writer":
        fields = {key: fields[key] for key in ("E", "Q", "V", "seen", "unseen") if key in fields}
        fields.update(name="t1", n_rel=4, head_sharing=.3, hops=(1,))
    elif evaluation and fields["name"] != "t1" and fields.get("base") != "t1":
        fields["hops"] = tuple(config["eval"]["hops"])
        fields["Q"] = len(fields["hops"])
    return TaskSpec(**fields)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class LLMExperiment:
    """An adapter installation with explicit, independently forkable supports.

    Keep the backbone in eval mode even during optimization: the scientific
    intervention is the adapter, and question activations must be repeatable.
    ``close`` restores the original projections for sequential model reuse.
    """

    def __init__(self, config: dict[str, Any], *, model=None, tokenizer=None,
                 writer_record: dict[str, Any] | None = None):
        configure_numerics()
        self.config = copy.deepcopy(config)
        self.model = load_model(device=config["device"]) if model is None else model
        self.tokenizer = load_tokenizer() if tokenizer is None else tokenizer
        if hasattr(self.model, "looplora_adapters"):
            raise ValueError("close the previous experiment before reusing a backbone")
        self.model.eval()
        self.model.requires_grad_(False)
        self._closed = False
        stage, variant = config["stage"], config["variant"]
        self.oracle = config.get("writer", "learned") == "oracle_random"
        self.layer = config["model"]["layer"]
        self.input_site = config["model"].get("input_site", "o_proj")
        self.adapter = None
        self.writer = None
        if variant == "B0":
            if stage != "baseline":
                raise ValueError("B0 requires the baseline stage")
            self.bank = AdapterBank({})
            self.model.looplora_adapters = self.bank
            return
        core_variant = {"B1": "lora", "B2": "parallel", "B3": "loop", "B5": "fixedq",
                        "B8": "unshared"}.get(variant, "iterq")
        m = config["model"]
        cfg = AdapterConfig(d_in=1, d_out=1, r=m["r"], variant=core_variant,
                            rho=m["rho"], alpha=m["alpha"],
                            activation=config.get("activation", "identity" if variant == "M1-lin" else "relu"),
                            gate=m.get("gate", "scalar"),
                            unshared_steps=4 if variant == "B8" else 1,
                            lambda_init=m.get("lambda_init", .99), beta_init=m.get("beta_init", .1),
                            dpfp_nu=m.get("dpfp_nu", 0))
        init_seed = derive_seed(config["seed"], "llm", stage, "reader", variant)
        self.bank = attach_adapters(self.model, cfg, layers=(self.layer,), seed=init_seed,
                                    input_site=self.input_site)
        self.adapter = self.bank.modules[self.layer].adapter
        self.writer = self.adapter if self.adapter.has_memory else None
        try:
            if self.oracle:
                if (stage not in {"reader", "joint"} or config["protocol"] != "memory_only"
                        or self.adapter.cfg.r_mem != 32):
                    raise ValueError("oracle_random requires a memory-only reader with r_mem=32")
                self.writer = OracleRandomWriter(config["task"]["V"], config["task"]["n_rel"],
                                                  config.get("oracle_seed", 17)).to(self.device)
                # A encodes the question and remains trainable. Learned writer
                # projections/gates have no role in this condition.
                for name in WRITER_NAMES[1:]:
                    if hasattr(self.adapter, name):
                        getattr(self.adapter, name).requires_grad_(False)
            elif variant == "B3":
                writer_cfg = AdapterConfig(**{**asdict(self.adapter.cfg), "variant": "iterq"})
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(derive_seed(config["seed"], "llm", "external-writer"))
                    self.writer = make_adapter(writer_cfg).to(device=self.device, dtype=torch.float32)
                self.writer.A = self.adapter.A
                # B3 has no loss path through K/V/gates or the external reader.
                self.writer.requires_grad_(False)
                self.adapter.A.requires_grad_(stage != "reader")
            if stage == "reader" and not self.oracle:
                if self.writer is None:
                    raise ValueError("reader stage requires a memory writer")
                if writer_record is None:
                    checkpoint = config.get("writer_checkpoint")
                    if not checkpoint:
                        raise ValueError("reader stage requires writer_checkpoint")
                    writer_record = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self._load_writer(writer_record)
                for name in WRITER_NAMES:
                    if hasattr(self.writer, name):
                        getattr(self.writer, name).requires_grad_(False)
        except BaseException:
            self.close()
            raise

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def writer_signature(self) -> dict[str, Any]:
        if self.writer is None:
            return {}
        if self.oracle:
            return dict(writer="oracle_random", version=1, r_mem=32,
                        V=self.writer.V, n_rel=self.writer.n_rel, seed=self.writer.seed,
                        keys="global_entity_relation_pairs", values="OracleWriter.global_codes")
        cfg = self.writer.cfg
        signature = {"model": MODEL_ID, "revision": MODEL_REVISION, "layer": self.layer,
                "d_in": cfg.d_in, "d_out": cfg.d_out, "r": cfg.r, "r_mem": cfg.r_mem,
                "gate": cfg.gate, "write_policy": self.config["write_policy"],
                "V": self.config["task"]["V"]}
        if self.input_site != "o_proj":
            signature["input_site"] = self.input_site
        if cfg.dpfp_nu:
            signature["dpfp_nu"] = cfg.dpfp_nu
        return signature

    def writer_state(self) -> dict[str, Tensor]:
        if self.writer is None:
            return {}
        if self.oracle:
            return {name: value.detach().cpu().clone() for name, value in self.writer.state_dict().items()}
        return {name: getattr(self.writer, name).detach().cpu().clone()
                for name in WRITER_NAMES if hasattr(self.writer, name)}

    def _load_writer(self, record: dict[str, Any]) -> None:
        if record.get("signature") != self.writer_signature():
            raise ValueError("writer signature does not match backbone, insertion, rank, policy or vocabulary")
        state = record.get("writer", {})
        expected = self.writer_state()
        if set(state) != set(expected):
            raise ValueError("writer checkpoint has missing or unexpected parameters")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != torch.float32:
                raise ValueError(f"writer parameter {name} must have matching shape and fp32 dtype")
        if self.oracle:
            self.writer.load_state_dict(state, strict=True)
            return
        with torch.no_grad():
            for name, value in state.items():
                getattr(self.writer, name).copy_(value)

    @contextmanager
    def _support_writer(self, protocol: str):
        module = self.bank.modules.get(self.layer)
        if module is None or self.writer is self.adapter or self.writer is None:
            yield
            return
        if protocol != "memory_only":
            raise ValueError("the B3 external-writer protocol is memory_only")
        reader = module.adapter
        module.adapter = self.writer
        try:
            yield
        finally:
            module.adapter = reader

    def prepare(self, arrays, protocol=None, *, truncate=True, tokens=None) -> PreparedBatch:
        if self._closed:
            raise RuntimeError("experiment is closed")
        protocol = self.config["protocol"] if protocol is None else protocol
        if protocol not in {"memory_only", "context_retained"}:
            raise ValueError(f"unknown protocol: {protocol}")
        if self.oracle and protocol != "memory_only":
            raise ValueError("oracle_random supports memory_only only")
        tokens = (tokenize_batch(arrays, self.tokenizer, device=self.device, include_support=not self.oracle)
                  if tokens is None else tokens.to(self.device))
        B = tokens.support_ids.shape[0]
        if self.oracle:
            tokens = replace(tokens, support_ids=tokens.support_ids[:, :0],
                             support_mask=tokens.support_mask[:, :0], fact_final=tokens.fact_final[:, :0])
            facts = torch.as_tensor(arrays.facts, dtype=torch.long, device=self.device)
            valid = torch.as_tensor(arrays.fact_valid, dtype=torch.bool, device=self.device)
            snapshot = SupportSnapshot({self.layer: self.writer(facts, valid)}, None,
                                       tokens.support_mask.clone(), protocol)
        elif protocol == "memory_only" and self.writer is None:
            # Support never enters either the backbone attention or a writer.
            snapshot = SupportSnapshot(self.bank.zero_states(B), None, tokens.support_mask.clone(), protocol)
        else:
            with self._support_writer(protocol):
                snapshot = encode_support(
                    self.model, tokens.support_ids, tokens.support_mask,
                    self.config["write_policy"], fact_final=tokens.fact_final,
                    protocol=protocol, loop_steps=4 if self.config["variant"] == "B8" else 1,
                    truncate=truncate and protocol == "memory_only",
                )
        permitted = tokens.support_mask if self.config["write_policy"] == "all_tokens" else tokens.fact_final & tokens.support_mask
        writes = (valid.sum(-1) if self.oracle else permitted.sum(-1) if self.writer is not None
                  else torch.zeros(B, dtype=torch.long, device=self.device))
        return PreparedBatch(snapshot, tokens.question_ids, tokens.question_mask,
                             tokens.answer_ids, tokens.candidate_ids, tokens.candidate_mask,
                             arrays, int(tokens.support_mask.sum()), writes,
                             tokens.support_ids, tokens.support_mask, tokens.fact_final)

    def score(self, prepared: PreparedBatch, L: int, *, capture_inputs=False):
        if self._closed:
            raise RuntimeError("experiment is closed")
        validate_steps(L)
        B, Q, T = prepared.question_ids.shape
        snapshot = repeat_snapshot(prepared.snapshot, Q)
        captured = []
        layer = self.model.model.layers[self.layer]
        projection = layer if self.input_site == "residual" else layer.self_attn.o_proj
        hook = None
        if capture_inputs:
            hook = projection.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
        try:
            result = answer_scores(
                self.model, snapshot, prepared.question_ids.reshape(B * Q, T),
                prepared.question_mask.reshape(B * Q, T),
                prepared.answer_ids.reshape(B * Q, 1), loop_steps=L,
            )
        finally:
            if hook is not None:
                hook.remove()
        inputs = None
        if capture_inputs:
            if len(captured) != 1:
                raise RuntimeError("scoring must make exactly one projection call")
            inputs = captured[0].reshape(B, Q, T, -1)
        return result.logits[:, 0].reshape(B, Q, -1), inputs

    def trainable_parameters(self):
        return [] if self.adapter is None else [p for p in self.adapter.parameters() if p.requires_grad]

    def close(self) -> None:
        if not self._closed:
            detach_adapters(self.model)
            self.model.zero_grad(set_to_none=True)
            self._closed = True


def _fit(experiment: LLMExperiment, out: Path) -> dict[str, Any]:
    config, options = experiment.config, experiment.config["train"]
    spec = make_task_spec(config)
    depths = [1] if config["stage"] in {"writer", "lora_ctx"} or config["variant"] == "B4" else options["depths"]
    if config["variant"] == "B8":
        depths = [4]
    depth_rng = np.random.default_rng(derive_seed(config["seed"], "llm", config["stage"], "depths"))
    data_seed = derive_seed(config["seed"], "llm", "writer-data" if config["stage"] == "writer" else "train-data")
    params = experiment.trainable_parameters()
    steps = 0 if config["stage"] == "baseline" else options["steps"]
    optimizer = torch.optim.AdamW(params, lr=options["lr"], weight_decay=options["weight_decay"]) if steps else None
    before = experiment.writer_state() if config["stage"] == "reader" else None
    durations, chosen = [], []
    final_loss, train_tokens = None, 0
    synchronize(experiment.device)
    started = time.perf_counter()
    with (out / "train_log.jsonl").open("w") as log, BatchStream(
        spec, "train", options["batch_size"], data_seed, steps,
        tokenizer=experiment.tokenizer, workers=options.get("prefetch_workers", 0),
        include_support=not experiment.oracle,
    ) as batches:
        for step in range(steps):
            synchronize(experiment.device)
            begin = time.perf_counter()
            arrays, tokens = next(batches)
            L = int(depth_rng.choice(depths))
            warmup = min(1., (step + 1) / max(1, options["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = options["lr"] * warmup
            optimizer.zero_grad(set_to_none=True)
            batch = experiment.prepare(arrays, tokens=tokens)
            logits, _ = experiment.score(batch, L)
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            synchronize(experiment.device)
            seconds = time.perf_counter() - begin
            durations.append(seconds)
            chosen.append(L)
            train_tokens += batch.support_tokens + int(batch.question_mask.sum())
            if step == 0 or (step + 1) % options["log_every"] == 0 or step + 1 == steps:
                final_loss = float(loss.detach())
                row = {"stage": config["stage"], "step": step + 1, "L": L,
                       "mean_L": float(np.mean(chosen)), "loss": final_loss,
                       "accuracy": float((logits.argmax(-1) == batch.answer_ids).float().mean()),
                       "lr": options["lr"] * warmup, "grad_norm": float(grad_norm),
                       "step_seconds": seconds, "data_seed": data_seed}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
    synchronize(experiment.device)
    if before is not None:
        after = experiment.writer_state()
        if any(not torch.equal(value, after[name]) for name, value in before.items()):
            raise RuntimeError("frozen writer changed during reader training")
    return {"train_seconds": time.perf_counter() - started,
            "train_step_seconds": float(np.mean(durations)) if durations else 0.,
            "train_step_p95_seconds": float(np.quantile(durations, .95)) if durations else 0.,
            "seconds_per_1000_steps": float(np.mean(durations)) * 1000 if durations else 0.,
            "mean_L": float(np.mean(chosen)) if chosen else None, "steps": steps,
            "final_loss": final_loss, "data_seed": data_seed, "train_tokens": train_tokens,
            "train_queries": steps * options["batch_size"] * spec.Q,
            "writer_frozen": config["stage"] == "reader" or experiment.oracle,
            "trainable_parameters": sum(p.numel() for p in params)}


def train(config: dict[str, Any], out: Path, *, model=None, tokenizer=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("checkpoint.pt", "train_log.jsonl", "writer.pt")):
        raise FileExistsError(f"training output already exists: {out}")
    resolved = copy.deepcopy(config)
    resolved["task"] = asdict(make_task_spec(resolved))
    if resolved["stage"] == "writer":
        resolved["variant"] = "B8" if resolved["variant"] == "B8" else "M1"
        resolved["protocol"] = "memory_only"
    if resolved["stage"] == "lora_ctx":
        resolved["variant"] = "B2" if resolved["variant"] == "B2" else "B1"
        resolved["protocol"] = "context_retained"
    experiment = LLMExperiment(resolved, model=model, tokenizer=tokenizer)
    try:
        summary = _fit(experiment, out)
        writer = {"writer": experiment.writer_state(), "signature": experiment.writer_signature()}
        if experiment.writer is not None:
            torch.save({**writer, "config": resolved}, out / "writer.pt")
        summary["writer_checkpoint"] = str(out / "writer.pt") if experiment.writer is not None else None
        adapter = {} if experiment.adapter is None else {
            key: value.detach().cpu().clone() for key, value in experiment.adapter.state_dict().items()
        }
        torch.save({"format_version": 1, "config": resolved, "adapter": adapter,
                    **writer, "summary": summary}, out / "checkpoint.pt")
    except BaseException:
        experiment.close()
        raise
    return experiment, summary


def load_checkpoint(path, model=None, tokenizer=None, device=None) -> LLMExperiment:
    record = torch.load(path, map_location="cpu", weights_only=True)
    if record.get("format_version") != 1:
        raise ValueError("unsupported LLM checkpoint format")
    config = copy.deepcopy(record["config"])
    if device is not None:
        config["device"] = str(device)
    experiment = LLMExperiment(config, model=model, tokenizer=tokenizer, writer_record=record)
    try:
        if experiment.adapter is not None:
            experiment.adapter.load_state_dict(record["adapter"], strict=True)
        elif record["adapter"]:
            raise ValueError("B0 checkpoint cannot contain adapter parameters")
        if experiment.writer is not None:
            experiment._load_writer(record)
    except BaseException:
        experiment.close()
        raise
    return experiment
