"""One-GH200 synthetic throughput pilot; run inside a Miyabi-G PBS allocation.

python -m looplora.llm.bench --config configs/pilot_throughput.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time
from typing import Any, Callable

from looplora.infra.seeds import derive_seed


@dataclass(frozen=True)
class PilotConfig:
    ranks: tuple[int, ...] = (16, 32)
    loop_steps: tuple[int, ...] = (0, 1, 2, 4, 8)
    batch_sizes: tuple[int, ...] = (16, 64)
    layers: tuple[int, ...] = (14,)
    support_tokens: int = 64
    question_tokens: int = 16
    answer_tokens: int = 1
    warmup: int = 2
    repeats: int = 5
    seed: int = 20261026
    learning_rate: float = 0.001
    protocol: str = "memory_only"
    write_policy: str = "all_tokens"
    max_elapsed_seconds: int = 1080
    output_dir: str = "runs/pilot"

    def __post_init__(self) -> None:
        for key, expected in (("ranks", (16, 32)), ("loop_steps", (0, 1, 2, 4, 8)),
                              ("batch_sizes", (16, 64))):
            if tuple(getattr(self, key)) != expected:
                raise ValueError(f"{key} must be {expected}: this pilot requires the full grid")
        if self.repeats < 5 or self.warmup < 1:
            raise ValueError("the pilot requires >=5 repeats and >=1 warmup")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be a nonempty list without duplicates")
        for key in ("support_tokens", "question_tokens", "answer_tokens", "max_elapsed_seconds"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.protocol != "memory_only" or self.write_policy != "all_tokens":
            raise ValueError("this pilot measures memory_only with all_tokens writes")

    @classmethod
    def from_yaml(cls, path: Path) -> PilotConfig:
        import yaml

        values = yaml.safe_load(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("pilot config must be a mapping")
        for key in ("ranks", "loop_steps", "batch_sizes", "layers"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def cells(self) -> list[tuple[int, int, int]]:
        return list(product(self.ranks, self.loop_steps, self.batch_sizes))


def _atomic_write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _summarize(samples: list[dict[str, float | int]], tokens: int) -> dict[str, Any]:
    seconds = [float(sample["seconds"]) for sample in samples]
    return {
        "samples": samples,
        "tokens_per_repeat": tokens,
        "median_seconds": statistics.median(seconds),
        "mean_seconds": statistics.mean(seconds),
        "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        "tokens_per_second": tokens / statistics.median(seconds),
        "peak_allocated_bytes": max(int(s["peak_allocated_bytes"]) for s in samples),
        "peak_reserved_bytes": max(int(s["peak_reserved_bytes"]) for s in samples),
        "peak_extra_allocated_bytes": max(int(s["peak_extra_allocated_bytes"]) for s in samples),
    }


def _measure(operation: Callable[[], None], *, cuda: Any, cfg: PilotConfig, tokens: int,
             prepare: Callable[[], None] | None = None) -> dict[str, Any]:
    for _ in range(cfg.warmup):
        if prepare is not None:
            prepare()
        operation()
        cuda.synchronize()
    samples = []
    for _ in range(cfg.repeats):
        if prepare is not None:
            prepare()
        gc.collect()
        cuda.synchronize()
        baseline = cuda.memory_allocated()
        cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        operation()
        cuda.synchronize()
        seconds = time.perf_counter() - start
        peak = cuda.max_memory_allocated()
        samples.append({
            "seconds": seconds,
            "tokens_per_second": tokens / seconds,
            "baseline_allocated_bytes": baseline,
            "peak_allocated_bytes": peak,
            "peak_reserved_bytes": cuda.max_memory_reserved(),
            "peak_extra_allocated_bytes": peak - baseline,
        })
    return _summarize(samples, tokens)


def _provenance(config_path: Path | None) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                                check=True)
        return result.stdout.strip()

    root = Path(__file__).resolve().parents[3]
    source_files = sorted((root / "src" / "looplora").rglob("*.py"))
    digests = {str(path.relative_to(root)): hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
               for path in source_files}
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_blake2b": digests,
        "config_path": str(config_path.resolve()) if config_path else None,
        "config_text": config_path.read_text() if config_path else None,
        "python": platform.python_version(),
        "hostname": socket.gethostname(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "hf_home": os.environ.get("HF_HOME"),
        "threads": {key: os.environ.get(key) for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
    }


def _detached_snapshot(snapshot: Any) -> Any:
    # A new dictionary and fresh tensors keep every timing call independent of
    # the bank's live context and prevent backward through the support writer.
    return replace(snapshot, states={key: value.detach().clone() for key, value in snapshot.states.items()},
                   attention_mask=snapshot.attention_mask.detach().clone(), past_key_values=None)


def _cell(model: Any, cfg: PilotConfig, rank: int, loops: int, batch: int,
          save_phase: Callable[[str, dict[str, Any]], None]) -> None:
    import torch

    from looplora.core.config import AdapterConfig
    from looplora.llm.inject import attach_adapters, detach_adapters
    from looplora.llm.protocols import answer_scores, encode_support

    cell_seed = derive_seed(cfg.seed, "throughput", rank, loops, batch)
    torch.manual_seed(cell_seed)
    torch.cuda.manual_seed_all(derive_seed(cell_seed, "cuda"))
    base_parameters = tuple(model.parameters())
    bank = attach_adapters(model, AdapterConfig(variant="iterq", d_in=2048, d_out=1024, r=rank),
                           layers=cfg.layers, seed=derive_seed(cell_seed, "adapters"))
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(derive_seed(cell_seed, "synthetic-tokens"))
    # Ordinary in-vocabulary IDs, excluding special IDs. No linguistic scores
    # are interpreted in this synthetic timing experiment.
    special_ids = {value for value in (model.config.bos_token_id, model.config.eos_token_id,
                                      model.config.pad_token_id) if isinstance(value, int)}

    def inputs(length: int) -> Any:
        ids = torch.randint(0, model.config.vocab_size, (batch, length), generator=generator,
                            device=device)
        replacement = next(value for value in range(model.config.vocab_size) if value not in special_ids)
        for value in special_ids:
            ids.masked_fill_(ids == value, replacement)
        return ids

    support_ids, question_ids, answer_ids = (inputs(length) for length in
                                            (cfg.support_tokens, cfg.question_tokens, cfg.answer_tokens))
    support_mask = torch.ones_like(support_ids)
    question_mask = torch.ones_like(question_ids)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=cfg.learning_rate)
    if not parameters or any(parameter.requires_grad for parameter in base_parameters):
        raise RuntimeError("expected trainable adapters on a frozen backbone")
    if any(parameter.dtype != torch.float32 for module in bank.modules.values()
           for parameter in module.adapter.parameters()):
        raise RuntimeError("adapter parameters must be fp32")

    def encode() -> Any:
        with torch.no_grad():
            return encode_support(model, support_ids, support_mask, write_policy=cfg.write_policy,
                                  protocol=cfg.protocol, loop_steps=loops)

    def support_step() -> None:
        encode()

    try:
        save_phase("support_encoding", _measure(support_step, cuda=torch.cuda, cfg=cfg,
                                                tokens=batch * cfg.support_tokens))
        snapshot = _detached_snapshot(encode())

        def read_step() -> None:
            with torch.no_grad():
                answer_scores(model, snapshot, question_ids, question_mask, answer_ids, loop_steps=loops)

        reader_tokens = batch * (cfg.question_tokens + cfg.answer_tokens - 1)
        save_phase("question_reading", _measure(read_step, cuda=torch.cuda, cfg=cfg, tokens=reader_tokens))

        def train_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            scores = answer_scores(model, snapshot, question_ids, question_mask, answer_ids,
                                   loop_steps=loops)
            if scores.loss is None or not scores.loss.requires_grad:
                raise RuntimeError("reader loss must retain its graph through frozen upper layers")
            scores.loss.backward()
            optimizer.step()

        # Only timing warmup: move B off its zero initialization so the measured
        # backward exercises internal reader gradients. Re-encode at those exact
        # weights, then restore the same weights before every subsequent step.
        train_step()
        optimizer.zero_grad(set_to_none=True)
        snapshot = _detached_snapshot(encode())
        baseline_parameters = [parameter.detach().clone() for parameter in parameters]

        def restore() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for parameter, baseline in zip(parameters, baseline_parameters, strict=True):
                    parameter.copy_(baseline)

        training = _measure(train_step, cuda=torch.cuda, cfg=cfg, tokens=reader_tokens, prepare=restore)
        if any(parameter.grad is not None for parameter in base_parameters):
            raise RuntimeError("frozen backbone received gradients")
        if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters
               if parameter.grad is not None):
            raise RuntimeError("reader training produced nonfinite adapter gradients")
        active_gradients = [name for name, parameter in model.named_parameters()
                            if parameter.requires_grad and parameter.grad is not None
                            and bool(parameter.grad.ne(0).any())]
        if not active_gradients:
            raise RuntimeError("reader training produced no nonzero adapter gradients")
        training.update({"optimizer": "SGD", "learning_rate": cfg.learning_rate,
                         "includes": ["zero_grad", "reader_forward", "loss_backward", "optimizer_step"],
                         "scored_answer_tokens": batch * cfg.answer_tokens,
                         "untimed_adapter_activation_steps": 1,
                         "identical_parameters_each_repeat": True,
                         "support_snapshot_detached": True,
                         "nonzero_gradient_parameters": active_gradients,
                         "backbone_gradients_absent": True})
        save_phase("reader_training_step", training)
    finally:
        optimizer.zero_grad(set_to_none=True)
        detach_adapters(model)


def run_benchmark(cfg: PilotConfig, output: Path, *, config_path: Path | None = None) -> dict[str, Any]:
    import torch
    import transformers

    from looplora.llm.inject import load_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("pilot requires exactly one visible CUDA GPU in a Miyabi-G allocation")
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": 1, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg), "provenance": _provenance(config_path), "expected_cells": len(cfg.cells()),
        "model": {"id": "Qwen/Qwen3-0.6B-Base", "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                  "backbone_dtype": "bfloat16", "adapter_dtype": "float32", "memory_dtype": "float32",
                  "attention": "sdpa", "backbone_frozen": True},
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "measurement": {"clock": "perf_counter with CUDA synchronize before/after",
                        "model_mode": "eval (dropout disabled), gradients enabled for reader training",
                        "support_and_reading": "no_grad inference",
                        "reader_token_definition": "batch * (question_tokens + answer_tokens - 1) backbone input tokens",
                        "peak_memory": "absolute process allocated/reserved and allocated growth per sample",
                        "synthetic_tokens": True, "task_accuracy_measured": False},
        "cells": [],
    }
    _atomic_write(output, record)
    try:
        model = load_model(dtype=torch.bfloat16, device="cuda")
        model.eval()
        for rank, loops, batch in cfg.cells():
            if time.monotonic() - started >= cfg.max_elapsed_seconds:
                raise TimeoutError("pilot time budget reached; completed cells are preserved")
            gc.collect()
            torch.cuda.empty_cache()
            cell: dict[str, Any] = {"rank": rank, "loop_steps": loops, "batch_size": batch,
                                    "seed": derive_seed(cfg.seed, "throughput", rank, loops, batch),
                                    "status": "running", "phases": {}}
            record["cells"].append(cell)
            _atomic_write(output, record)

            def save_phase(name: str, result: dict[str, Any]) -> None:
                cell["phases"][name] = result
                record["elapsed_seconds"] = time.monotonic() - started
                _atomic_write(output, record)

            _cell(model, cfg, rank, loops, batch, save_phase)
            cell["status"] = "complete"
            _atomic_write(output, record)
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        if record["cells"] and record["cells"][-1]["status"] == "running":
            record["cells"][-1]["status"] = "failed"
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(output, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_throughput.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = PilotConfig.from_yaml(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or Path(cfg.output_dir) / f"throughput-{stamp}.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace existing pilot record: {output}")
    run_benchmark(cfg, output, config_path=args.config)
    print(output)


if __name__ == "__main__":
    main()
