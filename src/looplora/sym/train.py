"""Deterministic episode minibatches, shared T1 writers, and reader training."""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from looplora.data.relations import TaskSpec, generate
from looplora.infra.seeds import derive_seed
from .config import pretrain_task_spec
from .model import SymbolicModel, make_task_spec
from .writers import FactorizedWriter, LearnedWriter


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _fit(model: SymbolicModel, config: dict[str, Any], out: Path,
         stage: str) -> dict[str, Any]:
    options = config["train"]
    spec = make_task_spec(config)
    depths = options["depths"]
    if config["variant"] in {"B4", "B6", "B7"}:
        depths = [1]
    elif config["variant"] == "B8":
        depths = [4]
    depth_rng = np.random.default_rng(derive_seed(config["seed"], stage, "depths"))
    data_seed = derive_seed(config["seed"], "train_data" if stage == "train" else "pretrain_data")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=options["lr"], weight_decay=options["weight_decay"])
    model.train()
    synchronize(model.device)
    started = time.perf_counter()
    elapsed_steps, chosen_depths, final_loss = [], [], None
    with (out / "train_log.jsonl").open("a") as log:
        for step in range(options["steps"]):
            synchronize(model.device)
            step_start = time.perf_counter()
            arrays = generate(spec, "train", step * options["batch_size"],
                              options["batch_size"], data_seed)
            L = int(depth_rng.choice(depths))
            warmup = min(1.0, (step + 1) / max(1, options["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = options["lr"] * warmup
            optimizer.zero_grad(set_to_none=True)
            logits, batch = model(arrays, L)
            if not batch.candidate_mask.gather(1, batch.answer).all():
                raise ValueError("training answer absent from episode candidate set")
            restricted = logits.masked_fill(~batch.candidate_mask[:, None, :], -torch.inf)
            loss = F.cross_entropy(restricted.flatten(0, 1), batch.answer.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {stage} loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            synchronize(model.device)
            step_seconds = time.perf_counter() - step_start
            elapsed_steps.append(step_seconds)
            chosen_depths.append(L)
            if (step + 1) % options["log_every"] == 0 or step == 0 or step + 1 == options["steps"]:
                final_loss = float(loss.detach())
                row = {"stage": stage, "step": step + 1, "L": L,
                       "mean_L": float(np.mean(chosen_depths)), "loss": final_loss,
                       "accuracy": float((restricted.argmax(-1) == batch.answer).float().mean()),
                       "lr": options["lr"] * warmup, "grad_norm": float(grad_norm),
                       "step_seconds": step_seconds, "data_seed": data_seed}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
    synchronize(model.device)
    return {"train_seconds": time.perf_counter() - started,
            "train_step_seconds": float(np.mean(elapsed_steps)) if elapsed_steps else 0.0,
            "train_step_p95_seconds": float(np.quantile(elapsed_steps, .95)) if elapsed_steps else 0.0,
            "mean_L": float(np.mean(chosen_depths)) if chosen_depths else None,
            "steps": options["steps"], "final_loss": final_loss, "data_seed": data_seed,
            "train_tokens": options["steps"] * options["batch_size"] * (spec.E + spec.Q),
            "train_queries": options["steps"] * options["batch_size"] * spec.Q}


def _pretrain_config(config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    # Task overrides preserve the reader's feature vocabulary and relation table.
    result["task"] = asdict(pretrain_task_spec(config))
    result["variant"], result["activation"], result["writer"] = "B4", "relu", "joint"
    if config["writer"] == "t1_frozen_fact":
        result["writer"] = "joint_fact"
    result["train"].update({key: value for key, value in config["pretrain"].items()
                            if key in result["train"]})
    result["train"]["depths"] = [1]
    return result


def train(config: dict[str, Any], out: Path) -> tuple[SymbolicModel, dict[str, Any]]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint.pt").exists() or (out / "train_log.jsonl").exists():
        raise FileExistsError(f"training output already exists: {out}")
    pretrain_seconds = 0.0
    pretrain_summary: dict[str, Any] | None = None
    model = SymbolicModel(config)
    if config["writer"] in {"t1_frozen", "t1_frozen_fact"}:
        checkpoint = config["pretrain"].get("writer_checkpoint")
        if checkpoint:
            writer_record = torch.load(checkpoint, map_location="cpu", weights_only=True)
        else:
            stage1 = _pretrain_config(config)
            teacher = SymbolicModel(stage1, reader_label="writer_pretrain_reader")
            pretrain_summary = _fit(teacher, stage1, out, "pretrain")
            pretrain_seconds = pretrain_summary["train_seconds"]
            writer_record = {"writer": {k: v.detach().cpu().clone() for k, v in teacher.writer.state_dict().items()},
                             "signature": teacher.writer_signature(), "pretrain": pretrain_summary,
                             "pretrain_config": stage1}
            del teacher
        if writer_record["signature"] != model.writer_signature():
            raise ValueError("pretrained writer vocabulary/backbone/rank does not match reader")
        model.writer.load_state_dict(writer_record["writer"], strict=True)
        model.freeze_writer()
        torch.save(writer_record, out / "writer.pt")
    summary = _fit(model, config, out, "train")
    summary["pretrain_seconds"] = pretrain_seconds
    summary["pretrain"] = pretrain_summary
    summary["writer_frozen"] = config["writer"] not in {"joint", "joint_fact"}
    exports_writer = isinstance(model.writer, (LearnedWriter, FactorizedWriter))
    summary["writer_checkpoint"] = str(out / "writer.pt") if exports_writer else None
    summary["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary["registered_parameters"] = sum(p.numel() for p in model.parameters())
    if config["writer"] in {"joint", "joint_fact"} and exports_writer:
        torch.save({"writer": {key: value.detach().cpu().clone()
                               for key, value in model.writer.state_dict().items()},
                    "signature": model.writer_signature(),
                    "training_config": copy.deepcopy(config),
                    "training": copy.deepcopy(summary)}, out / "writer.pt")
    torch.save({"config": config, "model": model.state_dict(), "summary": summary}, out / "checkpoint.pt")
    return model, summary


def load_checkpoint(path: str | Path, device: str | None = None) -> SymbolicModel:
    record = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint_config = copy.deepcopy(record["config"])
    config = copy.deepcopy(checkpoint_config)
    if device is not None:
        config["device"] = device
    model = SymbolicModel(config)
    model.checkpoint_config = checkpoint_config
    model.load_state_dict(record["model"], strict=True)
    if config["writer"] in {"t1_frozen", "t1_frozen_fact"}:
        model.freeze_writer()
    return model.eval()
