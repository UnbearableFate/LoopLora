#!/usr/bin/env python3
"""Validation-only, timing-only acceptance of the full frozen-reader eval path."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from audit_p4s_stopping import require_compute
from evaluate_p1l_main import inference_timing, readout, swap_successors
from looplora.data.relations import generate
from looplora.llm.config import recursive_merge, resolve_config
from looplora.llm.evaluate import evaluate
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.train import load_checkpoint, make_task_spec
from looplora.sym.evaluate import stratify_queries
from readout_p1l import reader_states
from run_p1l_main import ROOT, digest, runtime, save


@torch.no_grad()
def audit_answer_branches(experiment, cfg):
    spec = make_task_spec(cfg, evaluation=True)
    arrays = stratify_queries(generate(spec, "val", 0, 16, 4242), spec, cfg["eval"]["hops"], 4242)
    prepared = experiment.prepare(arrays)
    frames = []
    original = experiment.adapter.branch
    def capture(z):
        value = original(z)
        frames.append(value)
        return value
    with patch.object(experiment.adapter, "branch", side_effect=capture):
        _, inputs = experiment.score(prepared, 4, capture_inputs=True)
    valid = prepared.question_mask.flatten(0, 1)
    b, t = valid.shape
    last = torch.where(valid, torch.arange(t, device=valid.device), -1).amax(-1)
    actual = torch.zeros((b, experiment.adapter.cfg.d_out), device=valid.device)
    positions = valid.any(0).nonzero(as_tuple=True)[0]
    if len(frames) != len(positions):
        raise AssertionError("readout branch calls do not match valid token positions")
    for position, frame in zip(positions, frames, strict=True):
        rows = valid[:, position].nonzero(as_tuple=True)[0]
        keep = last[rows] == position
        actual[rows[keep]] = frame[keep]
    x = inputs.flatten(0, 1)[torch.arange(b, device=valid.device), last].float()
    u = experiment.adapter.u(x)
    memory = prepared.snapshot.states[experiment.layer].repeat_interleave(prepared.answer_ids.shape[1], 0)
    recomputed = experiment.adapter.branch(reader_states(experiment.adapter, u, memory)[-1])
    return dict(bitwise_equal=bool(torch.equal(actual, recomputed)),
                max_abs_error=float((actual - recomputed).abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(1)
    record = json.loads((args.checkpoint.parent / "record.json").read_text())
    if record["status"] != "complete" or record["checkpoint_hash"] != digest(args.checkpoint):
        raise ValueError("evaluation preflight requires a complete immutable pilot checkpoint")
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(), load_tokenizer()
    experiment = load_checkpoint(args.checkpoint, model=model, tokenizer=tokenizer)
    try:
        base = copy.deepcopy(experiment.config)
        base["smoke"] = True
        base["eval"].update(episodes=200, hops=list(range(1, 9)), splits=["val"],
                            allow_confirmation=False, counterfactuals=False, protocols=["memory_only"])
        if base["variant"] == "M1":
            base["eval"]["depths"] = list(range(1, 9))
        cfg = resolve_config(base)
        result = dict(status="running", episodes=200, timing_only=True, **runtime(),
                      checkpoint_hash=digest(args.checkpoint), phases={})
        started = time.perf_counter()
        for setting in ("primary", "typed"):
            current = cfg
            if setting == "typed":
                task = json.loads((ROOT / "configs/p1l/main/eval-typed-replication.json").read_text())["task"]
                current = resolve_config(recursive_merge(cfg, dict(task=task)))
            before = time.perf_counter()
            _, timing = evaluate(experiment, current, args.out / setting)
            result["phases"][setting] = dict(seconds=time.perf_counter()-before,
                                                evaluation_seconds=timing["evaluation_seconds"])
        before = time.perf_counter()
        swap_successors(experiment, cfg, args.out)
        result["phases"]["swap"] = dict(seconds=time.perf_counter()-before)
        direction_record = json.loads((args.directions.parent / "record.json").read_text())
        if direction_record["file_hash"] != digest(args.directions) or direction_record["trial"]:
            raise ValueError("invalid fixed name-gradient probe cache")
        directions = torch.load(args.directions, map_location="cpu", weights_only=True)
        before = time.perf_counter()
        readout(experiment, cfg, directions, args.out)
        result["phases"]["readout"] = dict(seconds=time.perf_counter()-before)
        inference_timing(experiment, cfg, args.out)
        result["branch_audit"] = audit_answer_branches(experiment, cfg)
        result.update(status="complete", total_seconds=time.perf_counter()-started)
        save(args.out / "timing.json", result)
        print(json.dumps(result), flush=True)
    finally:
        experiment.close()


if __name__ == "__main__":
    main()
