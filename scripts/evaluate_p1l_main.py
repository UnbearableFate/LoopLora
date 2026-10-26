#!/usr/bin/env python3
"""One-shot confirmation evaluation after global training and writer gates."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_p4s_stopping import require_compute
from looplora.data.relations import counterfactual, generate
from looplora.infra.seeds import derive_seed
from looplora.llm.config import recursive_merge, resolve_config
from looplora.llm.evaluate import _slice, evaluate
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.train import load_checkpoint, make_task_spec
from looplora.sym.evaluate import stratify_queries

from readout_p1l import path_entities, read_batch
from run_p1l_main import LABELS, ROOT, SEEDS, digest, runtime, save


def check_training(manifest: dict, authorization: dict) -> None:
    expected = {f"{label}-s{seed}" for label in LABELS for seed in SEEDS} | {"B0-s17"}
    if set(manifest["runs"]) != expected:
        raise ValueError("confirmation requires all21 trained runs and the untrained B0")
    if (authorization.get("status") != "allowed" or authorization.get("writer_conditions") not in {1, 2}
            or not authorization.get("orchestrator_decision")):
        raise ValueError("confirmation is held until the orchestrator decides WP-L4")
    for name, path in manifest["runs"].items():
        directory = Path(path)
        record = json.loads((directory / "record.json").read_text())
        checkpoint = directory / "checkpoint.pt"
        if record["status"] != "complete" or record["checkpoint_hash"] != digest(checkpoint):
            raise ValueError(f"incomplete or changed main checkpoint: {name}")
        cfg = record["config"]
        if cfg["smoke"] or cfg["eval"]["allow_confirmation"] or cfg["eval"]["splits"] != ["val"]:
            raise ValueError("main training accessed confirmation data or used smoke settings")
        if name != "B0-s17" and (cfg["train"]["steps"] != manifest["S"]
                or cfg["train"]["lr"] != manifest["lrs"][name.rsplit("-s", 1)[0]]):
            raise ValueError("main training differs from its committed S/lr lock")


@torch.no_grad()
def swap_successors(experiment, cfg: dict, out: Path) -> dict:
    spec = make_task_spec(cfg, evaluation=True)
    arrays = stratify_queries(generate(spec, "test", 100000, 2000, 4242),
                             spec, cfg["eval"]["hops"], 4242)
    swapped, _ = counterfactual(arrays, "swap_successors",
                               derive_seed(4242, "p1l", "confirm-swap-successors"))
    # swap_successors returns allsource query labels; no oracle answer enters scoring.
    source_episode = np.asarray(swapped.meta["source_episode"])
    source_query = np.asarray(swapped.meta["source_query"])
    rows, samples = [], []
    for begin in range(0, len(swapped.episode_seed), cfg["eval"]["batch_size"]):
        end = min(len(swapped.episode_seed), begin + cfg["eval"]["batch_size"])
        part = _slice(swapped, slice(begin, end))
        original_indices = source_episode[begin:end]
        original = _slice(arrays, original_indices)
        # Original source has8 queries; score exactly the selected query for the control.
        selected = source_query[begin:end]
        for field in ("q_start", "q_rel", "q_hop_given", "q_hop_true", "answer"):
            value = getattr(original, field)
            setattr(original, field, value[np.arange(len(value)), selected][:, None])
        original_batch = experiment.prepare(original)
        swapped_batch = experiment.prepare(part)
        if (not torch.equal(original_batch.question_ids, swapped_batch.question_ids)
                or not torch.equal(original_batch.question_mask, swapped_batch.question_mask)):
            raise AssertionError("swap_successors changed source question tokens")
        for depth in ([1, 4, 8] if cfg["variant"] == "M1" else [4]):
            before, _ = experiment.score(original_batch, depth)
            after, _ = experiment.score(swapped_batch, depth)
            answer = swapped_batch.answer_ids.cpu().numpy().ravel()
            old = before.argmax(-1).cpu().numpy().ravel()
            new = after.argmax(-1).cpu().numpy().ravel()
            samples.append(dict(episode_id=arrays.episode_seed[original_indices],
                query_id=selected, hop=part.q_hop_true.ravel(), L=np.full(len(answer), depth),
                original_prediction=old, prediction=new, answer=answer,
                original_answer=original_batch.answer_ids.cpu().numpy().ravel()))
            for hop in sorted(set(part.q_hop_true.ravel().tolist())):
                keep = part.q_hop_true.ravel() == hop
                rows.append(dict(hop=int(hop), L=depth, n=int(keep.sum()),
                                 follows_new=int((new[keep] == answer[keep]).sum()),
                                 original_predicts_new=int((old[keep] == answer[keep]).sum())))
    result = []
    for depth, hop in sorted({(row["L"], row["hop"]) for row in rows}):
        same = [row for row in rows if row["hop"] == hop and row["L"] == depth]
        n = sum(row["n"] for row in same)
        result.append(dict(hop=hop, L=depth, n=n,
            follows_new=sum(row["follows_new"] for row in same) / n,
            original_predicts_new=sum(row["original_predicts_new"] for row in same) / n))
    np.savez_compressed(out / "swap-successors-samples.npz",
                        **{key: np.concatenate([row[key] for row in samples]) for key in samples[0]})
    save(out / "swap-successors.json", result)
    return dict(episodes=len(arrays.episode_seed), eligible_queries=sum(row["n"] for row in rows), table=result)


@torch.no_grad()
def readout(experiment, cfg: dict, directions: dict, out: Path):
    spec = make_task_spec(cfg, evaluation=True)
    arrays = stratify_queries(generate(spec, "test", 100000, 2000, 4242),
                             spec, cfg["eval"]["hops"], 4242)
    totals, faithful = {}, {}
    for begin in range(0, 2000, cfg["eval"]["batch_size"]):
        part = _slice(arrays, slice(begin, min(2000, begin + cfg["eval"]["batch_size"])))
        prepared = experiment.prepare(part)
        outputs, actual = read_batch(experiment, prepared, directions)
        paths, hops = path_entities(part).reshape(-1, 4), part.q_hop_true.ravel()
        for reader_step, lenses in enumerate(outputs, 1):
            for method, scores in lenses.items():
                prediction = scores.argmax(-1)
                if reader_step == 4:
                    for hop in np.unique(hops):
                        keep = hops == hop
                        key = method, int(hop)
                        count, correct = faithful.get(key, (0, 0))
                        faithful[key] = count + int(keep.sum()), correct + int((prediction[keep] == actual[keep]).sum())
                for entity_step in range(1, 5):
                    targets = paths[:, entity_step - 1]
                    for hop in np.unique(hops):
                        keep = (hops == hop) & (targets >= 0) & (entity_step <= hops)
                        if not keep.any():
                            continue
                        selected, labels = scores[keep], targets[keep]
                        target_score = selected[np.arange(len(labels)), labels]
                        ranks = 1 + (selected > target_score[:, None]).sum(-1)
                        key = method, int(hop), reader_step, entity_step
                        old = totals.get(key, (0, 0., 0, 0))
                        totals[key] = (old[0] + len(labels), old[1] + float(ranks.sum()),
                            old[2] + int((prediction[keep] == labels).sum()),
                            old[3] + int((target_score == selected.max(-1)).sum()))
    rows = [dict(method=method, hop=hop, reader_step=step, entity_step=entity_step,
                 n=n, rank_sum=rank_sum, mean_rank=rank_sum/n, top1_count=correct,
                 top1_rate=correct/n, tied_top1_count=tied, tied_top1_rate=tied/n)
            for (method, hop, step, entity_step), (n, rank_sum, correct, tied) in sorted(totals.items())]
    faithfulness = [dict(method=method, hop=hop, n=n, matches=count, rate=count/n)
                    for (method, hop), (n, count) in sorted(faithful.items())]
    save(out / "readout.json", dict(table=rows, faithfulness=faithfulness))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--directions", type=Path)
    args = parser.parse_args()
    require_compute()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    authorization = json.loads(args.authorization.read_text())
    check_training(manifest, authorization)  # before any test episode is generated
    args.out.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(), load_tokenizer()
    for name in args.runs:
        if name not in manifest["runs"]:
            raise ValueError(f"unknown main run: {name}")
        run = Path(manifest["runs"][name])
        experiment = load_checkpoint(run / "checkpoint.pt", model=model, tokenizer=tokenizer)
        out = args.out / name
        out.mkdir()
        start = time.perf_counter()
        try:
            base = copy.deepcopy(experiment.config)
            template = json.loads((ROOT / "configs/p1l/main/eval-confirmation.json").read_text())
            cfg = resolve_config(recursive_merge(base, template))
            if cfg["variant"] == "M1":
                cfg["eval"]["depths"] = list(range(1, 9))
            for task in ("primary", "typed"):
                current = cfg
                if task == "typed":
                    template = json.loads((ROOT / "configs/p1l/main/eval-typed-replication.json").read_text())
                    current = resolve_config(recursive_merge(cfg, template))
                metrics, timing = evaluate(experiment, current, out / task)
                save(out / task / "timing.json", timing)
                save(out / task / "config.json", current)
            if cfg["variant"] in {"M1", "B8"}:
                swap_successors(experiment, cfg, out)
                if args.directions:
                    record = json.loads((args.directions.parent / "record.json").read_text())
                    if record["trial"] or record["questions"] != 512 or record["names"] != 1024:
                        raise ValueError("readout requires the full registered direction cache")
                    if record["file_hash"] != digest(args.directions):
                        raise ValueError("J-lens cache changed")
                    directions = torch.load(args.directions, map_location="cpu", weights_only=True)
                    readout(experiment, cfg, directions, out)
            save(out / "record.json", dict(status="complete", name=name,
                checkpoint_hash=digest(run / "checkpoint.pt"), seconds=time.perf_counter()-start,
                authorization_hash=digest(args.authorization), manifest_hash=digest(args.manifest), **runtime()))
        finally:
            experiment.close()


if __name__ == "__main__":
    main()
