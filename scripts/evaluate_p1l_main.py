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
from looplora.llm.config import recursive_merge, resolve_config, split_window
from looplora.llm.evaluate import _slice, evaluate
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.train import load_checkpoint, make_task_spec
from looplora.sym.evaluate import stratify_queries

from readout_p1l import path_entities, read_batch
from run_p1l_main import LABELS, ROOT, SEEDS, STUDIES, digest, labels_for_rank, runtime, save


def check_run(name: str, path: str, rank: int) -> dict:
    """Per-run checks shared by every study: complete record, unchanged checkpoint, rank, no smoke or confirmation use."""
    directory = Path(path)
    record = json.loads((directory / "record.json").read_text())
    checkpoint = directory / "checkpoint.pt"
    if record["status"] != "complete" or record["checkpoint_hash"] != digest(checkpoint):
        raise ValueError(f"incomplete or changed main checkpoint: {name}")
    cfg = record["config"]
    if cfg["model"]["r"] != rank:
        raise ValueError("main run rank differs from its condition manifest")
    if cfg["smoke"] or cfg["eval"]["allow_confirmation"] or cfg["eval"]["splits"] != ["val"]:
        raise ValueError("main training accessed confirmation data or used smoke settings")
    return cfg


def check_training(manifest: dict, authorization: dict, *, check_all: bool = True) -> None:
    """Before any test episode is generated: the training matrix, its lock and the authorization agree.

    A manifest without `study` is P1-L's; `study: "p1l-s"` selects check_study_training and the authorization must
    name the same study. P1-L's rules below run unchanged.
    """
    study = manifest.get("study", "p1l")
    if study not in STUDIES:
        raise ValueError(f"unknown study: {study!r}")
    if authorization.get("study", "p1l") != study:
        raise ValueError("the authorization names another study than the training manifest")
    if study != "p1l":
        return check_study_training(manifest, authorization, study, check_all=check_all)
    if "confirmation_window" in authorization:
        raise ValueError("a P1-L authorization names no confirmation window: its range is the registered one")
    rank = manifest.get("rank", 32)
    expected = {f"{label}-s{seed}" for label in labels_for_rank(rank) for seed in SEEDS}
    if rank == 32:
        expected.add("B0-s17")
    if set(manifest["runs"]) != expected:
        raise ValueError("confirmation requires all21 trained runs and B0 forr32, or all15 forr128")
    if authorization.get("status") != "allowed" or not authorization.get("orchestrator_decision"):
        raise ValueError("confirmation is held until the orchestrator decides WP-L4")
    conditions = authorization.get("condition_manifests", {})
    k = authorization.get("gated_conditions")
    if (not isinstance(k, int) or isinstance(k, bool) or k < 2 or k != len(conditions)
            or not {"oracle-r32", "oracle-r128"} <= conditions.keys()):
        raise ValueError("B3 requires the fixed count and manifests of every gated condition")
    if check_all:
        all_manifests = {label: json.loads(Path(path).read_text()) for label, path in conditions.items()}
        if any(other.get("condition") != label for label, other in all_manifests.items()):
            raise ValueError("global confirmation condition labels do not match their manifests")
        if (all_manifests["oracle-r32"].get("rank") != 32
                or all_manifests["oracle-r128"].get("rank") != 128):
            raise ValueError("global confirmation rank labels do not match their manifests")
        if manifest not in all_manifests.values():
            raise ValueError("current condition is absent from the global confirmation lock")
        for other in all_manifests.values():
            check_training(other, authorization, check_all=False)
    for name, path in manifest["runs"].items():
        cfg = check_run(name, path, rank)
        if name != "B0-s17" and (cfg["train"]["steps"] != manifest["S"]
                or cfg["train"]["lr"] != manifest["lrs"][name.rsplit("-s", 1)[0]]):
            raise ValueError("main training differs from its committed S/lr lock")


def check_study_training(manifest: dict, authorization: dict, study: str, *, check_all: bool = True) -> None:
    """P1L-S (docs/prereg/P1L-S.md): runs retrained under the oracle code WP-L6 selected plus P1-L's reused B3-L4 runs.

    One gated condition (k = 1, 95% CIs) and one registered confirmation window. The authorization names the committed
    WP-L6 decision by path and hash; every reused run must be the one P1-L's own training manifest lists.
    """
    profile = STUDIES[study]
    condition = profile["condition"]
    names = [f"{label}-s{seed}" for label in labels_for_rank(32, study) for seed in SEEDS]
    if manifest.get("rank") != 32 or manifest.get("condition") != condition or set(manifest["runs"]) != set(names):
        raise ValueError("P1L-S requires its retrained runs and the reused B3-L4 runs at r = 32, and no other run")
    if authorization.get("status") != "allowed" or not authorization.get("orchestrator_decision"):
        raise ValueError("confirmation is held until the orchestrator releases P1L-S")
    conditions, k = authorization.get("condition_manifests", {}), authorization.get("gated_conditions")
    if not isinstance(k, int) or isinstance(k, bool) or k != 1 or set(conditions) != {condition}:
        raise ValueError("P1L-S gates exactly one condition")
    if authorization.get("confirmation_window") != profile["window"]:
        raise ValueError(f"the authorization must name the {profile['window']} confirmation window")
    if check_all and json.loads(Path(conditions[condition]).read_text()) != manifest:
        raise ValueError("the training manifest is not the one the authorization locked")
    committed = ROOT / profile["decision"]
    if authorization.get("decision") != dict(path=profile["decision"], hash=digest(committed)):
        raise ValueError("the authorization does not name the committed WP-L6 decision")
    decision = json.loads(committed.read_text())
    if decision["status"] != "selected" or decision["selected"] != manifest.get("writer"):
        raise ValueError("the manifest's oracle code is not the one WP-L6 selected")
    reused = {name for name in names if name.rsplit("-s", 1)[0] in profile["reused"]}
    if set(manifest.get("reused", {})) != reused:
        raise ValueError("the reused runs are not exactly the registered ones")
    for name, path in manifest["runs"].items():
        label = name.rsplit("-s", 1)[0]
        cfg = check_run(name, path, 32)
        if name in reused:
            entry = manifest["reused"][name]
            source = json.loads(Path(entry["manifest"]).read_text())
            if (digest(Path(entry["manifest"])) != entry["manifest_hash"] or source.get("condition") != "oracle-r32"
                    or source.get("rank", 32) != 32 or "study" in source or source["runs"].get(name) != path
                    or entry["source"] != path or entry["checkpoint_hash"] != digest(Path(path) / "checkpoint.pt")):
                raise ValueError(f"reused run does not match P1-L's training manifest: {name}")
            if source["S"] != manifest["S"]:
                raise ValueError("P1L-S trains every reader with the S of P1-L's reused runs")
            steps, lr, writer = source["S"], source["lrs"][label], "oracle_random"
        else:
            steps, lr, writer = manifest["S"], manifest["lrs"][label], manifest["writer"]
        if cfg["writer"] != writer or cfg["oracle_seed"] != 17:
            raise ValueError(f"{name} was not trained with its registered oracle code")
        if cfg["train"]["steps"] != steps or cfg["train"]["lr"] != lr:
            raise ValueError("main training differs from its committed S/lr lock")


@torch.no_grad()
def swap_successors(experiment, cfg: dict, out: Path) -> dict:
    spec = make_task_spec(cfg, evaluation=True)
    split, = cfg["eval"]["splits"]
    generator_split, start = split_window(cfg["eval"], split)
    count, seed = cfg["eval"]["episodes"], cfg["eval"]["seed"]
    arrays = stratify_queries(generate(spec, generator_split, start, count, seed),
                             spec, cfg["eval"]["hops"], seed)
    swapped, _ = counterfactual(arrays, "swap_successors",
                               derive_seed(seed, "p1l", f"{split}-swap-successors"))
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
                query_id=selected, hop=original.q_hop_true.ravel(),
                counterfactual_hop=part.q_hop_true.ravel(), L=np.full(len(answer), depth),
                original_prediction=old, prediction=new, answer=answer,
                original_answer=original_batch.answer_ids.cpu().numpy().ravel()))
            for hop in sorted(set(original.q_hop_true.ravel().tolist())):
                keep = original.q_hop_true.ravel() == hop
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
    split, = cfg["eval"]["splits"]
    generator_split, start = split_window(cfg["eval"], split)
    episode_count, seed = cfg["eval"]["episodes"], cfg["eval"]["seed"]
    arrays = stratify_queries(generate(spec, generator_split, start, episode_count, seed),
                             spec, cfg["eval"]["hops"], seed)
    totals, faithful = {}, {}
    for begin in range(0, episode_count, cfg["eval"]["batch_size"]):
        part = _slice(arrays, slice(begin, min(episode_count, begin + cfg["eval"]["batch_size"])))
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
                        previous_n, correct = faithful.get(key, (0, 0))
                        faithful[key] = previous_n + int(keep.sum()), correct + int((prediction[keep] == actual[keep]).sum())
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


@torch.no_grad()
def inference_timing(experiment, cfg: dict, out: Path):
    """B4 descriptive score-path timing; fixed validation questions, no audit hook."""
    spec = make_task_spec(cfg, evaluation=True)
    arrays = stratify_queries(generate(spec, "val", 0, 16, 4242), spec, cfg["eval"]["hops"], 4242)
    depth = 1 if cfg["variant"] in {"B0", "B1", "B2", "B4"} else 4
    rows = []
    for protocol in cfg["eval"]["protocols"]:
        prepared = experiment.prepare(arrays, protocol=protocol)
        for _ in range(5):
            logits, _ = experiment.score(prepared, depth)
            logits.argmax(-1)
        durations = []
        for _ in range(20):
            torch.cuda.synchronize()
            before = time.perf_counter()
            logits, _ = experiment.score(prepared, depth, capture_inputs=False)
            logits.argmax(-1)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - before)
        questions = int(prepared.answer_ids.numel())
        rows.append(dict(protocol=protocol, L=depth, batch_episodes=16, batch_questions=questions,
            question_shape=list(prepared.question_ids.shape), warmups=5, repeats=20,
            mean_seconds_per_question=float(np.mean(durations)) / questions,
            median_seconds_per_question=float(np.median(durations)) / questions,
            batch_seconds=durations))
    adapter = experiment.adapter
    # oracle_shared's table C is a trained writer parameter outside the adapter: counted apart, and only when present.
    writer_parameters = sum(p.numel() for p in experiment.writer_trainable_parameters())
    save(out / "inference.json", dict(rank=cfg["model"]["r"],
        adapter_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters()),
        trainable_adapter_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        **({"writer_parameters": writer_parameters} if writer_parameters else {}),
        measurement="Amortized batched question score+full-vocabulary argmax, CUDA synchronized, "
            "single GPU process; includes frozen backbone, adapter and score-path CE; excludes "
            "support preparation/writing, tokenization and audit capture/hashing; fixed val[0,16), "
            "seed4242,8questions/episode,5warmups+20measured repeats.",
        timings=rows, **runtime()))


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
            if "confirmation_window" in authorization:  # P1L-S only; check_training rejects it for P1-L
                template = recursive_merge(template, {"eval": {"confirmation": authorization["confirmation_window"]}})
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
                        raise ValueError("name-gradient probe cache changed")
                    directions = torch.load(args.directions, map_location="cpu", weights_only=True)
                    readout(experiment, cfg, directions, out)
            inference_timing(experiment, cfg, out)
            save(out / "record.json", dict(status="complete", name=name,
                checkpoint_hash=digest(run / "checkpoint.pt"), seconds=time.perf_counter()-start,
                authorization_hash=digest(args.authorization), manifest_hash=digest(args.manifest), **runtime()))
        finally:
            experiment.close()


if __name__ == "__main__":
    main()
