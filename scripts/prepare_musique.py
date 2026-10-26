"""Prepare offline MuSiQue and retention data; execute via scripts/submit.py."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

from looplora.data.musique import (
    DATASET_ID, DATASET_REVISION, DEFAULT_ROOT, HOP_TYPES, SOURCE_COUNTS, VERSION,
    MuSiQueEpisode, answer_substitution, build_donor_pool, episode_path, file_digest,
    make_splits, read_examples, remove_final_support, render_example, write_episodes,
)

OFFICIAL_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    result = {"count": len(values), "min": ordered[0], "max": ordered[-1],
              "mean": sum(values) / len(values)}
    for percentile in (50, 90, 95, 99):
        result[f"p{percentile}"] = ordered[math.ceil(len(values) * percentile / 100) - 1]
    return result


def length_stats(episodes: list[MuSiQueEpisode]) -> dict[str, Any]:
    return {hop: {name: distribution([len(getattr(episode, name + "_ids"))
                                      for episode in episodes if episode.hop_type == hop])
                  for name in ("context", "question", "answer")}
            for hop in HOP_TYPES}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_musique(raw_dir: Path, output_dir: Path, *, base_seed: int = 20261026,
                    n_distractors: int = 0, context_cap: int | None = None) -> dict[str, Any]:
    from looplora.llm.inject import MODEL_ID, MODEL_REVISION, load_tokenizer

    begin = perf_counter()
    examples, sources = {}, {}
    for source_split, count in SOURCE_COUNTS.items():
        path = raw_dir / f"musique_ans_v1.0_{source_split}.jsonl"
        examples[source_split] = read_examples(path, expected_count=count)
        sources[source_split] = {"path": str(path.resolve()), "count": count,
                                 "bytes": path.stat().st_size, "blake2b_128": file_digest(path),
                                 "url": f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
                                        f"{DATASET_REVISION}/{path.name}"}
    splits = make_splits(examples["train"], examples["dev"])
    tokenizer = load_tokenizer()
    rendered = {}
    for split, pool in splits.items():
        rendered[split] = [render_example(example, tokenizer=tokenizer, base_seed=base_seed,
                                          n_distractors=n_distractors) for example in pool]
        print(json.dumps({"stage": "rendered", "split": split, "count": len(pool)}), flush=True)
    cap_method = {"source": "train", "percentile": 95, "round_up_to": 128}
    if context_cap is None:
        train_lengths = [len(episode.context_ids) for episode in rendered["train"]]
        context_cap = math.ceil(distribution(train_lengths)["p95"] / 128) * 128
    else:
        if context_cap < 1:
            raise ValueError("context cap must be positive")
        cap_method = {"source": "explicit_argument"}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": VERSION, "dataset": DATASET_ID, "revision": DATASET_REVISION,
        "source_files": sources, "tokenizer": {"model": MODEL_ID, "revision": MODEL_REVISION},
        "source_counts_reference": "https://aclanthology.org/2022.tacl-1.31.pdf (Table 2)",
        "base_seed": base_seed, "n_distractors": n_distractors,
        "split_rule": {"val_count": 1000, "stratification": "proportional_largest_remainder",
                       "ranking": "blake2b-128(UTF-8 id), then id",
                       "test": "big-endian blake2b-128(id) modulo 3 == 0", "confirm": "remaining dev ids"},
        "context_cap": context_cap, "context_cap_method": cap_method,
        "official_metric": {
            "source": f"https://github.com/StonyBrookNLP/musique/blob/{OFFICIAL_REVISION}/metrics/answer.py",
            "differences": "No per-example answer EM/F1 difference; return unrounded scores. "
                           "Official CLI rounds aggregate scores to 3 decimals. No support/sufficiency metric.",
        },
        "protocols": {
            "context_retained": "Support context stays in backbone attention/KV.",
            "memory_only": "Encode support, discard support backbone KV, retain declared memory only.",
            "closed_book": "Question only; no support encoding, support KV or carried memory.",
            "write_policies": "All context tokens, or paragraph_final_mask passed as fact_final.",
            "remove_final_support": "Sensitivity only: unchanged targets; report accuracy drop; "
                                    "do not treat refusal as an answer error.",
        },
        "leakage_note": "val may share single-hop sub-questions with train; MuSiQue limits "
                        "single-hop leakage only across its official splits.",
        "splits": {}, "files": {},
    }
    train_subquestions = {step for example in splits["train"] for step in example.decomposition_ids}
    val_subquestions = {step for example in splits["val"] for step in example.decomposition_ids}
    manifest["shared_train_val_subquestion_ids"] = len(train_subquestions & val_subquestions)
    for split, all_episodes in rendered.items():
        retained = [episode for episode in all_episodes if len(episode.context_ids) <= context_cap]
        dropped = [episode for episode in all_episodes if len(episode.context_ids) > context_cap]
        retained_ids = {episode.id for episode in retained}
        donors = build_donor_pool([example for example in splits[split] if example.id in retained_ids])
        substitutions, removals = [], []
        cf_over_cap = Counter()
        for episode in retained:
            substituted = answer_substitution(episode, donors, tokenizer=tokenizer, base_seed=base_seed)
            if substituted is not None:
                if len(substituted.context_ids) <= context_cap:
                    substitutions.append(substituted)
                else:
                    cf_over_cap[episode.hop_type] += 1
            removals.append(remove_final_support(episode, tokenizer=tokenizer))
        hop_counts = Counter(episode.hop_type for episode in all_episodes)
        kept_counts = Counter(episode.hop_type for episode in retained)
        coverage_counts = Counter(episode.hop_type for episode in substitutions)
        shuffle = {}
        for hop in HOP_TYPES:
            group = [episode for episode in all_episodes if episode.hop_type == hop]
            matches = 0
            expected = 0.0
            for episode in group:
                order = tuple(paragraph.index for paragraph in episode.paragraphs if paragraph.is_supporting)
                decomposition = tuple(dict.fromkeys(episode.meta["decomposition_order"]))
                matches += order == decomposition
                expected += 1 / math.factorial(len(order))
            shuffle[hop] = {"count": len(group), "matches_decomposition": matches,
                            "chance_expected_matches": expected}
        budget = max((len(tokenizer.encode(" " + answer, add_special_tokens=False))
                      for episode in retained
                      for answer in (episode.answer_text[1:], *episode.answer_aliases)), default=1)
        manifest["splits"][split] = {
            "source_count": len(all_episodes), "count": len(retained),
            "source_ids": [episode.id for episode in all_episodes],
            "ids": [episode.id for episode in retained], "dropped_ids": [episode.id for episode in dropped],
            "hop_counts_before_cap": dict(hop_counts), "hop_counts": dict(kept_counts),
            "hop_proportions_before_cap": {hop: hop_counts[hop] / len(all_episodes) for hop in HOP_TYPES},
            "hop_proportions": {hop: kept_counts[hop] / len(retained) if retained else 0 for hop in HOP_TYPES},
            "dropped_by_hop": dict(Counter(episode.hop_type for episode in dropped)),
            "lengths_before_cap": length_stats(all_episodes), "lengths_after_cap": length_stats(retained),
            "paragraph_order": shuffle,
            "answer_substitution": {
                "count": len(substitutions), "by_hop": dict(coverage_counts),
                "coverage_by_hop": {hop: coverage_counts[hop] / kept_counts[hop] if kept_counts[hop] else 0
                                    for hop in HOP_TYPES},
                "over_cap_by_hop": dict(cf_over_cap), "donor_pool": "same retained split only",
            },
            "generation": {"greedy": True, "stop_at_newline": True, "stop_at_eos": True,
                           "max_answer_tokens": max(1, budget),
                           "budget_rule": "maximum retained gold-or-alias leading-space token length"},
        }
        for condition, episodes in (("original", retained), ("answer_substitution", substitutions),
                                    ("remove_final_support", removals)):
            path = episode_path(output_dir, split, condition)
            write_episodes(path, episodes)
            manifest["files"][path.name] = {"count": len(episodes), "bytes": path.stat().st_size,
                                            "blake2b_128": file_digest(path)}
        print(json.dumps({"stage": "prepared", "split": split, "count": len(retained),
                          "dropped": len(dropped), "substituted": len(substitutions),
                          "max_answer_tokens": budget}), flush=True)
    for name in ("official_answer.py", "official_evaluate.py"):
        source = raw_dir.parent / "provenance" / name
        if source.exists():
            manifest["official_metric"][name + "_blake2b_128"] = file_digest(source)
    manifest["preparation_seconds"] = perf_counter() - begin
    job_spec = os.environ.get("LL_SPEC")
    source = json.loads(Path(job_spec).read_text())["source"] if job_spec else {}
    manifest["job"] = {"id": os.environ.get("PBS_JOBID"), "host": socket.gethostname(), "source": source}
    if source.get("kind") == "commit":
        manifest["code_commit"] = source["ref"]
    else:
        try:
            manifest["code_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            manifest["code_commit"] = None
    manifest["code_files_blake2b_128"] = {
        "prepare_musique.py": file_digest(__file__),
        "musique.py": file_digest(Path(__file__).parents[1] / "src/looplora/data/musique.py"),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_ROOT / "raw")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hf-home", type=Path, default=Path("/work/xg24i002/x10041/hf_home"))
    parser.add_argument("--retention-output", type=Path, default=DEFAULT_ROOT.parent / "retention")
    parser.add_argument("--section", choices=("all", "musique", "retention"), default="all")
    parser.add_argument("--base-seed", type=int, default=20261026)
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--context-cap", type=int)
    parser.add_argument("--retention-split", action="append", default=[], metavar="NAME=SPLIT")
    parser.add_argument("--allow-smaller-retention", action="append", default=[], metavar="NAME")
    args = parser.parse_args()
    if args.section in ("all", "musique"):
        result = prepare_musique(args.raw_dir, args.output_dir, base_seed=args.base_seed,
                                 n_distractors=args.n_distractors, context_cap=args.context_cap)
        print(json.dumps({"musique_manifest": str(args.output_dir / "manifest.json"),
                          "context_cap": result["context_cap"], "seconds": result["preparation_seconds"]}), flush=True)
    if args.section in ("all", "retention"):
        from looplora.data.retention import prepare_retention

        overrides = {}
        for item in args.retention_split:
            name, separator, split = item.partition("=")
            if not separator or name in overrides:
                parser.error("retention overrides must be unique NAME=SPLIT pairs")
            overrides[name] = split
        result = prepare_retention(args.hf_home, args.retention_output, count=500,
                                    split_overrides=overrides, allow_smaller=args.allow_smaller_retention)
        print(json.dumps({"retention_manifest": str(args.retention_output / "manifest.json"),
                          "datasets": list(result.get("datasets", {}))}), flush=True)


if __name__ == "__main__":
    main()
