#!/usr/bin/env python3
"""Compute-only diagnosis of the registered S1B calibration stop; no training.

Authorized by WP-L14's 2026-10-30 00:56 JST instruction. Only S1B
calibration and the non-sealed S1A T1 pilot reference are generated. T1-worded
T2 is an explicitly diagnostic prompt intervention, never a design amendment.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.nn import functional as F

from looplora.core.memory_backend import MatrixBackend, MatrixState, MemoryRead
from looplora.data.render import load_name_pool, render_episodes
from looplora.llm import s1b_artifacts as artifacts, s1b_data as data, s1b_model as model
from looplora.llm import slot_data, slot_experiment
from looplora.llm.evaluate import _slice
from looplora.llm.run import source_metadata
from looplora.llm.slot_read import ReadSpec, bind_batch, final_query, true_values
from looplora.llm.train import synchronize
from looplora.llm.writer_fix import true_fact_indices


@dataclass(frozen=True)
class OracleAllRead(MemoryRead):
    """Diagnostic only: supply the true hop-1 written value at every token."""
    oracle: torch.Tensor | None = None

    def read(self, query):
        return self.oracle.index_select(0, self.owner)


def summary(samples):
    return dict(n=len(samples["correct"]), correct=int(samples["correct"].sum()),
                em=float(samples["correct"].mean()),
                nll=float(samples["nll"].astype(np.float64).mean()))


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=int(values.size), mean=float(values.mean()),
                p10=float(np.quantile(values, .1)), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)))


def decode(tokenizer, ids):
    return tokenizer.decode(list(map(int, ids)), clean_up_tokenization_spaces=False)


def read_stats(experiment, batch):
    """Use captured actual q/m at every valid token, with owner/position checks."""
    B, Q, T = batch.question_ids.shape
    assert Q == 1
    target = true_fact_indices(batch.arrays, experiment.device)[:, 0]
    valid = batch.question_mask[:, 0]
    lengths = valid.long().sum(-1)
    top = np.full((B, T), -1, dtype=np.int16)
    margin = np.full((B, T), np.nan, dtype=np.float32)
    read_error = np.full((B, T), np.nan, dtype=np.float32)
    final_q = torch.empty((B, 1, batch.keys.shape[-1]), device=experiment.device)
    assert len(experiment.adapter.trace) == T
    model.verify_trace(experiment.adapter)
    for t, event in enumerate(experiment.adapter.trace):
        owner, query, read = event["_view"].owner, event["_q"], event["_m"]
        assert event["step"] == 1
        assert torch.equal(owner, valid[:, t].nonzero(as_tuple=True)[0])
        scores = (batch.keys.index_select(0, owner) * query[:, None]).sum(-1)
        gold = target.index_select(0, owner)
        actual = scores.gather(1, gold[:, None])[:, 0]
        other = scores.masked_fill(F.one_hot(gold, scores.shape[-1]).bool(), -torch.inf).amax(-1)
        clean = batch.values[owner, gold]
        error = (read - clean).square().sum(-1) / (clean.square().sum(-1) + 1e-6)
        rows = owner.cpu().numpy()
        top[rows, t] = (scores.argmax(-1) == gold).cpu().numpy()
        margin[rows, t] = (actual - other).cpu().numpy()
        read_error[rows, t] = error.cpu().numpy()
        last = lengths.index_select(0, owner) == t + 1
        final_q[owner[last], 0] = query[last]
    assert np.array_equal(top >= 0, valid.cpu().numpy())
    return dict(key_top1=top, key_margin=margin, read_relative_error=read_error,
                question_length=lengths.cpu().numpy()), final_q


def stats_summary(samples):
    lengths = samples["question_length"]
    top, margin = samples["key_top1"], samples["key_margin"]
    final_top = top[np.arange(len(top)), lengths - 1]
    final_margin = margin[np.arange(len(top)), lengths - 1]
    error = samples["read_relative_error"][np.arange(len(top)), lengths - 1]
    by_offset = {}
    for offset in range(int(lengths.max())):
        take = np.flatnonzero(lengths > offset)
        positions = lengths[take] - offset - 1
        by_offset[str(-offset - 1)] = dict(n=len(take),
            key_top1=float(top[take, positions].mean()),
            margin=distribution(margin[take, positions]))
    return dict(final_key_top1=float(final_top.mean()), final_margin=distribution(final_margin),
                final_read_relative_error=distribution(error),
                all_valid_key_top1=float(top[top >= 0].mean()),
                all_valid_margin=distribution(margin[np.isfinite(margin)]),
                by_token_offset_from_answer=by_offset)


def prompt_record(experiment, arrays, index, condition, logits, batch):
    row = render_episodes(_slice(arrays, slice(index, index + 1)), tokenizer=experiment.tokenizer)[0]
    query = row.queries[0]
    length = int(batch.question_mask[index, 0].sum())
    ids = batch.question_ids[index, 0, :length].tolist()
    assert ids == list(query.question_ids)
    assert int(batch.answer_ids[index, 0]) == query.answer_token_id
    values, tokens = logits[index, 0].float().topk(5)
    logz = logits[index, 0].float().logsumexp(-1)
    top5 = [dict(token_id=int(token), token=decode(experiment.tokenizer, [token]),
                 logit=float(value), probability=float((value - logz).exp()))
            for token, value in zip(tokens.tolist(), values.tolist())]
    spans = []
    for span in row.fact_spans:
        positions = batch.positions[index, span.fact_index].tolist()
        assert positions[0] == span.token_end - 1
        tail = int(arrays.facts[index, span.fact_index, 2])
        assert row.context_ids[positions[2]] == load_name_pool().token_ids[tail]
        spans.append(dict(fact_index=span.fact_index, positions=positions,
            fact_text=decode(experiment.tokenizer, row.context_ids[span.token_start:span.token_end]),
            write_token=decode(experiment.tokenizer, [row.context_ids[positions[0]]])))
    return dict(episode_id=int(arrays.episode_seed[index]), condition=condition,
                protocol="memory_only: support encoded separately; only question reaches read forward",
                support_text=row.context_text, question_text=query.question_text,
                question_token_ids=ids, question_tokens=[decode(experiment.tokenizer, [i]) for i in ids],
                answer_position=length - 1, answer_position_token=decode(experiment.tokenizer, [ids[-1]]),
                target_entity=int(arrays.answer[index, 0]), target_text=query.answer_text,
                target_tokens=[query.answer_token_id], top5=top5, fact_positions=spans)


@torch.no_grad()
def diagnose_set(experiment, arrays, specs, value_head, out, label):
    """One support encoding per chunk, all conditions on that exact snapshot."""
    pools = {name: {} for name in specs}
    prompts, value_correct, value_nll = [], [], []
    start = time.perf_counter()
    for begin in range(0, len(arrays.episode_seed), 16):
        part = _slice(arrays, slice(begin, begin + 16))
        batch = experiment.prepare(part)
        assert bool((batch.write_count == arrays.facts.shape[1]).all())
        target = true_fact_indices(part, experiment.device)[:, 0]
        # At hop 1 the true written value must encode the actual answer entity.
        facts = torch.as_tensor(part.facts, device=experiment.device)
        assert torch.equal(facts[torch.arange(len(target), device=target.device), target, 2],
                           torch.as_tensor(part.answer[:, 0], device=target.device))
        clean = true_values(batch, experiment.device)[:, 0]
        head_logits = value_head(clean)
        answer_entities = torch.as_tensor(part.answer[:, 0], device=target.device, dtype=torch.long)
        value_correct.append((head_logits.argmax(-1) == answer_entities).cpu().numpy())
        value_nll.append(F.cross_entropy(head_logits, answer_entities, reduction="none").cpu().numpy())
        for name, spec in specs.items():
            capture = name in {"matrix", "soft"}
            experiment.adapter.reset_observation(capture=capture)
            if name == "memory_off":
                zero = torch.zeros_like(batch.snapshot.states[experiment.layer])
                view = MemoryRead.bind(MatrixBackend(), MatrixState(zero))
                bound = replace(batch, snapshot=replace(batch.snapshot, states={experiment.layer: view}))
            elif name == "oracle_all":
                matrix = batch.snapshot.states[experiment.layer]
                native = MemoryRead.bind(MatrixBackend(), MatrixState(matrix))
                view = OracleAllRead(native.backend, native.state, native.owner, oracle=clean)
                bound = replace(batch, snapshot=replace(batch.snapshot, states={experiment.layer: view}))
            else:
                bound = bind_batch(experiment, batch, spec)
            logits, inputs = experiment.score(bound, 1, capture_inputs=capture)
            assert bool(torch.isfinite(logits).all())
            row = dict(correct=(logits.argmax(-1) == batch.answer_ids)[:, 0].cpu().numpy(),
                prediction=logits.argmax(-1)[:, 0].cpu().numpy(), answer=batch.answer_ids[:, 0].cpu().numpy(),
                nll=F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten(), reduction="none").cpu().numpy())
            if capture:
                stats, actual_query = read_stats(experiment, batch)
                # The diagnostic uses the actual captured query, and checks its position.
                torch.testing.assert_close(actual_query, final_query(experiment, batch, inputs), rtol=1e-5, atol=1e-6)
                row.update(stats)
                if begin == 0:
                    prompts.extend(prompt_record(experiment, part, i, name, logits, batch) for i in range(5))
            for field, value in row.items():
                pools[name].setdefault(field, []).append(value)
            experiment.adapter.reset_observation()
    results = {}
    for name, fields in pools.items():
        packed = {}
        for key, chunks in fields.items():
            if chunks[0].ndim == 2:
                width = max(c.shape[1] for c in chunks)
                fill = -1 if key == "key_top1" else np.nan
                chunks = [np.pad(c, ((0, 0), (0, width - c.shape[1])), constant_values=fill) for c in chunks]
            packed[key] = np.concatenate(chunks)
        packed["episode_id"] = arrays.episode_seed
        results[name] = summary(packed)
        if name in {"matrix", "soft"}:
            results[name]["read_positions"] = stats_summary(packed)
        np.savez_compressed(out / f"{label}-{name}.npz", **packed)
    results["true_written_value_head"] = dict(n=len(arrays.episode_seed),
        em=float(np.concatenate(value_correct).mean()), nll=float(np.concatenate(value_nll).astype(np.float64).mean()),
        role="Descriptive training-only S-RE value head; never used in model forward/read")
    synchronize(experiment.device)
    return dict(conditions=results, prompts=prompts, seconds=time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--t1-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from looplora.llm.s1b_run import output
    out = output(args.out)
    started = time.perf_counter()
    lock = artifacts.require_lock(args.preparation / "manifest.lock.json")
    branch = artifacts.require_branch(args.preparation / "branch.json")
    assert branch["branch"] == "advance"
    calibration_path = args.preparation / "calibration/calibration.json"
    cal = artifacts.read(calibration_path)
    assert cal["lock_blake2b_128"] == data.old.file_digest(args.preparation / "manifest.lock.json")
    assert cal["branch_blake2b_128"] == data.old.file_digest(args.preparation / "branch.json")
    assert cal["selected"]["matrix"]["per_hop"]["1"]["em"] < .1
    reference = artifacts.read(args.t1_reference)
    references = [r["step0"] for r in reference["runs"] if r["arm"] == "matrix_L1"]
    expected = references[0]
    assert all(r["correct"] == expected["correct"] and r["nll"] == expected["nll"] for r in references)
    assert expected["correct"] == 733 and expected["n"] == 1000
    (out / "calibration.input.json").write_bytes(calibration_path.read_bytes())
    (out / "t1-reference.input.json").write_bytes(args.t1_reference.read_bytes())
    t1 = slot_experiment.episodes(slot_data.eval_set("S1A", "pilot"))
    full_t2 = data.episodes(data.eval_set("calibration"))
    t2 = data.episodes(data.eval_set("calibration"), [1])
    assert np.array_equal(t2.episode_seed, full_t2.episode_seed)
    assert np.array_equal(t2.q_start[:, 0], full_t2.q_start[:, 0])
    assert np.array_equal(t2.answer[:, 0], full_t2.answer[:, 0])
    # A paired prompt-only counterfactual: support/identity/target unchanged.
    reworded = replace(t2, q_hop_given=np.ones_like(t2.q_hop_given),
        meta={**t2.meta, "spec": {**t2.meta["spec"], "name": "t1"}})
    assert np.array_equal(reworded.facts, t2.facts) and np.array_equal(reworded.answer, t2.answer)
    result = dict(stage="S1B-DIAGNOSIS", status="running", **source_metadata(),
        job_id=os.environ.get("PBS_JOBID"), host=socket.gethostname(),
        source_blake2b_128=lock["source_blake2b_128"],
        script_blake2b_128=data.old.file_digest(__file__),
        lock_blake2b_128=data.old.file_digest(args.preparation / "manifest.lock.json"),
        calibration_blake2b_128=data.old.file_digest(calibration_path),
        t1_reference_blake2b_128=data.old.file_digest(args.t1_reference),
        scope="S1B calibration and non-sealed S1A T1 pilot reference only; no optimization",
        oracle_final_definition="S0 clean_final: true written next value at final valid token, native Matrix g=1 at earlier tokens",
        oracle_all_definition="Additional diagnostic: true written next value at every valid token, gain 1",
        memory_off_definition="Zero m at every valid token; identical L1 reader R/U/B path retained",
        prompt_intervention="T1 wording on identical T2 hop-1 calibration supports/queries/targets, diagnostic only")
    exp = model.load_parent("advance", expected=lock["inventory"])
    before = slot_experiment.tensor_digests(exp.adapter)
    try:
        t1_read = ReadSpec("matrix", gain=1., capacity=16)
        replay = model.evaluate(exp, t1, t1_read, 1)
        replay_summary = summary({k: v[:, 0] for k, v in replay["samples"].items() if k in {"correct", "nll"}})
        # Also compare the inherited evaluator on the exact same loaded instance.
        inherited = slot_experiment.evaluate(exp, t1, {"native": None, "matrix": t1_read}, diagnostics=False)
        equal = {name: all(np.array_equal(replay["samples"][k][:, 0], samples[k])
            for k in ("correct", "nll", "prediction", "answer")) for name, samples in inherited["samples"].items()}
        result["t1_replay"] = dict(actual=replay_summary, recorded=expected,
            nll_gap=replay_summary["nll"] - expected["nll"],
            inherited_per_sample_bitwise=equal,
            matches_recorded=replay_summary["correct"] == expected["correct"] and abs(replay_summary["nll"] - expected["nll"]) < 1e-7)
        result["calibration_reproduction"] = {}
        for kind, arm in (("matrix", "M1"), ("soft", "S1")):
            row = cal["selected"][kind]
            spec = ReadSpec(kind, row["gain"], row["tau"], capacity=12)
            value = model.evaluate(exp, full_t2, spec, 1)
            actual = value["summary"]
            result["calibration_reproduction"][kind] = dict(actual=actual, recorded=row,
                matches=actual == {k: row[k] for k in actual})
            assert actual == {k: row[k] for k in actual}
            np.savez_compressed(out / f"calibration-reproduction-{arm}.npz", **value["samples"])
        value_head = slot_experiment.load_value_head("S-RE", exp.device)
        specs = {kind: ReadSpec(kind, row["gain"], row["tau"], capacity=12) for kind, row in cal["selected"].items()}
        specs.update(oracle_final=ReadSpec("clean_final", scope="final_only", capacity=12),
                     oracle_all=None, memory_off=None)
        for label, arrays in (("t1", t1), ("t2", t2), ("t2_t1_wording", reworded)):
            selected = dict(specs)
            if label == "t1":
                selected["matrix"] = t1_read
                selected["soft"] = ReadSpec("soft", gain=1., tau=.1, capacity=16)
                selected["oracle_final"] = ReadSpec("clean_final", scope="final_only", capacity=16)
            result[label] = diagnose_set(exp, arrays, selected, value_head, out, label)
            artifacts.dump(out / "diagnosis.partial.json", result)
        result["adapter_unchanged"] = before == slot_experiment.tensor_digests(exp.adapter)
        assert result["adapter_unchanged"]
    finally:
        exp.close()
    assert data.old.source_digest() == lock["source_blake2b_128"]
    result.update(status="complete", seconds=time.perf_counter() - started)
    artifacts.dump(out / "diagnosis.json", result)
    lines = ["# S1B transfer diagnosis: first five fixed examples per set", ""]
    for label in ("t2", "t1", "t2_t1_wording"):
        for row in result[label]["prompts"]:
            lines.extend([f"## {label}, episode {row['episode_id']}, {row['condition']}", "",
                "Support (encoded separately):", "```text", row["support_text"].rstrip(), "```", "",
                "Read question:", "```text", row["question_text"], "```", "",
                f"Target: {row['target_text']!r}; token IDs {row['target_tokens']}; "
                f"answer-position index {row['answer_position']} / token {row['answer_position_token']!r}.", "",
                "Top five next tokens: " + json.dumps(row["top5"], ensure_ascii=False), ""])
    (out / "prompts.md").write_text("\n".join(lines))
    artifacts.dump(out / "artifacts.json", {p.name: data.old.file_digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(stage=result["stage"], status=result["status"],
        t1_reference_matches=result["t1_replay"]["matches_recorded"],
        digest=data.old.file_digest(out / "diagnosis.json"), seconds=result["seconds"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
