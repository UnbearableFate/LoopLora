"""Exact frozen-writer support reproducibility across real BF16 GPU processes.

Run only through the compute test workflow. The 200-episode validation set and
its 1600 replacement forks reproduce the full smoke shapes, including the
previously process-dependent replacement batch at offset 112.
"""
from __future__ import annotations

from dataclasses import fields
import multiprocessing
from pathlib import Path
from queue import Empty
import traceback

import torch

from looplora.data.relations import EpisodeArrays, TaskSpec, counterfactual, generate
from looplora.infra.seeds import derive_seed
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_model, load_tokenizer
from looplora.llm.train import LLMExperiment, make_task_spec
from looplora.sym.evaluate import stratify_queries, tensor_hash

SMOKE_SEED = 20261027
EVAL_SEED = derive_seed(SMOKE_SEED, "p1l-smoke-eval")


def _rows(arrays: EpisodeArrays, start: int, stop: int) -> EpisodeArrays:
    return EpisodeArrays(**{
        field.name: (getattr(arrays, field.name) if field.name == "meta"
                     else getattr(arrays, field.name)[start:stop])
        for field in fields(arrays)
    })


def _worker(worker: int, writer_path: str, ready, result) -> None:
    """Each spawned worker loads its own pinned model exactly once."""
    experiment = None
    try:
        model, tokenizer = load_model(), load_tokenizer()
        if worker == 0:
            teacher = LLMExperiment(resolve_config({"stage": "writer", "seed": SMOKE_SEED}),
                                    model=model, tokenizer=tokenizer)
            try:
                torch.save({"signature": teacher.writer_signature(),
                            "writer": teacher.writer_state()}, writer_path)
            finally:
                teacher.close()
        ready.wait(timeout=120)
        cfg = resolve_config({"stage": "reader", "variant": "M1" if worker == 0 else "M1-lin",
                              "seed": SMOKE_SEED, "smoke": True,
                              "writer_checkpoint": writer_path,
                              "eval": {"episodes": 200, "seed": EVAL_SEED,
                                       "protocols": ["memory_only"]}})
        experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
        with torch.no_grad():
            # Distinct real name tokens, sequence/batch sizes, and reader depths
            # exercise separate allocator/kernel histories before identical data.
            warmups = ([(1, 5, 3), (8, 24, 41), (2, 11, 103)] if worker == 0 else
                       [(3, 9, 57), (2, 32, 211), (4, 17, 307), (1, 7, 439)])
            warmup_shapes = []
            for number, (batch, facts, start) in enumerate(warmups):
                task = TaskSpec("t1", E=facts, Q=1, n_rel=4, head_sharing=.3)
                arrays = generate(task, "val", start, batch,
                                  derive_seed(SMOKE_SEED, "determinism-warmup", worker, number))
                prepared = experiment.prepare(arrays)
                warmup_shapes.append(tuple(prepared.support_ids.shape))
                experiment.score(prepared, 1 if worker == 0 else 4)
            spec = make_task_spec(cfg, evaluation=True)
            arrays = stratify_queries(generate(spec, "val", 0, 200, EVAL_SEED),
                                      spec, cfg["eval"]["hops"], EVAL_SEED)
            replacements, _ = counterfactual(arrays, "replace_key_fact",
                                             derive_seed(EVAL_SEED, "eval-replace", "val"))
            captured = []
            projection = model.model.layers[14].self_attn.o_proj
            # Hook before the wrapper: raw BF16 attention output, before A.
            hook = projection.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
            hashes = []
            try:
                for start in range(0, len(replacements.episode_seed), 16):
                    captured.clear()
                    prepared = experiment.prepare(_rows(replacements, start, start + 16))
                    if len(captured) != 1:
                        raise AssertionError("support must reach the wrapped projection exactly once")
                    x = captured[0]
                    hashes.append({"start": start, "x": tensor_hash(x),
                                   "u": tensor_hash(experiment.adapter.u(x.float())),
                                   "M": tensor_hash(prepared.snapshot.states[14]),
                                   "ids": tensor_hash(prepared.support_ids),
                                   "mask": tensor_hash(prepared.support_mask),
                                   "writes": tensor_hash(prepared.write_count)})
            finally:
                hook.remove()
            writer = {name: tensor_hash(value) for name, value in experiment.writer_state().items()}
        result.put({"worker": worker, "hashes": hashes, "writer": writer,
                    "warmup_shapes": warmup_shapes})
    except BaseException:
        ready.abort()
        result.put({"worker": worker, "error": traceback.format_exc()})
    finally:
        if experiment is not None:
            experiment.close()


def test_frozen_support_exact_across_processes_with_different_histories(tmp_path: Path):
    context = multiprocessing.get_context("spawn")
    ready, result = context.Barrier(2), context.Queue()
    writer_path = str(tmp_path / "writer.pt")
    processes = [context.Process(target=_worker, args=(worker, writer_path, ready, result))
                 for worker in range(2)]
    for process in processes:
        process.start()
    rows = []
    try:
        for _ in processes:
            try:
                rows.append(result.get(timeout=180))
            except Empty:
                raise AssertionError("cross-process support regression timed out") from None
        errors = [row["error"] for row in rows if "error" in row]
        assert not errors, "\n".join(errors)
        rows.sort(key=lambda row: row["worker"])
        left, right = rows
        assert left["warmup_shapes"] != right["warmup_shapes"]
        assert left["writer"] == right["writer"]
        assert len(left["hashes"]) == len(right["hashes"]) == 100
        assert any(row["start"] == 112 for row in left["hashes"])
        for expected, observed in zip(left["hashes"], right["hashes"]):
            assert expected == observed, {"expected": expected, "observed": observed}
    finally:
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result.close()
    assert all(process.exitcode == 0 for process in processes)
