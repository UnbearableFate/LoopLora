#!/usr/bin/env python3
"""Fingerprints of every oracle_random path that WP-L6 touched; one run per source tree.

    PYTHONPATH=<tree>/src python tests/assets/oracle_offpath_probe.py --configs configs/p1l --out probe.json

Compute node only (tiny models train on the GPU). tests/test_llm_oracle_offpath.py runs it on the current tree and on
a copy whose four WP-L6 files (llm/config.py, oracle.py, run.py, train.py) are taken back to the baseline commit,
then compares the two JSON files exactly. Timing, host and path fields are never recorded.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import looplora
from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm.config import resolve_config
from looplora.llm.inject import load_tokenizer
from looplora.llm.oracle import OracleRandomWriter
from looplora.llm.run import _parameter_counts, compare_suite, run_one
from looplora.llm.train import make_task_spec, train
from looplora.sym.evaluate import tensor_hash

VARIANTS = ("M1", "B8", "B4", "B5", "B3")
VOLATILE = ("train_seconds", "train_step_seconds", "train_step_p95_seconds", "seconds_per_1000_steps",
            "writer_checkpoint")
STABLE = ("status", "config_hash", "data_seed", "eval_data_seed", "writer", "stage", "protocol", "variant", "rank",
          "r_mem", "L", "mean_L", "train_tokens", "eval_samples", "parameters", "state_bytes", "scores",
          "writer_signature", "writer_hash", "consumed_writer_checkpoint", "failure_reason")


def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "oracle-tiny-qwen"))
        return Qwen3ForCausalLM(Qwen3Config(
            vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()


def config(variant: str) -> dict:
    return resolve_config(dict(
        name="oracle-offpath", writer="oracle_random", stage="reader", variant=variant,
        model=dict(r=32, layer=1), task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
        train=dict(steps=3, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
        eval=dict(episodes=2, batch_size=2, hops=[1, 2, 3, 4], counterfactuals=False)))


def writer_part() -> dict:
    """CPU: the code tables and the memory written from a fixed fact list, at both oracle ranks."""
    result = {}
    for rank in (32, 128):
        writer = OracleRandomWriter(V=1024, n_rel=4, seed=17, r_mem=rank)
        generator = torch.Generator().manual_seed(derive_seed(17, "offpath", "facts", rank))
        facts = torch.stack((torch.randint(0, 1024, (8, 12), generator=generator),
                             torch.randint(0, 4, (8, 12), generator=generator),
                             torch.randint(0, 1024, (8, 12), generator=generator)), dim=-1)
        valid = torch.rand(8, 12, generator=generator) > .25
        facts = facts.masked_fill(~valid[..., None], -1)
        result[str(rank)] = dict(code_hash=stable_hash({k: tensor_hash(v) for k, v in writer.state_dict().items()}),
                                 state_hash=tensor_hash(writer(facts, valid)))
    return result


def config_part(directory: Path) -> dict:
    """Every committed P1-L config must resolve to the same dict (the config hash is stored in checkpoints)."""
    result = {}
    for path in sorted(directory.rglob("*.json")):
        key = str(path.relative_to(directory))
        try:
            result[key] = stable_hash(resolve_config(json.loads(path.read_text())))
        except Exception as error:  # both trees must also reject the same files with the same message
            result[key] = f"{type(error).__name__}: {error}"
    return result


def train_part(model, tokenizer, root: Path) -> dict:
    result = {}
    for variant in VARIANTS:
        cfg, out = config(variant), root / "train" / variant
        experiment, summary = train(cfg, out, model=model, tokenizer=tokenizer)
        try:
            arrays = generate(make_task_spec(cfg), "train", 0, 2, derive_seed(17, "oracle-unit-data"))
            with torch.no_grad():
                batch = experiment.prepare(arrays)
                logits, _ = experiment.score(batch, 1 if variant == "B4" else 4)
            saved = torch.load(out / "checkpoint.pt", weights_only=True)
            result[variant] = dict(
                summary={k: v for k, v in summary.items() if k not in VOLATILE},
                log=[{k: v for k, v in json.loads(line).items() if k != "step_seconds"}
                     for line in (out / "train_log.jsonl").read_text().splitlines()],
                adapter={k: tensor_hash(v) for k, v in experiment.adapter.state_dict().items()},
                checkpoint_adapter={k: tensor_hash(v) for k, v in saved["adapter"].items()},
                checkpoint_writer={k: tensor_hash(v) for k, v in saved["writer"].items()},
                signature=experiment.writer_signature(),
                writer_hash=stable_hash({k: tensor_hash(v) for k, v in experiment.writer_state().items()}),
                memory=tensor_hash(batch.snapshot.states[1]), logits=tensor_hash(logits),
                parameters=_parameter_counts(experiment))
        finally:
            experiment.close()
    return result


def run_part(model, tokenizer, root: Path) -> dict:
    result, jobs = {}, []
    for variant in ("M1", "B4"):
        cfg = config(variant)
        cfg["name"] = variant
        record = run_one(cfg, root / "run" / variant, model=model, tokenizer=tokenizer)
        result[variant] = {key: record[key] for key in STABLE}
        jobs.append(dict(name=variant, config=cfg))
    result["compare"] = json.loads(json.dumps(compare_suite(jobs, root / "run"), default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer, model = load_tokenizer(), tiny_model()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        result = dict(source=str(Path(looplora.__file__).resolve().parents[1]), torch=torch.__version__,
                      writer=writer_part(), configs=config_part(args.configs),
                      train=train_part(model, tokenizer, root), run=run_part(model, tokenizer, root))
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
