"""WP-L5 no-training audit of final DPFP, LSQT (LT-*) and SQ2 (S-*) checkpoints (descriptive diagnostics); compute only.

For each arm folder (record.json, writer.pt) it reloads the export with the study's own `load_inference` and scores the 1,000 validation
episodes of the arm's own evaluation (eval seed 4242, one query each, as `read_fix_run.validate`) with the deployed memory
(original), a zeroed memory and the memory of another episode (shuffle: a one-step roll of a seeded permutation, so no episode keeps
its own memory). Reports EM and NLL per condition and whether the original reproduces the final validation of the record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import dpfp_run, lsqt_run, sq2_run
from looplora.llm.evaluate import _slice
from looplora.llm.read_fix import controlled_score
from looplora.llm.run import _write_json, source_metadata
from looplora.llm.train import make_task_spec
from looplora.sym.evaluate import stratify_queries

CONDITIONS = ('original', 'zero', 'shuffle')


def load_inference(path, *, model=None, tokenizer=None, arm=''):
    """The loader of the arm's study: LT-* from lsqt_run, S-* from sq2_run, the DPFP arms from dpfp_run."""
    runner = lsqt_run if arm.startswith('LT-') else sq2_run if arm.startswith('S-') else dpfp_run
    return runner.load_inference(path, model=model, tokenizer=tokenizer)


@torch.no_grad()
def audit_one(folder, out, count=1000, *, model=None, tokenizer=None):
    folder, out = Path(folder), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = json.loads((folder / 'record.json').read_text())
    assert record['status'] == 'complete' and not record.get('smoke')
    experiment = load_inference(folder / 'writer.pt', model=model, tokenizer=tokenizer, arm=record['arm'])
    try:
        spec = make_task_spec(experiment.config)
        arrays = stratify_queries(generate(spec, 'val', 0, count, 4242), spec, [1], 4242)
        order = np.random.default_rng(derive_seed(17, 'dpfp', 'audit-memory-shuffle')).permutation(count)
        pairing = np.empty(count, dtype=np.int64)
        pairing[order] = np.roll(order, 1)
        assert count < 2 or not (pairing == np.arange(count)).any()
        correct = {name: [] for name in CONDITIONS}
        nll = {name: [] for name in CONDITIONS}
        drift = 0.
        for begin in range(0, count, 16):
            end = min(begin + 16, count)
            batch = experiment.prepare(_slice(arrays, slice(begin, end)))
            donor = experiment.prepare(_slice(arrays, pairing[begin:end]))
            original, x = controlled_score(experiment, batch)
            repeat, _ = controlled_score(experiment, batch)
            drift = max(drift, float((repeat - original).abs().max()))
            scores = {'original': original}
            for name in ('zero', 'shuffle'):
                scores[name], changed_x = controlled_score(experiment, batch, name, donor if name == 'shuffle' else None)
                assert torch.equal(changed_x, x), 'question input changed under a memory-only intervention'
            for name, logits in scores.items():
                correct[name].append((logits.argmax(-1) == batch.answer_ids).flatten().cpu().numpy())
                nll[name].append(F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten(), reduction='none').cpu().numpy())
        result = dict(**source_metadata(), arm=record['arm'], E=record['config']['task']['E'], rank=record['rank'], dpfp_nu=record.get('dpfp_nu'),
                      checkpoint_job_id=record['job_id'], checkpoint_git_commit=record['git_commit'], steps=record['training']['actual_steps'],
                      episodes=count, eval_seed=4242, optimizer_steps=0, max_repeated_logit_drift=drift)
        for name in CONDITIONS:
            hits, loss = np.concatenate(correct[name]), np.concatenate(nll[name])
            result[name] = dict(correct=int(hits.sum()), em=float(hits.mean()), nll=float(loss.mean()))
        final = record['training']['metrics'][str(record['training']['actual_steps'])]
        result['original_matches_record'] = dict(correct=result['original']['correct'] == final['correct'],
                                                 record_correct=final['correct'], record_nll=final['nll'])
        _write_json(out / 'summary.json', result)
        return result
    finally:
        experiment.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new folder; one subfolder per arm folder')
    parser.add_argument('--folders', type=Path, nargs='+', required=True, help='arm folders holding record.json and writer.pt')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    runs = {}
    for folder in args.folders:
        record = json.loads((folder / 'record.json').read_text())
        label = f'{record["arm"]}-E{record["config"]["task"]["E"]}'
        runs[label] = audit_one(folder, args.out / label)
        print(label, {k: runs[label][k] for k in (*CONDITIONS, 'original_matches_record', 'max_repeated_logit_drift')}, flush=True)
    _write_json(args.out / 'summary.json', dict(**source_metadata(), runs=runs))
