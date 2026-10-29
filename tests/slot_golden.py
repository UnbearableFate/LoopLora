"""Old-path digests of SLOT-LOOP-v1 (compute only): the tiny S-RE-style experiment's matrix path, hashed.

`tests/assets/slot_golden_digests.json` holds the digests this file prints when run against the source of the commit before any
slot code: the state, the logits at L = 1, 2 and 4 for a training-shaped batch (four questions per episode) and an
evaluation-shaped one (one question), and the adapter gradients of one training loss. `tests/test_slot_experiment.py` recomputes
them with the current code in a fresh process: equal digests mean the old matrix path is bitwise unchanged. This file imports
only modules that exist before the study, so it runs on both source trees.

    python tests/slot_golden.py                                  # digests of the tree on PYTHONPATH, as one JSON line
    python tests/slot_golden.py --old-commit HASH --out FILE     # capture: old tree twice, current tree once, one record
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm import sq2
from looplora.llm.inject import load_tokenizer
from looplora.llm.train import make_task_spec
from looplora.sym.evaluate import stratify_queries


def tiny_model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'slot-golden-model'))
        return Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()


def tiny_config():
    cfg = sq2.configuration('S-RE', 10, eta=1e-2, numerics_digest='n' * 32)
    cfg['model'].update(layer=1)
    return cfg


def tiny_experiment(model, tokenizer):
    """An S-RE-style experiment whose reads matter: a nonzero B and non-trivial input standardization buffers."""
    experiment = sq2.SQ2Experiment(tiny_config(), 'S-RE', model=model, tokenizer=tokenizer)
    generator = torch.Generator().manual_seed(derive_seed(17, 'slot-golden-weights'))
    with torch.no_grad():
        adapter = experiment.adapter
        adapter.B.copy_(torch.randn(adapter.B.shape, generator=generator) * 0.05)
        adapter.input_mean.copy_(torch.randn(adapter.input_mean.shape, generator=generator) * 0.1)
        adapter.input_std.copy_(0.5 + torch.rand(adapter.input_std.shape, generator=generator))
    return experiment


def digest(*tensors):
    h = hashlib.blake2b(digest_size=16)
    for t in tensors:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def batches(cfg):
    spec = make_task_spec(cfg)
    train = generate(spec, 'train', 0, 8, derive_seed(17, 'slot-golden-data'))
    return train, stratify_queries(generate(spec, 'val', 0, 8, derive_seed(17, 'slot-golden-eval')), spec, [1], 4242)


def digests(model, tokenizer):
    experiment = tiny_experiment(model, tokenizer)
    out = {}
    try:
        train, evaluation = batches(experiment.config)
        for name, arrays in (('train', train), ('eval', evaluation)):
            with torch.no_grad():
                batch = experiment.prepare(arrays)
                out[f'{name}_matrix_state'] = digest(batch.snapshot.states[experiment.layer], batch.keys, batch.values)
                for L in (1, 2, 4):
                    logits, x = experiment.score(batch, L, capture_inputs=True)
                    out[f'{name}_logits_L{L}'] = digest(logits)
                out[f'{name}_inputs'] = digest(x)
        batch = experiment.prepare(train)
        logits, _ = experiment.score(batch, 1)
        F.cross_entropy(logits.float().flatten(0, 1), batch.answer_ids.flatten()).backward()
        grads = {n: getattr(experiment.adapter, n).grad for n in ('Q', 'R', 'U', 'B', 'A', 'K', 'V', 'a_lambda', 'a_beta')}
        out['train_gradients'] = digest(*[g for g in grads.values() if g is not None])
        out['train_gradient_names'] = ','.join(n for n, g in grads.items() if g is not None)
    finally:
        experiment.close()
    return out


def child(source):
    """Digests of the tree whose `src` directory is `source` (None: the current tree), computed in a fresh process."""
    env = dict(os.environ)
    if source is not None:
        env['PYTHONPATH'] = str(Path(source) / 'src')
    run = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=env, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f'digest child failed (source {source}, exit {run.returncode}):\n{run.stderr[-6000:]}')
    return json.loads(run.stdout.strip().splitlines()[-1])


REPO = '/work/xg24i002/x10041/LoopLora'  # the object store job.pbs archives from (a snapshot has no .git)


def capture(old_commit, out):
    with tempfile.TemporaryDirectory(dir=os.environ.get('LL_SCRATCH')) as scratch:
        archive = subprocess.run(['git', '-C', REPO, 'archive', old_commit, 'src'], check=True, capture_output=True).stdout
        subprocess.run(['tar', '-x', '-C', scratch], input=archive, check=True)
        old, again = child(scratch), child(scratch)
    current = child(None)
    record = dict(old_commit=old_commit, digests=old, old_repeat_equal=old == again, current_equal=old == current,
                  current=current, torch=torch.__version__, gpu=torch.cuda.get_device_name())
    Path(out).write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True), flush=True)
    return record['old_repeat_equal'] and record['current_equal']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--old-commit')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.old_commit:
        sys.exit(0 if capture(args.old_commit, args.out) else 1)
    from looplora.llm.inject import configure_numerics
    configure_numerics()
    print(json.dumps(digests(tiny_model(), load_tokenizer()), sort_keys=True), flush=True)
