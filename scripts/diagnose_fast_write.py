"""WP-C5 bounded numerical probe; execute through submit.py on compute."""
import json
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from looplora.data.relations import generate
from looplora.infra.seeds import derive_seed
from looplora.llm.pilot1_select import writer_config
from looplora.llm.train import LLMExperiment, make_task_spec
from looplora.llm.inject import load_model, load_tokenizer


def probe(model, tokenizer, site, policy):
    cfg = writer_config('probe', 16, 'fact_final', .0003, 32, 1)
    cfg['write_policy'] = policy
    cfg['model']['input_site'] = site
    cfg['train']['batch_size'] = 2
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        generator = torch.Generator(device='cuda').manual_seed(derive_seed(17, 'c5-real-B'))
        with torch.no_grad():
            experiment.adapter.B.normal_(0, .03, generator=generator)
        arrays = generate(make_task_spec(cfg), 'train', 0, 2, derive_seed(17, 'c5-fixed-equivalence'))
        fast_scan = experiment.adapter.scan_final
        outcomes = []
        captured = []

        def scan(self, *args, **kwargs):
            if not captured:
                captured.append(tuple(v.detach().clone() for v in args))
            return fast_scan(*args, **kwargs) if fast else self.scan_prewrite(*args, **kwargs)[1]

        experiment.adapter.scan_final = MethodType(scan, experiment.adapter)
        with torch.no_grad():
            for fast in (False, False, True, True):
                prepared = experiment.prepare(arrays)
                logits, _ = experiment.score(prepared, 1)
                loss = F.cross_entropy(logits.float().flatten(0, 1), prepared.answer_ids.flatten())
                outcomes.append((float(loss), prepared.snapshot.states[14].clone()))
            initial, u, valid, write, reset = captured[0]
            mask = valid & write
            count = int(mask.sum(1).max())
            positions = torch.arange(u.shape[1], device=u.device).expand_as(mask)
            positions = positions.masked_fill(~mask, u.shape[1]).sort(1).values[:, :count].clamp_max(u.shape[1]-1)
            packed = u.gather(1, positions[..., None].expand(-1, -1, u.shape[-1]))
            bulk = F.linear(packed, experiment.adapter.K)
            stepwise = torch.stack([F.linear(packed[:, s], experiment.adapter.K) for s in range(count)], 1)
        reference = outcomes[0][1]
        result = dict(input_site=site, policy=policy, losses=[r[0] for r in outcomes],
            state_abs_errors=[float((r[1]-reference).abs().max()) for r in outcomes],
            state_relative_errors=[float(torch.linalg.vector_norm(r[1]-reference)/torch.linalg.vector_norm(reference))
                                   for r in outcomes],
            bulk_step_projection_abs_error=float((bulk-stepwise).abs().max()),
            u_shape=list(u.shape), packed_shape=list(packed.shape),
            float32_precision=torch.get_float32_matmul_precision(),
            tf32=torch.backends.cuda.matmul.allow_tf32)
        print(json.dumps(result), flush=True)
        return result
    finally:
        experiment.close()


def main():
    torch.set_num_threads(1)
    model, tokenizer = load_model(), load_tokenizer()
    rows = [probe(model, tokenizer, site, policy) for site in ('o_proj', 'residual')
            for policy in ('fact_final', 'all_tokens')]
    path = Path('/work/xg24i002/x10041/LoopLora/runs/fast-write/numerical-probe-all.json')
    path.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
