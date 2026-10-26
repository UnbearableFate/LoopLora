"""WP-L5: with DPFP off, the current code reproduces WP-L4 R's first 200 training steps bitwise (compute only).

Compared against the committed WP-L4 record reports/p1l-read-fix/loss_curves.csv: the arm-R rows at steps
1, 50, 100, 150 and 200 for rank 32 and rank 128 (every logged field except the step time). The first 200
steps do not depend on the total step count of record (23590 and 24645), because the learning-rate warm-up
(20 steps), the data stream and the optimizer do not read it, and no validation runs before step 200.

Run this file alone in its own job. Each rank trains in a fresh Python process, as WP-L4's arms did, so the
CUDA and cuBLAS state matches the reference runs. Run as a script, it trains one rank and writes the log:
    python tests/test_llm_dpfp_repro.py --rank 32 --out DIR
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / 'reports' / 'p1l-read-fix' / 'loss_curves.csv'
STEPS = 200
LOGGED = [1, 50, 100, 150, 200]
FIELDS = ('loss', 'main_loss', 'address_loss', 'value_loss', 'read_loss', 'decorrelation_loss',
          'grad_norm', 'clipped', 'lr', 'data_seed')


def reproduce(rank, out):
    """WP-L4 R unchanged (DPFP never enabled): standardize, then 200 training steps without validation."""
    from looplora.llm.read_fix import ReadFixExperiment, configuration
    from looplora.llm.read_fix_run import fit
    experiment = ReadFixExperiment(configuration('R', STEPS, rank=rank), 'R')
    try:
        assert experiment.adapter.cfg.dpfp_nu == 0 and experiment.adapter.cfg.key_dim == rank
        experiment.standardize()
        fit(experiment, Path(out), STEPS, evaluate=False)
    finally:
        experiment.close()


def recorded(rank):
    with CURVES.open() as handle:
        rows = [row for row in csv.DictReader(handle)
                if (row['rank'], row['arm'], row['E']) == (str(rank), 'R', '16') and int(row['step']) <= STEPS]
    return {int(row['step']): dict(
        {field: float(row[field]) for field in FIELDS if field not in ('clipped', 'data_seed')},
        clipped=row['clipped'] == 'True', data_seed=int(row['data_seed'])) for row in rows}


@pytest.mark.parametrize('rank', [32, 128])
def test_dpfp_off_reproduces_wp_l4_R_training_losses_bitwise(rank, tmp_path):
    reference = recorded(rank)
    assert sorted(reference) == LOGGED
    # The child imports looplora from the same source tree as this test (job.pbs exports the same PYTHONPATH).
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(filter(None, [str(ROOT / 'src'), os.environ.get('PYTHONPATH')]))}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--rank', str(rank), '--out', str(tmp_path)],
                   check=True, env=env, timeout=900)
    rows = [json.loads(line) for line in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
    observed = {row['step']: row for row in rows}
    assert sorted(observed) == LOGGED
    mismatches = [(step, field, reference[step][field], observed[step][field])
                  for step in LOGGED for field in FIELDS if reference[step][field] != observed[step][field]]
    assert not mismatches, f'rank {rank}: DPFP-off training differs from WP-L4 R at (step, field, WP-L4, now): {mismatches}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=(32, 128), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.rank, args.out)
