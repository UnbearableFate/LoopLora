"""WP-L12 SQ2 without torch: the registered arms, the parent checkpoint and the checks of a schedule release (docs/prereg/SQ2.md).

The runner (sq2_run), the numerics script and scripts/prepare_sq2_release.py share these definitions. The release script runs on
the login node, so nothing here imports torch.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

# arm -> (store, coefficient loss); S-RC against S-DE is the registered primary contrast
ARMS=('S-DE','S-DC','S-RE','S-RC')
SPECS={'S-DE':('delta',False),'S-DC':('delta',True),'S-RE':('ridge',False),'S-RC':('ridge',True)}
N_MAX=6000
# the parent: DPFP's D128-3 at its final step (config hash, file digests and validation as recorded at drafting)
PARENT=dict(arm='D128-3',steps=22370,config_hash='2207a847e4967fb8e60d1663237fd979',
    folder='/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp/arms/D128-3',record='reports/p1l-dpfp/arms/D128-3/record.json',
    digests={'checkpoint.pt':'b6acff8b772a692311e630293add553f','writer.pt':'31d98c01c604c05a54bc20e4b4c4f436'})
ETA_GRID=(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.)   # RFX's grid
ETA_BOUNDS=(1e-4,1.0)
WEIGHT_BOUNDS=(1/16,16.)
NUMERICS_BATCHES=64
PREREG='docs/prereg/SQ2.md'
WALL_SECONDS=3600  # one arm job: <= 1 h, no --scale-up
EVALUATIONS=6      # the five checkpoints and the step-0 evaluation (the step-0 evaluation is also 1,000 episodes)


def file_digest(path):
    digest=hashlib.blake2b(digest_size=16)
    with open(path,'rb') as handle:
        for chunk in iter(lambda:handle.read(1<<22),b''):
            digest.update(chunk)
    return digest.hexdigest()


def prereg_digest(root=None):
    root=Path(__file__).resolve().parents[3] if root is None else Path(root)
    return file_digest(root/PREREG)


def choose_eta(curve):
    """The registered rule: the grid value with the lowest mean normalized read error (ties: the larger eta), clipped to [1e-4, 1]."""
    best=min(curve,key=lambda row:(row['error'],-row['eta']))['eta']
    return min(max(float(best),ETA_BOUNDS[0]),ETA_BOUNDS[1])


def round_weight(ratios):
    """The registered rule: the median gradient-norm ratio, rounded to the nearest power of two, clipped to [1/16, 16]."""
    ordered=sorted(ratios)
    n=len(ordered)
    if n==0 or any(not math.isfinite(r) or r<=0 for r in ordered):
        raise ValueError('the gradient-norm ratios must be positive and finite')
    median=ordered[n//2] if n%2 else 0.5*(ordered[n//2-1]+ordered[n//2])
    return min(max(2.0**round(math.log2(median)),WEIGHT_BOUNDS[0]),WEIGHT_BOUNDS[1])


def scheduled_steps(arm,steps,schedule,*,smoke=False):
    """Smoke: the explicit step count. Science: the frozen release's count (the same N for every arm)."""
    if arm not in SPECS:
        raise ValueError(f'{arm} is not a registered SQ2 arm')
    if smoke:
        if schedule is not None or steps is None or steps<5:
            raise ValueError('smoke requires its explicit step count')
        return steps
    if steps is not None or schedule is None:
        raise ValueError('scientific arms require an immutable schedule release')
    if schedule['study']!='SQ2' or schedule['E']!=16:
        raise ValueError('schedule study/capacity mismatch')
    by_arm=schedule['steps_by_arm']
    if set(by_arm)!=set(ARMS) or len(set(by_arm.values()))!=1:
        raise ValueError('the SQ2 release fixes one step count for all four arms')
    result=by_arm[arm]
    if isinstance(result,bool) or not isinstance(result,int) or result<5 or result%5 or result>N_MAX:
        raise ValueError('scientific schedules require five equal integer intervals and at most 6,000 steps')
    return result
