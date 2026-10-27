"""WP-L7 LSQ without torch: the registered arms and the checks of a schedule release.

The runner (lsq_run), RFX (rfx_run) and scripts/prepare_lsq_release.py share these definitions. The script runs on the
login node, so nothing here imports torch.
"""
from __future__ import annotations

import math

# Candidate order of docs/prereg/LSQ.md: rank 32 first, then 128. U (the deployed delta store with the unit query; it
# runs under RFX's normalization verdict) and K4 and RG (the least-squares stores; they run under the storage verdict) are
# mutually exclusive per rank; K4 (no new operator, no regularizer) comes before RG.
ARMS=('L32-U','L32-K4','L32-RG','L128-U','L128-K4','L128-RG')
SPECS={'L32-U':(32,'delta'),'L32-K4':(32,'sweep'),'L32-RG':(32,'ridge'),
       'L128-U':(128,'delta'),'L128-K4':(128,'sweep'),'L128-RG':(128,'ridge')}
ETA_BOUNDS=(1e-4,1e-1)


def lsq_eta(rfx_eta):
    """Ridge strength of an LSQ arm: the RFX fitting-set selection for the arm's rank, clipped to [1e-4, 1e-1]."""
    if isinstance(rfx_eta,bool) or not math.isfinite(rfx_eta) or rfx_eta<=0:
        raise ValueError('the RFX-selected eta must be positive and finite')
    return min(max(float(rfx_eta),ETA_BOUNDS[0]),ETA_BOUNDS[1])


def check_arms(arms):
    """A set of triggered arms: each arm once, in candidate order, and per rank U alone or K4 together with RG."""
    arms=list(arms)
    if not arms or any(arm not in SPECS for arm in arms) or arms!=[a for a in ARMS if a in arms]:
        raise ValueError('triggered arms must be registered LSQ arms, each once, in candidate order')
    for rank in (32,128):
        stores=[SPECS[arm][1] for arm in arms if SPECS[arm][0]==rank]
        if stores not in ([],['delta'],['sweep','ridge']):
            raise ValueError(f'rank {rank}: the arms are U alone (normalization verdict) or K4 with RG (storage verdict)')
    return arms


def release_arms(schedule):
    """The candidate arms a release covers: RFX's `lsq_arms`, in candidate order (per rank U alone, or K4 with RG)."""
    return check_arms(schedule['arms'])


def release_eta(schedule,arm):
    """The registered ridge strength of an RG arm: lsq_eta of RFX's fitting-set selection for its rank (U and K4 have none)."""
    rank,store=SPECS[arm]
    if store!='ridge':
        return None
    eta=lsq_eta(schedule['rfx_eta_by_rank'][str(rank)])
    if schedule['eta_by_rank'][str(rank)]!=eta:
        raise ValueError('the release eta differs from the clipped RFX selection')
    return eta


def scheduled_steps(arm,steps,schedule,*,smoke=False,E=16):
    """Smoke: the explicit step count. Science: the frozen release's count for this arm."""
    if smoke:
        if schedule is not None or steps is None or steps<5:
            raise ValueError('smoke requires its explicit step count')
        return steps
    if steps is not None or schedule is None:
        raise ValueError('scientific arms require an immutable schedule release')
    if schedule['study']!='LSQ' or schedule['E']!=E:
        raise ValueError('schedule study/capacity mismatch')
    arms=release_arms(schedule)
    by_arm=schedule['steps_by_arm']
    if arm not in arms or set(by_arm)!=(set(arms) if E==16 else {arm}):
        raise ValueError('the E16 release fixes the step counts of every triggered arm at once, the E8 release only the '
                         'retrained arm')
    result=by_arm[arm]
    if isinstance(result,bool) or not isinstance(result,int) or result<5 or result%5:
        raise ValueError('scientific schedules require five equal integer intervals')
    return result


def eligible_e8(lock,arm):
    choice=lock['choice']
    if choice['action']!='retry_e8' or choice['arm']!=arm or choice['arm'] not in ARMS:
        raise ValueError('E8 requires the one registered best arm')
    return 8
