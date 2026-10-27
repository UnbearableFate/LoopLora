"""WP-L5 DPFP without torch: the registered arms and the checks of a schedule release.

The runner (dpfp_run) and scripts/prepare_dpfp_release.py share these definitions. The script runs on the login node, so
nothing here imports torch.
"""
from __future__ import annotations

# Candidate order of docs/prereg/DPFP.md: r = 32 first, then r = 128 with the smaller feature map first.
ARMS=('D32-3','D128-1','D128-3')
SPECS={'D32-3':(32,3),'D128-1':(128,1),'D128-3':(128,3)}


def scheduled_steps(arm,steps,schedule,*,smoke=False,E=16):
    """Smoke: the explicit step count. Science: the frozen release's count for this arm."""
    if smoke:
        if schedule is not None or steps is None or steps<5:
            raise ValueError('smoke requires its explicit step count')
        return steps
    if steps is not None or schedule is None:
        raise ValueError('scientific arms require an immutable schedule release')
    if schedule['study']!='DPFP' or schedule['E']!=E:
        raise ValueError('schedule study/capacity mismatch')
    by_arm=schedule['steps_by_arm']
    if arm not in by_arm or (E==16 and set(by_arm)!=set(ARMS)):
        raise ValueError('the E16 release fixes the step counts of all three arms at once (E8: the retrained arm)')
    result=by_arm[arm]
    if isinstance(result,bool) or not isinstance(result,int) or result<5 or result%5:
        raise ValueError('scientific schedules require five equal integer intervals')
    return result


def eligible_e8(lock,arm):
    choice=lock['choice']
    if choice['action']!='retry_e8' or choice['arm']!=arm or choice['arm'] not in ARMS:
        raise ValueError('E8 requires the one registered best arm')
    return 8
