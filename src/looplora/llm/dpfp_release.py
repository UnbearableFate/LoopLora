"""WP-L5 DPFP without torch: the registered arms and the checks of a schedule release.

The runner (dpfp_run) and scripts/prepare_dpfp_release.py share these definitions. The script runs on the login node, so
nothing here imports torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Candidate order of docs/prereg/DPFP.md: r = 32 first, then r = 128 with the smaller feature map first.
ARMS=('D32-3','D128-1','D128-3')
SPECS={'D32-3':(32,3),'D128-1':(128,1),'D128-3':(128,3)}

# docs/prereg/DPFP-X.md (WP-L8): the best DPFP arm at rank 256 on the 0.6B backbone (X256, rank 256 is not a WP-L4 run) and on
# Qwen3-1.7B-Base at rank 128 (X17). Both nu = 3, E16 only: no E8 retrain follows. X256 keeps the default backbone.
BACKBONE_17B=dict(id='Qwen/Qwen3-1.7B-Base',revision='ea980cb0a6c2ae4b936e82123acc929f1cec04c1')
XARMS=('X256','X17')
XSPECS={'X256':(256,3,None),'X17':(128,3,BACKBONE_17B)}  # rank, nu, backbone (None: the pinned 0.6B)


def arm_spec(arm):
    """(rank, nu, backbone or None) of a registered DPFP or DPFP-X arm."""
    if arm in SPECS:
        return (*SPECS[arm],None)
    if arm in XSPECS:
        return XSPECS[arm]
    raise ValueError(f'{arm} is not a registered DPFP or DPFP-X arm')


PREREG_X='docs/prereg/DPFP-X.md'


def prereg_digest(root=None):
    """blake2b-128 of the DPFP-X preregistration in a source tree (default: the tree this module is part of)."""
    root=Path(__file__).resolve().parents[3] if root is None else Path(root)
    return hashlib.blake2b((root/PREREG_X).read_bytes(),digest_size=16).hexdigest()


def study_of(arm):
    return 'DPFP-X' if arm in XSPECS else 'DPFP'


def scheduled_steps(arm,steps,schedule,*,smoke=False,E=16):
    """Smoke: the explicit step count. Science: the frozen release's count for this arm."""
    if smoke:
        if schedule is not None or steps is None or steps<5:
            raise ValueError('smoke requires its explicit step count')
        return steps
    if steps is not None or schedule is None:
        raise ValueError('scientific arms require an immutable schedule release')
    if schedule['study']!=study_of(arm) or schedule['E']!=E:
        raise ValueError('schedule study/capacity mismatch')
    by_arm=schedule['steps_by_arm']
    if study_of(arm)=='DPFP-X':
        if E!=16 or set(by_arm)!=set(XARMS):
            raise ValueError('the DPFP-X release fixes the step counts of both arms at once, at E16 only')
    elif arm not in by_arm or (E==16 and set(by_arm)!=set(ARMS)):
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
