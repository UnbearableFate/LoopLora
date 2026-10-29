"""WP-L12 LSQT without torch: the registered arms and the checks of a schedule release (docs/prereg/LSQT.md).

The runner (lsqt_run) and scripts/prepare_dpfp_release.py share these definitions. The script runs on the login node, so nothing
here imports torch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Candidate order of docs/prereg/LSQT.md: K4 first (no hyperparameter), then the smaller eta. LT-U is the conditional control
# of the registration (it runs only after the three registered arms, if the best of them reaches 40 %); it is not a candidate
# arm and has no release of its own here.
ARMS=('LT-K4','LT-RG001','LT-RG1')
CONTROL='LT-U'
RANK=128
# arm -> (store, ridge eta); eta is fixed by the registration, not taken from RFX (which produced none)
SPECS={'LT-K4':('sweep',None),'LT-RG001':('ridge',1e-2),'LT-RG1':('ridge',1.0),'LT-U':('delta',None)}
ETA_BOUNDS=(1e-4,1.0)  # LSQ's clip is (1e-4, 1e-1); the registration widens it to include eta = 1 for LSQT only
PREREG='docs/prereg/LSQT.md'


def prereg_digest(root=None):
    """blake2b-128 of the LSQT registration in a source tree (default: the tree this module is part of)."""
    root=Path(__file__).resolve().parents[3] if root is None else Path(root)
    return hashlib.blake2b((root/PREREG).read_bytes(),digest_size=16).hexdigest()


def scheduled_steps(arm,steps,schedule,*,smoke=False):
    """Smoke: the explicit step count. Science: the frozen release's count for this arm (E16 only; LT-U only from a LSQT-U release)."""
    if arm not in SPECS:
        raise ValueError(f'{arm} is not a registered LSQT arm')
    if smoke:
        if schedule is not None or steps is None or steps<5:
            raise ValueError('smoke requires its explicit step count')
        return steps
    if steps is not None or schedule is None:
        raise ValueError('scientific arms require an immutable schedule release')
    if schedule['study'] not in ('LSQT','LSQT-U') or schedule['E']!=16:
        raise ValueError('schedule study/capacity mismatch')
    by_arm=schedule['steps_by_arm']
    # The three registered arms are released together (study LSQT); the conditional control U is released alone (study LSQT-U),
    # after the three have finished (docs/prereg/LSQT.md, decision 2).
    registered=ARMS if schedule['study']=='LSQT' else (CONTROL,)
    if arm not in registered or set(by_arm)!=set(registered):
        raise ValueError('the LSQT release fixes the step counts of the three registered arms at once, the LSQT-U release that of the control alone, at E16 only')
    result=by_arm[arm]
    if isinstance(result,bool) or not isinstance(result,int) or result<5 or result%5:
        raise ValueError('scientific schedules require five equal integer intervals')
    return result
