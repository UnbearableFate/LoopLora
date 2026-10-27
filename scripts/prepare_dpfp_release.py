"""Immutable DPFP schedule release (docs/prereg/DPFP.md): the timing of the 10% smokes of the three arms.

Standard library only, for the login node. The release holds `study`, `E`, `steps_by_arm` (the largest schedule that fits
the job, from the smokes' timings; E16 names all three arms, E8 only the retrained one) and `science_commit` (the commit
the smokes ran at). Only identity and timing fields of the smoke records are read, never a score. An existing release is
never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from looplora.llm.dpfp_release import ARMS,SPECS,eligible_e8,scheduled_steps
from looplora.llm.release import read_json,require,schedule_fields,smoke_timing,write_new


def check_release(release,*,E):
    """What the runner will do with the release: it must accept the step count of every arm the release names."""
    require(release['study']=='DPFP' and release['E']==E,'the release names another study or capacity')
    for arm,steps in release['steps_by_arm'].items():
        require(scheduled_steps(arm,None,release,E=E)==steps,f'{arm}: the runner does not accept this release')


def smoke_row(root,arm,*,steps,E):
    record=read_json(Path(root)/arm/'record.json')
    rank,nu=SPECS[arm]
    require(record['rank']==rank and record['dpfp_nu']==nu and record['config']['dpfp']['arm']==arm,
            f'{arm}: the smoke record belongs to another arm')
    return smoke_timing(record,arm=arm,steps=steps,E=E)


def prepare(root,out,*,steps=2400,e8_decision=None):
    """Build, check and write the release; E16 by default, E8 for the one arm of an audited retry_e8 lock."""
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists: a frozen release is never overwritten')
    E,sized_arms=16,list(ARMS)
    if e8_decision is not None:
        lock=read_json(e8_decision)
        retrained=lock['choice']['arm']
        E,sized_arms=eligible_e8(lock,retrained),[retrained]
    rows={arm:smoke_row(root,arm,steps=steps,E=E) for arm in sized_arms}
    release=dict(study='DPFP',E=E,**schedule_fields(rows),scale_up='docs/prereg/DPFP.md')
    check_release(release,E=E)
    write_new(out,release)
    return release


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='smoke output root: ROOT/ARM/record.json for every sized arm')
    parser.add_argument('--out',type=Path,required=True,help='the release file (must not exist)')
    parser.add_argument('--steps',type=int,default=2400,help='step count of the smokes')
    parser.add_argument('--e8-decision',type=Path,help='audited lock with choice retry_e8: an E8 release for that arm')
    args=parser.parse_args()
    release=prepare(args.root,args.out,steps=args.steps,e8_decision=args.e8_decision)
    print(json.dumps(dict(study=release['study'],E=release['E'],steps_by_arm=release['steps_by_arm'],
                          science_commit=release['science_commit'],quality_scores_exposed=False)),flush=True)
