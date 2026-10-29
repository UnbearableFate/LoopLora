"""Immutable DPFP schedule release (docs/prereg/DPFP.md): the timing of the 10% smokes of the three arms.

`--study DPFP-X` (WP-L8, docs/prereg/DPFP-X.md): the same rule for the two arms X256 and X17 at E16. The release also holds
`config_hashes` (the science configs at the released step counts, computed on compute by scripts/dpfp_config_hashes.py from
`--steps-only`'s output) and `prereg_blake2b_128`.

Standard library only, for the login node. The release holds `study`, `E`, `steps_by_arm` (the largest schedule that fits
the job, from the smokes' timings; E16 names all three arms, E8 only the retrained one) and `science_commit` (the commit
the smokes ran at). Only identity and timing fields of the smoke records are read, never a score. An existing release is
never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from looplora.llm.dpfp_release import (ARMS,PREREG_X,XARMS,arm_spec,eligible_e8,prereg_digest,scheduled_steps,
                                       study_of)
from looplora.llm.release import read_json,require,schedule_fields,smoke_timing,write_new


def check_release(release,*,E):
    """What the runner will do with the release: it must accept the step count of every arm the release names."""
    require(release['study'] in ('DPFP','DPFP-X') and release['E']==E,'the release names another study or capacity')
    for arm,steps in release['steps_by_arm'].items():
        require(study_of(arm)==release['study'],f'{arm}: not an arm of the {release["study"]} study')
        require(scheduled_steps(arm,None,release,E=E)==steps,f'{arm}: the runner does not accept this release')


def smoke_row(root,arm,*,steps,E):
    record=read_json(Path(root)/arm/'record.json')
    rank,nu,backbone=arm_spec(arm)
    require(record['rank']==rank and record['dpfp_nu']==nu and record['config']['dpfp']['arm']==arm
            and record['config'].get('backbone')==backbone,f'{arm}: the smoke record belongs to another arm')
    return smoke_timing(record,arm=arm,steps=steps,E=E)


def prepare(root,out,*,steps=2400,e8_decision=None,study='DPFP',config_hashes=None):
    """Build, check and write the release; E16 by default, E8 for the one arm of an audited retry_e8 lock. DPFP-X needs the
    science config hashes of its two arms (a mapping arm -> hash); with out=None it only returns the release, unwritten."""
    if out is not None and Path(out).exists():
        raise FileExistsError(f'{out} exists: a frozen release is never overwritten')
    if study=='DPFP-X':
        require(e8_decision is None,'DPFP-X has no E8 retrain')
        rows={arm:smoke_row(root,arm,steps=steps,E=16) for arm in XARMS}
        release=dict(study='DPFP-X',E=16,**schedule_fields(rows),scale_up=PREREG_X,prereg_blake2b_128=prereg_digest())
        check_release(release,E=16)
        if config_hashes is not None:
            require(set(config_hashes)==set(XARMS) and all(isinstance(v,str) and len(v)==32 for v in config_hashes.values()),
                    'config_hashes must name both DPFP-X arms')
            release['config_hashes']=dict(config_hashes)
        if out is not None:
            require(config_hashes is not None,'a DPFP-X release needs the science config hashes')
            write_new(Path(out),release)
        return release
    out=Path(out)
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
    parser.add_argument('--out',type=Path,help='the release file (must not exist)')
    parser.add_argument('--study',choices=('DPFP','DPFP-X'),default='DPFP')
    parser.add_argument('--steps',type=int,help='step count of the smokes (default 2400; DPFP-X: 1800)')
    parser.add_argument('--steps-only',action='store_true',help='DPFP-X: print the released step counts, write nothing')
    parser.add_argument('--config-hashes',type=Path,help='DPFP-X: JSON of scripts/dpfp_config_hashes.py')
    parser.add_argument('--e8-decision',type=Path,help='audited lock with choice retry_e8: an E8 release for that arm')
    args=parser.parse_args()
    steps=args.steps if args.steps is not None else (1800 if args.study=='DPFP-X' else 2400)
    if args.steps_only:
        if args.study!='DPFP-X' or args.out is not None:
            parser.error('--steps-only is for DPFP-X and writes no release')
        print(json.dumps(prepare(args.root,None,steps=steps,study='DPFP-X')['steps_by_arm']))
        raise SystemExit(0)
    if args.out is None:
        parser.error('--out is required')
    hashes=None if args.config_hashes is None else json.loads(args.config_hashes.read_text())['config_hashes']
    release=prepare(args.root,args.out,steps=steps,e8_decision=args.e8_decision,study=args.study,config_hashes=hashes)
    print(json.dumps(dict(study=release['study'],E=release['E'],steps_by_arm=release['steps_by_arm'],
                          science_commit=release['science_commit'],quality_scores_exposed=False)),flush=True)
