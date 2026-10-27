"""Immutable LSQ schedule release (docs/prereg/LSQ.md): RFX's summary.json plus the timing of the 10% smokes.

Standard library only, for the login node. The release holds `study`, `E`, `arms` (RFX's `lsq_arms`), `steps_by_arm`
(the largest schedule that fits the job, from the smokes' timings), `rfx_eta_by_rank` and `eta_by_rank` (copied from the
summary) and `science_commit` (the commit the smokes ran at). Only identity and timing fields of the smoke records are
read, never a score. An existing release is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from looplora.llm.lsq_release import SPECS,check_arms,eligible_e8,release_arms,release_eta,scheduled_steps
from looplora.llm.release import read_json,require,schedule_fields,smoke_timing,write_new


def check_summary(summary):
    """RFX's science summary must be complete and clean; returns the arms it triggers."""
    require(summary['smoke'] is False,'an RFX smoke summary releases nothing')
    require(summary['source_dirty'] is False,'the RFX run used dirty source')
    require(not any(summary['instrumentation_failures'].values()),'the RFX run failed its instrumentation checks')
    require(summary.get('entity_em_guard',{}).get('passed') is True,'the RFX run did not pass the entity-restricted EM guard')
    arms=check_arms(summary['lsq_arms'])  # raises when RFX triggered nothing or an impossible set
    ridge_ranks={str(SPECS[arm][0]) for arm in arms if SPECS[arm][1]=='ridge'}
    require(set(summary['rfx_eta_by_rank'])==ridge_ranks==set(summary['eta_by_rank']),
            'the eta maps must hold exactly the ranks of the triggered RG arms')
    return arms


def check_release(release,summary,*,E):
    """What the runner will do with the release must agree with RFX's summary (the runner cannot check this itself)."""
    require(release['arms']==summary['lsq_arms'],'the release arms differ from the summary lsq_arms')
    require(release['rfx_eta_by_rank']==summary['rfx_eta_by_rank'] and release['eta_by_rank']==summary['eta_by_rank'],
            'the release eta maps differ from the summary')
    require(release_arms(release)==summary['lsq_arms'],'the release arms are not RFX\'s arms')
    for arm in release['arms']:
        eta=release_eta(release,arm)
        rank,store=SPECS[arm]
        require(eta==(summary['eta_by_rank'][str(rank)] if store=='ridge' else None),f'{arm}: wrong release eta')
    for arm,steps in release['steps_by_arm'].items():
        require(scheduled_steps(arm,None,release,E=E)==steps,f'{arm}: the runner does not accept this release')


def smoke_row(root,arm,*,steps,E):
    record=read_json(Path(root)/arm/'record.json')
    rank,store=SPECS[arm]
    require(record['rank']==rank and record['store']==store and record['config']['lsq']['arm']==arm,
            f'{arm}: the smoke record belongs to another arm')
    require((record['store_eta'] is not None)==(store=='ridge'),f'{arm}: only the RG arms take an eta')
    return smoke_timing(record,arm=arm,steps=steps,E=E)


def prepare(summary_path,root,out,*,steps=2400,e8_decision=None):
    """Build, check and write the release; E16 by default, E8 for the one arm of an audited retry_e8 lock."""
    out=Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists: a frozen release is never overwritten')
    summary=read_json(summary_path)
    arms=check_summary(summary)
    E,sized_arms=16,arms
    if e8_decision is not None:
        lock=read_json(e8_decision)
        retrained=lock['choice']['arm']
        E,sized_arms=eligible_e8(lock,retrained),[retrained]
        require(retrained in arms,f'{retrained} is not one of the arms RFX triggered')
    rows={arm:smoke_row(root,arm,steps=steps,E=E) for arm in sized_arms}
    release=dict(study='LSQ',E=E,arms=arms,rfx_eta_by_rank=dict(summary['rfx_eta_by_rank']),
                 eta_by_rank=dict(summary['eta_by_rank']),**schedule_fields(rows),
                 rfx=dict(summary_blake2b=hashlib.blake2b(Path(summary_path).read_bytes(),digest_size=16).hexdigest(),
                          git_commit=summary['git_commit'],job_id=summary['job_id'],verdicts=summary['verdicts']),
                 scale_up='docs/prereg/LSQ.md')
    check_release(release,summary,E=E)
    write_new(out,release)
    return release


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True,help="RFX's science summary.json")
    parser.add_argument('--root',type=Path,required=True,help='smoke output root: ROOT/ARM/record.json for every sized arm')
    parser.add_argument('--out',type=Path,required=True,help='the release file (must not exist)')
    parser.add_argument('--steps',type=int,default=2400,help='step count of the smokes')
    parser.add_argument('--e8-decision',type=Path,help='audited lock with choice retry_e8: an E8 release for that arm')
    args=parser.parse_args()
    release=prepare(args.summary,args.root,args.out,steps=args.steps,e8_decision=args.e8_decision)
    print(json.dumps(dict(study=release['study'],E=release['E'],arms=release['arms'],steps_by_arm=release['steps_by_arm'],
                          science_commit=release['science_commit'],quality_scores_exposed=False)),flush=True)
