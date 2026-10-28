"""Torch-free helpers of the DPFP and LSQ schedule releases (scripts/prepare_dpfp_release.py, prepare_lsq_release.py).

A release freezes the step count of every arm before any scientific training: the largest multiple of 5 whose estimated
job time fits 2 h, from the timing of the 10% smokes (WP-L4's rule). The scripts run on the login node, so this module
imports only the standard library, and it reads identity and timing fields of the smoke records, never a score.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from looplora.infra.seeds import stable_hash

WALL_SECONDS,RESERVE,MARGIN,EVALUATIONS,EVAL_SCALE=7200,90,1.5,5,10
SMOKE_EPISODES=100  # a 1000-episode scientific evaluation costs EVAL_SCALE times the smoke's 100-episode one


def require(condition,message):
    if not condition:
        raise ValueError(message)


def largest_schedule(step_seconds,eval_seconds,*,evaluations=EVALUATIONS,wall_seconds=WALL_SECONDS,reserve=RESERVE,
                     margin=MARGIN):
    """The rule of read_fix_run.largest_schedule (WP-L4; tests/test_release_scripts.py checks that they agree)."""
    if not all(math.isfinite(v) and v>0 for v in (step_seconds,eval_seconds,wall_seconds,margin)) or reserve<0:
        raise ValueError('schedule requires positive finite timings and nonnegative reserve')
    available=(wall_seconds-reserve)/margin-evaluations*eval_seconds
    steps=5*math.floor(available/step_seconds/5)
    if steps<5:
        raise ValueError('evaluations/setup leave no training budget')
    return steps


def smoke_timing(record,*,arm,steps,E):
    """The timing row of one 10% smoke record: its identity is checked, its timings are read, no score is."""
    require(record['status']=='complete' and record['smoke'] is True and record['arm']==arm,
            f'{arm}: expected a complete smoke record of this arm')
    require(record['source_dirty'] is False and record['config_hash']==stable_hash(record['config']),
            f'{arm}: the smoke ran from dirty source or its config hash does not match')
    commit=record['git_commit']
    require(isinstance(commit,str) and re.fullmatch(r'[0-9a-f]{40}',commit) is not None,
            f'{arm}: the smoke must name its full 40-hex commit')
    config,training=record['config'],record['training']
    require(config['smoke'] is True and config['task']['E']==E and config['eval']['episodes']==SMOKE_EPISODES,
            f'{arm}: expected a {SMOKE_EPISODES}-episode smoke at E{E}')
    marks=[round(steps*k/5) for k in range(1,6)]
    require(training['actual_steps']==steps and training['training_episodes']==32*steps
            and training['eval_steps']==marks and training['primary_eval_steps']==marks
            and training['matched_eval_steps']==[],
            f'{arm}: expected the {steps}-step smoke with five evaluations')
    metrics=training['metrics']
    require(set(metrics)=={str(m) for m in marks} and all(metrics[str(m)]['n']==SMOKE_EPISODES for m in marks),
            f'{arm}: expected one {SMOKE_EPISODES}-episode evaluation at each of the five marks')
    evaluation=[metrics[str(m)]['seconds'] for m in marks]
    timings=[training['steady_step_mean_seconds'],training['step_p95_seconds'],training['seconds'],*evaluation]
    require(all(not isinstance(t,bool) and isinstance(t,(int,float)) and math.isfinite(t) and t>0 for t in timings),
            f'{arm}: the smoke timings must be positive and finite')
    return dict(arm=arm,git_commit=commit,source_hash=record['source_hash'],job_id=record['job_id'],host=record['host'],
                steps=steps,steady_step_mean_seconds=training['steady_step_mean_seconds'],
                step_p95_seconds=training['step_p95_seconds'],total_seconds=training['seconds'],
                eval100_seconds=evaluation)


def sized(row):
    """The largest schedule that fits the job for one smoke row: (steps, sizing record)."""
    step=row['steady_step_mean_seconds']
    full_eval=max(row['eval100_seconds'])*EVAL_SCALE
    target=largest_schedule(step,full_eval)
    total=MARGIN*(target*step+EVALUATIONS*full_eval)+RESERVE
    require(total<=WALL_SECONDS<MARGIN*((target+5)*step+EVALUATIONS*full_eval)+RESERVE,
            f"{row['arm']}: {target} steps are not the largest schedule that fits {WALL_SECONDS} s")
    return target,dict(step_seconds=step,eval1000_seconds=full_eval,evaluations=EVALUATIONS,margin=MARGIN,
                       setup_reserve_seconds=RESERVE,expected_seconds_with_margin=total,
                       smoke_fraction_of_fixed_steps=row['steps']/target)


def schedule_fields(rows):
    """What every release freezes, from the smoke rows (arm -> row): step counts, five evaluation marks, sizing,
    the one commit the smokes ran at (the science commit) and the timing evidence."""
    require(len(rows)>0,'a release needs at least one arm')
    commits={row['git_commit'] for row in rows.values()}
    require(len(commits)==1,'the smokes ran at different commits')
    steps,marks,sizing={},{},{}
    for arm,row in rows.items():
        steps[arm],sizing[arm]=sized(row)
        marks[arm]=[steps[arm]*k//5 for k in range(1,6)]
    return dict(steps_by_arm=steps,primary_eval_steps_by_arm=marks,sizing_by_arm=sizing,science_commit=commits.pop(),
                smoke=dict(steps=next(iter(rows.values()))['steps'],evaluation_episodes=SMOKE_EPISODES,
                           rows=list(rows.values()),quality_scores_exposed=False),
                max_job_seconds=WALL_SECONDS,walltime='02:00:00',status='released_after_smoke_timing_gate')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path,value):
    """Write a release as JSON; an existing file is never overwritten (a release is frozen)."""
    path=Path(path)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        handle.write(text)
