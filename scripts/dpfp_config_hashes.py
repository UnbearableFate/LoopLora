"""Science config hashes of the arms of DPFP-X (WP-L8), LSQT or SQ2 (WP-L12) at their released step counts.

`configuration` imports torch, so this runs on compute. Its output goes to `prepare_dpfp_release.py --config-hashes`, which freezes
the hashes in the release; the runner then refuses any other config.
Usage: dpfp_config_hashes.py [--study DPFP-X|LSQT|SQ2] [--numerics FILE (SQ2)] --out FILE ARM=STEPS ... (STEPS from `prepare_dpfp_release.py --steps-only`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from looplora.infra.seeds import stable_hash
from looplora.llm import dpfp, lsqt, lsqt_release, sq2, sq2_release, sq2_run
from looplora.llm.dpfp_release import XARMS



def sq2_config(arm, steps, numerics):
    store, coefficient = sq2_release.SPECS[arm]
    return sq2.configuration(arm, steps, eta=numerics['eta'] if store == 'ridge' else None,
                             w_c=numerics['w_c'] if coefficient else None, numerics_digest=numerics['digest'])


STUDIES = {'DPFP-X': (XARMS, lambda arm, steps, numerics: dpfp.configuration(arm, steps, E=16)),
           'LSQT': (lsqt_release.ARMS, lambda arm, steps, numerics: lsqt.configuration(arm, steps, E=16)),
           'SQ2': (sq2_release.ARMS, sq2_config)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=tuple(STUDIES), default='DPFP-X')
    parser.add_argument('--numerics', type=Path, help='SQ2: the committed numerics file')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('steps', nargs='+', help='ARM=STEPS for each arm of the study')
    args = parser.parse_args()
    arms, build = STUDIES[args.study]
    steps = {key: int(value) for key, value in (item.split('=') for item in args.steps)}
    if set(steps) != set(arms) or len(args.steps) != len(arms):
        raise SystemExit(f'name every arm of {args.study} once: {", ".join(arms)}')
    if (args.study == 'SQ2') != (args.numerics is not None):
        raise SystemExit('--numerics belongs to the SQ2 study, and SQ2 needs it')
    numerics = None if args.numerics is None else sq2_run.numerics_values(args.numerics)
    hashes = {arm: stable_hash(build(arm, steps[arm], numerics)) for arm in arms}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(dict(steps_by_arm=steps, config_hashes=hashes), indent=2, sort_keys=True) + '\n')
    print(json.dumps(hashes), flush=True)
