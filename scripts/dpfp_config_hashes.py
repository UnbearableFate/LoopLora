"""Science config hashes of the DPFP-X arms at their released step counts (docs/prereg/DPFP-X.md, WP-L8).

`configuration` imports torch, so this runs on compute. Its output goes to `prepare_dpfp_release.py --study DPFP-X
--config-hashes`, which freezes the hashes in the release; the runner then refuses any other config.
Usage: dpfp_config_hashes.py --out FILE X256=STEPS X17=STEPS (STEPS from `prepare_dpfp_release.py --steps-only`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from looplora.infra.seeds import stable_hash
from looplora.llm.dpfp import configuration
from looplora.llm.dpfp_release import XARMS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('steps', nargs='+', help='ARM=STEPS for each of X256 and X17')
    args = parser.parse_args()
    steps = {key: int(value) for key, value in (item.split('=') for item in args.steps)}
    if set(steps) != set(XARMS) or len(args.steps) != len(XARMS):
        raise SystemExit('name both X256 and X17 once')
    hashes = {arm: stable_hash(configuration(arm, steps[arm], E=16)) for arm in XARMS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(dict(steps_by_arm=steps, config_hashes=hashes), indent=2, sort_keys=True) + '\n')
    print(json.dumps(hashes), flush=True)
