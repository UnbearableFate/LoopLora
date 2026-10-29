"""SQ2 numerics (docs/prereg/SQ2.md): eta and w_c from training-only numerics at the D128-3 checkpoint; compute only.

The output (new file, never overwritten) is committed before any smoke output is read and its blake2b-128 is frozen in the release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from looplora.llm.run import source_metadata
from looplora.llm.sq2_numerics import numerics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = numerics()
    result.update(**source_metadata(), job_id=os.environ.get('PBS_JOBID'), torch=torch.__version__, gpu=torch.cuda.get_device_name())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('eta', 'w_c', 'eta_curve', 'gradient_ratio')}, sort_keys=True, default=float), flush=True)
