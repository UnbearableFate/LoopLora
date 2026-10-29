#!/usr/bin/env python3
"""Frozen S1B analysis entry point. Execute on compute through scripts/submit.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from looplora.llm.s1b_run import analysis_main

if __name__ == "__main__":
    raise SystemExit(analysis_main())
