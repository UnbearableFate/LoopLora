"""scripts/submit.py: the user's per-job cap (2026-10-29), nodes x walltime <= 16 node-h."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('submit', ROOT / 'scripts' / 'submit.py')
submit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submit)


@pytest.mark.parametrize('nodes, wall', [(1, 16 * 3600), (2, 8 * 3600), (8, 2 * 3600)])
def test_jobs_up_to_16_node_hours_pass(nodes, wall):
    submit.check_cap(nodes, wall)


@pytest.mark.parametrize('nodes, wall', [(1, 16 * 3600 + 1), (2, 8 * 3600 + 1), (0, 60)])
def test_jobs_above_16_node_hours_or_without_a_node_are_refused(nodes, wall):
    with pytest.raises(SystemExit):
        submit.check_cap(nodes, wall)
