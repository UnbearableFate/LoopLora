"""Parse every executable submission in the state runbook without submitting or running a workload."""
import argparse
import importlib.util
import re
import shlex
import string
import sys
from pathlib import Path

import pytest

from looplora.llm import slot_run

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'docs/reports/SLOT-LOOP-v1-state.md'


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def commands(text):
    found = []
    for block in re.findall(r'```bash\n(.*?)```', text, re.S):
        found += [line.strip() for line in block.splitlines() if line.strip().startswith('scripts/submit.py ')]
    prose = re.sub(r'```.*?```', '', text, flags=re.S)
    found += [span for span in re.findall(r'`([^`\n]+)`', prose)
              if span.startswith('scripts/submit.py ') and ' -- ' in span]
    return found


class Parsed(Exception):
    def __init__(self, value):
        self.value = value


def parsed_call(monkeypatch, function, argv, script):
    original = argparse.ArgumentParser.parse_args
    def capture(parser, *args, **kwargs):
        raise Parsed(original(parser, *args, **kwargs))
    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, 'parse_args', capture)
        patch.setattr(sys, 'argv', [script, *argv])
        with pytest.raises(Parsed) as result:
            function() if script == 'scripts/submit.py' else function(argv)
    return result.value.value


def test_command_extraction_selects_submission_instead_of_first_backticked_word():
    text = '1. On `debug-g`, run `scripts/submit.py --name example --walltime 00:03:00 --commit "' + 'c' * 40 + '" -- -m target`.'
    assert commands(text) == [text.split('run `')[1][:-2]]


def test_all_submission_commands_pass_submit_and_target_parsers(monkeypatch):
    submit = module('submit')
    targets = {'scripts/analyze_slot_study.py': module('analyze_slot_study').main,
               'scripts/prepare_slot_release.py': module('prepare_slot_release').main}
    raw = commands(STATE.read_text())
    assert raw, 'no executable runbook commands found'
    modes, stages, builder = set(), set(), 0
    for arm in ('matrix_L1', 'soft_slot_L1'):
        for seed in ('17', '29', '43'):
            for set_name in ('confirmation', 'unseen'):
                values = dict(R=str(slot_run.RUNS), W=str(ROOT), REVIEW_COMMIT='a' * 40, LOCK_COMMIT='b' * 40,
                    SCIENCE_COMMIT='c' * 40, arm=arm, seed=seed, set_name=set_name,
                    TRAIN_WALLTIME='00:25:00', TRAIN_QUEUE='debug-g', OPEN_ROOT=str(slot_run.RUNS / 'open-root'))
                for command in raw:
                    argv = [string.Template(token).substitute(values) for token in shlex.split(command)]
                    assert argv[0] == 'scripts/submit.py'
                    sub = parsed_call(monkeypatch, submit.main, argv[1:], argv[0])
                    assert sub.worktree is None and re.fullmatch('[0-9a-f]{40}', sub.commit)
                    submit.check_cap(sub.nodes, submit.parse_walltime(sub.walltime))
                    assert sub.name and sub.queue in submit.QUEUES
                    if sub.queue == 'debug-g':
                        assert submit.parse_walltime(sub.walltime) <= 1800
                    target = sub.argv[1:] if sub.argv[:1] == ['--'] else sub.argv
                    if target[:2] == ['-m', 'looplora.llm.slot_run']:
                        parsed = parsed_call(monkeypatch, slot_run.main, target[2:], target[1])
                        modes.add(parsed.mode)
                        assert parsed.out.is_absolute()
                        if parsed.mode != 'preflight':
                            assert parsed.manifest.is_absolute()
                        if parsed.mode in {'preflight', 's0', 'smoke', 'pilot', 'train'}:
                            assert parsed.config.is_absolute()
                        if parsed.mode in {'smoke', 'pilot'}:
                            assert parsed.s0_metrics.is_absolute()
                        if parsed.mode in {'reload-check', 'train', 'evaluate'}:
                            assert parsed.release.is_absolute()
                        if parsed.mode == 'evaluate':
                            assert parsed.entries.is_absolute() and parsed.open_root.is_absolute()
                        if parsed.mode == 'reload-check':
                            assert parsed.export.is_absolute() and parsed.predictions.is_absolute()
                    else:
                        assert target[0] in targets, f'extra interpreter or unknown target: {target[0]}'
                        parsed = parsed_call(monkeypatch, targets[target[0]], target[1:], target[0])
                        if target[0].endswith('analyze_slot_study.py'):
                            stages.add(parsed.stage)
                            assert Path(parsed.records).is_absolute() and Path(parsed.out).is_absolute()
                            assert Path(parsed.manifest).is_absolute()
                            if parsed.stage == 'S1A':
                                assert Path(parsed.open_record).is_absolute() and Path(parsed.release).is_absolute()
                        else:
                            builder += 1
                            for field in ('manifest', 'pilot', 's0_metrics', 'smoke', 'out'):
                                assert getattr(parsed, field).is_absolute()
                            assert re.fullmatch('[0-9a-f]{40}', parsed.science_commit)
    assert modes >= {'preflight', 's0', 'smoke', 'pilot', 'reload-check', 'train', 'evaluate'}
    assert stages >= {'S0', 'S1A-DEV', 'S1A-ABL', 'S1A'}
    assert builder > 0
