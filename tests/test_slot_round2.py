"""Artifact-boundary regressions for review round 2; compute only, synthetic parents and data."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import slot_golden
from test_slot_experiment import tokenizer
from test_slot_release import inputs, builder
from looplora.infra.seeds import derive_seed
from looplora.llm import slot_data, slot_rules, slot_run
from looplora.llm.slot_experiment import adopt
from looplora.llm.slot_read import ReadSpec

ROOT = Path(__file__).resolve().parents[1]


def test_round2_smoke_json_builder_and_cli_reload_are_one_strict_path(inputs, tokenizer, tmp_path, monkeypatch):
    rows, paths = inputs
    # Match the registered adapter dimensions without loading any real parent weights.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'slot-json-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=1024, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, head_dim=512,
            attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    def synthetic_parent(name, **kwargs):
        config = slot_golden.tiny_config()
        config['model']['layer'] = 0
        from looplora.llm.sq2 import SQ2Experiment
        experiment = adopt(SQ2Experiment(config, 'S-RE', model=model, tokenizer=tokenizer))
        with torch.no_grad():
            experiment.adapter.B.normal_(0, 0.05)
        return experiment
    monkeypatch.setattr(slot_run, 'load_parent', synthetic_parent)
    monkeypatch.setattr(slot_run, 'require_lock', lambda path: rows['manifest'])
    slot_run.run_smoke(tmp_path / 'smoke', ROOT / 'configs/slot_loop_v1/s1a.json', paths['manifest'], paths['s0_metrics'])
    candidate = tmp_path / 'candidate.json'
    builder.prepare(paths['manifest'], paths['pilot'], paths['s0_metrics'], tmp_path / 'smoke/smoke.json',
                    science_commit=slot_run.provenance()['git_commit'], out=candidate)
    loaded = json.loads(candidate.read_text())
    assert loaded['status'] == 'pending_reload' and len(loaded['config_hashes']) == 8
    monkeypatch.setattr(slot_run, 'RUNS', tmp_path)
    from looplora.llm import slot_experiment
    original_load = slot_experiment.load_export
    monkeypatch.setattr(slot_run, 'load_export', lambda path, **kwargs: original_load(path, model=model, tokenizer=tokenizer))
    for arm in slot_rules.ARMS:
        report = tmp_path / f'reload-{arm}.json'
        assert slot_run.main(['--mode', 'reload-check', '--manifest', str(paths['manifest']), '--release', str(candidate),
            '--export', str(tmp_path / 'smoke' / arm / 'selected.pt'),
            '--predictions', str(tmp_path / 'smoke' / arm / 'smoke_predictions.npz'), '--out', str(report)]) == 0
        assert json.loads(report.read_text())['config_matches_release'] is True


@pytest.fixture
def evaluation_boundary(tmp_path, monkeypatch):
    root = tmp_path / 'opening'
    (root / 'docs/prereg').mkdir(parents=True)
    (root / 'scripts').mkdir()
    (root / 'reports/slot-loop-v1').mkdir(parents=True)
    shutil.copy(ROOT / slot_data.PREREG, root / slot_data.PREREG)
    (root / slot_data.ANALYSIS_SCRIPT).write_text('frozen = True\n')
    selected = {f'{arm}/{seed}': f'{i + 1:032x}' for i, (arm, seed) in enumerate(
        (arm, seed) for arm in slot_rules.ARMS for seed in slot_data.RUN_SEEDS)}
    record = dict(prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128, selected_checkpoints=selected,
                  analysis_script_blake2b_128=slot_data.file_digest(root / slot_data.ANALYSIS_SCRIPT))
    (root / slot_data.OPEN_RECORD).write_text(json.dumps(record))
    reads = {arm: ReadSpec(slot_rules.ARM_KIND[arm], tau=0.3 if arm == 'soft_slot_L1' else None).as_dict()
             for arm in slot_rules.ARMS}
    release = dict(status='ready', prereg_blake2b_128=slot_data.PREREG_BLAKE2B_128,
                   analysis_script_blake2b_128=record['analysis_script_blake2b_128'],
                   arms={arm: dict(read=read) for arm, read in reads.items()},
                   parent=dict(writer_blake2b_128='e' * 32))
    release_path = root / 'release.json'
    release_path.write_text(json.dumps(release))
    inventory = {'S-RE': {'writer.pt': 'e' * 32}}
    monkeypatch.setattr(slot_run, 'require_lock', lambda path: dict(inventory=inventory))
    monkeypatch.setattr(slot_run, 'RUNS', tmp_path)
    # No sealed payload is generated: synthetic records stand in for the model and episodes.
    monkeypatch.setattr(slot_run, 'episodes', lambda s: SimpleNamespace(episode_seed=np.arange(2, dtype=np.int64)))
    monkeypatch.setattr(slot_run, 'ids_record', lambda s, a: 'synthetic')
    monkeypatch.setattr(slot_run, 'file_digest', lambda path: Path(path).name)
    metadata = {digest: dict(arm=key.split('/')[0], run=dict(seed=int(key.split('/')[1]), ablation=False))
                for key, digest in selected.items()}
    def fake_load(path, **kwargs):
        slot = metadata[Path(path).name]
        return SimpleNamespace(slot=slot, read_spec=ReadSpec.from_dict(reads[slot['arm']]), model=None, tokenizer=None, close=lambda: None)
    monkeypatch.setattr(slot_run, 'load_export', fake_load)
    monkeypatch.setattr(slot_run, 'load_parent', lambda *a, **kw: SimpleNamespace(model=None, tokenizer=None, close=lambda: None))
    monkeypatch.setattr(slot_run, 'evaluate', lambda *a, **kw: dict(samples={'read': dict(
        episode_id=np.arange(2, dtype=np.int64), correct=np.ones(2, dtype=bool), nll=np.zeros(2, dtype=np.float32))}))
    entries = [dict(label=f'{arm}_s{seed}', export=digest, checkpoint_blake2b_128=digest)
               for key, digest in selected.items() for arm, seed in [key.split('/')]]
    entries += [dict(label=label, parent_writer_blake2b_128='e' * 32, read=read)
                for label, read in [('s_re_native', None), ('matrix_step0', reads['matrix_L1']), ('soft_step0', reads['soft_slot_L1'])]]
    return SimpleNamespace(root=root, release=release, release_path=release_path, entries=entries, metadata=metadata,
                           inventory=inventory, record=record, tmp=tmp_path)


def evaluate_cli(boundary, entries):
    path = boundary.tmp / 'entries.json'
    path.write_text(json.dumps(entries))
    return slot_run.main(['--mode', 'evaluate', '--set', 'confirmation', '--open-root', str(boundary.root),
        '--release', str(boundary.release_path), '--entries', str(path), '--manifest', 'synthetic.lock.json',
        '--out', str(boundary.tmp / 'evaluated')])


@pytest.mark.parametrize('defect', ['swapped', 'duplicate_checkpoint', 'reference_as_main', 'export_metadata', 'duplicate_label'])
def test_round2_checkpoint_label_mismatch_is_refused(evaluation_boundary, defect):
    b = evaluation_boundary
    entries = json.loads(json.dumps(b.entries))
    if defect == 'swapped':
        for field in ('export', 'checkpoint_blake2b_128'):
            entries[0][field], entries[3][field] = entries[3][field], entries[0][field]
    elif defect == 'duplicate_checkpoint':
        entries[1].update(export=entries[0]['export'], checkpoint_blake2b_128=entries[0]['checkpoint_blake2b_128'])
    elif defect == 'reference_as_main':
        entries[0] = dict(entries[6], label=entries[0]['label'])
    elif defect == 'export_metadata':
        b.metadata[entries[0]['checkpoint_blake2b_128']]['run']['ablation'] = True
    else:
        entries.append(dict(entries[0]))
    try:
        result = evaluate_cli(b, entries)
    except slot_data.SealedSetError:
        result = 3
    assert result == 3


def test_round2_changed_script_with_matching_open_record_is_refused(evaluation_boundary, monkeypatch):
    b = evaluation_boundary
    (b.root / slot_data.ANALYSIS_SCRIPT).write_text('changed_after_release = True\n')
    b.record['analysis_script_blake2b_128'] = slot_data.file_digest(b.root / slot_data.ANALYSIS_SCRIPT)
    (b.root / slot_data.OPEN_RECORD).write_text(json.dumps(b.record))
    with pytest.raises(slot_data.SealedSetError):
        evaluate_cli(b, b.entries)


@pytest.mark.parametrize('step', [1, 2, 3, 4])
def test_round2_runbook_uses_persistent_paths_and_the_right_snapshot(step):
    import shlex
    state = (ROOT / 'docs/reports/SLOT-LOOP-v1-state.md').read_text()
    line = next(line for line in state.splitlines() if line.startswith(f'  {step}. '))
    command = line.split('`')[1]
    argv = shlex.split(command)
    if step == 2:
        assert argv[:2] == ['/work/xg24i002/x10041/LoopLora/.venv/bin/python', 'scripts/prepare_slot_manifest.py']
    else:
        assert argv[0] == 'scripts/submit.py'
        assert argv[argv.index('--commit') + 1] == ('$REVIEW_COMMIT' if step == 1 else '$LOCK_COMMIT')
        after = argv[argv.index('--') + 1:]
        assert after[:2] == ['-m', 'looplora.llm.slot_run'] if step != 4 else after[0] == 'scripts/analyze_slot_study.py'
    for flag in ('--out', '--manifest'):
        if flag in argv:
            value = argv[argv.index(flag) + 1].replace('$R', str(slot_run.RUNS)).replace('$W', str(ROOT))
            assert Path(value).is_absolute()
    if step == 4:
        assert argv[argv.index('--out') + 1] == '$R/s0/decision'
