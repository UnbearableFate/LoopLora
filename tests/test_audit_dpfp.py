"""scripts/audit_dpfp.py on a tiny exported DPFP writer (compute only): the memory-intervention audit of the final checkpoints."""
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from looplora.infra.seeds import derive_seed
from looplora.llm.dpfp import DPFPExperiment, configuration
from looplora.llm import lsqt, sq2
from looplora.llm.inject import load_tokenizer
from looplora.llm.read_fix_run import validate

ROOT = Path(__file__).resolve().parents[1]
RANK = 8  # as tests/test_llm_dpfp.py: at rank 4 the phi supports of keys and query do not overlap and the read is exactly zero


def script():
    spec = importlib.util.spec_from_file_location('audit_dpfp', ROOT / 'scripts' / 'audit_dpfp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope='module')
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, 'dpfp-test-model'))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield model
    assert not hasattr(model, 'looplora_adapters')


def export(folder, model, tokenizer, count, arm='D32-3'):
    """A tiny writer of `arm` with a memory that matters (B = .2), its writer.pt and a record as the runner writes it.
    D arms get the tiny rank; X256 keeps its registered rank 256 (a [256, 1536] memory)."""
    cfg = configuration(arm, 10)
    cfg['model'].update(layer=1, **({} if arm == 'X256' else dict(r=RANK)))
    experiment = DPFPExperiment(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        experiment.standardize()
        with torch.no_grad():
            experiment.adapter.B.fill_(.2)
        torch.save(dict(read_fix_version=1, arm=arm, config=experiment.config, step=5, signature=experiment.writer_signature(),
                        adapter=experiment.inference_state()), folder / 'writer.pt')
        metrics = validate(experiment, folder / 'val-5', count)
    finally:
        experiment.close()
    record = dict(status='complete', smoke=False, arm=arm, rank=cfg['model']['r'], dpfp_nu=3, job_id='fixture', git_commit='f' * 40, config=cfg,
                  training=dict(actual_steps=5, metrics={'5': metrics}))
    (folder / 'record.json').write_text(json.dumps(record))
    return metrics


def test_audit_reproduces_the_final_validation_and_both_memory_interventions_change_it(model, tokenizer, tmp_path):
    audit = script()
    metrics = export(tmp_path, model, tokenizer, 32)
    result = audit.audit_one(tmp_path, tmp_path / 'audit', 32, model=model, tokenizer=tokenizer)
    saved = json.loads((tmp_path / 'audit' / 'summary.json').read_text())
    assert saved['original'] == result['original'] and saved['arm'] == 'D32-3' and saved['E'] == 16 and saved['episodes'] == 32
    # The original is the record's own validation (same episodes, same weights), and it is repeatable bit for bit.
    assert result['original_matches_record']['correct'] is True and result['original']['correct'] == metrics['correct']
    assert result['original']['nll'] == pytest.approx(metrics['nll'], rel=1e-5) and result['max_repeated_logit_drift'] == 0.
    for name in audit.CONDITIONS:
        assert 0 <= result[name]['correct'] <= 32 and result[name]['em'] == result[name]['correct'] / 32
        assert math.isfinite(result[name]['nll'])
    # Zeroing and shuffling the memory must change the scores of a model whose memory matters.
    assert result['zero']['nll'] != result['original']['nll'] and result['shuffle']['nll'] != result['original']['nll']
    assert result['zero']['nll'] != result['shuffle']['nll'] and result['optimizer_steps'] == 0
    with pytest.raises(FileExistsError):
        audit.audit_one(tmp_path, tmp_path / 'audit', 32, model=model, tokenizer=tokenizer)


def test_audit_refuses_a_smoke_or_an_unfinished_record(model, tokenizer, tmp_path):
    audit = script()
    export(tmp_path, model, tokenizer, 16)
    record = json.loads((tmp_path / 'record.json').read_text())
    for change in (dict(smoke=True), dict(status='running')):
        (tmp_path / 'record.json').write_text(json.dumps({**record, **change}))
        with pytest.raises(AssertionError):
            audit.audit_one(tmp_path, tmp_path / f'audit-{next(iter(change))}', 16, model=model, tokenizer=tokenizer)


def test_audit_takes_an_x256_export_with_the_rank_256_memory(model, tokenizer, tmp_path):
    audit = script()
    metrics = export(tmp_path, model, tokenizer, 16, arm='X256')
    result = audit.audit_one(tmp_path, tmp_path / 'audit', 16, model=model, tokenizer=tokenizer)
    assert (result['arm'], result['rank'], result['dpfp_nu'], result['E']) == ('X256', 256, 3, 16)
    assert result['original_matches_record']['correct'] is True and result['original']['correct'] == metrics['correct']
    assert result['max_repeated_logit_drift'] == 0. and result['optimizer_steps'] == 0
    assert result['zero']['nll'] != result['original']['nll'] and result['shuffle']['nll'] != result['original']['nll']
    assert all(math.isfinite(result[name]['nll']) for name in audit.CONDITIONS)


def export_study(folder, model, tokenizer, count, arm):
    """A tiny exported writer of an LSQT (LT-*) or SQ2 (S-*) arm and its record, as the study's runner writes them."""
    if arm.startswith('LT-'):
        cfg, cls = lsqt.configuration(arm, 10), lsqt.LSQTExperiment
    else:
        cfg, cls = sq2.configuration(arm, 10, eta=1e-2 if arm in ('S-RE', 'S-RC') else None, w_c=2. if arm in ('S-DC', 'S-RC') else None), sq2.SQ2Experiment
    cfg['model'].update(layer=1)
    experiment = cls(cfg, arm, model=model, tokenizer=tokenizer)
    try:
        experiment.standardize()
        with torch.no_grad():
            experiment.adapter.B.fill_(.2)
        torch.save(dict(read_fix_version=1, arm=arm, config=experiment.config, step=5, signature=experiment.writer_signature(),
                        adapter=experiment.inference_state()), folder / 'writer.pt')
        metrics = validate(experiment, folder / 'val-5', count)
    finally:
        experiment.close()
    record = dict(status='complete', smoke=False, arm=arm, rank=128, job_id='fixture', git_commit='f' * 40, config=cfg,
                  training=dict(actual_steps=5, metrics={'5': metrics}))
    if arm.startswith('S-'):
        record['dpfp_nu'] = 3
    (folder / 'record.json').write_text(json.dumps(record))
    return metrics


@pytest.mark.parametrize('arm', ['LT-K4', 'LT-RG001', 'S-DE', 'S-RC'])
def test_audit_takes_the_lsqt_and_sq2_exports_with_their_own_loaders(model, tokenizer, tmp_path, arm):
    audit = script()
    metrics = export_study(tmp_path, model, tokenizer, 16, arm)
    result = audit.audit_one(tmp_path, tmp_path / 'audit', 16, model=model, tokenizer=tokenizer)
    assert (result['arm'], result['rank'], result['E']) == (arm, 128, 16) and result['dpfp_nu'] == (3 if arm.startswith('S-') else None)
    assert result['original_matches_record']['correct'] is True and result['original']['correct'] == metrics['correct']
    assert result['max_repeated_logit_drift'] == 0. and result['optimizer_steps'] == 0
    assert result['zero']['nll'] != result['original']['nll'] and result['shuffle']['nll'] != result['original']['nll']
    assert all(math.isfinite(result[name]['nll']) for name in audit.CONDITIONS)
