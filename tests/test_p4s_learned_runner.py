"""Synthetic head-only runner checks; run the GPU canaries in an allocation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from looplora.infra.seeds import derive_seed, stable_hash

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import run_p4s_learned as runner


def training_design():
    return dict(training=dict(split='train', generator_seed=31337, episodes=20000,
                             steps=2000, optimizer='adam', lr=.001, batch_episodes=256,
                             hops_per_episode=8, steps_per_query=16))


def feature_cache(device='cpu', episodes=11):
    """Detached post-state features with distinct episode/query/step labels."""
    generator = torch.Generator(device=device).manual_seed(31337)
    states = torch.randn(episodes, 8, 16, 32, generator=generator, device=device) * .1
    states[..., 0] = torch.arange(episodes, device=device)[:, None, None]
    states[..., 1] = torch.arange(8, device=device)[None, :, None]
    step = torch.eye(16, device=device).expand(episodes, 8, -1, -1)
    features = torch.cat((states, states.norm(dim=-1, keepdim=True),
                          torch.ones(episodes, 8, 16, 2, device=device), step), dim=-1)
    labels = (torch.arange(episodes, device=device)[:, None, None] +
              torch.arange(8, device=device)[None, :, None] +
              torch.arange(16, device=device)[None, None, :]) % 3 == 0
    return features.detach(), labels


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fit_head_replays_initialization_and_episode_sampler_without_global_rng_changes(device, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA runner canary requires a compute GPU')
    if device == 'cpu':
        # The production runner is CUDA-only; its synchronization is irrelevant
        # to testing the same head optimizer and dedicated sampler on CPU.
        monkeypatch.setattr(runner.torch.cuda, 'synchronize', lambda: None)
    features, targets = feature_cache(device)
    before_features, before_targets = features.clone(), targets.clone()
    run = dict(name='synthetic-head', seed=29, family='oracle_random/r32/M1-lin')
    random_before = torch.random.get_rng_state().clone()
    first, first_record = runner.fit_head(features, targets, run, training_design(), steps=2)
    second, second_record = runner.fit_head(features, targets, dict(run, name='same-identity-new-name'), training_design(), steps=2)
    assert torch.equal(torch.random.get_rng_state(), random_before)
    for name, parameter in first.state_dict().items():
        torch.testing.assert_close(parameter, second.state_dict()[name], rtol=0, atol=0)
    assert first_record['logs'] == second_record['logs']
    assert first_record['steps'] == 2 and [row['step'] for row in first_record['logs']] == [1, 2]
    assert first_record['init_seed'] == derive_seed(29, 'p4s-l', run['family'], 'head_init')
    assert first_record['sampler_seed'] == derive_seed(29, 'p4s-l', run['family'], 'episode_sampler')
    assert first_record['parameters'] == 3393
    assert not first.training and features.grad is None and targets.grad is None
    assert torch.equal(features, before_features) and torch.equal(targets, before_targets)


def test_fit_head_samples_256_whole_episodes_and_optimizes_only_the_head(monkeypatch):
    monkeypatch.setattr(runner.torch.cuda, 'synchronize', lambda: None)
    features, targets = feature_cache()
    # This reader parameter produces the cached states but must be disconnected
    # from the subsequent head optimizer, even if it was originally trainable.
    reader = nn.Linear(32, 32)
    with torch.no_grad():
        features[..., :32] = reader(features[..., :32])
    reader_before = {name: value.detach().clone() for name, value in reader.state_dict().items()}
    head_class, adam_class = runner.HaltingHead, runner.torch.optim.Adam
    randint, bce = runner.torch.randint, runner.F.binary_cross_entropy_with_logits
    captured = dict(indices=[], inputs=[], labels=[], optimizers=[])

    class ObservedHead(head_class):
        def __init__(self, seed):
            super().__init__(seed=seed)
            captured['init_seed'] = seed
            captured['initial_head'] = {name: value.detach().clone() for name, value in self.state_dict().items()}

        def logits(self, inputs):
            captured['inputs'].append(inputs.detach().clone())
            return super().logits(inputs)

    def observed_randint(high, size, **kwargs):
        assert size == (256,) and high == len(features)
        captured['generator_seed'] = kwargs['generator'].initial_seed()
        indices = randint(high, size, **kwargs)
        captured['indices'].append(indices.detach().clone())
        return indices

    def observed_bce(logits, labels, **kwargs):
        captured['labels'].append(labels.detach().clone())
        assert logits.shape == labels.shape == (256 * 8 * 16,)
        return bce(logits, labels, **kwargs)

    def observed_adam(parameters, **kwargs):
        parameters = list(parameters)
        captured['optimizers'].append((parameters, kwargs))
        return adam_class(parameters, **kwargs)

    monkeypatch.setattr(runner, 'HaltingHead', ObservedHead)
    monkeypatch.setattr(runner.torch, 'randint', observed_randint)
    monkeypatch.setattr(runner.F, 'binary_cross_entropy_with_logits', observed_bce)
    monkeypatch.setattr(runner.torch.optim, 'Adam', observed_adam)
    run = dict(name='whole-episode-cache', seed=17, family='joint_fact/r32/M1')
    head, record = runner.fit_head(features, targets, run, training_design(), steps=2)
    expected_generator = torch.Generator().manual_seed(derive_seed(17, 'p4s-l', run['family'], 'episode_sampler'))
    for indices, inputs, labels in zip(captured['indices'], captured['inputs'], captured['labels']):
        assert torch.equal(indices, randint(len(features), (256,), generator=expected_generator))
        assert inputs.shape == (256 * 8 * 16, 51)
        assert torch.equal(inputs.reshape(256, 8, 16, 51), features[indices])
        assert torch.equal(labels.reshape(256, 8, 16), targets[indices].float())
    assert len(captured['indices']) == 2 and len(captured['optimizers']) == 1
    assert captured['init_seed'] == derive_seed(17, 'p4s-l', run['family'], 'head_init')
    parameters, kwargs = captured['optimizers'][0]
    assert kwargs == {'lr': .001}
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in head.parameters()}
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters())
    assert any(not torch.equal(value, captured['initial_head'][name]) for name, value in head.state_dict().items())
    for name, value in reader.state_dict().items():
        assert torch.equal(value, reader_before[name])
    assert all(parameter.grad is None for parameter in reader.parameters())
    assert not features.requires_grad and features.grad is None and record['steps'] == 2


@pytest.mark.parametrize('variant', ['M1', 'M1-lin'])
def test_gpu_collect_matches_frozen_candidate_decoding_cache_targets_and_execution_counts(variant):
    if not torch.cuda.is_available():
        pytest.skip('Frozen-reader collection requires a compute GPU')
    from looplora.sym.config import resolve_config
    from looplora.sym.model import SymbolicModel
    from looplora.sym.pilot import base_config

    config = base_config('p4l-synthetic-reader', 'oracle_random', variant, 32, 12, .003,
                         't2_chain_end', [8, 4])
    config['device'] = 'cuda:0'
    model = SymbolicModel(resolve_config(config)).eval()
    with torch.no_grad():
        generator = torch.Generator(device=model.device).manual_seed(1729)
        model.adapter.B.normal_(std=.15, generator=generator)
    model.requires_grad_(False)
    before = runner.state_hash(model)
    design = dict(hops=list(range(1, 9)))
    (features, targets), timing = runner.collect(model, design, 'train', 31337, 4, training=True)
    assert features.shape == (4, 8, 16, 51) and targets.shape == (4, 8, 16)
    assert features.device == targets.device == model.device and targets.dtype == torch.bool
    assert not features.requires_grad and not features.is_inference() and not targets.requires_grad
    assert timing['queries'] == 32 and timing['episodes'] == 4 and timing['generator_seed'] == 31337
    spec = runner.make_task_spec(config, evaluation=True)
    original = runner.stratify_queries(runner.generate(spec, 'train', 0, 4, 31337), spec, design['hops'], 31337)
    prepared = model.prepare(original)
    expected = []
    for depth in range(1, 17):
        output = model.read(prepared, depth)['logits']
        prediction = output.masked_fill(~prepared.candidate_mask[:, None], -torch.inf).argmax(-1)
        expected.append(prediction == torch.as_tensor(original.answer, device=model.device))
    assert torch.equal(targets, torch.stack(expected, dim=-1))
    head, _ = runner.fit_head(features, targets, dict(name='tiny-gpu-head', seed=17, family=f'oracle_random/r32/{variant}'), training_design(), steps=2)
    assert runner.state_hash(model) == before and all(parameter.grad is None for parameter in model.parameters())

    class ConstantHead(nn.Module):
        def __init__(self, probability):
            super().__init__()
            self.probability = probability

        def forward(self, inputs):
            return inputs.new_full(inputs.shape[:-1], self.probability)

    for probability, executed, reads in ((.8, 1, 2), (.2, 16, 17)):
        values, measured, first = runner.collect(model, design, 'val', 4242, 4,
                                                 head=ConstantHead(probability), theta=.5)
        assert values['predictions'].shape == (4, 8, 17)
        assert values['probabilities'].shape == (4, 8, 16)
        assert np.array_equal(values['hop'], np.broadcast_to(np.arange(1, 9), (4, 8)))
        assert bool(np.all(values['selected_steps'] == 1))
        assert bool(np.all(values['executed_steps'] == executed))
        assert bool(np.all(values['read_calls'] == reads))
        assert first.u.shape == (4, 32) and measured['queries'] == 32
        result = runner.learned_adaptive_read(model.adapter, first, ConstantHead(probability), .5)
        assert bool(torch.all(result.steps == 1)) and bool(torch.all(result.executed_updates == executed))
        assert bool(torch.all(result.read_calls == reads))
        assert runner.state_hash(model) == before
    # The actually trained head is also usable on detached evaluation features.
    values, _, _ = runner.collect(model, design, 'val', 4242, 4, head=head, theta=.5)
    assert np.isfinite(values['probabilities']).all() and runner.state_hash(model) == before


def _git(repository, *arguments):
    return subprocess.check_output(['git', '-C', str(repository), *arguments], text=True).strip()


def _commit(repository, message):
    _git(repository, 'add', '.')
    _git(repository, '-c', 'user.name=P4L Synthetic Test', '-c', 'user.email=p4l-test@example.invalid',
         'commit', '-qm', message)
    return _git(repository, 'rev-parse', 'HEAD')


@pytest.fixture
def locked_study(tmp_path):
    repository = tmp_path / 'repo'
    repository.mkdir()
    _git(repository, 'init', '-q')
    root = tmp_path / 'raw'
    root.mkdir()
    runs = []
    for writer in ('joint_fact', 't1_frozen_fact', 'oracle_random', 'joint'):
        for variant in ('M1', 'M1-lin'):
            for seed in (17, 29, 43):
                name = f'synthetic-{writer}-{variant}-{seed}'
                checkpoint = tmp_path / 'released' / name / 'checkpoint.pt'
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b'synthetic frozen reader bytes')
                runs.append(dict(name=name, writer=writer, r=32, variant=variant, seed=seed,
                                 family=f'{writer}/r32/{variant}', checkpoint=str(checkpoint),
                                 checkpoint_blake2b=runner.digest_checkpoint(checkpoint)))
    design = dict(runs=runs, hops=list(range(1, 9)), max_steps=16, r=32,
                  theta_grid=[round(i / 20, 2) for i in range(1, 20)],
                  validation=dict(split='val', generator_seed=4242, episodes=2000),
                  confirmation=dict(split='test', generator_seed=999002, episodes=2000),
                  **training_design())
    design_path = repository / 'reports/p4s-l/design.json'
    runner.save(design_path, design)
    design_commit = _commit(repository, 'Lock synthetic design before training')
    trained_runs = copy.deepcopy(runs)
    for run in trained_runs:
        head = root / 'heads' / f"{run['name']}.pt"
        head.parent.mkdir(parents=True, exist_ok=True)
        head.write_bytes(b'synthetic trained head bytes')
        run.update(head_checkpoint=str(head), head_blake2b=runner.digest_checkpoint(head))
    grid = [dict(family=family, theta=theta, validation_accuracy=.5, validation_mean_steps=2.,
                 validation_mean_executed_steps=16.)
            for family in sorted({run['family'] for run in runs}) for theta in design['theta_grid']]
    selection = dict(design_hash=stable_hash(design), grid=grid,
                     selected=[row for row in grid if row['theta'] == .05], locked_at='2026-10-27T00:00:00+00:00')
    theta_path = repository / 'reports/p4s-l/theta.json'
    runner.save(theta_path, selection)
    theta_commit = _commit(repository, 'Lock synthetic validation theta before confirmation')
    runner.save(root / 'theta.json', selection)
    trained = dict(status='complete', smoke=False, design=design, design_hash=stable_hash(design),
                   design_path=str(design_path), design_lock_commit=design_commit, runs=trained_runs)
    runner.save(root / 'train_manifest.json', trained)
    return dict(root=root, design=design, design_path=design_path, design_commit=design_commit,
                theta_path=theta_path, theta_commit=theta_commit, trained=trained)


@pytest.mark.parametrize('failure', ['running', 'smoke', 'missing_theta_commit', 'unknown_theta_commit',
                                      'theta_not_committed', 'design_not_committed', 'design_drift',
                                      'reader_seal', 'head_seal'])
def test_confirm_rejects_unfinished_training_or_unlocked_inputs_before_model_or_confirmation(locked_study, monkeypatch, failure):
    study = locked_study
    trained = copy.deepcopy(study['trained'])
    theta_commit = study['theta_commit']
    design = copy.deepcopy(study['design'])
    if failure == 'running':
        trained['status'] = 'running'
    elif failure == 'smoke':
        trained['smoke'] = True
    elif failure == 'missing_theta_commit':
        theta_commit = ''
    elif failure == 'unknown_theta_commit':
        theta_commit = '0' * 40
    elif failure == 'theta_not_committed':
        with study['theta_path'].open('a') as stream:
            stream.write('\n')
    elif failure == 'design_not_committed':
        with study['design_path'].open('a') as stream:
            stream.write('\n')
    elif failure == 'design_drift':
        design['training']['lr'] = .002
    else:
        changed = trained['runs'][0]['checkpoint' if failure == 'reader_seal' else 'head_checkpoint']
        Path(changed).write_bytes(b'changed after locking')
    runner.save(study['root'] / 'train_manifest.json', trained)
    monkeypatch.setattr(runner, 'load_model', lambda *args: pytest.fail('model opened before training/theta lock checks'))
    monkeypatch.setattr(runner, 'collect', lambda *args, **kwargs: pytest.fail('confirmation evaluated before training/theta lock checks'))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        runner.confirm(study['root'], design, study['theta_path'], theta_commit)
    assert not (study['root'] / 'manifest.json').exists()


@pytest.mark.parametrize('failure', ['missing_commit', 'unknown_commit', 'uncommitted_bytes', 'different_design'])
def test_training_requires_actual_committed_design_before_output_or_cache(locked_study, monkeypatch, failure):
    study = locked_study
    design, commit = copy.deepcopy(study['design']), study['design_commit']
    if failure == 'missing_commit':
        commit = ''
    elif failure == 'unknown_commit':
        commit = '0' * 40
    elif failure == 'uncommitted_bytes':
        with study['design_path'].open('a') as stream:
            stream.write('\n')
    else:
        design['training']['steps'] = 3
    root = study['root'] / 'new-head-training'
    monkeypatch.setattr(runner, 'load_model', lambda *args: pytest.fail('reader loaded before committed design validation'))
    monkeypatch.setattr(runner, 'collect', lambda *args, **kwargs: pytest.fail('features cached before design validation'))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        runner.train_all(root, design, commit, design_path=study['design_path'])
    assert not root.exists()
