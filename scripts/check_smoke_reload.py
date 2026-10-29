"""Compute-only reload gate of one DPFP or LSQ smoke arm, run before the schedule release is committed.

`prepare_dpfp_release.py` and `prepare_lsq_release.py` run on the login node and read only the identity and timing fields of
the smoke records. This job, one per smoke arm, covers what they cannot, as WP-L4's A2 gate did: the arm's export reloads with
the study's own `load_inference` (the loader of the audit); the reloaded config is the smoke record's and the one the release
implies, `configuration(arm, smoke steps, E of the release, smoke=True)`; and a fresh 100-episode validation of the reloaded
writer predicts exactly what the smoke's last evaluation predicted. Prints identity and booleans only, never a score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from looplora.infra.seeds import stable_hash
from looplora.llm import dpfp, dpfp_run, lsq, lsq_run, lsqt, lsqt_run, lsqt_release
from looplora.llm.dpfp_release import prereg_digest
from looplora.llm.read_fix_run import validate
from looplora.llm.run import source_metadata
from looplora.sym.stats import load_samples

STUDIES = {'DPFP': (dpfp, dpfp_run), 'DPFP-X': (dpfp, dpfp_run), 'LSQ': (lsq, lsq_run), 'LSQT': (lsqt, lsqt_run), 'LSQT-U': (lsqt, lsqt_run)}
PREREG_DIGESTS = {'DPFP-X': prereg_digest, 'LSQT': lsqt_release.prereg_digest, 'LSQT-U': lsqt_release.prereg_digest}
EPISODES = 100  # the smoke's evaluation size


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(config):
    """A config as the smoke record holds it (JSON): tuples become lists."""
    return json.loads(json.dumps(config))


def check(release_path, root, arm):
    release = json.loads(Path(release_path).read_text())
    study, E, steps = release['study'], release['E'], release['smoke']['steps']
    require(study in STUDIES, f'unknown study {study!r}')
    module, runner = STUDIES[study]
    rows = [row for row in release['smoke']['rows'] if row['arm'] == arm]
    require(len(rows) == 1 and arm in release['steps_by_arm'], f'{arm}: not an arm of this {study} release')
    folder = Path(root) / arm
    record = json.loads((folder / 'record.json').read_text())
    identity = ('job_id', 'git_commit', 'source_hash')
    require([record[key] for key in identity] == [rows[0][key] for key in identity],
            f'{arm}: this smoke record is not the one the release was sized from')
    options = dict(eta=record['store_eta']) if study == 'LSQ' else {}  # the smoke's own ridge strength (RG arms only)
    expected = normalized(module.configuration(arm, steps, E=E, smoke=True, **options))
    require(record['config'] == expected, f'{arm}: the smoke ran another config than the code builds for E{E}, {steps} steps')
    if study in PREREG_DIGESTS:  # the science config the runner will build at the released steps must be the frozen one
        science = stable_hash(module.configuration(arm, release['steps_by_arm'][arm], E=E))
        require(science == release['config_hashes'][arm], f'{arm}: the release holds another science config hash')
        require(release['prereg_blake2b_128'] == PREREG_DIGESTS[study](), f'{arm}: the registration differs from the release')
    experiment = runner.load_inference(folder / 'writer.pt')
    try:
        require(normalized(experiment.config) == expected, f'{arm}: the reloaded export has another config')
        require(experiment.value_head is None, f'{arm}: the reloaded export holds a training-only value head')
        require(all(bool(torch.isfinite(value).all()) for value in experiment.adapter.state_dict().values()
                    if value.is_floating_point()), f'{arm}: the reloaded export has non-finite parameters')
        reloaded_folder = Path(root) / f'export-check-{arm}'
        validate(experiment, reloaded_folder, EPISODES)
    finally:
        experiment.close()
    reloaded = load_samples(reloaded_folder / 'samples.npz')
    saved = load_samples(folder / f'val-{steps}' / 'samples.npz')
    require(all(np.array_equal(reloaded[key], saved[key]) for key in ('episode_id', 'answer')),
            f'{arm}: the reloaded validation is not on the smoke\'s episodes')
    mismatches = int((reloaded['prediction'] != saved['prediction']).sum())
    require(mismatches == 0, f'{arm}: the reloaded writer differs from the smoke in {mismatches} of {EPISODES} predictions')
    result = dict(study=study, arm=arm, E=E, steps=steps, config_equal=True, reload_identical=True,
                  prediction_mismatches=mismatches, science_commit=release['science_commit'],
                  checked_at_commit=source_metadata()['git_commit'], quality_scores_exposed=False)
    (reloaded_folder / 'reload-check.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True,
                        help='the release written by prepare_dpfp_release.py or prepare_lsq_release.py (not yet committed)')
    parser.add_argument('--root', type=Path, required=True,
                        help='the smoke root of that release: ROOT/ARM/ is read, ROOT/export-check-ARM/ is written')
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.release, args.root, args.arm)), flush=True)
