"""Behavior regressions for the S1B cross-review; synthetic data, compute only."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from looplora.llm import s1b_analysis as stats, s1b_data as data, s1b_rules as rules
from looplora.llm import s1b_model as model, s1b_run as run, s1b_artifacts as artifacts, s1b_preflight as preflight
from looplora.llm.s1b_train import fit
from test_s1b_artifacts import bundle
from test_s1b_model import factory, backbone, tokenizer, raw, evaluation
from test_s1b_data_analysis import paired_fixture


def test_review_zero_boundary_keeps_storage_null_and_selects_matrix():
    """RVC-01: the registered 1/120 percentile has two exactly-zero bracketing sums."""
    n = 2000
    vector = np.roll(np.r_[np.ones(1000, dtype=np.int64), np.zeros(103, dtype=np.int64),
                           -np.ones(897, dtype=np.int64)], 3)
    main = {(arm, seed): dict(correct=np.zeros((n, 8), dtype=np.int64),
                             nll=np.ones((n, 8)), episode_id=np.arange(n, dtype=np.int64), hops=np.arange(1, 9))
            for arm in ("M1", "M4", "S1", "S4") for seed in data.RUN_SEEDS}
    for sign, arm in ((1, "S4"), (-1, "M4")):
        for j, i in enumerate(np.flatnonzero(vector == sign)):
            main[arm, data.RUN_SEEDS[j % 3]]["correct"][i, 1] = 1
    sweeps = {(arm, seed, depth): dict(main[arm, seed], correct=(main[arm, seed]["correct"].copy()
              if depth >= 4 else np.zeros((n, 8), dtype=np.int64)))
              for arm in ("M4", "S4") for seed in data.RUN_SEEDS for depth in (1, 2, 4, 8)}
    result = stats.paired(main, sweeps, "advance")
    storage = result["contrasts"]["S4-M4"]
    assert storage["interval_pp"][0] == 0
    assert storage["tests"]["lower_positive"] is False
    assert storage["passed"] is False
    assert result["continuation"]["M"]["qualified"] and result["continuation"]["S"]["qualified"]
    assert result["s2_primary"] == "matrix"


def test_review_percentile_interpolation_and_sign_are_exact():
    assert stats.exact_percentile(np.array([-1, 1]), Fraction(1, 2)) == 0
    assert stats.exact_percentile(np.array([-1, 1]), Fraction(2, 3)) == Fraction(1, 3)


@pytest.mark.parametrize("passing", [True, False])
def test_review_confirmation_tables_present_for_positive_and_null_results(passing):
    main, sweep = paired_fixture()
    if not passing:
        for sample in [*main.values(), *sweep.values()]:
            sample["correct"][:] = 0
    for j, sample in enumerate(main.values()):
        sample["nll"][:] = j + .25
    result = stats.paired(main, sweep, "advance")
    table = result["descriptive"]
    assert "descriptive" in table["label"]
    assert table["episodes"] == 100 and len(table["arms"]) == 12 and len(table["depth_sweeps"]) == 24
    assert table["arms"]["M1/17"]["per_hop"]["1"] == dict(n=100, correct=0, em=0., nll=.25)
    assert set(table["contrasts"]) == {"S1-M1", "Memory_x_Depth"}
    assert table["contrasts"]["S1-M1"]["delta_pp"] == 0
    assert result["contrasts"]["M4-M1"]["passed"] is passing
    assert len(table["depth_sweeps"]["M4/17@8"]["per_hop"]) == 8
    report = stats.confirmation_markdown(result)
    assert "Descriptive arm/seed/hop counts, EM and NLL" in report
    assert "M4/17@8" in report and "S1-M1" in report and "Memory_x_Depth" in report
    assert result["verdict"] in report


@pytest.mark.parametrize("defect", ["untriggered", "missing", "decision", "ablation"])
def test_review_branch_records_require_trigger_and_one_release(defect):
    digest = "c" * 32
    decision = dict(stage="S1A", verdict="NOT ADVANCED", release_blake2b_128=digest)
    dev = dict(stage="S1A-DEV", outcome="ablation", release_blake2b_128=digest)
    abl = dict(stage="S1A-ABL", flag=True, release_blake2b_128=digest)
    if defect == "untriggered":
        dev["outcome"] = "no_trigger"
    elif defect == "missing":
        dev.pop("release_blake2b_128")
    elif defect == "decision":
        decision["release_blake2b_128"] = "d" * 32
    else:
        abl["release_blake2b_128"] = "d" * 32
    with pytest.raises(ValueError):
        rules.resolve_branch(decision, dev, abl)


@pytest.mark.parametrize("input_name", ["s1a", "stopped"])
def test_a1_rejects_changed_source_release_or_stopped_evidence(bundle, input_name):
    p = bundle.paths
    p[input_name].write_text(p[input_name].read_text() + " ")
    with pytest.raises(ValueError):
        artifacts.require_read_settings(p["cal"], "advance", p["lock"])


def test_a1_calibration_cli_removed_before_data_or_model_work(tmp_path):
    with pytest.raises(SystemExit):
        run.main(["--mode", "calibration", "--out", str(tmp_path / "cal")])
    assert not (tmp_path / "cal").exists()


def test_review_real_t2_batchstream_runs_prefetched_token_positions(factory, monkeypatch, tmp_path):
    """Exercise source=None/workers=2, not the test's prebuilt source iterator."""
    from looplora.llm import writer_fix
    original = writer_fix.token_fact_positions
    captures = []
    def positions(arrays, tokens, device="cpu"):
        assert tokens.support_ids.device.type == "cpu"
        found = original(arrays, tokens, device)
        captures.append(tuple(found.shape))
        return found
    monkeypatch.setattr(writer_fix, "token_fact_positions", positions)
    exp = factory()
    record = fit(exp, rules.read_config("M4"), arm="M4", seed=17, lr=1e-3, steps=2, out=tmp_path,
                 dev=evaluation(2, [1, 2, 3]), dev_steps=[2], support=raw(2),
                 parent_digest="f" * 32, science_hash="stream-test", workers=2)
    assert captures == [(32, 12, 3), (32, 12, 3)]
    assert record["all_finite"] and record["train_episodes"] == 64 and record["train_questions"] == 256
    assert stats.audit_checkpoints(record)


def test_review_retained_candidate_exports_bind_every_row_and_selected_file(factory, monkeypatch, tmp_path):
    exp = factory()
    actual = model.evaluate
    calls = []
    def improving(*args, **kwargs):
        value = actual(*args, **kwargs)
        value["samples"]["nll"].fill(12 - len(calls))
        value["summary"] = model.per_hop(value["samples"])
        calls.append(True)
        return value
    monkeypatch.setattr(model, "evaluate", improving)
    support = raw(2)
    record = fit(exp, rules.read_config("M4"), arm="M4", seed=17, lr=1e-3, steps=2, out=tmp_path,
                 dev=evaluation(2, [1, 2, 3]), dev_steps=[1, 2], support=support,
                 parent_digest="f" * 32, science_hash="retention-test", source=iter([(support, None)] * 2))
    assert record["selected_step"] == 2 and stats.audit_checkpoints(record)
    assert [row["checkpoint"]["step"] for row in record["dev"]] == [0, 1, 2]
    assert all((tmp_path / row["checkpoint"]["file"]).is_file() for row in record["dev"])
    assert len({row["checkpoint"]["blake2b_128"] for row in record["dev"]}) == 3
    selected = tmp_path / "selected.pt"
    selected.write_bytes((tmp_path / record["dev"][1]["checkpoint"]["file"]).read_bytes())
    with pytest.raises(ValueError, match="winning"):
        stats.audit_checkpoints(record)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_review_saved_trajectory_matches_actual_final_token_rows(factory, monkeypatch, tmp_path, depth):
    exp = factory()
    original = model.trace_diagnostics
    captured = {}
    def capture(experiment, batch, arrays, spec, d):
        for step in range(1, d + 1):
            events = [t for t in experiment.adapter.trace if t["step"] == step and len(t["owner"])]
            rows = torch.cat([t["question_row"] for t in events])
            for name in ("q", "m", "z"):
                captured[name, step] = torch.cat([t[name] for t in events])[rows.argsort()].cpu().numpy()
        return original(experiment, batch, arrays, spec, d)
    monkeypatch.setattr(model, "trace_diagnostics", capture)
    arrays = evaluation(2)
    result = model.evaluate(exp, arrays, rules.read_config("S4"), depth, diagnostics=True)
    path = tmp_path / f"trajectory-L{depth}.npz"
    run.save_samples(path, result["samples"], evaluated_depth=depth, label="S4/17",
                     checkpoint_blake2b_128="f" * 32, trajectory_order=["episode", "hop", "step", "feature"])
    samples, provenance = run.load_samples(path)
    stats.validate_trajectories(samples, depth)
    np.testing.assert_array_equal(samples["episode_id"], arrays.episode_seed)
    assert provenance["evaluated_depth"] == depth
    for step in range(1, depth + 1):
        for name in ("q", "m", "z"):
            np.testing.assert_array_equal(samples[f"trajectory_{name}"][:, :, step - 1].reshape(16, -1), captured[name, step])


def test_review_preflight_owner_all_depths_and_capture_dimensions(factory):
    exp = factory()
    owner = preflight.owner_checks(exp)
    assert owner["passed"] and owner["depths"] == [1, 2, 4, 8]
    assert all(row["passed"] and row["questions"] == 8 for row in owner["evidence"])
    shapes = preflight.capture_shapes(exp, raw(2))
    assert shapes["passed"] and shapes["keys"] == [2, 12, 768] and shapes["values"] == [2, 12, 128]


def test_review_preflight_t1_identity_at_all_registered_depths(factory):
    from looplora.llm.slot_experiment import raw_episodes
    exp = factory("query_copy")
    t1 = raw_episodes(replace(data.old.SETS["S0", "smoke"], count=2))
    result = preflight.checks.matrix_identity(exp, t1, depths=(1, 2, 4))
    assert result["passed"] and result["depths"] == [1, 2, 4] and result["max_abs_difference"] == 0
