"""WP-L6 pilot runner: the registered decision rule and the pack-or-separate sizing (stdlib only; no GPU needed)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_wp_l6
from run_wp_l6 import ARMS, SMOKE_STEPS, STEPS, THRESHOLD, arm_outcome, config, decide, diverged, hhmmss, size


def table(hop1_at_L4, others, depths=(1, 2, 3, 4), hops=range(1, 9), n=1000):
    return [dict(L=L, hop=hop, n=n, split="val", protocol="memory_only", condition="memory_only/original",
                 full_accuracy=hop1_at_L4 if (hop, L) == (1, 4) else others, accuracy=0., candidate_accuracy=.1,
                 chance_accuracy=.07) for L in depths for hop in hops]


def write_pilot(directory, arm, hop1=.5, others=0., edit=None, status="complete", **cells):
    cfg = config(arm)
    if edit is not None:
        edit(cfg)
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(json.dumps(dict(
        status=status, config=cfg, checkpoint_hash="c" * 32, writer_hash="w" * 32, job_id="1.opbs",
        validation=table(hop1, others, **cells))))
    return directory


def pilots(tmp_path, rot, shared, **kwargs):
    return {"oracle_rot": write_pilot(tmp_path / "rot", "oracle_rot", rot, **kwargs),
            "oracle_shared": write_pilot(tmp_path / "shared", "oracle_shared", shared, **kwargs)}


def test_registered_constants():
    assert ARMS == ("oracle_rot", "oracle_shared")
    assert STEPS == 5666 == 17000 // 3 and SMOKE_STEPS == 567 and THRESHOLD == .30
    assert config("oracle_rot")["writer"] == "oracle_rot" and config("oracle_shared")["writer"] == "oracle_shared"
    with pytest.raises(ValueError):
        config("oracle_random")


def test_import_is_login_safe():
    code = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import run_wp_l6; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("rot,shared,selected", [
    (.31, .60, "oracle_rot"),      # the fixed code wins whenever it passes
    (.30, .00, "oracle_rot"),      # the threshold is inclusive
    (.2999, .30, "oracle_shared"),
    (.20, .10, None),
])
def test_decision_rule(tmp_path, rot, shared, selected):
    result = decide(pilots(tmp_path, rot, shared), tmp_path / "decision.json")
    assert result["selected"] == selected and result["status"] == ("selected" if selected else "stop")
    assert result["arms"]["oracle_rot"]["passed"] is (rot >= THRESHOLD)
    assert result["arms"]["oracle_shared"]["passed"] is (shared >= THRESHOLD)
    assert json.loads((tmp_path / "decision.json").read_text()) == result


def test_selection_reads_hop_1_at_L4_only(tmp_path):
    # O-rot is perfect everywhere except the registered cell; O-shared is at chance except the registered cell.
    dirs = {"oracle_rot": write_pilot(tmp_path / "rot", "oracle_rot", hop1=.10, others=.99),
            "oracle_shared": write_pilot(tmp_path / "shared", "oracle_shared", hop1=.35, others=0.)}
    result = decide(dirs)
    assert result["selected"] == "oracle_shared"
    assert result["arms"]["oracle_rot"]["hop1_L4_full_em"] == .10
    assert result["arms"]["oracle_shared"]["hop1_L4_full_em"] == .35


def test_diverged_arm_fails_and_the_other_arm_still_decides(tmp_path):
    dirs = {"oracle_rot": tmp_path / "rot", "oracle_shared": write_pilot(tmp_path / "shared", "oracle_shared", .4)}
    dirs["oracle_rot"].mkdir()
    (dirs["oracle_rot"] / "failure.json").write_text(json.dumps(
        dict(error="FloatingPointError: non-finite loss at step 1200")))
    result = decide(dirs)
    assert result["selected"] == "oracle_shared" and result["arms"]["oracle_rot"]["passed"] is False
    assert "FloatingPointError" in result["arms"]["oracle_rot"]["failure"]


def test_error_classification():
    assert diverged("FloatingPointError: non-finite loss at step 7")
    assert diverged("FloatingPointError: non-finite answer logits during LLM evaluation")
    assert diverged("RuntimeError: The total norm of order 2.0 for gradients from `parameters` is non-finite, "
                    "so it cannot be clipped.")
    assert not diverged("RuntimeError: CUDA error: an illegal memory access was encountered")
    assert not diverged("OutOfMemoryError: CUDA out of memory.")
    assert not diverged("FileNotFoundError: [Errno 2] No such file or directory")


@pytest.mark.parametrize("error", ["OutOfMemoryError: CUDA out of memory.", "KeyboardInterrupt: ",
                                   "RuntimeError: CUDA error: an illegal memory access was encountered"])
def test_infrastructure_failures_are_not_outcomes(tmp_path, error):
    (tmp_path / "rot").mkdir()
    (tmp_path / "rot" / "failure.json").write_text(json.dumps(dict(error=error)))
    with pytest.raises(ValueError, match="rerun the unchanged config"):
        arm_outcome(tmp_path / "rot", "oracle_rot")


@pytest.mark.parametrize("edit", [
    lambda cfg: cfg["train"].update(steps=1000), lambda cfg: cfg["train"].update(lr=.001),
    lambda cfg: cfg.update(seed=18), lambda cfg: cfg.update(oracle_seed=18), lambda cfg: cfg.update(smoke=True),
    lambda cfg: cfg.update(variant="B8"), lambda cfg: cfg["model"].update(r=128),
    lambda cfg: cfg["train"].update(depths=[1]), lambda cfg: cfg["eval"].update(seed=1),
    lambda cfg: cfg["eval"].update(episodes=500), lambda cfg: cfg["eval"].update(splits=["val", "test"]),
    lambda cfg: cfg["eval"].update(allow_confirmation=True), lambda cfg: cfg.update(writer="oracle_shared"),
])
def test_runs_that_are_not_the_registered_pilot_are_rejected(tmp_path, edit):
    directory = write_pilot(tmp_path / "rot", "oracle_rot", .9, edit=edit)
    with pytest.raises(ValueError, match="registered pilot configuration"):
        arm_outcome(directory, "oracle_rot")


def test_unfinished_run_and_incomplete_cell_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="registered pilot configuration"):
        arm_outcome(write_pilot(tmp_path / "a", "oracle_rot", .9, status="running"), "oracle_rot")
    with pytest.raises(ValueError, match="cell"):
        arm_outcome(write_pilot(tmp_path / "b", "oracle_rot", .9, depths=(1, 2, 3)), "oracle_rot")
    with pytest.raises(ValueError, match="cell"):
        arm_outcome(write_pilot(tmp_path / "c", "oracle_rot", .9, n=999), "oracle_rot")


def test_walltime_rounds_up_to_five_minutes():
    assert [hhmmss(s) for s in (1, 300, 301, 3600, 3601)] == [
        "00:05:00", "00:05:00", "00:10:00", "01:00:00", "01:05:00"]


def smoke(directory, arm, step, validation=60.):
    cfg = config(arm)
    cfg["smoke"], cfg["train"]["steps"] = True, SMOKE_STEPS
    directory.mkdir(parents=True)
    (directory / "record.json").write_text(json.dumps(dict(
        status="complete", config=cfg, steady_step_seconds=step, train_seconds=SMOKE_STEPS * step + 8.,
        validation_seconds=validation, peak_memory_bytes=7_200_000_000)))


def test_sizing_packs_only_when_both_arms_fit_one_hour_with_the_margin(tmp_path):
    for name, step in (("fast", .185), ("slow", .200)):
        for arm in ARMS:
            smoke(tmp_path / name / arm, arm, step)
    fast, slow = size(tmp_path / "fast", tmp_path / "fast.json"), size(tmp_path / "slow")
    for result, step in ((fast, .185), (slow, .200)):
        for arm in ARMS:
            row = result["arms"][arm]
            assert row["expected_seconds"] == pytest.approx(STEPS * step + 8. + 60.)
            assert row["step_seconds"] == step and row["overhead_seconds"] == pytest.approx(8.)
    assert fast["recommendation"] == "packed" and fast["packed"]["fits_one_hour_with_margin"]
    assert fast["packed"]["walltime"] == "01:00:00"
    assert slow["recommendation"] == "separate" and not slow["packed"]["fits_one_hour_with_margin"]
    assert slow["arms"]["oracle_rot"]["walltime"] == "00:35:00"
    assert fast["expected_node_hours"] == pytest.approx(2 * (STEPS * .185 + 68.) / 3600)
    assert json.loads((tmp_path / "fast.json").read_text()) == fast


@pytest.mark.parametrize("damage", [
    lambda record: record.update(status="running"),
    lambda record: record["config"].update(smoke=False),
    lambda record: record["config"]["train"].update(steps=100),
])
def test_sizing_requires_complete_smokes_of_the_registered_length(tmp_path, damage):
    for arm in ARMS:
        smoke(tmp_path / arm, arm, .19)
    path = tmp_path / "oracle_rot/record.json"
    record = json.loads(path.read_text())
    damage(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="complete 567-step smoke"):
        size(tmp_path)


def test_command_line_decide_and_size(tmp_path):
    dirs = pilots(tmp_path, .45, .10)
    out = subprocess.run([sys.executable, str(SCRIPTS / "run_wp_l6.py"), "decide", "--oracle_rot",
                          str(dirs["oracle_rot"]), "--oracle_shared", str(dirs["oracle_shared"]),
                          "--out", str(tmp_path / "decision.json")], check=True, capture_output=True, text=True)
    assert json.loads(out.stdout)["selected"] == "oracle_rot"
    assert json.loads((tmp_path / "decision.json").read_text())["selected"] == "oracle_rot"
    for arm in ARMS:
        smoke(tmp_path / "smoke" / arm, arm, .18)
    out = subprocess.run([sys.executable, str(SCRIPTS / "run_wp_l6.py"), "size", "--smoke", str(tmp_path / "smoke")],
                         check=True, capture_output=True, text=True)
    assert json.loads(out.stdout)["recommendation"] == "packed"
    assert run_wp_l6.PACK_LIMIT == 3600


def test_runbook_runs_the_file_directly_on_the_login_node(tmp_path):
    # docs/reports/WP-L6.md calls `$W/scripts/run_wp_l6.py size|decide` as a command: the file must carry its
    # shebang and the executable bit (git mode 100755), which the sys.executable calls above cannot show.
    script = SCRIPTS / "run_wp_l6.py"
    assert script.read_text().startswith("#!/usr/bin/env python3\n") and os.access(script, os.X_OK)
    dirs = pilots(tmp_path, .45, .10)
    out = subprocess.run([str(script), "decide", "--oracle_rot", str(dirs["oracle_rot"]),
                          "--oracle_shared", str(dirs["oracle_shared"])], check=True, capture_output=True, text=True)
    assert json.loads(out.stdout)["selected"] == "oracle_rot"
