"""P1L-S pipeline behaviour (WP-L6 step 1c; docs/reports/WP-L6.md): what `--study p1l-s` / `study: "p1l-s"` add.

tests/test_p1ls_defaults.py proves that every P1-L default is unchanged. This module tests the new paths: the named
confirmation windows; check_training on 12 retrained + 3 reused runs (one gated condition, one window, the committed
WP-L6 decision); finalize's per-group oracle-code audit, writer labels, B6 and B7 text and the writer parameters; the
launcher, config writer, analysis, ledger and run profile on a stand-in repository; evaluate_p1l_main.main's window and
evaluate()'s episode range. Inputs come from tests/assets/p1l_matrix.py (files only, no model) except the tiny-model
tests, which need the GPU. Run only through the compute test workflow.
"""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests" / "assets")]
import analyze_p1l_main as analyze
import evaluate_p1l_main as evaluator
import finalize_p1l_main as finalizer
import launch_p1l_main as launcher
import looplora.llm.evaluate as llm_evaluate
import make_p1ls_configs as maker
import p1l_job_ledger as ledger
import p1l_matrix as matrix
import run_p1l_main as run
from looplora.infra.seeds import derive_seed
from looplora.llm.config import CONFIRMATION_WINDOWS, DEFAULTS, recursive_merge, resolve_config, split_window
from looplora.llm.inject import load_tokenizer
from looplora.llm.p1lc import REGISTERED_RANGE
from looplora.llm.train import LLMExperiment
from p1ls_defaults_probe import SOURCE, STEP, FakeSubprocess

read, edited = matrix.read_json, matrix.edited
SEEDS = matrix.SEEDS
RETRAINED = [f"{label}-s{seed}" for label in matrix.RETRAINED for seed in SEEDS]
REUSED = [f"B3-L4-s{seed}" for seed in SEEDS]
WRITERS = ("oracle_rot", "oracle_shared")
OTHER = dict(zip(WRITERS, reversed(WRITERS)))
STUDY = ("--study", "p1l-s")


# ---- fixtures -----------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def p1l(tmp_path_factory):
    """P1-L's r = 32 and r = 128 matrices with their confirmation artifacts; P1L-S reuses the B3-L4 runs from here."""
    return matrix.build_p1l(tmp_path_factory.mktemp("p1l"))


def build(factory, name, p1l, writer, **kwargs):
    """P1L-S under `writer`, plus a stand-in repository root holding the committed WP-L6 decision and the templates."""
    root = factory.mktemp(name)
    repository = root / "repository"
    shutil.copytree(ROOT / "configs/p1l/main", repository / "configs/p1l/main")
    return SimpleNamespace(writer=writer, root=root, repository=repository,
                           **matrix.build_p1ls(root, p1l, writer, repository, **kwargs))


@pytest.fixture(scope="module")
def matrices(p1l, tmp_path_factory):
    """The P1L-S matrix of an oracle code, each built once: matrices("oracle_rot")."""
    made = {}

    def get(writer):
        if writer not in made:
            made[writer] = build(tmp_path_factory, writer, p1l, writer)
        return made[writer]

    return get


@pytest.fixture(scope="module", params=WRITERS)
def built(request, matrices):
    return matrices(request.param)


@pytest.fixture
def study(built, monkeypatch):
    """`built`, with evaluate_p1l_main reading the committed decision and the templates from the stand-in root."""
    monkeypatch.setattr(evaluator, "ROOT", built.repository)
    return built


@pytest.fixture
def rot(matrices, monkeypatch):
    """The oracle_rot matrix alone, for the rules that only a fixed code has."""
    built = matrices("oracle_rot")
    monkeypatch.setattr(evaluator, "ROOT", built.repository)
    return built


def finalize(manifest, evaluations, authorization, out):
    with contextlib.redirect_stdout(io.StringIO()):
        finalizer.finalize(SimpleNamespace(manifest=Path(manifest), evaluations=Path(evaluations),
                                           authorization=Path(authorization), out=Path(out)))
    return Path(out)


def finalize_study(study, out):
    return finalize(study.manifest, study.evaluations, study.authorization, out)


@pytest.fixture(scope="module")
def finalized(built, tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluator, "ROOT", built.repository)
        return finalize_study(built, tmp_path_factory.mktemp("finalized"))


@pytest.fixture(scope="module")
def unused_memory(p1l, tmp_path_factory):
    """oracle_rot with a B8 that scores below the no-memory loop: the B7 check fails, B6 still holds."""
    unused = build(tmp_path_factory, "unused", p1l, "oracle_rot", overrides={"B8": .05})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluator, "ROOT", unused.repository)
        return finalize_study(unused, tmp_path_factory.mktemp("finalized-unused"))


# ---- confirmation windows -----------------------------------------------------------------------------------------

def test_confirmation_windows_are_named_registered_and_disjoint():
    assert CONFIRMATION_WINDOWS == {"p1l": 100000, "p1l-s": 110000}
    assert "confirmation" not in DEFAULTS["eval"]
    ranges = sorted([(start, start + 2000) for start in CONFIRMATION_WINDOWS.values()]
                    + [(REGISTERED_RANGE["start"], REGISTERED_RANGE["start"] + REGISTERED_RANGE["count"])])
    assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:])) and (110000, 112000) in ranges


@pytest.mark.parametrize("name", ["confirmation", "typed-replication"])
def test_the_evaluation_templates_name_no_window(name):
    evaluation = read(ROOT / "configs/p1l/main" / f"eval-{name}.json")["eval"]
    assert evaluation["splits"] == ["confirm"]
    assert not {"confirmation", "start", "episode_start"} & evaluation.keys()


def test_split_window_maps_only_the_confirm_split():
    assert split_window({}, "confirm") == split_window({"confirmation": "p1l"}, "confirm") == ("test", 100000)
    assert split_window({"confirmation": "p1l-s"}, "confirm") == ("test", 110000)
    for split in ("val", "test", "test_unseen"):
        assert split_window({"confirmation": "p1l-s"}, split) == split_window({}, split) == (split, 0)


def confirmation_config(**evaluation):
    template = read(ROOT / "configs/p1l/main/eval-confirmation.json")
    template["eval"].update(evaluation)
    return resolve_config(recursive_merge(run.config("M1-L4"), template))


def test_a_named_window_adds_one_key_and_changes_nothing_else():
    default, named = confirmation_config(), confirmation_config(confirmation="p1l-s")
    assert "confirmation" not in default["eval"] and named["eval"].pop("confirmation") == "p1l-s"
    assert named == default


@pytest.mark.parametrize("update,message", [
    (dict(confirmation="p1l-x"), "must be one of"),
    (dict(confirmation=110000), "must be one of"),
    (dict(confirmation=None), "must be one of"),
    (dict(confirmation="p1l-s", splits=["val"], allow_confirmation=False), "requires the confirm split"),
    (dict(start=110000), "unknown eval fields"),
    (dict(episode_start=110000), "unknown eval fields"),
    (dict(confirmation="p1l-s", start=110000), "unknown eval fields"),
])
def test_resolve_config_rejects_free_integers_unknown_names_and_a_window_without_confirmation(update, message):
    with pytest.raises(ValueError, match=message):
        confirmation_config(**update)


# ---- check_training: P1L-S ----------------------------------------------------------------------------------------

def check(study, *, manifest=None, authorization=None, **kwargs):
    return evaluator.check_training(read(study.manifest) if manifest is None else manifest,
                                    read(study.authorization) if authorization is None else authorization, **kwargs)


def damaged(value, edit):
    value = copy.deepcopy(value)
    edit(value)
    return value


def test_the_matrix_is_accepted_with_and_without_the_lock_check(study):
    assert check(study) is None and check(study, check_all=False) is None
    assert sorted(read(study.manifest)["runs"]) == sorted(RETRAINED + REUSED)


def test_the_launchers_reused_entries_are_the_ones_the_checker_verifies(p1l, study):
    assert launcher.reused_runs(Path(p1l["manifest32"]), ("B3-L4",)) == study.reused == read(study.manifest)["reused"]


AUTHORIZATION = {
    "no_study": (lambda a: a.pop("study"), "another study"),
    "p1l_study": (lambda a: a.update(study="p1l"), "another study"),
    "held": (lambda a: a.update(status="held"), "held until"),
    "no_decision": (lambda a: a.pop("orchestrator_decision"), "held until"),
    "two_gated": (lambda a: a.update(gated_conditions=2), "exactly one condition"),
    "boolean_count": (lambda a: a.update(gated_conditions=True), "exactly one condition"),
    "no_count": (lambda a: a.pop("gated_conditions"), "exactly one condition"),
    "no_condition": (lambda a: a.update(condition_manifests={}), "exactly one condition"),
    "p1l_condition": (lambda a: a.update(condition_manifests={"oracle-r32": "x"}), "exactly one condition"),
    "extra_condition": (lambda a: a["condition_manifests"].update({"oracle-r32": "x"}), "exactly one condition"),
    "no_window": (lambda a: a.pop("confirmation_window"), "p1l-s confirmation window"),
    "p1l_window": (lambda a: a.update(confirmation_window="p1l"), "p1l-s confirmation window"),
    "free_integer_window": (lambda a: a.update(confirmation_window=110000), "p1l-s confirmation window"),
    "no_decision_entry": (lambda a: a.pop("decision"), "committed WP-L6 decision"),
    "wrong_decision_hash": (lambda a: a["decision"].update(hash="0" * 32), "committed WP-L6 decision"),
    "wrong_decision_path": (lambda a: a["decision"].update(path="reports/wp-l6/other.json"),
                            "committed WP-L6 decision"),
}


@pytest.mark.parametrize("case", AUTHORIZATION)
def test_the_authorization_names_the_study_one_condition_the_window_and_the_decision(study, case):
    edit, message = AUTHORIZATION[case]
    with pytest.raises(ValueError, match=message):
        check(study, authorization=damaged(read(study.authorization), edit))


MANIFEST = {
    "missing_run": (lambda m: m["runs"].pop("B8-s17"), "retrained runs"),
    "extra_B0": (lambda m: m["runs"].update({"B0-s17": "/nowhere"}), "retrained runs"),
    "extra_B1": (lambda m: m["runs"].update({"B1-s17": "/nowhere"}), "retrained runs"),
    "extra_seed": (lambda m: m["runs"].update({"B8-s99": "/nowhere"}), "retrained runs"),
    "p1l_condition": (lambda m: m.update(condition="oracle-r32"), "retrained runs"),
    "rank_128": (lambda m: m.update(rank=128), "retrained runs"),
    "no_rank": (lambda m: m.pop("rank"), "retrained runs"),
    "unknown_study": (lambda m: m.update(study="p1l-x"), "unknown study"),
    "other_code": (lambda m: m.update(writer=OTHER[m["writer"]]), "not the one WP-L6 selected"),
    "no_code": (lambda m: m.pop("writer"), "not the one WP-L6 selected"),
    "reused_missing": (lambda m: m["reused"].pop("B3-L4-s29"), "reused runs are not exactly"),
    "reused_extra": (lambda m: m["reused"].update({"B8-s17": m["reused"]["B3-L4-s17"]}), "reused runs are not exactly"),
    "reused_manifest_hash": (lambda m: m["reused"]["B3-L4-s17"].update(manifest_hash="0" * 32),
                             "reused run does not match"),
    "reused_checkpoint_hash": (lambda m: m["reused"]["B3-L4-s29"].update(checkpoint_hash="0" * 32),
                               "reused run does not match"),
    "reused_source": (lambda m: m["reused"]["B3-L4-s43"].update(source="/elsewhere"), "reused run does not match"),
}


@pytest.mark.parametrize("case", MANIFEST)
def test_the_manifest_lists_exactly_the_registered_runs_code_and_reused_entries(study, case):
    edit, message = MANIFEST[case]
    with edited(study.manifest, edit):  # written to the locked file, so the lock comparison passes and the rule fires
        with pytest.raises(ValueError, match=message):
            check(study, manifest=read(study.manifest))


def test_a_manifest_that_differs_from_the_locked_one_is_refused_only_with_the_lock_check(study):
    changed = damaged(read(study.manifest), lambda m: m.update(lock_commit="deadbeef"))
    with pytest.raises(ValueError, match="not the one the authorization locked"):
        check(study, manifest=changed)
    assert check(study, manifest=changed, check_all=False) is None


@pytest.mark.parametrize("selected", ["stop", "other_code"])
def test_the_decision_must_select_the_manifests_code(study, selected):
    edit = ((lambda d: d.update(status="stop", selected=None)) if selected == "stop"
            else (lambda d: d.update(selected=OTHER[study.writer])))
    with edited(study.decision, edit):
        authorization = read(study.authorization)
        authorization["decision"]["hash"] = matrix.digest(study.decision)  # the authorization names the edited file
        with pytest.raises(ValueError, match="not the one WP-L6 selected"):
            check(study, authorization=authorization)
    with edited(study.decision, lambda d: d.update(note="changed after the authorization")):
        with pytest.raises(ValueError, match="committed WP-L6 decision"):
            check(study)


RECORDS = {
    "new_other_code": ("B8-s17", lambda r: r["config"].update(writer="oracle_random"), "registered oracle code"),
    "new_other_oracle_seed": ("M1-L4-s29", lambda r: r["config"].update(oracle_seed=29), "registered oracle code"),
    "reused_under_a_new_code": ("B3-L4-s17", lambda r: r["config"].update(writer="oracle_rot"),
                                "registered oracle code"),
    "new_steps": ("B4-s43", lambda r: r["config"]["train"].update(steps=100), "S/lr lock"),
    "new_lr": ("B5-L4-s17", lambda r: r["config"]["train"].update(lr=.5), "S/lr lock"),
    "reused_steps": ("B3-L4-s29", lambda r: r["config"]["train"].update(steps=100), "S/lr lock"),
    "reused_lr": ("B3-L4-s43", lambda r: r["config"]["train"].update(lr=.5), "S/lr lock"),
    "incomplete": ("B8-s29", lambda r: r.update(status="running"), "incomplete or changed"),
    "changed_checkpoint": ("B8-s43", lambda r: r.update(checkpoint_hash="0" * 32), "incomplete or changed"),
    "smoke": ("M1-L4-s17", lambda r: r["config"].update(smoke=True), "smoke settings"),
    "confirmation_access": ("B4-s17", lambda r: r["config"]["eval"].update(allow_confirmation=True), "confirmation"),
    "test_split": ("B5-L4-s29", lambda r: r["config"]["eval"].update(splits=["test"]), "confirmation"),
    "rank": ("B8-s17", lambda r: r["config"]["model"].update(r=128), "rank differs"),
    "reused_rank": ("B3-L4-s17", lambda r: r["config"]["model"].update(r=128), "rank differs"),
}


@pytest.mark.parametrize("case", RECORDS)
def test_every_run_is_complete_unchanged_under_its_registered_code_and_locked_settings(study, case):
    name, edit, message = RECORDS[case]
    with edited(Path(study.runs[name]) / "record.json", edit):
        with pytest.raises(ValueError, match=message):
            check(study)


P1L_MANIFEST = {
    "condition": (lambda m: m.update(condition="oracle-r128"), "does not match P1-L's training manifest"),
    "rank": (lambda m: m.update(rank=128), "does not match P1-L's training manifest"),
    "study_key": (lambda m: m.update(study="p1l-s"), "does not match P1-L's training manifest"),
    "other_path": (lambda m: m["runs"].update({"B3-L4-s17": "/elsewhere"}), "does not match P1-L's training manifest"),
    "S": (lambda m: m.update(S=8500), "S of P1-L's reused runs"),
    "lr": (lambda m: m["lrs"].update({"B3-L4": .003}), "S/lr lock"),
}


@pytest.mark.parametrize("case", P1L_MANIFEST)
def test_reused_runs_must_be_the_ones_p1ls_r32_manifest_lists(p1l, study, case):
    edit, message = P1L_MANIFEST[case]
    with edited(p1l["manifest32"], edit):
        def retarget(manifest):  # the entries follow the edited file's hash, so only the rule under test can fire
            for entry in manifest["reused"].values():
                entry["manifest_hash"] = matrix.digest(p1l["manifest32"])

        with edited(study.manifest, retarget):
            with pytest.raises(ValueError, match=message):
                check(study, manifest=read(study.manifest))


def test_the_two_studies_do_not_accept_each_others_manifests_or_authorizations(p1l, study):
    manifest, authorization = read(p1l["manifest32"]), read(p1l["authorization"])
    assert evaluator.check_training(manifest, authorization) is None
    with pytest.raises(ValueError, match="another study"):
        evaluator.check_training(manifest, dict(authorization, study="p1l-s"))
    with pytest.raises(ValueError, match="another study"):
        check(study, authorization=authorization)
    with pytest.raises(ValueError, match="names no confirmation window"):
        evaluator.check_training(manifest, dict(authorization, confirmation_window="p1l-s"))


# ---- evaluate_p1l_main.main: the window comes from the authorization ----------------------------------------------

def run_evaluator(monkeypatch, tmp_path, manifest, authorization, name, writer):
    """main() with the model, checkpoint load and evaluate() replaced; returns the configs it evaluates and its output."""
    seen = []
    base = resolve_config({**run.config("B8", 17), "writer": writer})

    class Loaded:
        config = base

        def close(self):
            pass

    monkeypatch.setattr(evaluator, "require_compute", lambda: None)
    monkeypatch.setattr(evaluator, "load_model", lambda: None)
    monkeypatch.setattr(evaluator, "load_tokenizer", lambda: None)
    monkeypatch.setattr(evaluator, "load_checkpoint", lambda path, model, tokenizer: Loaded())
    monkeypatch.setattr(evaluator, "evaluate", lambda experiment, cfg, out: (seen.append(copy.deepcopy(cfg)), ({}, {}))[1])
    monkeypatch.setattr(evaluator, "swap_successors", lambda experiment, cfg, out: None)
    monkeypatch.setattr(evaluator, "inference_timing", lambda experiment, cfg, out: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate_p1l_main.py", "--manifest", str(manifest),
                                      "--authorization", str(authorization), "--out", str(out), "--runs", name])
    evaluator.main()
    return seen, out / name


def test_the_confirmation_configs_carry_the_authorizations_window(study, monkeypatch, tmp_path):
    seen, out = run_evaluator(monkeypatch, tmp_path, study.manifest, study.authorization, "B8-s17", study.writer)
    assert [cfg["task"]["name"] for cfg in seen] == ["t2_chain_end", "t2_typed"]
    for cfg in seen:
        assert cfg["eval"]["confirmation"] == "p1l-s" and cfg["eval"]["splits"] == ["confirm"]
        assert cfg["eval"]["allow_confirmation"] is True and (cfg["eval"]["episodes"], cfg["eval"]["seed"]) == (2000, 4242)
        assert split_window(cfg["eval"], "confirm") == ("test", 110000) and cfg["writer"] == study.writer
    assert [read(out / task / "config.json")["eval"]["confirmation"] for task in ("primary", "typed")] == ["p1l-s"] * 2
    assert read(out / "record.json")["authorization_hash"] == matrix.digest(study.authorization)


def test_a_p1l_authorization_leaves_the_configs_on_p1ls_registered_range(p1l, monkeypatch, tmp_path):
    seen, _ = run_evaluator(monkeypatch, tmp_path, p1l["manifest32"], p1l["authorization"], "B8-s17", "oracle_random")
    assert len(seen) == 2
    for cfg in seen:
        assert "confirmation" not in cfg["eval"] and split_window(cfg["eval"], "confirm") == ("test", 100000)


# ---- finalize: P1L-S ----------------------------------------------------------------------------------------------

def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def test_finalize_records_the_study_the_window_and_the_writers(built, finalized):
    writer = built.writer
    writers = {"B8": writer, "M1-L4": writer, "B4": writer, "B5-L4": writer, "B3-L4": "oracle_random"}
    endpoints = read(finalized / "endpoints.json")
    assert (endpoints["study"], endpoints["confirmation_window"], endpoints["writer"]) == ("p1l-s", "p1l-s", writer)
    assert endpoints["writer_by_reader"] == writers and endpoints["rank"] == 32
    assert endpoints["confidence"] == pytest.approx(.95) and endpoints["authorization"]["gated_conditions"] == 1
    assert endpoints["G1_L"] and endpoints["G2_L"] and endpoints["passes"] and endpoints["B6_equal_parameter"] is True
    assert endpoints["memory_use_passed"] is True and endpoints["memory_use"]["confidence"] == pytest.approx(.95)
    counts = table(finalized / "accuracy.csv")
    assert {row["reader"]: row["writer"] for row in counts} == writers and {row["rank"] for row in counts} == {"32"}
    assert {row["seed"] for row in counts} == {str(seed) for seed in SEEDS}


def test_finalize_counts_the_trained_table_as_a_writer_parameter_only_where_there_is_one(built, finalized):
    expected = matrix.SHARED_ROWS if built.writer == "oracle_shared" else 0
    rows = table(finalized / "inference.csv")
    assert {row["reader"]: int(row["writer_parameters"]) for row in rows} == {
        "B8": expected, "M1-L4": expected, "B4": expected, "B5-L4": expected, "B3-L4": 0}
    assert len(rows) == len(RETRAINED + REUSED)


def test_run_manifest_audits_the_retrained_and_the_reused_runs_as_two_groups(built, finalized):
    manifest = read(finalized / "run-manifest.json")
    assert (manifest["status"], manifest["study"], manifest["condition"]) == ("audited", "p1l-s", "p1l-s-r32")
    assert manifest["writer"] == built.writer and sorted(manifest["reused"]) == REUSED
    groups = manifest["groups"]
    assert set(groups) == {built.writer, "oracle_random"}
    assert sorted(groups["oracle_random"]["runs"]) == REUSED and groups["oracle_random"]["writer_hash"] == "oracle_random-code"
    assert sorted(groups[built.writer]["runs"]) == sorted(RETRAINED)
    if built.writer == "oracle_rot":
        assert groups["oracle_rot"]["writer_hash"] == "oracle_rot-code"
        return
    shared = groups["oracle_shared"]
    assert len(set(shared["trained_table_hashes"].values())) == len(RETRAINED)  # a table per run, not one code
    assert shared["writer_trainable_parameters"] == matrix.SHARED_ROWS
    assert set(shared["code_drift"]) == set(RETRAINED) and {"global_codes_hash", "rotations_hash"} <= shared.keys()
    memories = {read(Path(read(built.evaluations)["runs"][name]) / "primary/metrics.json")["memory_hashes"][0]["memory"]
                for name in RETRAINED}
    assert len(memories) == len(RETRAINED)  # the memories differ by design and finalize accepted them


def test_results_report_the_window_the_b6_flag_and_the_p1ls_b7_text(built, finalized):
    text = (finalized / "results.md").read_text()
    assert text.startswith(f"# P1L-S {built.writer} r32 confirmation")
    assert "test [110000, 112000)" in text and "B3-L4 is reused from P1-L (oracle_random)" in text
    assert "meets the G1-L criterion: **yes**" in text and "Memory-use check: **pass**" in text
    assert "docs/prereg/P1L-S.md" in text and "Rule 2" not in text and "Rule 3" not in text
    for name in ("accuracy.png", "accuracy.pdf", "M1-depth.png", "M1-depth.pdf", "memory-use.csv"):
        assert (finalized / name).exists()
    assert {row["confidence"] for row in table(finalized / "memory-use.csv") if row["seed"] == "pooled"} == {"0.95"}


def test_a_reader_that_does_not_use_the_memory_fails_b7_whatever_b6_says(unused_memory):
    endpoints = read(unused_memory / "endpoints.json")
    assert endpoints["memory_use_passed"] is False and endpoints["B6_equal_parameter"] is True
    assert endpoints["memory_use"]["contrasts"]["B8_vs_B3_hop1"]["ci_low"] < 0 and not endpoints["G1_L"]
    text = (unused_memory / "results.md").read_text()
    assert "Memory-use check: **fail**" in text and "did not learn to use the oracle memory at r = 32 with this code" in text
    assert "no multi-read claim at r = 32 follows from this study" in text and "B7 passes" not in text


def evaluation_dir(study, name):
    return Path(read(study.evaluations)["runs"][name])


WINDOW = {
    "primary_range_is_p1ls": ("B4-s29", "primary/metrics.json",
                              lambda m: m["episode_ranges"]["confirm"].update(start=100000)),
    "typed_range_is_p1ls": ("M1-L4-s17", "typed/metrics.json",
                            lambda m: m["episode_ranges"]["confirm"].update(start=100000)),
    "config_without_the_window": ("B8-s43", "primary/config.json", lambda c: c["eval"].pop("confirmation")),
    "config_with_p1ls_window": ("B3-L4-s17", "typed/config.json", lambda c: c["eval"].update(confirmation="p1l")),
}


@pytest.mark.parametrize("case", WINDOW)
def test_finalize_rejects_a_range_that_is_not_the_authorizations_window(study, tmp_path, case):
    name, relative, edit = WINDOW[case]
    with edited(evaluation_dir(study, name) / relative, edit):
        with pytest.raises(ValueError, match=f"range/protocol mismatch: {name}"):
            finalize_study(study, tmp_path / "out")


P1L_WINDOW = {
    "p1ls_range": ("primary/metrics.json", lambda m: m["episode_ranges"]["confirm"].update(start=110000)),
    "p1ls_window_in_the_config": ("typed/config.json", lambda c: c["eval"].update(confirmation="p1l-s")),
}


@pytest.mark.parametrize("case", P1L_WINDOW)
def test_finalize_rejects_the_p1ls_window_under_a_p1l_authorization(p1l, tmp_path, case):
    relative, edit = P1L_WINDOW[case]
    directory = Path(read(p1l["evaluations"])["runs"]["B8-s17"])
    with edited(directory / relative, edit):
        with pytest.raises(ValueError, match="range/protocol mismatch: B8-s17"):
            finalize(p1l["manifest32"], p1l["evaluations"], p1l["authorization"], tmp_path / "out")


@pytest.mark.parametrize("edit,message", [
    (lambda a: a.update(confirmation_window="p1l"), "p1l-s confirmation window"),
    (lambda a: a.update(study="p1l"), "another study"),
])
def test_finalize_runs_the_study_checks_before_it_opens_any_evaluation(study, tmp_path, edit, message):
    with edited(study.authorization, edit):
        with pytest.raises(ValueError, match=message):
            finalize_study(study, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_finalize_requires_an_evaluation_for_every_run_of_the_matrix(study, tmp_path):
    evaluations = read(study.evaluations)
    evaluations["runs"].pop("B3-L4-s43")
    path = matrix.write_json(tmp_path / "evaluations.json", evaluations)
    with pytest.raises(ValueError, match="complete training matrix"):
        finalize(study.manifest, path, study.authorization, tmp_path / "out")


def write_count(metrics):
    metrics["memory_hashes"][0]["write_count"] = "another-write-count"


def memory(metrics):
    metrics["memory_hashes"][1]["memory"] = "another-memory"


def writer_hash(record):
    record["writer_hash"] = "another-code"


GROUP = {
    "write_count_in_the_new_group": ("evaluation", "B4-s29", "primary/metrics.json", write_count, "write counts differ"),
    "write_count_in_the_reused_group": ("evaluation", "B3-L4-s17", "primary/metrics.json", write_count,
                                        "write counts differ"),
    "memory_in_the_reused_group": ("evaluation", "B3-L4-s29", "primary/metrics.json", memory,
                                   "oracle_random support snapshots or code hashes differ"),
    "code_hash_in_the_reused_group": ("training", "B3-L4-s43", "record.json", writer_hash,
                                      "oracle_random support snapshots or code hashes differ"),
}
# oracle_shared trains a table per run, so its memories and code hashes differ by design; only a fixed code has these.
FIXED_CODE = {
    "memory": ("evaluation", "B8-s43", "primary/metrics.json", memory,
               "oracle_rot support snapshots or code hashes differ"),
    "code_hash": ("training", "B5-L4-s17", "record.json", writer_hash,
                  "oracle_rot support snapshots or code hashes differ"),
}


def finalize_damaged(study, tmp_path, where, name, relative, edit, message):
    directory = evaluation_dir(study, name) if where == "evaluation" else Path(study.runs[name])
    with edited(directory / relative, edit):
        with pytest.raises(ValueError, match=message):
            finalize_study(study, tmp_path / "out")


@pytest.mark.parametrize("case", GROUP)
def test_finalize_audits_the_reused_group_and_the_write_counts_across_groups(study, tmp_path, case):
    finalize_damaged(study, tmp_path, *GROUP[case])


@pytest.mark.parametrize("case", FIXED_CODE)
def test_finalize_requires_one_snapshot_and_one_code_hash_per_fixed_code_group(rot, tmp_path, case):
    finalize_damaged(rot, tmp_path, *FIXED_CODE[case])


# ---- finalize helpers ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("difference,ci_low,per_seed,expected", [
    (.03, .001, (.04, .02, .03), True),
    (.0299, .001, (.04, .02, .03), False),   # below 3 pp
    (.05, .0, (.04, .02, .03), False),       # the interval must lie above zero
    (.05, .001, (.04, .0, .03), False),      # positive in every seed
    (.05, .001, (.04, -.01, .03), False),
])
def test_b6_follows_the_g1_criterion(difference, ci_low, per_seed, expected):
    row = dict(difference=difference, ci_low=ci_low, per_seed=[dict(difference=value) for value in per_seed])
    assert finalizer.meets_g1_criterion(row) is expected


def audited(name, writer, code="code", write="writes", memory="memory"):
    return dict(name=name, training=dict(config=dict(writer=writer), writer_hash=code), directory=None,
                snapshot={("confirm", "memory_only", "memory_only/original", 0): (memory, write)})


def group_rows(**changes):
    rows = [audited(name, "oracle_rot", "rot") for name in RETRAINED]
    rows += [audited(name, "oracle_random", "random", memory="memory-random") for name in REUSED]
    for index, row in changes.items():
        rows[int(index.lstrip("r"))] = row
    return rows


GROUP_MANIFEST = dict(writer="oracle_rot", runs={name: "-" for name in RETRAINED + REUSED},
                      reused={name: {} for name in REUSED})


def test_writer_groups_pass_when_each_group_is_one_code():
    groups = finalizer.audit_writer_groups(group_rows(), GROUP_MANIFEST)  # memories differ across groups, as they must
    assert groups == {"oracle_rot": dict(runs=RETRAINED, writer_hash="rot"),
                      "oracle_random": dict(runs=REUSED, writer_hash="random")}


@pytest.mark.parametrize("rows,message", [
    (group_rows(r0=audited(RETRAINED[0], "oracle_random")), "training writers do not match"),
    (group_rows(r14=audited(REUSED[-1], "oracle_rot")), "training writers do not match"),
    (group_rows(r4=audited(RETRAINED[4], "oracle_shared")), "training writers do not match"),
    (group_rows()[:-1], "training writers do not match"),
    (group_rows(r5=audited(RETRAINED[5], "oracle_rot", "rot", write="other")), "write counts differ"),
    (group_rows(r12=audited(REUSED[0], "oracle_random", "random", write="other")), "write counts differ"),
    (group_rows(r3=audited(RETRAINED[3], "oracle_rot", "other-code")), "oracle_rot support snapshots or code hashes"),
    (group_rows(r3=audited(RETRAINED[3], "oracle_rot", "rot", memory="other")),
     "oracle_rot support snapshots or code hashes"),
    (group_rows(r13=audited(REUSED[1], "oracle_random", "other-code", memory="memory-random")),
     "oracle_random support snapshots or code hashes"),
])
def test_writer_groups_reject_a_mixed_or_inconsistent_group(rows, message):
    with pytest.raises(ValueError, match=message):
        finalizer.audit_writer_groups(rows, GROUP_MANIFEST)


def shared_rows(tmp_path, edit=None, names=("B8-s17", "B8-s29", "B4-s43")):
    rows = []
    for name in names:
        state = matrix.shared_state(name)
        record = dict(writer_trainable_parameters=matrix.SHARED_ROWS)
        if edit:
            edit(name, state, record)
        directory = tmp_path / name
        matrix.make_run(directory, name.split("-")[0], int(name.rsplit("-s", 1)[1]), 32, "oracle_shared", 17000, .001,
                        state=state, extra=record)
        rows.append(dict(name=name, training=read(directory / "record.json"), snapshot={}, directory=directory))
    return rows


def test_shared_audit_accepts_trained_tables_and_records_one_hash_per_run(tmp_path):
    audit = finalizer.audit_shared_code(shared_rows(tmp_path))
    assert list(audit["trained_table_hashes"]) == ["B8-s17", "B8-s29", "B4-s43"]
    assert len(set(audit["trained_table_hashes"].values())) == 3 and audit["writer_trainable_parameters"] == matrix.SHARED_ROWS
    assert audit["code_drift"] == {name: None for name in audit["trained_table_hashes"]}


@pytest.mark.parametrize("field", ["global_codes", "rotations"])
@pytest.mark.parametrize("target", ["B8-s17", "B4-s43"])
def test_shared_audit_rejects_fixed_parts_that_differ_from_the_seeded_writer(tmp_path, field, target):
    def edit(name, state, record):
        if name == target:
            state[field] = state[field] + 1e-3

    with pytest.raises(ValueError, match=f"global codes or rotations differ from the seeded writer: {target}"):
        finalizer.audit_shared_code(shared_rows(tmp_path, edit))


@pytest.mark.parametrize("count", [0, matrix.SHARED_ROWS - 1])
def test_shared_audit_rejects_a_table_size_that_the_training_record_does_not_state(tmp_path, count):
    def edit(name, state, record):
        if name == "B8-s29":
            record["writer_trainable_parameters"] = count

    with pytest.raises(ValueError, match="trained-table size differs from the training record: B8-s29"):
        finalizer.audit_shared_code(shared_rows(tmp_path, edit))


# ---- launcher, config writer, analysis, ledger, run profile (stand-in repository) ----------------------------------

@pytest.fixture
def control(tmp_path, monkeypatch):
    """A stand-in repository for the control plane: P1-L's configs, the WP-L6 decision (oracle_rot), the P1L-S configs
    made from it, the committed lock, and a sizing file. run, launch and make read it instead of the checkout."""
    root = tmp_path / "repository"
    shutil.copytree(ROOT / "configs/p1l/main", root / "configs/p1l/main")
    for module in (run, launcher, maker):
        monkeypatch.setattr(module, "ROOT", root)
    decision = matrix.write_json(root / "reports/wp-l6/decision.json", dict(status="selected", selected="oracle_rot"))
    maker.write(maker.make(read(decision)))
    lock = matrix.write_json(root / "reports/p1l-s/lr-S-lock.json", dict(
        status="locked", S=17000, lrs={label: matrix.LRS[label] for label in matrix.RETRAINED}))
    sizing = matrix.write_json(tmp_path / "sizing.json", dict(
        rank=32, S=17000, pilot_steps=5666, study="p1l-s", timing_rows=[
            dict(label=label, step_seconds=STEP[32][label], overhead_seconds=20.) for label in matrix.RETRAINED]))
    return SimpleNamespace(root=root, decision=decision, lock=lock, sizing=sizing, out=tmp_path / "out")


def launch(control, monkeypatch, stage, *flags, fake=None, sizing=None):
    fake = fake or FakeSubprocess(control.root)
    monkeypatch.setattr(launcher, "subprocess", fake)
    monkeypatch.setattr(sys, "argv", ["launch_p1l_main.py", stage, "--root", str(control.out),
                                      "--sizing", str(sizing or control.sizing), *flags])
    with contextlib.redirect_stdout(io.StringIO()):
        launcher.main()
    return fake.commands


def option(command, flag):
    return command[command.index(flag) + 1]


def arguments(command):
    return command[command.index("--") + 1:]


def reuse(p1l):
    return "--reuse", str(p1l["manifest32"])


def test_main_submits_one_single_run_job_per_retrained_reader_and_seed(control, p1l, monkeypatch):
    commands = launch(control, monkeypatch, "main", *STUDY, *reuse(p1l))
    assert sorted(option(command, "--name") for command in commands) == sorted(f"ps-m-{name}" for name in RETRAINED)
    for command in commands:
        name = option(command, "--name")[len("ps-m-"):]
        label, seed = name.rsplit("-s", 1)
        assert option(command, "--scale-up") == "docs/prereg/P1L-S.md" and option(command, "--commit") == SOURCE
        assert option(command, "--walltime") == launcher.walltime(17000 * STEP[32][label] + 20. + 30.)
        assert option(command, "--queue") == "regular-g"
        assert arguments(command) == ["scripts/run_p1l_main.py", "main", "--out", str(control.out / "main" / name),
                                      "--seed", seed, "--rank", "32", "--labels", label, "--study", "p1l-s"]
    receipts = read(control.out / "main-jobs.json")
    assert receipts["source"] == SOURCE and sorted(receipts["jobs"]) == sorted(RETRAINED)


def test_main_writes_the_manifest_the_checker_reads(control, p1l, monkeypatch):
    launch(control, monkeypatch, "main", *STUDY, *reuse(p1l))
    reused = launcher.reused_runs(Path(p1l["manifest32"]), ("B3-L4",))
    assert read(control.out / "training-manifest.json") == dict(
        study="p1l-s", condition="p1l-s-r32", rank=32, writer="oracle_rot", S=17000,
        lrs=read(control.lock)["lrs"], lock_commit=SOURCE, lock_file_hash=matrix.digest(control.lock), reused=reused,
        runs={**{name: str(control.out / "main" / name / name) for name in RETRAINED},
              **{name: row["source"] for name, row in reused.items()}})


def test_pilot_runs_the_four_retrained_readers_at_seed_17(control, monkeypatch):
    commands = launch(control, monkeypatch, "pilot", *STUDY)
    assert [option(command, "--name") for command in commands] == [f"ps-p-{label}" for label in matrix.RETRAINED]
    for command, label in zip(commands, matrix.RETRAINED):
        assert option(command, "--walltime") == launcher.walltime(2 * 5666 * STEP[32][label] + (2 * 20. + 210.))
        assert option(command, "--scale-up") == "docs/prereg/P1L-S.md"
        assert arguments(command) == ["scripts/run_p1l_main.py", "pilot", "--out", str(control.out / "pilot" / label),
                                      "--seed", "17", "--rank", "32", "--labels", label, "--steps", "5666",
                                      "--study", "p1l-s"]
    assert not (control.out / "training-manifest.json").exists()


def test_a_second_launch_submits_nothing_new(control, p1l, monkeypatch):
    launch(control, monkeypatch, "main", *STUDY, *reuse(p1l))
    assert launch(control, monkeypatch, "main", *STUDY, *reuse(p1l)) == []


def test_study_job_names_never_enter_the_other_studys_ledger(control, p1l, monkeypatch):
    commands = launch(control, monkeypatch, "pilot", *STUDY) + launch(control, monkeypatch, "main", *STUDY, *reuse(p1l))
    names = [option(command, "--name") for command in commands]
    p1l_prefixes, p1ls_prefixes = ledger.LEDGERS["p1l"]["prefixes"], ledger.LEDGERS["p1l-s"]["prefixes"]
    assert len(names) == 16 and all(name.startswith(p1ls_prefixes) and not name.startswith(p1l_prefixes) for name in names)
    assert not any(f"{prefix}-m-B8-s17".startswith(p1ls_prefixes) for prefix in p1l_prefixes)
    assert not any(f"{prefix}-r128-m-B8-s17".startswith(p1ls_prefixes) for prefix in p1l_prefixes)


@pytest.mark.parametrize("stage,flags", [
    ("main", STUDY),                                  # the reused runs must be named
    ("pilot", (*STUDY, "--reuse", "x")),              # nothing is reused before the pilot
    ("main", ("--reuse", "x")),                       # P1-L reuses nothing
    ("main", (*STUDY, "--rank", "128", "--reuse", "x")),
    ("pilot", (*STUDY, "--rank", "128")),
])
def test_launcher_argument_rules(control, monkeypatch, stage, flags):
    with pytest.raises(SystemExit) as error:
        launch(control, monkeypatch, stage, *flags)
    assert error.value.code == 2


def test_launch_needs_a_selected_decision_that_the_configs_follow(control, p1l, monkeypatch):
    matrix.write_json(control.decision, dict(status="stop", selected=None))
    with pytest.raises(ValueError, match="only if WP-L6 selected"):
        launch(control, monkeypatch, "pilot", *STUDY)
    matrix.write_json(control.decision, dict(status="selected", selected="oracle_rot"))
    path = control.root / "configs/p1l-s/main/B4-s29.json"
    path.write_text(json.dumps({**read(path), "writer": "oracle_shared"}, indent=2) + "\n")
    with pytest.raises(ValueError, match="B4-s29 does not use the selected code"):
        launch(control, monkeypatch, "main", *STUDY, *reuse(p1l))


def test_launch_refuses_a_decision_that_is_not_the_committed_one(control, monkeypatch):
    fake = FakeSubprocess(control.root)
    committed = fake.check_output
    fake.check_output = lambda command, text=True: (
        json.dumps(dict(status="selected", selected="oracle_shared"))
        if command[3] == "show" and command[4].endswith("decision.json") else committed(command, text=text))
    with pytest.raises(ValueError, match="not the committed one"):
        launch(control, monkeypatch, "pilot", *STUDY, fake=fake)
    assert fake.commands == []


def test_launch_validates_the_reused_manifest_the_sizing_and_the_lock(control, p1l, monkeypatch, tmp_path):
    mismatched = {"study": lambda m: m.update(study="p1l-s"), "condition": lambda m: m.update(condition="oracle-r128")}
    for name, edit in mismatched.items():
        path = matrix.write_json(tmp_path / f"{name}.json", damaged(read(p1l["manifest32"]), edit))
        with pytest.raises(ValueError, match="needs P1-L's r = 32 training-manifest.json"):
            launch(control, monkeypatch, "main", *STUDY, "--reuse", str(path))
    p1l_sizing = matrix.write_json(tmp_path / "p1l-sizing.json", {**read(control.sizing), "study": "p1l"})
    unmarked = read(control.sizing)
    unmarked.pop("study")
    for sizing in (p1l_sizing, matrix.write_json(tmp_path / "unmarked.json", unmarked)):
        with pytest.raises(ValueError, match="sizing rank or study differs"):
            launch(control, monkeypatch, "main", *STUDY, *reuse(p1l), sizing=sizing)
    shorter = matrix.write_json(tmp_path / "shorter.json", {**read(control.sizing), "S": 8500})
    with pytest.raises(ValueError, match="committed final S/lr lock"):
        launch(control, monkeypatch, "main", *STUDY, *reuse(p1l), sizing=shorter)
    with pytest.raises(ValueError, match="sizing rank or study differs"):
        launch(control, monkeypatch, "main", sizing=control.sizing)  # a P1L-S sizing file is not P1-L's


def test_launch_does_not_overwrite_an_output_that_has_no_receipt(control, p1l, monkeypatch):
    (control.out / "main" / "B8-s17").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="without a submission receipt"):
        launch(control, monkeypatch, "main", *STUDY, *reuse(p1l))


def test_the_profile_selects_labels_locks_and_configs(control):
    assert tuple(run.STUDIES) == ("p1l", "p1l-s")
    assert run.labels_for_rank(32, "p1l-s") == ("B8", "M1-L4", "B4", "B5-L4", "B3-L4")
    assert run.trained_labels(32, "p1l-s") == ("B8", "M1-L4", "B4", "B5-L4")
    assert run.trained_labels(32) == run.LABELS and run.trained_labels(128) == run.LABELS[:5]
    assert run.lock_file(32, "p1l-s") == "reports/p1l-s/lr-S-lock.json"
    assert (run.lock_file(32), run.lock_file(128)) == ("reports/p1l/lr-S-lock.json", "reports/p1l/r128/lr-S-lock.json")
    assert (ROOT / run.STUDIES["p1l-s"]["prereg"]).is_file()
    assert run.config("B8", 29, 32, "p1l-s")["writer"] == "oracle_rot"
    assert run.config("B8", 29)["writer"] == "oracle_random"


@pytest.mark.parametrize("flags", [("--rank", "128"), ("--labels", "B3-L4"), ("--labels", "B1"), ("--labels", "B0"),
                                   ("--labels", "B8", "B3-L4")])
def test_run_trains_only_the_four_retrained_readers_at_r32(tmp_path, monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["run_p1l_main.py", "main", "--out", str(tmp_path / "out"), *STUDY, *flags])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2 and not (tmp_path / "out").exists()


@pytest.mark.parametrize("writer", WRITERS)
def test_p1ls_configs_differ_from_p1ls_files_in_name_and_writer_only(writer):
    configs = maker.make(dict(status="selected", selected=writer))
    assert sorted(configs) == sorted(RETRAINED)
    for name, value in configs.items():
        label, seed = name.rsplit("-s", 1)
        original = run.config(label, int(seed), 32, "p1l")
        assert value.keys() == original.keys() and {key for key in value if value[key] != original[key]} == {"name", "writer"}
        assert value["name"] == f"p1l-s-{name}" and value["writer"] == writer
        assert resolve_config(value)["writer"] == writer


@pytest.mark.parametrize("decision", [dict(status="stop"), dict(status="stop", selected="oracle_rot"),
                                      dict(status="selected"), dict(status="selected", selected="oracle_random"),
                                      dict(status="selected", selected="learned")])
def test_p1ls_configs_need_a_selected_oracle_code(decision):
    with pytest.raises(ValueError, match="needs a WP-L6 decision"):
        maker.make(decision)


def test_p1ls_configs_are_written_once_and_never_rewritten(control, capsys, monkeypatch):
    directory = control.root / "configs/p1l-s/main"
    paths = sorted(directory.glob("*.json"))
    assert len(paths) == 12 and all(path.read_text() == json.dumps(read(path), indent=2) + "\n" for path in paths)
    before = {path: path.read_text() for path in paths}
    configs = maker.make(read(control.decision))
    assert sorted(maker.write(configs)) == paths and {path: path.read_text() for path in paths} == before
    victim, other = paths[0], paths[1]
    victim.write_text("{}\n")
    other.unlink()
    with pytest.raises(FileExistsError, match="never rewritten"):
        maker.write(configs)
    assert victim.read_text() == "{}\n" and not other.exists()  # the conflict was found before anything was written
    with pytest.raises(FileExistsError):
        maker.write(maker.make(dict(status="selected", selected="oracle_shared")))
    victim.write_text(before[victim])
    monkeypatch.setattr(sys, "argv", ["make_p1ls_configs.py", "--decision", str(control.decision)])
    maker.main()
    assert json.loads(capsys.readouterr().out) == dict(written=12, directory=str(directory))
    assert {path: path.read_text() for path in paths} == before


def smoke(root, label, index, step):
    directory = root / "smoke" / f"{label}-s17"
    directory.mkdir(parents=True)
    matrix.write_json(directory / "record.json", dict(
        status="complete", config=dict(model=dict(r=32), smoke=True, train=dict(steps=100),
                                       eval=dict(splits=["val"], allow_confirmation=False)),
        train_seconds=20 + 100 * step, peak_memory_bytes=1000 + index, checkpoint_hash=f"c{index}",
        job_id=f"{index}.opbs", host="mg0001"))
    (directory / "train_log.jsonl").write_text("".join(
        json.dumps(dict(step=number, step_seconds=step)) + "\n" for number in range(1, 41)))


def pilot(root, label, index, sizing, rate_two_better):
    for rate in (.001, .003):
        directory = root / label / f"{label}-lr{rate:g}"
        accuracy = .3 + .01 * index + (.02 if rate_two_better and rate == .003 else 0.)
        matrix.write_json(directory / "record.json", dict(
            status="complete", checkpoint_hash=f"p{index}", config=dict(
                seed=17, train=dict(steps=sizing["pilot_steps"]), model=dict(r=32), protocol="memory_only",
                eval=dict(splits=["val"], hops=[1, 2, 3])),
            validation=[dict(condition="memory_only/original", hop=hop, n=1000, accuracy=accuracy) for hop in (1, 2, 3)]))


def test_analysis_sizes_and_locks_the_four_retrained_readers(tmp_path):
    for index, label in enumerate(matrix.RETRAINED):
        smoke(tmp_path, label, index, .1 + .01 * index)
    sizing_path = tmp_path / "sizing.json"
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.size(SimpleNamespace(root=tmp_path, rank=32, out=sizing_path, study="p1l-s"))
    sizing = read(sizing_path)
    assert (sizing["study"], sizing["S"], sizing["pilot_steps"]) == ("p1l-s", 17000, 5666)
    assert [row["label"] for row in sizing["timing_rows"]] == list(matrix.RETRAINED)
    for index, label in enumerate(matrix.RETRAINED):
        pilot(tmp_path / "pilot", label, index, sizing, rate_two_better=index in (0, 3))  # B4 ties: the lower rate wins
    lock_path = tmp_path / "lock.json"
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.select(SimpleNamespace(root=tmp_path / "pilot", sizing=sizing_path, rank=32, out=lock_path, study="p1l-s"))
    lock = read(lock_path)
    assert (lock["status"], lock["study"], lock["S"]) == ("locked", "p1l-s", 17000)
    assert lock["lrs"] == {"B8": .003, "M1-L4": .001, "B4": .001, "B5-L4": .003}
    assert [row["label"] for row in lock["selection"]] == list(matrix.RETRAINED)


@pytest.mark.parametrize("sizing_study,requested", [(None, "p1l-s"), ("p1l-s", "p1l")])
def test_a_sizing_file_locks_only_for_its_own_study(tmp_path, sizing_study, requested):
    sizing = dict(S=17000, pilot_steps=5666, packing="x", main_plus_pilot_node_hours=1.)
    if sizing_study:
        sizing["study"] = sizing_study
    path = matrix.write_json(tmp_path / "sizing.json", sizing)
    with pytest.raises(ValueError, match="another study"):
        analyze.select(SimpleNamespace(root=tmp_path, sizing=path, rank=32, out=tmp_path / "lock.json", study=requested))


def test_analysis_runs_p1ls_at_r32_only(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_p1l_main.py", "size", "--root", str(tmp_path), "--out",
                                      str(tmp_path / "out.json"), *STUDY, "--rank", "128"])
    with pytest.raises(SystemExit) as error:
        analyze.main()
    assert error.value.code == 2


def test_each_ledger_counts_only_its_own_jobs(tmp_path, monkeypatch):
    (tmp_path / "logs/pbs").mkdir(parents=True)
    names = ["p1l-m-B8-s17", "ev-p1l-B8-s17", "ps-m-B8-s17", "ps-ev-B8-s17", "wp-l6-pilot"]
    entries = [dict(submitted="2026-10-27T05:00", name=name, job_id=f"{100 + index}.opbs", log=str(tmp_path / f"{name}.log"),
                    walltime="01:00:00", source=SOURCE, argv=["scripts/run_p1l_main.py"]) for index, name in enumerate(names)]
    (tmp_path / "logs/pbs/submissions.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    jobs = [dict(job_id=entry["job_id"], job_name=entry["name"], status="FINISH", elapsed_seconds=3600, nodes=1 + index)
            for index, entry in enumerate(entries)]
    answer = SimpleNamespace(stdout=json.dumps(dict(ok=True, jobs=jobs)))
    monkeypatch.setattr(ledger, "REPO", tmp_path)
    monkeypatch.setattr(ledger, "subprocess", SimpleNamespace(run=lambda *args, **kwargs: answer))
    results = {}
    for study in ledger.LEDGERS:
        monkeypatch.setattr(sys, "argv", ["p1l_job_ledger.py", "--out", str(tmp_path / f"{study}.json"), "--study", study])
        with contextlib.redirect_stdout(io.StringIO()):
            ledger.main()
        results[study] = read(tmp_path / f"{study}.json")
    assert [row["name"] for row in results["p1l"]["jobs"]] == ["p1l-m-B8-s17", "ev-p1l-B8-s17"]
    assert [row["name"] for row in results["p1l-s"]["jobs"]] == ["ps-m-B8-s17", "ps-ev-B8-s17"]
    assert (results["p1l"]["task"], results["p1l-s"]["task"]) == ("WP-P1L", "P1L-S")
    assert results["p1l"]["allocated_node_hours"] == 1 + 2 and results["p1l-s"]["allocated_node_hours"] == 3 + 4


def test_p1l_ledger_keeps_its_registered_literals():
    assert ledger.LEDGERS["p1l"] == dict(
        name="P1-L", task="WP-P1L", since="2026-10-27T04:07", budget_rule="B2: per-job8nodes x2h; no total cap",
        prefixes=("pf-p1l", "size-p1l", "sm-p1l", "t-p1l", "p1l-", "ev-p1l", "fin-p1l"))


def test_the_control_plane_scripts_import_without_torch():
    code = ("import sys; sys.path.insert(0, %r); import run_p1l_main, launch_p1l_main, make_p1ls_configs, "
            "analyze_p1l_main, p1l_job_ledger; assert 'torch' not in sys.modules" % str(ROOT / "scripts"))
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


# ---- tiny model (GPU) ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "oracle-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def tiny(writer, variant="M1", **evaluation):
    return resolve_config(dict(
        name="p1ls-pipeline", writer=writer, stage="reader", variant=variant, model=dict(r=32, layer=1),
        task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
        train=dict(steps=2, batch_size=2, depths=[4], log_every=1, warmup_steps=1),
        eval=dict(episodes=2, batch_size=2, depths=[1, 4], hops=[1, 2, 3, 4], counterfactuals=False, **evaluation)))


@pytest.mark.parametrize("writer,expected", [("oracle_random", None), ("oracle_rot", 0), ("oracle_shared", matrix.SHARED_ROWS)])
def test_fit_adds_the_code_fields_for_the_new_writers_only(model, tokenizer, tmp_path, writer, expected):
    record = run.fit(tiny(writer), tmp_path / "run", model, tokenizer, expose_scores=False)
    fields = {"writer_trainable_parameters", "parameters", "code_drift"}
    assert read(tmp_path / "run" / "record.json")["status"] == "complete"
    if expected is None:
        assert not fields & record.keys()  # P1-L records keep their fields
        return
    assert record["writer_trainable_parameters"] == expected and record["parameters"]["external_writer"] == expected
    assert ("code_drift" in record) is (writer == "oracle_shared")
    if writer == "oracle_shared":
        seen, unseen = record["code_drift"]["seen"], record["code_drift"]["unseen"]
        assert (seen["rows"], unseen["rows"]) == (768, 256) and unseen["rows_changed"] == 0


@pytest.mark.parametrize("writer,expected", [("oracle_random", None), ("oracle_rot", None), ("oracle_shared", matrix.SHARED_ROWS)])
def test_inference_json_counts_the_trained_table_only_when_there_is_one(model, tokenizer, tmp_path, writer, expected):
    cfg = tiny(writer, "B8")
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        evaluator.inference_timing(experiment, cfg, tmp_path)
    finally:
        experiment.close()
    measured = read(tmp_path / "inference.json")
    assert measured.get("writer_parameters") == expected and ("writer_parameters" in measured) is (expected is not None)
    assert measured["rank"] == 32 and measured["trainable_adapter_parameters"] > 0


@pytest.mark.parametrize("splits,window,calls,ranges", [
    (["confirm"], "p1l-s", [["test", 110000, 2, 4242]], {"confirm": dict(split="test", start=110000, count=2)}),
    (["confirm"], None, [["test", 100000, 2, 4242]], {"confirm": dict(split="test", start=100000, count=2)}),
    (["val", "confirm"], "p1l-s", [["val", 0, 2, 4242], ["test", 110000, 2, 4242]],
     {"val": dict(split="val", start=0, count=2), "confirm": dict(split="test", start=110000, count=2)}),
    (["val"], None, [["val", 0, 2, 4242]], {"val": dict(split="val", start=0, count=2)}),
])
def test_evaluate_reads_the_named_window_of_the_test_split(model, tokenizer, tmp_path, monkeypatch, splits, window, calls, ranges):
    asked, real = [], llm_evaluate.generate

    def recorder(spec, split, start, count, seed):  # validation episodes stand in: no test episode is ever generated
        asked.append([split, start, count, seed])
        return real(spec, "val", 0, count, seed)

    monkeypatch.setattr(llm_evaluate, "generate", recorder)
    cfg = tiny("oracle_rot", splits=splits, allow_confirmation="confirm" in splits, **({"confirmation": window} if window else {}))
    experiment = LLMExperiment(cfg, model=model, tokenizer=tokenizer)
    try:
        metrics, _ = llm_evaluate.evaluate(experiment, cfg, tmp_path / "evaluation")
    finally:
        experiment.close()
    assert asked == calls and metrics["episode_ranges"] == ranges
