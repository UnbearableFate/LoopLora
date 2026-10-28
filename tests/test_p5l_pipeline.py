"""P5L step 2: the confirmation checks, evaluator configs and finalizer, on synthetic artifacts (no model).

The matrix builders are those of the P1-L tests (tests/assets/p1l_matrix.py). Compute node only (torch, looplora).
"""
import contextlib
import copy
import io
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests" / "assets")]
import evaluate_p5l as evaluator
import finalize_p5l as finalizer
import make_p5l_manifest as manifest_maker
import p1l_matrix as matrix
import run_p5l
from looplora.llm.config import split_window

BACKBONE = evaluator.BACKBONE
SEEDS = matrix.SEEDS
TRAINED = [f"{label}-s{seed}" for label in run_p5l.LABELS for seed in SEEDS]
LRS = {"B8": .003, "M1-L4": .003, "B4": .003, "B5-L4": .003, "B3-L4": .001}
read, edited = matrix.read_json, matrix.edited


def build(root: Path, overrides=None, window_start=104000) -> SimpleNamespace:
    """A stand-in repository (prereg, lock, templates, the 0.6B endpoints), 15 trained runs and B0, the manifest, the
    authorization and the confirmation artifacts of every run."""
    repository = root / "repository"
    shutil.copytree(ROOT / "configs/p1l/main", repository / "configs/p1l/main")
    (repository / "docs/prereg").mkdir(parents=True)
    (repository / "docs/prereg/P5L.md").write_text("registered prereg\n")
    lock = matrix.write_json(repository / "reports/p5l/r128/lr-S-lock.json", dict(
        status="locked", rank=128, S=17000, lrs=LRS, backbone=BACKBONE))
    shutil.copy(ROOT / "reports/p1l/r128/endpoints.json", matrix.write_json(repository / "reports/p1l/r128/endpoints.json", {}))
    runs = {}
    for name in (*TRAINED, "B0-s17"):
        label, seed = name.rsplit("-s", 1)
        directory = root / "runs" / name
        matrix.make_run(directory, label, int(seed), 128, "oracle_random" if label != "B0" else "learned",
                        17000, LRS.get(label, .001))
        for file in ("config.json", "record.json"):                                  # the fields this study adds
            value = read(directory / file)
            target = value["config"] if file == "record.json" else value
            target["backbone"] = BACKBONE
            target["variant"] = label if label == "B0" else "M1"
            matrix.write_json(directory / file, value)
        runs[name] = str(directory)
    manifest = matrix.write_json(repository / "reports/p5l/r128/training-manifest.json", dict(
        study="p5l", condition="p5l-r128", rank=128, S=17000, lrs=LRS, backbone=BACKBONE, runs=runs,
        lock=dict(path="reports/p5l/r128/lr-S-lock.json", hash=matrix.digest(lock))))
    authorization = matrix.write_json(repository / "reports/p5l/confirmation-authorization.json", dict(
        study="p5l", status="allowed", orchestrator_decision="fixture", gated_conditions=2, confidence=.975,
        confirmation_window="p5l", prereg=dict(path="docs/prereg/P5L.md", hash=matrix.digest(repository / "docs/prereg/P5L.md")),
        lock=dict(path="reports/p5l/r128/lr-S-lock.json", hash=matrix.digest(lock)),
        condition_manifests={"p5l-r128": str(manifest)}, manifest_hash=matrix.digest(manifest)))
    accuracy = matrix.accuracy_function(overrides)
    evaluations = {}
    for name, train in runs.items():
        out = Path(matrix.eval_run(root, name, train, authorization, memory="memory-oracle_random", window="p5l",
                                   start=window_start, accuracy=accuracy, rank=128))
        for setting in ("primary", "typed"):
            with_backbone = read(out / setting / "config.json")
            with_backbone["backbone"] = BACKBONE
            matrix.write_json(out / setting / "config.json", with_backbone)
        evaluations[name] = str(out)
    for name in runs:                                                              # the synthetic writer hash is per label
        record = read(Path(runs[name]) / "record.json")
        record["writer_hash"] = "oracle_random-code"
        matrix.write_json(Path(runs[name]) / "record.json", record)
    return SimpleNamespace(repository=repository, manifest=manifest, authorization=authorization, runs=runs,
                           evaluations=matrix.write_json(root / "evaluation-manifest.json", dict(runs=evaluations)))


@pytest.fixture(scope="module")
def p5l(tmp_path_factory):
    return build(tmp_path_factory.mktemp("p5l"))


@pytest.fixture
def repo(p5l, monkeypatch):
    for module in (evaluator, finalizer):
        monkeypatch.setattr(module, "ROOT", p5l.repository)
    return p5l


def check(p5l, manifest=None, authorization=None, **kwargs):
    evaluator.check_training(manifest or read(p5l.manifest), authorization or read(p5l.authorization), **kwargs)


# ---- the window and the registered decision --------------------------------------------------------------------

def test_the_p5l_window_is_the_unused_range_after_p1ls_and_p1lc():
    assert split_window({"confirmation": "p5l"}, "confirm") == ("test", 104000)
    assert evaluator.WINDOW == "p5l"


@pytest.mark.parametrize("memory, b6, prefix", [(False, True, "B7_failed"), (False, False, "B7_failed"),
                                               (True, True, "holds_at_second_scale"), (True, False, "LP4_narrowed_to_0.6B")])
def test_the_decision_follows_the_registered_use(memory, b6, prefix):
    assert finalizer.decision(memory, b6).startswith(prefix)


# ---- check_training ---------------------------------------------------------------------------------------------

def test_the_matrix_is_accepted_with_and_without_the_lock_check(repo):
    check(repo)
    check(repo, check_all=False)


@pytest.mark.parametrize("edit", [
    lambda a: a.update(study="p1l"), lambda a: a.update(status="held"), lambda a: a.update(orchestrator_decision=""),
    lambda a: a.update(gated_conditions=1), lambda a: a.update(gated_conditions=3), lambda a: a.update(confidence=.95),
    lambda a: a.update(confirmation_window="p1l"), lambda a: a.pop("confirmation_window"),
    lambda a: a["prereg"].update(hash="0" * 32), lambda a: a["lock"].update(hash="0" * 32),
    lambda a: a["condition_manifests"].update(extra="x"), lambda a: a.update(manifest_hash="0" * 32)])
def test_the_authorization_names_the_study_the_prereg_the_lock_the_window_and_one_condition(repo, edit):
    authorization = copy.deepcopy(read(repo.authorization))
    edit(authorization)
    with pytest.raises(ValueError):
        check(repo, authorization=authorization)


@pytest.mark.parametrize("edit", [
    lambda m: m.update(study="p1l"), lambda m: m.update(rank=32), lambda m: m.update(condition="oracle-r128"),
    lambda m: m.update(S=16000), lambda m: m["lrs"].update(B8=.001),
    lambda m: m["backbone"].update(revision="0" * 40), lambda m: m["runs"].pop("B0-s17"),
    lambda m: m["runs"].pop("B8-s17"), lambda m: m["runs"].update(extra="x")])
def test_the_manifest_is_exactly_the_registered_matrix(repo, edit):
    manifest = copy.deepcopy(read(repo.manifest))
    edit(manifest)
    with pytest.raises(ValueError):
        check(repo, manifest=manifest, check_all=False)


def test_a_manifest_that_differs_from_the_locked_one_is_refused_only_with_the_lock_check(repo):
    manifest = copy.deepcopy(read(repo.manifest))
    manifest["lock"]["hash"] = "0" * 32
    check(repo, manifest=manifest, check_all=False)
    with pytest.raises(ValueError, match="locked"):
        check(repo, manifest=manifest)


@pytest.mark.parametrize("name, edit", [
    ("B8-s17", lambda r: r["config"].pop("backbone")),
    ("B8-s17", lambda r: r["config"]["backbone"].update(id="Qwen/Qwen3-0.6B-Base")),
    ("M1-L4-s29", lambda r: r["config"]["train"].update(lr=.001)),
    ("B3-L4-s43", lambda r: r["config"]["train"].update(steps=16000)),
    ("B4-s17", lambda r: r["config"].update(writer="learned")),
    ("B4-s17", lambda r: r["config"].update(oracle_seed=18)),
    ("B5-L4-s17", lambda r: r["config"].update(smoke=True)),
    ("B5-L4-s17", lambda r: r["config"]["eval"].update(allow_confirmation=True)),
    ("B5-L4-s17", lambda r: r["config"]["model"].update(r=32)),
    ("B0-s17", lambda r: r["config"].update(variant="M1")),
    ("B8-s43", lambda r: r.update(status="running")),
    ("B8-s43", lambda r: r.update(checkpoint_hash="0" * 32))])
def test_every_run_is_complete_unchanged_on_the_registered_backbone_under_the_locked_settings(repo, name, edit):
    with edited(Path(repo.runs[name]) / "record.json", edit):
        with pytest.raises(ValueError):
            check(repo, check_all=False)
    check(repo, check_all=False)


# ---- the evaluator ---------------------------------------------------------------------------------------------

def run_evaluator(monkeypatch, tmp_path, repo, name):
    """main() with the model, checkpoint load and evaluate() replaced; returns what it loaded and the configs it scores."""
    seen, loaded = [], []
    base = {**read(Path(repo.runs[name]) / "config.json"), "device": "cpu"}
    from looplora.llm.config import resolve_config
    base = resolve_config({k: v for k, v in base.items() if k != "device"})

    class Loaded:
        config = base

        def close(self):
            pass

    monkeypatch.setattr(evaluator, "require_compute", lambda: None)
    monkeypatch.setattr(evaluator, "load_model", lambda backbone=None: loaded.append(("model", backbone)))
    monkeypatch.setattr(evaluator, "load_tokenizer", lambda backbone=None: loaded.append(("tokenizer", backbone)))
    monkeypatch.setattr(evaluator, "load_checkpoint", lambda path, model, tokenizer: Loaded())
    monkeypatch.setattr(evaluator, "evaluate", lambda experiment, cfg, out: (seen.append(copy.deepcopy(cfg)), ({}, {}))[1])
    monkeypatch.setattr(evaluator, "swap_successors", lambda experiment, cfg, out: None)
    monkeypatch.setattr(evaluator, "inference_timing", lambda experiment, cfg, out: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    out = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["evaluate_p5l.py", "--manifest", str(repo.manifest), "--authorization",
                                      str(repo.authorization), "--out", str(out), "--runs", name])
    evaluator.main()
    return seen, loaded, out / name


@pytest.mark.parametrize("name", ["B8-s17", "B0-s17"])
def test_the_evaluator_loads_the_manifests_backbone_and_scores_the_p5l_window(repo, monkeypatch, tmp_path, name):
    seen, loaded, out = run_evaluator(monkeypatch, tmp_path, repo, name)
    assert loaded == [("model", BACKBONE), ("tokenizer", BACKBONE)]
    assert [cfg["task"]["name"] for cfg in seen] == ["t2_chain_end", "t2_typed"]
    for cfg in seen:
        assert cfg["eval"]["confirmation"] == "p5l" and cfg["eval"]["splits"] == ["confirm"] and cfg["backbone"] == BACKBONE
        assert cfg["eval"]["allow_confirmation"] is True and (cfg["eval"]["episodes"], cfg["eval"]["seed"]) == (2000, 4242)
        assert split_window(cfg["eval"], "confirm") == ("test", 104000)
    assert read(out / "record.json")["authorization_hash"] == matrix.digest(repo.authorization)


def test_the_evaluator_scores_nothing_when_the_checks_fail(repo, monkeypatch, tmp_path):
    with edited(repo.authorization, lambda a: a.update(status="held")):
        with pytest.raises(ValueError):
            run_evaluator(monkeypatch, tmp_path, repo, "B8-s17")
    assert not (tmp_path / "evaluation").exists()


# ---- the finalizer ---------------------------------------------------------------------------------------------

def finalize(p5l, out):
    with contextlib.redirect_stdout(io.StringIO()):
        finalizer.finalize(SimpleNamespace(manifest=p5l.manifest, evaluations=p5l.evaluations,
                                           authorization=p5l.authorization, out=Path(out)))
    return Path(out)


@pytest.fixture(scope="module")
def finalized(p5l, tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        for module in (evaluator, finalizer):
            patch.setattr(module, "ROOT", p5l.repository)
        return finalize(p5l, tmp_path_factory.mktemp("final"))


def test_the_finalizer_records_the_study_window_backbone_decision_and_files(finalized):
    result = read(finalized / "endpoints.json")
    assert (result["study"], result["rank"], result["confirmation_window"], result["backbone"]) == ("p5l", 128, "p5l", BACKBONE)
    assert result["confidence"] == .975 and result["memory_use_passed"] and result["B6_equal_parameter"] is True
    assert result["decision"].startswith("holds_at_second_scale")
    assert {path.name for path in finalized.iterdir()} == {"endpoints.json", "accuracy.csv", "swap-successors.csv",
        "inference.csv", "memory-use.csv", "run-manifest.json", "results.md"}
    assert {row["reader"] for row in map(lambda line: dict(zip(("rank", "reader"), line.split(",")[:2])),
           (finalized / "accuracy.csv").read_text().splitlines()[1:])} == {"B8", "M1-L4", "B4", "B5-L4", "B3-L4", "B0"}
    manifest = read(finalized / "run-manifest.json")
    assert manifest["backbone"] == BACKBONE and len(manifest["runs"]) == 16


def test_the_results_report_the_decision_the_window_and_the_cross_scale_table(finalized):
    text = (finalized / "results.md").read_text()
    assert "test [104000, 106000)" in text and "holds_at_second_scale" in text and "Rule 3" in text
    assert "Cross-scale, descriptive" in text and "| M1_vs_B4 |" in text


@pytest.mark.parametrize("overrides, memory, b6, prefix", [
    ({"M1-L4": .31}, True, False, "LP4_narrowed_to_0.6B"),                                   # M1 - B4 below 3 pp
    ({"B4": .12, "B8": .12, "M1-L4": .12}, False, False, "B7_failed")])                      # no reader above B3-L4
def test_the_decision_changes_with_b6_and_b7(overrides, memory, b6, prefix, tmp_path, monkeypatch):
    p5l = build(tmp_path, overrides)
    for module in (evaluator, finalizer):
        monkeypatch.setattr(module, "ROOT", p5l.repository)
    result = read(finalize(p5l, tmp_path / "out") / "endpoints.json")
    assert (result["memory_use_passed"], result["B6_equal_parameter"]) == (memory, b6)
    assert result["decision"].startswith(prefix)


def test_the_finalizer_refuses_another_window_a_missing_run_and_a_missing_backbone(p5l, repo, tmp_path):
    first = Path(read(p5l.evaluations)["runs"]["B8-s17"])
    with edited(first / "primary" / "metrics.json", lambda m: m["episode_ranges"]["confirm"].update(start=102000)):
        with pytest.raises(ValueError, match="range"):
            finalize(p5l, tmp_path / "a")
    with edited(first / "typed" / "config.json", lambda c: c.pop("backbone")):
        with pytest.raises(ValueError, match="backbone"):
            finalize(p5l, tmp_path / "b")
    with edited(p5l.evaluations, lambda e: e["runs"].pop("B0-s17")):
        with pytest.raises(ValueError, match="complete training matrix"):
            finalize(p5l, tmp_path / "c")


# ---- the manifest builder --------------------------------------------------------------------------------------

def test_the_manifest_builder_refuses_a_lock_that_is_not_the_registered_one(p5l, tmp_path):
    with edited(p5l.repository / "reports/p5l/r128/lr-S-lock.json", lambda lock: lock.update(status="draft")):
        with pytest.raises(ValueError, match="registered"):
            manifest_maker.verify(tmp_path, tmp_path / "receipts.jsonl", p5l.repository / "reports/p5l/r128/lr-S-lock.json", "a" * 40)
