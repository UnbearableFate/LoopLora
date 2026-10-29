"""B8W step 2: the manifest builder, the confirmation checks and the finalizer, on synthetic artifacts (no model).

The matrix builders are those of the P1-L tests (tests/assets/p1l_matrix.py). Compute node only (torch, looplora).
"""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests" / "assets")]
import evaluate_b8w as evaluator
import finalize_b8w as finalizer
import make_b8w_manifest as maker
import p1l_matrix as matrix
from looplora.llm.config import split_window

SEEDS = matrix.SEEDS
read, write = matrix.read_json, matrix.write_json
VARIANT = {"M1-L4": "M1", "B4": "B4", "B8": "B8"}     # training label -> config variant
SOURCE = "a" * 40
DATA_SEED = 7406709751464271312
ACCURACY = {"start": .60, "shared": .62, "untied": .64, "B4": .30, "B8": .20}


def make_run(root: Path, name: str, label: str, seed: int, steps: int = 17000, lr: float = .003) -> Path:
    directory = root / name
    matrix.make_run(directory, label, seed, 128, "oracle_random", steps, lr)
    adapter = {key: torch.randn(4, 4, generator=torch.Generator().manual_seed(seed + i)) for i, key in enumerate("RUQ")}
    if VARIANT[label] == "B8":
        adapter = {f"{key}.{step}": value + step for key, value in adapter.items() for step in range(4)}
    torch.save({"writer": {}, "adapter": adapter}, directory / "checkpoint.pt")
    for file in ("config.json", "record.json"):
        value = read(directory / file)
        target = value["config"] if file == "record.json" else value
        target["variant"] = VARIANT[label]
        if file == "record.json":
            value["checkpoint_hash"] = matrix.digest(directory / "checkpoint.pt")
        write(directory / file, value)
    return directory


def train_log(path: Path, steps=(1, *range(100, 2001, 100)), data_seed=DATA_SEED, first_loss=3.2):
    path.write_text("".join(json.dumps(dict(step=step, loss=first_loss if step == 1 else 3.0 - step * 1e-4, accuracy=.4,
                                             data_seed=data_seed)) + "\n" for step in steps))


def build(root: Path, overrides=None, window_start=108000) -> SimpleNamespace:
    repository = root / "repository"
    shutil.copytree(ROOT / "configs/p1l/main", repository / "configs/p1l/main")
    (repository / "docs/prereg").mkdir(parents=True)
    (repository / "docs/prereg/B8W.md").write_text("registered prereg\n")
    # the P1-L checkpoints (start = M1-L4, references B4 and B8) and the six continuations
    runs, lock_checkpoints = {}, {}
    for seed in SEEDS:
        for label, group in (("M1-L4", "start"), ("B4", "B4"), ("B8", "B8")):
            directory = make_run(root / "p1l", f"{label}-s{seed}", label, seed)
            train_log(directory / "train_log.jsonl", steps=(1, 100))
            runs[f"{group}-s{seed}"] = str(directory)
            lock_checkpoints[f"{label}-s{seed}"] = dict(path=str(directory / "checkpoint.pt"), hash=matrix.digest(directory / "checkpoint.pt"))
    lock = write(repository / "reports/b8w/checkpoint-lock.json", dict(
        status="locked", study="b8w", window="b8w", steps=2000, lr=.003, data_start_step=17000, oracle_hash="oracle_random-code",
        prereg_hash=matrix.digest(repository / "docs/prereg/B8W.md"), checkpoints=lock_checkpoints))
    jobs, receipts = {}, []
    for seed in SEEDS:
        base = root / "b8w" / f"s{seed}"
        converted = base / "converted.pt"
        base.mkdir(parents=True)
        torch.save({"adapter": {}}, converted)
        write(base / "gate.json", dict(passed=True, max_abs_difference=0.0, tolerance=0.0, control_equal=True,
              start_checkpoint_hash=lock_checkpoints[f"M1-L4-s{seed}"]["hash"], converted_checkpoint_hash=matrix.digest(converted)))
        log = root / "logs" / f"tr-b8w-s{seed}-20261029-000000-1.log"
        log.parent.mkdir(exist_ok=True, parents=True)
        Path(str(log) + ".exit").write_text("0\n")
        receipts.append(json.dumps(dict(job_id=f"{seed}.opbs", log=str(log), source=dict(kind="commit", ref=SOURCE))))
        for arm in ("shared", "untied"):
            directory = make_run(base, arm, "M1-L4" if arm == "shared" else "B8", seed, steps=2000)
            record = read(directory / "record.json")
            cfg = record["config"]
            cfg.update(seed=seed, name=f"b8w-{arm}-s{seed}")
            cfg["train"] = dict(steps=2000, lr=.003, data_start_step=17000, warmup_steps=20, batch_size=64, depths=[4])
            record.update(steps=2000, start_checkpoint_hash=(lock_checkpoints[f"M1-L4-s{seed}"]["hash"] if arm == "shared"
                          else matrix.digest(converted)), gate=dict(passed=True), git_commit=SOURCE, source_dirty=False,
                          writer_hash="oracle_random-code", config_hash=f"{arm}{seed}", train_seconds=380.)
            write(directory / "record.json", record)
            write(directory / "config.json", cfg)
            train_log(directory / "train_log.jsonl")
            runs[f"{arm}-s{seed}"] = str(directory)
            jobs[f"{arm}-s{seed}"] = dict(checkpoint_hash=record["checkpoint_hash"],
                                         converted=str(converted) if arm == "untied" else None)
    receipt_file = root / "receipts.jsonl"
    receipt_file.write_text("\n".join(receipts) + "\n")
    manifest = write(repository / "reports/b8w/continuation-manifest.json", dict(
        study="b8w", steps=2000, lr=.003, window="b8w", runs=runs, jobs=jobs,
        lock=dict(path="reports/b8w/checkpoint-lock.json", hash=matrix.digest(lock))))
    authorization = write(repository / "reports/b8w/confirmation-authorization.json", dict(
        study="b8w", status="allowed", orchestrator_decision="fixture", gated_conditions=1, confidence=.95,
        confirmation_window="b8w", prereg=dict(path="docs/prereg/B8W.md", hash=matrix.digest(repository / "docs/prereg/B8W.md")),
        lock=dict(path="reports/b8w/checkpoint-lock.json", hash=matrix.digest(lock)), manifest=str(manifest),
        manifest_hash=matrix.digest(manifest)))
    accuracy = matrix.accuracy_function({**ACCURACY, **(overrides or {})})
    evaluations = {}
    for name, train in runs.items():
        out = Path(matrix.eval_run(root, name, train, authorization, memory="memory-oracle_random", window="b8w",
                                   start=window_start, accuracy=accuracy, rank=128))
        record = read(out / "record.json")
        record["manifest_hash"] = matrix.digest(manifest)
        write(out / "record.json", record)
        evaluations[name] = str(out)
    return SimpleNamespace(repository=repository, manifest=manifest, authorization=authorization, runs=runs, lock=lock,
                           receipts=receipt_file, root=root, evaluations=write(root / "evaluation-manifest.json", dict(runs=evaluations)))


@pytest.fixture(scope="module")
def b8w(tmp_path_factory):
    return build(tmp_path_factory.mktemp("b8w"))


@pytest.fixture
def repo(b8w, monkeypatch):
    for module in (evaluator, finalizer, maker):
        monkeypatch.setattr(module, "ROOT", b8w.repository)
    monkeypatch.setattr(finalizer, "check_training", evaluator.check_training)
    return b8w


def check(b8w, manifest=None, authorization=None, **kwargs):
    evaluator.check_training(manifest or read(b8w.manifest), authorization or read(b8w.authorization), **kwargs)


def test_the_b8w_window_is_the_registered_range():
    assert split_window({"confirmation": "b8w"}, "confirm") == ("test", 108000)
    assert evaluator.WINDOW == "b8w" and len(evaluator.NAMES) == 15


# ---- check_training ---------------------------------------------------------------------------------------------

def test_the_matrix_is_accepted_with_and_without_the_manifest_check(repo):
    check(repo)
    check(repo, check_all=False)


@pytest.mark.parametrize("edit", [
    lambda a: a.update(study="p5l"), lambda a: a.update(status="held"), lambda a: a.update(orchestrator_decision=""),
    lambda a: a.update(gated_conditions=2), lambda a: a.update(confidence=.975), lambda a: a.update(confirmation_window="p5l"),
    lambda a: a.pop("confirmation_window"), lambda a: a["prereg"].update(hash="0" * 32), lambda a: a["lock"].update(hash="0" * 32),
    lambda a: a.update(manifest_hash="0" * 32)])
def test_the_authorization_names_the_study_the_prereg_the_lock_the_window_and_one_contrast(repo, edit):
    authorization = copy.deepcopy(read(repo.authorization))
    edit(authorization)
    with pytest.raises(ValueError):
        check(repo, authorization=authorization)


@pytest.mark.parametrize("edit", [
    lambda m: m.update(study="p5l"), lambda m: m["runs"].pop("untied-s17"), lambda m: m["runs"].update(extra="x"),
    lambda m: m["lock"].update(hash="0" * 32)])
def test_the_manifest_is_exactly_the_registered_fifteen_runs(repo, edit):
    manifest = copy.deepcopy(read(repo.manifest))
    edit(manifest)
    with pytest.raises(ValueError):
        check(repo, manifest=manifest, check_all=False)


def test_checkpoints_must_be_the_locked_ones_and_variants_the_registered_ones(repo):
    run = Path(read(repo.manifest)["runs"]["B4-s29"])
    original = (run / "checkpoint.pt").read_bytes()
    torch.save({"writer": {}, "adapter": {"R": torch.zeros(2)}}, run / "checkpoint.pt")
    record = read(run / "record.json")
    with matrix.edited(run / "record.json", lambda value: value.update(checkpoint_hash=matrix.digest(run / "checkpoint.pt"))):
        with pytest.raises(ValueError, match="locked"):
            check(repo, check_all=False)
    (run / "checkpoint.pt").write_bytes(original)
    with matrix.edited(Path(read(repo.manifest)["runs"]["untied-s43"]) / "record.json", lambda value: value["config"].update(variant="M1")):
        with pytest.raises(ValueError, match="registered reader"):
            check(repo, check_all=False)
    check(repo)


# ---- the finalizer ----------------------------------------------------------------------------------------------

def finalize(b8w, out):
    args = SimpleNamespace(manifest=b8w.manifest, evaluations=b8w.evaluations, authorization=b8w.authorization, out=out)
    finalizer.finalize(args)
    return read(out / "endpoints.json")


def test_equal_arms_keep_the_solution(repo, tmp_path):
    result = finalize(repo, tmp_path / "out")
    assert result["keeps"]["keeps"] and result["decision"].startswith("untying keeps the solution")
    delta = result["contrasts"]["DELTA_untied_vs_shared"]
    assert delta["comparison"] == "untied@4 - shared@4" and delta["difference"] > 0
    assert set(result["contrasts"]) == {name for name, *_ in finalizer.CONTRASTS} == set(result["contrasts_975"])
    assert result["contrasts"]["untied_vs_B4"]["comparison"] == "untied@4 - B4@1"
    assert not result["protocol_check_shared_below_start_by_more_than_2pp"]
    assert "Decision (docs/prereg/B8W.md)" in (tmp_path / "out/results.md").read_text()
    assert {"accuracy.csv", "drift.csv", "train-loss.csv", "run-manifest.json"} <= {p.name for p in (tmp_path / "out").iterdir()}


def test_a_large_loss_loses_the_solution(tmp_path_factory, monkeypatch):
    lossy = build(tmp_path_factory.mktemp("lossy"), overrides={"untied": .50})
    for module in (evaluator, finalizer):
        monkeypatch.setattr(module, "ROOT", lossy.repository)
    out = tmp_path_factory.mktemp("lossy-out")
    result = finalize(lossy, out / "x")
    assert not result["keeps"]["keeps"] and result["decision"].startswith("untying loses the solution")
    assert "not lack of expressivity" in result["decision"]


def test_a_shared_arm_far_below_the_start_is_flagged_without_changing_the_rule(tmp_path_factory, monkeypatch):
    flagged = build(tmp_path_factory.mktemp("flag"), overrides={"shared": .40, "untied": .45})
    for module in (evaluator, finalizer):
        monkeypatch.setattr(module, "ROOT", flagged.repository)
    result = finalize(flagged, tmp_path_factory.mktemp("flag-out") / "x")
    assert result["protocol_check_shared_below_start_by_more_than_2pp"] and result["keeps"]["keeps"]


def test_the_finalizer_rejects_a_wrong_window_and_unlocked_provenance(tmp_path_factory, monkeypatch):
    wrong = build(tmp_path_factory.mktemp("wrong"), window_start=104000)
    for module in (evaluator, finalizer):
        monkeypatch.setattr(module, "ROOT", wrong.repository)
    with pytest.raises(ValueError, match="range/protocol"):
        finalize(wrong, tmp_path_factory.mktemp("wrong-out") / "x")


def test_the_finalizer_rejects_a_changed_checkpoint(repo, tmp_path):
    name = "start-s17"
    record_path = Path(read(repo.evaluations)["runs"][name]) / "record.json"
    with matrix.edited(record_path, lambda value: value.update(checkpoint_hash="0" * 32)):
        with pytest.raises(ValueError, match="provenance"):
            finalize(repo, tmp_path / "out")


def test_drift_and_train_loss_tables(repo):
    manifest = read(repo.manifest)
    rows = finalizer.drift(manifest)
    assert len(rows) == 9 and {row["tensor"] for row in rows} == {"R", "U", "Q"}
    assert all(row["untied_mean_pairwise_cosine_distance"] >= 0 for row in rows)
    losses = finalizer.train_loss(manifest)
    assert len(losses) == 2 * 3 * 21 and {row["arm"] for row in losses} == {"shared", "untied"}


# ---- the manifest builder ---------------------------------------------------------------------------------------

def verify(b8w):
    return maker.verify(b8w.root / "b8w", b8w.receipts, b8w.lock, SOURCE)


def test_the_builder_verifies_six_continuations_and_names_the_fifteen_runs(repo):
    manifest = verify(repo)
    assert set(manifest["runs"]) == set(evaluator.NAMES) and manifest["lock"]["path"] == "reports/b8w/checkpoint-lock.json"
    assert manifest["jobs"]["untied-s29"]["converted"].endswith("s29/converted.pt")
    assert manifest["training_source"] == SOURCE


def edit_run(b8w, name, edit):
    return matrix.edited(Path(b8w.root / "b8w" / f"s{name[-2:]}" / name[:-4] / "record.json"), edit)


@pytest.mark.parametrize("name, edit", [
    ("shared-s17", lambda r: r.update(status="running")),
    ("shared-s17", lambda r: r.update(steps=1999)),
    ("shared-s17", lambda r: r["config"]["train"].update(lr=.001)),
    ("shared-s17", lambda r: r["config"]["train"].update(data_start_step=0)),
    ("shared-s17", lambda r: r.update(start_checkpoint_hash="0" * 32)),
    ("untied-s29", lambda r: r.update(start_checkpoint_hash="0" * 32)),
    ("untied-s29", lambda r: r.update(git_commit="b" * 40)),
    ("untied-s43", lambda r: r.update(source_dirty=True)),
    ("untied-s43", lambda r: r.update(writer_hash="other")),
    ("untied-s43", lambda r: r["gate"].update(passed=False)),
    ("shared-s29", lambda r: r["config"]["eval"].update(allow_confirmation=True)),
    ("untied-s17", lambda r: r["config"].update(variant="M1")),
    ("shared-s17", lambda r: r["config"].update(smoke=True)),
    ("untied-s17", lambda r: r["config"]["model"].update(r=32))])
def test_the_builder_rejects_a_continuation_that_disagrees_with_the_lock_or_the_registration(repo, name, edit):
    with edit_run(repo, name, edit):
        with pytest.raises(ValueError):
            verify(repo)
    verify(repo)


def test_the_builder_rejects_gates_logs_streams_and_receipts(repo):
    gate = repo.root / "b8w/s17/gate.json"
    for change in (dict(passed=False), dict(max_abs_difference=1e-9), dict(control_equal=False), dict(start_checkpoint_hash="0" * 32),
                   dict(converted_checkpoint_hash="0" * 32)):
        with matrix.edited(gate, lambda value, change=change: value.update(change)):
            with pytest.raises(ValueError, match="gate"):
                verify(repo)
    log = repo.root / "b8w/s29/shared/train_log.jsonl"
    original = log.read_text()
    train_log(log, steps=(1, *range(100, 1901, 100)))
    with pytest.raises(ValueError, match="2000"):
        verify(repo)
    train_log(log, data_seed=1)
    with pytest.raises(ValueError, match="stream"):
        verify(repo)
    train_log(log, first_loss=3.3)
    with pytest.raises(ValueError, match="first-step"):
        verify(repo)
    log.write_text(original)
    exit_file = Path(json.loads(repo.receipts.read_text().splitlines()[1])["log"] + ".exit")
    exit_file.write_text("1\n")
    with pytest.raises(ValueError, match="exit 0"):
        verify(repo)
    exit_file.write_text("0\n")
    with pytest.raises(ValueError, match="source"):
        maker.verify(repo.root / "b8w", repo.receipts, repo.lock, "c" * 40)
    verify(repo)


def test_the_builder_rejects_arms_that_differ_beyond_name_and_variant(repo):
    with edit_run(repo, "untied-s17", lambda r: r["config"]["train"].update(log_every=7)):
        with pytest.raises(ValueError, match="more than name and variant"):
            verify(repo)
