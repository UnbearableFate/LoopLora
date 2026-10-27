"""TSM runner and analysis scripts on tiny CPU runs (lock, smoke, train, evaluate, audit, analysis).

Run only inside a Miyabi-G compute allocation, including the CPU cases. The training children are
run in-process here (``children`` scripts how each one ends); their process management is exercised
by the smoke job.
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from looplora.sym import tsm, uns
from looplora.sym.config import recursive_merge, resolve_config
from looplora.sym.model import SymbolicModel
from looplora.sym.train import train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_tsm  # noqa: E402
import run_tsm  # noqa: E402

HOPS = [1, 2, 3, 4]
EPISODES = 6


def _config(writer, variant, seed):
    base = {
        "name": "tsm-script-tiny", "device": "cpu", "seed": seed, "writer": writer, "variant": variant,
        "model": {"d_e": 8, "r": 16, "r_mem": 16, "max_hop": 16},
        "task": {"name": "t2_chain_end", "E": 8, "Q": 4, "V": 96, "n_rel": 4, "seen": [0, 72], "unseen": [72, 96],
                 "chain_lengths": [4, 4], "hops": [1, 2, 3]},
        "train": {"steps": 3, "batch_size": 4, "lr": 0.01, "warmup_steps": 0, "depths": [1] if variant == "B4" else [1, 2, 4],
                  "log_every": 1},
        "eval": {"episodes": EPISODES, "batch_size": 3, "splits": ["test"], "depths": [1, 4], "hops": HOPS,
                 "counterfactuals": False},
    }
    return resolve_config(recursive_merge(base, {}))


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


class Exited:
    """A finished child: the in-process stand-in for the handle ``start_child`` returns."""

    def __init__(self, code=0):
        self.code = code

    def poll(self):
        return self.code

    def stop(self):
        pass


class Hanging:
    """A child that is still running until it is stopped."""

    def __init__(self, name, stopped):
        self.name, self.stopped, self.polls = name, stopped, 0

    def poll(self):
        self.polls += 1
        assert self.polls < 10000, f"the running child {self.name} was never stopped"  # not an endless wait
        return None

    def stop(self):
        self.stopped.append(self.name)


def _raise(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def _kill(root, name):
    """Leave a finished run as a job killed by its walltime leaves it: status running, no checkpoint."""
    out = root / "runs" / name
    record = json.loads((out / "record.json").read_text())
    for key in ("finished_at", "seconds", "summary", "checkpoint", "checkpoint_blake2b", "state_hash",
                "peak_memory_bytes"):
        record.pop(key, None)
    (out / "record.json").write_text(json.dumps(dict(record, status="running")))
    (out / "checkpoint.pt").unlink()


def _age(root, name, seconds):
    """Make everything the run wrote look ``seconds`` older, as if its job had ended that long ago."""
    out = root / "runs" / name
    for path in [out, root / "logs" / f"{name}.log", *out.rglob("*")]:
        stat = path.stat()
        os.utime(path, (stat.st_atime - seconds, stat.st_mtime - seconds))


@contextlib.contextmanager
def _edited(path, change):
    """Rewrite a JSON file (``change`` maps its content to the new content) for the duration of the block."""
    original = path.read_text()
    path.write_text(json.dumps(change(json.loads(original))))
    try:
        yield
    finally:
        path.write_text(original)


@pytest.fixture
def children():
    """How the next children end, by run name (``outcomes[name]`` is consumed one entry per start; the default is
    ``complete``): ``nonfinite`` (a non-finite loss), ``gradnorm`` (a non-finite gradient norm), ``fault`` (an I/O
    error), ``oom`` (CUDA out of memory), ``defect`` (any other exception: a code defect), ``killed`` (ended by
    walltime), ``crash`` (dies before it creates its run directory), ``hang`` (still running until it is stopped) or
    ``overlap`` (another job has just created its log, so it cannot be started). ``started`` lists every child that was
    started and ``stopped`` every child that was stopped while it was running."""
    return types.SimpleNamespace(outcomes={}, started=[], stopped=[])


@pytest.fixture
def tiny(tmp_path, monkeypatch, children):
    """Tiny released M1 and B4 checkpoints in the released layout, with the registered constants shrunk to match."""
    monkeypatch.setattr(run_tsm, "require_compute", lambda: None)
    monkeypatch.setattr(run_tsm, "RANK", 16)
    monkeypatch.setattr(run_tsm, "TRAINING", dict(steps=3, lr=0.01, batch_size=4, released_depths=[1, 2, 4],
                                                  t0_depth=2, stay_depth=4, hops=[1, 2, 3]))
    monkeypatch.setattr(run_tsm, "EVALUATION", dict(split="test", seed=999007, episodes=EPISODES, hops=HOPS,
                                                    depths=list(tsm.DEPTHS), batch_size=3))
    monkeypatch.setattr(run_tsm, "SMOKE", dict(steps=2, split="val", seed=4242, episodes=EPISODES, training_seed=17))
    monkeypatch.setattr(run_tsm, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(run_tsm, "QUIET_SECONDS", 0)  # the in-process children end at once; see the test of the guard

    def start_child(name, design_path, root, *, device, steps):
        plan = children.outcomes.get(name, [])
        outcome = plan.pop(0) if plan else "complete"
        (root / "logs").mkdir(parents=True, exist_ok=True)
        if outcome == "overlap":  # another job has just started this run
            (root / "logs" / f"{name}.log").write_text("")
        (root / "logs" / f"{name}.log").open("x").close()  # as the real child: a log never overwrites another
        children.started.append(name)
        if outcome == "hang":
            return Hanging(name, children.stopped)
        if outcome == "crash":
            return Exited(1)
        design = json.loads(Path(design_path).read_text())
        try:
            with pytest.MonkeyPatch.context() as patch:
                error = {"nonfinite": FloatingPointError("non-finite train loss at step 1"),
                         "gradnorm": RuntimeError("The total norm of order 2.0 for gradients from `parameters` is "
                                                  "non-finite, so it cannot be clipped."),
                         "fault": OSError("[Errno 5] Input/output error"),
                         "oom": torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
                         "defect": ValueError("training answer absent from episode candidate set")}.get(outcome)
                if error is not None:
                    patch.setattr(tsm, "train_arm", _raise(error))
                    patch.setattr(run_tsm, "train", _raise(error))
                run_tsm.train_one(design, root, name, device=device, steps=steps)
        except Exception:  # train_one has recorded the failure; the child process would exit with an error
            return Exited(1)
        if outcome == "killed":
            _kill(root, name)
            return Exited(-15)
        return Exited(0)

    monkeypatch.setattr(run_tsm, "start_child", start_child)
    p1s, f2 = tmp_path / "p1s", tmp_path / "f2"
    for writer in run_tsm.WRITERS:
        for variant in ("M1", "B4"):
            for seed in run_tsm.SEEDS:
                config = _config(writer, variant, seed)
                torch.manual_seed(seed)
                path = uns.run_root(writer, variant, seed, p1s_root=p1s, f2_root=f2) / "checkpoint.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"config": config, "model": SymbolicModel(config).state_dict(), "summary": {}}, path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return tmp_path, p1s, f2, repo


def _locked_design(tmp_path, p1s, f2, repo):
    design = tmp_path / "design.json"
    assert run_tsm.main(["lock", "--p1s-root", str(p1s), "--f2-root", str(f2), "--out", str(design)]) == 0
    locked = repo / "reports" / "tsm" / "design.json"
    locked.parent.mkdir(parents=True)
    locked.write_bytes(design.read_bytes())
    _git(repo, "add", "reports/tsm/design.json")
    _git(repo, "commit", "-q", "-m", "lock")
    return locked, _git(repo, "rev-parse", "HEAD")


def test_lock_seals_the_released_configurations_and_b4_references(tiny):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    assert len(design["templates"]) == 6 and len(design["references"]) == 6
    assert design["decision"]["confidence"] == pytest.approx(0.9875) and design["evaluation"]["seed"] == 999007
    run_tsm.verify_design(design)
    with pytest.raises(ValueError):  # a drifted registered setting
        run_tsm.verify_design({**design, "gate_init": [0.0, 3.0]})
    with pytest.raises(ValueError):  # a template whose released training settings differ
        bad = json.loads(json.dumps(design))
        bad["templates"][0]["config"]["train"]["lr"] = 0.5
        run_tsm.verify_design(bad)
    path = Path(design["references"][0]["path"])
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="B4 reference changed"):
        run_tsm.verify_design(design)


def test_arm_configurations_differ_from_the_released_one_only_in_name_depths_and_evaluation(tiny):
    _, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    template = next(t for t in design["templates"] if t["writer"] == "joint_fact" and t["seed"] == 29)["config"]
    configs = {arm: run_tsm.make_config(design, "joint_fact", 29, arm, device="cpu") for arm in tsm.ARMS}
    assert template["train"]["depths"] == [1, 2, 4]  # the released mixed-depth recipe
    for arm, depth in (("T0", 2), ("T1", 4), ("T2", 4)):  # the registered training depths of the tiny fixture
        assert configs[arm]["train"]["depths"] == [depth]
        assert {k: v for k, v in configs[arm]["train"].items() if k != "depths"} == {
            k: v for k, v in template["train"].items() if k != "depths"}
    for arm, config in configs.items():
        assert config["name"] == f"tsm-joint_fact-{arm}-s29" and config["seed"] == 29
        assert config["model"] == template["model"] and config["task"] == template["task"]
        assert config["eval"]["seed"] == 999007 and config["eval"]["splits"] == ["test"]
        assert config["eval"]["hops"] == HOPS and config["eval"]["counterfactuals"] is False
    assert run_tsm.make_config(design, "joint_fact", 29, "T0", device="cpu", steps=2)["train"]["steps"] == 2


def test_t0_is_the_unchanged_training_function(tiny, tmp_path_factory):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design))
    root = tmp_path / "root"
    record = run_tsm.train_one(design, root, "tsm-oracle_random-T0-s17", device="cpu")
    assert record["status"] == "complete" and record["arm"] == "T0" and record["steps"] == 3
    assert record["attempt"] == 1 and set(record["source"]) == {"git_commit", "source_dirty", "source_hash"}
    reference, _ = train(run_tsm.make_config(design, "oracle_random", 17, "T0", device="cpu"),
                         tmp_path_factory.mktemp("reference"))
    model, _, label = tsm.load_tsm(record["checkpoint"], device="cpu")
    assert label == "T0"
    for name, value in reference.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    with pytest.raises(FileExistsError):  # a run directory is never reused
        run_tsm.train_one(design, root, "tsm-oracle_random-T0-s17", device="cpu")


def test_schedule_limits_concurrency_and_stops_everything_on_the_first_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tsm, "POLL_SECONDS", 0.0)
    active, peak, stopped = set(), [0], []

    class Stub:
        def __init__(self, name):
            self.name, self.polls = name, 0

        def poll(self):
            self.polls += 1
            return None if self.polls < 2 else (1 if self.name == "c" else 0)

        def stop(self):
            active.discard(self.name)
            stopped.append(self.name)

    def start(name, design_path, root, *, device, steps):
        active.add(name)
        peak[0] = max(peak[0], len(active))
        return Stub(name)

    monkeypatch.setattr(run_tsm, "start_child", start)
    assert run_tsm.schedule(["a", "b"], 2, tmp_path / "d.json", tmp_path, device="cpu") == {"a": 0, "b": 0}
    assert peak[0] == 2 and sorted(stopped) == ["a", "b"] and not active
    active.clear()
    peak[0] = 0
    with pytest.raises(ValueError, match="run c exited 1"):
        run_tsm.schedule(["a", "b", "c", "d"], 2, tmp_path / "d.json", tmp_path, device="cpu")
    assert peak[0] <= 2 and not active  # every started child was stopped
    # with a fatal predicate only the failures it names stop the others; the rest are returned
    codes = run_tsm.schedule(["a", "b", "c", "d"], 2, tmp_path / "d.json", tmp_path, device="cpu",
                             fatal=lambda name: name == "d")
    assert codes == {"a": 0, "b": 0, "c": 1, "d": 0} and not active
    with pytest.raises(ValueError, match="run c exited 1"):
        run_tsm.schedule(["a", "b", "c", "d"], 2, tmp_path / "d.json", tmp_path, device="cpu",
                         fatal=lambda name: name == "c")
    assert not active

    def broken(name, design_path, root, *, device, steps):
        if name == "c":
            raise FileExistsError("the log exists")  # another job runs it
        return start(name, design_path, root, device=device, steps=steps)

    monkeypatch.setattr(run_tsm, "start_child", broken)
    stopped.clear()
    with pytest.raises(ValueError, match="run c could not be started: the log exists"):
        run_tsm.schedule(["a", "b", "c"], 3, tmp_path / "d.json", tmp_path, device="cpu")
    assert sorted(stopped) == ["a", "b"] and not active  # the children that were running are stopped


def test_pipeline_runs_end_to_end_and_the_analysis_audits_every_seal(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)

    smoke = tmp_path / "smoke"
    assert run_tsm.main(["smoke", "--design", str(locked), "--root", str(smoke), "--procs", "2",
                         "--device", "cpu"]) == 0
    report = json.loads((smoke / "smoke.json").read_text())
    assert report["status"] == "passed" and len(report["train_step_seconds"]) == 6
    assert len(report["endpoints"]) == 4 and len(report["evaluation_rows"]) == 8 and len(report["gate_profiles"]) == 2
    assert report["suggested_walltime_seconds"] > 0 and report["extrapolated_total_run_seconds"] > 0
    assert all(row["steps"] == 2 for row in (json.loads((smoke / "runs" / n / "record.json").read_text())
                                              for n in report["train_step_seconds"]))

    root = tmp_path / "run"
    for writer, label in (("oracle_random", "a"), ("joint_fact", "b")):
        assert run_tsm.main(["train", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                             "--procs", "3", "--label", label, "--writers", writer, "--device", "cpu"]) == 0
    manifests = [json.loads((root / f"train_manifest-{label}.json").read_text()) for label in ("a", "b")]
    assert all(m["status"] == "complete" and len(m["names"]) == 9 and m["training_failures"] == []
               and m["reruns"] == [] and m["unfinished"] == [] for m in manifests)
    assert len(list((root / "runs").glob("*/checkpoint.pt"))) == 18 and not (root / "attempts").exists()
    started = list(children.started)
    with pytest.raises(ValueError, match="needs its own --label"):  # a manifest is never overwritten
        run_tsm.main(["train", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                      "--label", "a", "--writers", "joint_fact", "--arms", "T0", "--seeds", "17", "--device", "cpu"])
    assert run_tsm.main(["train", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                         "--procs", "1", "--label", "again", "--writers", "joint_fact", "--arms", "T0",
                         "--seeds", "17", "--device", "cpu"]) == 0
    again = json.loads((root / "train_manifest-again.json").read_text())
    assert children.started == started  # a complete run is never trained again
    assert again["status"] == "complete" and again["started"] == [] and again["final_before"] == again["names"]
    locked.write_text(locked.read_text() + " ")
    with pytest.raises(ValueError, match="differs from its Git lock"):
        run_tsm.main(["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                      "--device", "cpu"])
    _git(repo, "checkout", "-q", "--", "reports/tsm/design.json")
    assert run_tsm.main(["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                         "--device", "cpu"]) == 0
    with pytest.raises(ValueError, match="already started"):
        run_tsm.main(["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root),
                      "--device", "cpu"])
    manifest = json.loads((root / "evaluation" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["evaluations"]) == 24
    assert manifest["training_failures"] == []
    assert manifest["attempt"] == 1 and manifest["restarts"] == [] and not (root / "attempts").exists()
    assert manifest["checkpoints"] == {name: run_tsm.read_record(root, name)["checkpoint_blake2b"]
                                       for name in sorted(manifest["checkpoints"])} and len(manifest["checkpoints"]) == 18
    assert len(list((root / "evaluation" / "samples").glob("*.npz"))) == 24
    assert len(json.loads((root / "evaluation" / "gate_profiles.json").read_text())) == 6

    out = tmp_path / "analysis"
    assert analyze_tsm.main(["--root", str(root), "--out", str(out)]) == 0
    with (out / "endpoints.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and {(r["writer"], r["arm"]) for r in rows} == {
        (w, a) for w in run_tsm.WRITERS for a in ("T1", "T2")}
    assert all(float(r["confidence"]) == pytest.approx(0.9875) and r["passes"] in {"True", "False"} for r in rows)
    assert all(r["training_failed"] == "False" and r["completed_seeds"] == "17;29;43" and r["failed_seeds"] == ""
               and r["stay_difference"] != "" for r in rows)
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["arms_pass"]) == {"T1", "T2"} and summary["training_failures"] == []
    assert summary["failed_cells"] == []
    assert summary["training_source_hash"] == run_tsm.read_record(root, NAME)["source"]["source_hash"]
    assert summary["evaluation_attempt"] == 1 and summary["evaluation_restarts"] == []
    for arm, passed in summary["arms_pass"].items():  # an arm passes only if it passes for both writers
        assert passed == all(summary["writer_passes"][f"{w}/{arm}"] for w in run_tsm.WRITERS)
        assert (summary["arm_outcome"][arm] == "both") == passed
    assert "Registered conditions" in (out / "summary.md").read_text()
    assert "## Training failures" not in (out / "summary.md").read_text()
    assert all(f"- {writer}: " in (out / "summary.md").read_text() for writer in run_tsm.WRITERS)  # per-writer reading
    with (out / "training.csv").open(newline="") as stream:
        training = list(csv.DictReader(stream))
    assert len(training) == 18 and {r["status"] for r in training} == {"complete"}
    assert {r["attempt"] for r in training} == {"1"}

    victim = next((root / "evaluation" / "samples").glob("*.npz"))
    original = victim.read_bytes()
    victim.write_bytes(original[:-4] + b"zzzz")
    with pytest.raises(ValueError, match="samples missing or changed"):
        analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])
    victim.write_bytes(original)
    with _edited(root / "evaluation" / "gate_profiles.json", lambda p: dict(list(p.items())[1:])):
        with pytest.raises(ValueError, match="gate profiles do not match"):  # a completed T2 run has no profile
            analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])
    checkpoint = next((root / "runs").glob("*/checkpoint.pt"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checkpoint missing or changed"):
        analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])


NAME = "tsm-joint_fact-T1-s17"


def _train_argv(locked, commit, root, label, *extra):
    return ["train", "--design", str(locked), "--design-commit", commit, "--root", str(root), "--procs", "3",
            "--label", label, "--device", "cpu", *extra]


def _evaluate_argv(locked, commit, root, *extra):
    return ["evaluate", "--design", str(locked), "--design-commit", commit, "--root", str(root), "--device", "cpu",
            *extra]


def _one_run(locked, commit, root, label, *extra):
    return run_tsm.main(_train_argv(locked, commit, root, label, "--writers", "joint_fact", "--arms", "T1",
                                    "--seeds", "17", *extra))


@pytest.mark.parametrize("record, expected", [
    (None, "incomplete"),  # a run directory whose record was never written
    ('{"status": "comp', "incomplete"),  # a record cut off while it was being written
    ({"status": "running"}, "incomplete"),
    ({"status": "complete"}, "complete"),
    ({"status": "failed", "failure_kind": "numerical",
      "failure": "FloatingPointError: non-finite train loss at step 12"}, "training_failure"),
    ({"status": "failed", "failure_kind": "numerical",
      "failure": "RuntimeError: The total norm of order 2.0 for gradients from `parameters` is non-finite, so it "
                 "cannot be clipped."}, "training_failure"),
    ({"status": "failed", "failure_kind": "infrastructure", "failure": "OSError: [Errno 5] Input/output error"},
     "incomplete"),
    ({"status": "failed", "failure_kind": "infrastructure", "failure": "OutOfMemoryError: CUDA out of memory"},
     "incomplete"),
    ({"status": "failed", "failure_kind": "code_defect",
      "failure": "ValueError: training answer absent from episode candidate set"}, "code_defect"),
    ({"status": "failed", "failure": "ValueError: a record that says nothing of its kind"}, "code_defect"),
])
def test_run_state_separates_registered_training_failures_from_unfinished_runs(tmp_path, record, expected):
    assert run_tsm.run_state(tmp_path, NAME) == "missing"
    out = tmp_path / "runs" / NAME
    out.mkdir(parents=True)
    if record is not None:
        (out / "record.json").write_text(record if isinstance(record, str) else json.dumps(record))
    assert run_tsm.run_state(tmp_path, NAME) == expected


@pytest.mark.parametrize("make_error, expected", [
    (lambda: FloatingPointError("non-finite train loss at step 12"), "numerical"),
    (lambda: RuntimeError("The total norm of order 2.0 for gradients from `parameters` is non-finite, so it cannot be "
                          "clipped."), "numerical"),
    (lambda: OSError(5, "Input/output error"), "infrastructure"),
    (lambda: FileNotFoundError("checkpoint.pt"), "infrastructure"),
    (lambda: torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), "infrastructure"),
    (lambda: RuntimeError("CUDA error: out of memory"), "infrastructure"),
    (lambda: KeyboardInterrupt(), "infrastructure"),  # an interruption is not an exception of the code
    (lambda: ValueError("training answer absent from episode candidate set"), "code_defect"),
    (lambda: KeyError("hop"), "code_defect"),
    (lambda: AssertionError(), "code_defect"),
    (lambda: RuntimeError("mat1 and mat2 shapes cannot be multiplied (4x8 and 16x8)"), "code_defect"),
    (lambda: RuntimeError("CUDA error: an illegal memory access was encountered"), "code_defect"),
    (lambda: RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 1099511627776 bytes"),
     "code_defect"),  # host memory is not CUDA out-of-memory
    (lambda: RuntimeError("out of memory while allocating the host buffer"), "code_defect"),  # not CUDA
    (lambda: MemoryError(), "code_defect"),
    (lambda: ValueError("non-finite values in the batch"), "code_defect"),  # not the error of the gradient clipping
])
def test_a_failure_is_read_as_the_registration_reads_it(make_error, expected):
    assert run_tsm.classify_failure(make_error()) == expected


def test_an_attempt_must_be_identical_to_the_earlier_one_and_within_three_attempts(tmp_path):
    source = dict(git_commit="a" * 40, source_dirty=False, source_hash="h1")
    first_run = run_tsm.check_rerun(tmp_path, NAME, config_hash="c9", source=dict(source, source_hash="h9"))
    assert first_run == 1  # there is nothing to match yet
    out = tmp_path / "runs" / NAME
    out.mkdir(parents=True)
    (out / "record.json").write_text(json.dumps(dict(status="running", config_hash="c1", source=source)))
    assert run_tsm.check_rerun(tmp_path, NAME, config_hash="c1", source=source) == 2  # the unfinished run is attempt 1
    with pytest.raises(ValueError, match="configuration differs"):
        run_tsm.check_rerun(tmp_path, NAME, config_hash="c2", source=source)
    with pytest.raises(ValueError, match="code differs"):
        run_tsm.check_rerun(tmp_path, NAME, config_hash="c1", source=dict(source, source_hash="h2"))
    # a rerun moved it aside and stopped before its child started: it is still the reference for the next attempt
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    out.rename(attempts / f"{NAME}-attempt1")
    assert run_tsm.check_rerun(tmp_path, NAME, config_hash="c1", source=source) == 2
    with pytest.raises(ValueError, match="code differs"):
        run_tsm.check_rerun(tmp_path, NAME, config_hash="c1", source=dict(source, source_hash="h2"))
    (attempts / f"{NAME}-attempt2").mkdir()  # an attempt that left no record does not hide the first one
    assert run_tsm.check_rerun(tmp_path, NAME, config_hash="c1", source=source) == 3
    with pytest.raises(ValueError, match="configuration differs"):
        run_tsm.check_rerun(tmp_path, NAME, config_hash="c2", source=source)
    (attempts / f"{NAME}-attempt3").mkdir()  # three attempts are used: there is no fourth
    with pytest.raises(ValueError, match="used 3 of 3 attempts"):
        run_tsm.check_rerun(tmp_path, NAME, config_hash="c1", source=source)


def test_an_infrastructure_failure_is_trained_again_identically_after_the_job_has_ended(tiny, children, monkeypatch):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    children.outcomes[NAME] = ["killed", "fault", "complete"]
    with pytest.raises(ValueError, match="did not finish"):  # attempt 1: the job ended before the run did
        _one_run(locked, commit, root, "a")
    first = json.loads((root / "train_manifest-a.json").read_text())
    assert first["status"] == "incomplete" and first["unfinished"] == [NAME] and first["training_failures"] == []
    with pytest.raises(ValueError, match="--rerun-incomplete"):  # a rerun is deliberate: its job must have ended
        _one_run(locked, commit, root, "b")
    assert children.started == [NAME] and not (root / "train_manifest-b.json").exists()
    real_source = run_tsm.source_metadata
    monkeypatch.setattr(run_tsm, "source_metadata", lambda: dict(real_source(), source_hash="another-code-version"))
    with pytest.raises(ValueError, match="code differs"):  # identical code is required, not only allowed
        _one_run(locked, commit, root, "x", "--rerun-incomplete")
    monkeypatch.setattr(run_tsm, "source_metadata", real_source)
    assert children.started == [NAME] and not (root / "attempts").exists()  # nothing moved, nothing started
    assert not (root / "train_manifest-x.json").exists()
    with pytest.raises(ValueError, match="did not finish"):  # attempt 2: an I/O fault is not a registered failure
        _one_run(locked, commit, root, "c", "--rerun-incomplete")
    attempts = root / "attempts"
    assert (attempts / f"{NAME}-attempt1").is_dir() and (attempts / f"{NAME}-attempt1.log").is_file()
    assert json.loads((attempts / f"{NAME}-attempt1" / "record.json").read_text())["status"] == "running"
    assert run_tsm.run_state(root, NAME) == "incomplete"
    assert _one_run(locked, commit, root, "d", "--rerun-incomplete") == 0  # attempt 3 completes
    assert (attempts / f"{NAME}-attempt2.log").is_file() and len(run_tsm.moved_attempts(root, NAME)) == 2
    final = run_tsm.read_record(root, NAME)
    assert final["status"] == "complete" and final["attempt"] == 3
    for number in (1, 2):  # every attempt is the same configuration on the same code
        earlier = json.loads((attempts / f"{NAME}-attempt{number}" / "record.json").read_text())
        assert earlier["attempt"] == number and earlier["config_hash"] == final["config_hash"]
        assert earlier["source"]["source_hash"] == final["source"]["source_hash"]
    assert json.loads((root / "train_manifest-d.json").read_text())["reruns"] == [NAME]


def test_a_child_that_died_before_creating_its_run_directory_has_used_no_attempt(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    children.outcomes[NAME] = ["crash", "complete"]
    with pytest.raises(ValueError, match="did not finish"):
        _one_run(locked, commit, root, "a")
    assert run_tsm.run_state(root, NAME) == "missing" and (root / "logs" / f"{NAME}.log").is_file()
    assert _one_run(locked, commit, root, "b") == 0  # nothing was trained: no --rerun-incomplete is needed
    assert (root / "attempts" / f"{NAME}-notstarted1.log").is_file() and not run_tsm.moved_attempts(root, NAME)
    assert run_tsm.read_record(root, NAME)["attempt"] == 1


def test_a_run_that_never_finishes_is_refused_after_three_attempts(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    children.outcomes[NAME] = ["killed"] * 3 + ["complete"]
    with pytest.raises(ValueError, match="did not finish"):
        _one_run(locked, commit, root, "a")
    for label in ("b", "c"):  # the first run and two identical reruns
        with pytest.raises(ValueError, match="did not finish"):
            _one_run(locked, commit, root, label, "--rerun-incomplete")
    with pytest.raises(ValueError, match="used 3 of 3 attempts"):
        _one_run(locked, commit, root, "d", "--rerun-incomplete")
    assert children.started == [NAME] * 3
    assert run_tsm.run_state(root, NAME) == "incomplete" and len(run_tsm.moved_attempts(root, NAME)) == 2


def test_an_aborted_rerun_does_not_loosen_the_identity_rule(tiny, children, monkeypatch):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    children.outcomes[NAME] = ["killed", "crash", "complete"]
    with pytest.raises(ValueError, match="did not finish"):
        _one_run(locked, commit, root, "a")
    with pytest.raises(ValueError, match="did not finish"):  # the rerun moves attempt 1 aside; its child dies at once
        _one_run(locked, commit, root, "b", "--rerun-incomplete")
    assert run_tsm.run_state(root, NAME) == "missing" and len(run_tsm.moved_attempts(root, NAME)) == 1
    real_source = run_tsm.source_metadata
    monkeypatch.setattr(run_tsm, "source_metadata", lambda: dict(real_source(), source_hash="another-code-version"))
    with pytest.raises(ValueError, match="code differs"):  # the run is missing now, but attempt 2 is still a rerun
        _one_run(locked, commit, root, "c")
    assert children.started == [NAME, NAME] and not (root / "train_manifest-c.json").exists()
    monkeypatch.setattr(run_tsm, "source_metadata", real_source)
    assert _one_run(locked, commit, root, "d") == 0  # the identical rerun needs no flag: nothing is running
    final = run_tsm.read_record(root, NAME)
    assert final["status"] == "complete" and final["attempt"] == 2
    assert json.loads((root / "train_manifest-d.json").read_text())["reruns"] == [NAME]


def test_a_run_that_may_still_be_running_is_not_moved_aside(tiny, children, monkeypatch):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    killed, faulted = "tsm-joint_fact-T1-s17", "tsm-joint_fact-T1-s29"
    children.outcomes[killed], children.outcomes[faulted] = ["killed", "complete"], ["fault", "complete"]

    def both(label, *extra):
        return run_tsm.main(_train_argv(locked, commit, root, label, "--writers", "joint_fact", "--arms", "T1",
                                        "--seeds", "17", "29", *extra))

    with pytest.raises(ValueError, match="did not finish"):
        both("a")
    monkeypatch.setattr(run_tsm, "QUIET_SECONDS", 3600)  # a live run writes to its directory every few seconds
    with pytest.raises(ValueError, match=f"run {killed} was written to .* its job may still be running"):
        both("b", "--rerun-incomplete")
    assert children.started == [killed, faulted] and not (root / "attempts").exists()  # nothing moved or started
    assert not (root / "train_manifest-b.json").exists()
    _age(root, killed, 7200)  # its job ended two hours ago; the failed run recorded its failure and is gone at once
    out = root / "runs" / killed
    for fresh in (out / "train_log.jsonl", root / "logs" / f"{killed}.log"):  # a live run appends to a file, which
        folder = out.stat().st_mtime  # leaves the directory old: any fresh file shows that the run is still alive
        fresh.touch()
        os.utime(out, (folder, folder))
        with pytest.raises(ValueError, match=f"run {killed} was written to .* its job may still be running"):
            both("c", "--rerun-incomplete")
        _age(root, killed, 7200)
    assert children.started == [killed, faulted] and not (root / "attempts").exists()
    assert both("d", "--rerun-incomplete") == 0
    assert [run_tsm.read_record(root, name)["attempt"] for name in (killed, faulted)] == [2, 2]


def test_a_child_that_cannot_be_started_fails_the_job_cleanly(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    children.outcomes[NAME] = ["overlap"]
    with pytest.raises(ValueError, match=f"run {NAME} could not be started"):  # its log appeared: another job runs it
        _one_run(locked, commit, root, "a")
    manifest = json.loads((root / "train_manifest-a.json").read_text())
    assert manifest["status"] == "failed" and f"run {NAME} could not be started" in manifest["reason"]
    assert children.started == [] and run_tsm.run_state(root, NAME) == "missing"


def test_train_one_refuses_a_fourth_attempt_and_creates_nothing(tiny):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    root = tmp_path / "root"
    for number in (1, 2, 3):
        (root / "attempts" / f"{NAME}-attempt{number}").mkdir(parents=True)
    with pytest.raises(ValueError, match="used 3 of 3 attempts"):
        run_tsm.train_one(design, root, NAME, device="cpu")
    assert not (root / "runs").exists()


def test_an_error_anywhere_after_the_run_directory_exists_is_recorded_with_its_kind(tiny, monkeypatch):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    root = tmp_path / "root"
    with monkeypatch.context() as patch:  # in the seals that follow a completed training: a code defect
        patch.setattr(run_tsm, "state_hash", _raise(KeyError("model")))
        with pytest.raises(KeyError):
            run_tsm.train_one(design, root, NAME, device="cpu")
    record = run_tsm.read_record(root, NAME)
    assert record["status"] == "failed" and record["failure_kind"] == "code_defect"
    assert record["failure"].startswith("KeyError") and "checkpoint_blake2b" not in record
    assert run_tsm.run_state(root, NAME) == "code_defect"
    other = "tsm-joint_fact-T1-s29"
    with monkeypatch.context() as patch:  # in the setup, before training starts: an I/O error
        patch.setattr(run_tsm, "runtime_metadata", _raise(OSError(5, "Input/output error")))
        with pytest.raises(OSError):
            run_tsm.train_one(design, root, other, device="cpu")
    record = run_tsm.read_record(root, other)
    assert record["status"] == "failed" and record["failure_kind"] == "infrastructure" and record["attempt"] == 1
    assert run_tsm.run_state(root, other) == "incomplete"


def test_the_evaluation_itself_refuses_a_checkpoint_that_changed_after_training(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    assert run_tsm.main(_train_argv(locked, commit, root, "a", "--writers", "joint_fact", "--seeds", "17")) == 0
    design = json.loads(locked.read_text())
    arguments = dict(writers=["joint_fact"], seeds=[17], evaluation=design["evaluation"], device="cpu", out=None)
    _, rows, profiles = run_tsm.evaluate_runs(design, root, **arguments)
    assert len(rows) == 4 and list(profiles) == ["tsm-joint_fact-T2-s17"]  # T0, T1, T2 and the B4 reference
    checkpoint = root / "runs" / NAME / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    with pytest.raises(ValueError, match=f"checkpoint changed: {NAME}"):
        run_tsm.evaluate_runs(design, root, **arguments)


def test_nonfinite_loss_fails_that_arm_for_that_writer_and_the_other_arms_are_evaluated(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    partial = "tsm-joint_fact-T1-s29"  # one seed of an arm
    whole = [f"tsm-oracle_random-T2-s{seed}" for seed in run_tsm.SEEDS]  # every seed of an arm
    for name in [partial, *whole]:
        children.outcomes[name] = ["nonfinite"]
    assert run_tsm.main(_train_argv(locked, commit, root, "a")) == 0  # a registered failure does not fail the job
    manifest = json.loads((root / "train_manifest-a.json").read_text())
    assert manifest["status"] == "complete" and manifest["unfinished"] == []
    assert {row["name"] for row in manifest["training_failures"]} == {partial, *whole}
    assert all(row["failure"].startswith("FloatingPointError: non-finite") for row in manifest["training_failures"])
    assert run_tsm.run_state(root, partial) == "training_failure"
    started = list(children.started)  # no retry and no re-tuning, not even with a deliberate rerun
    assert run_tsm.main(_train_argv(locked, commit, root, "b", "--rerun-incomplete")) == 0
    assert children.started == started and not (root / "attempts").exists()

    assert run_tsm.main(_evaluate_argv(locked, commit, root)) == 0
    evaluation = json.loads((root / "evaluation" / "manifest.json").read_text())
    assert len(evaluation["evaluations"]) == 20  # the 14 completed runs and the 6 B4 references
    assert {row["name"] for row in evaluation["training_failures"]} == {partial, *whole}
    assert not any((root / "evaluation" / "samples" / f"{name}.npz").exists() for name in [partial, *whole])
    assert len(json.loads((root / "evaluation" / "gate_profiles.json").read_text())) == 3  # joint_fact T2 only

    out = tmp_path / "analysis"
    assert analyze_tsm.main(["--root", str(root), "--out", str(out)]) == 0
    with (out / "endpoints.csv").open(newline="") as stream:
        rows = {(r["writer"], r["arm"]): r for r in csv.DictReader(stream)}
    for key, seeds, lost in ((("joint_fact", "T1"), "17;43", "29"), (("oracle_random", "T2"), "", "17;29;43")):
        cell = rows[key]
        assert cell["training_failed"] == "True" and cell["passes"] == "False"
        assert cell["completed_seeds"] == seeds and cell["failed_seeds"] == lost
        assert cell["stay_difference"] == "" and cell["stay_passes"] == ""  # no registered endpoint for that arm
    assert all(row["training_failed"] == "False" and row["stay_difference"] != ""
               for key, row in rows.items() if key not in {("joint_fact", "T1"), ("oracle_random", "T2")})
    with (out / "descriptive_contrasts.csv").open(newline="") as stream:
        descriptive = list(csv.DictReader(stream))
    cell = [r for r in descriptive if (r["writer"], r["arm"]) == ("joint_fact", "T1")]
    assert {"stay", "far", "gain"} <= {r["contrast"] for r in cell}  # reported, not registered
    assert {r["seeds"] for r in cell} == {"17;43"} and {r["training_failed"] for r in cell} == {"True"}
    assert not [r for r in descriptive if (r["writer"], r["arm"]) == ("oracle_random", "T2")]  # nothing to report
    summary = json.loads((out / "summary.json").read_text())
    assert summary["failed_cells"] == ["oracle_random/T2", "joint_fact/T1"]
    assert summary["arms_pass"] == {"T1": False, "T2": False}
    assert "joint_fact" not in summary["passing_writers"]["T1"] and "oracle_random" not in summary["passing_writers"]["T2"]
    assert {row["name"] for row in summary["training_failures"]} == {partial, *whole}
    text = (out / "summary.md").read_text()
    assert "## Training failures" in text and partial in text and whole[0] in text
    reading = {line.split(":")[0]: line for line in text.splitlines() if line.startswith("- ")}  # per-writer reading
    assert reading["- oracle_random"].endswith("training failure: T2") and reading["- joint_fact"].endswith(
        "training failure: T1")
    with (out / "training.csv").open(newline="") as stream:
        training = {r["name"]: r for r in csv.DictReader(stream)}
    assert len(training) == 18
    for name in [partial, *whole]:
        assert training[name]["status"] == "failed" and training[name]["attempt"] == "1"
        assert training[name]["failure"].startswith("FloatingPointError") and training[name]["final_loss"] == ""
    assert {r["status"] for name, r in training.items() if name not in {partial, *whole}} == {"complete"}

    path = root / "runs" / partial / "record.json"
    original = path.read_text()
    record = json.loads(original)
    path.write_text(json.dumps(dict(record, failure="FloatingPointError: non-finite train loss at step 99")))
    with pytest.raises(ValueError, match="training failures differ"):  # the manifest is checked against the records
        analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])
    path.write_text(json.dumps(dict(record, failure="OSError: [Errno 5] Input/output error",
                                    failure_kind="infrastructure")))
    with pytest.raises(ValueError, match=f"run {partial} is incomplete"):  # not a registered failure any more
        analyze_tsm.main(["--root", str(root), "--out", str(tmp_path / "again")])
    path.write_text(original)


def test_nonfinite_loss_in_t0_stops_the_study_before_the_confirmation_set_is_opened(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    failed, sibling = "tsm-oracle_random-T0-s17", "tsm-oracle_random-T0-s29"
    children.outcomes[failed], children.outcomes[sibling] = ["nonfinite"], ["hang"]  # the sibling is still training
    with pytest.raises(ValueError, match="the study stops"):
        run_tsm.main(_train_argv(locked, commit, root, "a"))
    stopped = json.loads((root / "train_manifest-a.json").read_text())
    assert stopped["status"] == "stopped" and failed in stopped["reason"]
    assert children.stopped == [sibling]  # a run that is still training when the study stops is stopped
    assert "tsm-joint_fact-T2-s43" not in children.started  # nothing more is started once the study has stopped
    with pytest.raises(ValueError, match="the study stops"):  # no later job starts anything either
        run_tsm.main(_train_argv(locked, commit, root, "b", "--rerun-incomplete"))
    assert not (root / "train_manifest-b.json").exists()
    with pytest.raises(ValueError, match="the study stops"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    assert not (root / "evaluation").exists()  # the confirmation set was never touched


@pytest.mark.parametrize("failed", ["tsm-oracle_random-T0-s17", "tsm-oracle_random-T1-s17", "tsm-oracle_random-T2-s17"])
def test_a_code_defect_in_any_run_stops_the_study_before_the_confirmation_set_is_opened(tiny, children, failed):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    sibling = failed.replace("-s17", "-s29")  # started in the same batch: still training when the first one ends
    children.outcomes[failed], children.outcomes[sibling] = ["defect"], ["hang"]
    with pytest.raises(ValueError, match=r"ended with a code defect \(ValueError: training answer absent.*the study stops"):
        run_tsm.main(_train_argv(locked, commit, root, "a"))
    record = run_tsm.read_record(root, failed)
    assert record["status"] == "failed" and record["failure_kind"] == "code_defect"
    assert run_tsm.run_state(root, failed) == "code_defect"
    stopped = json.loads((root / "train_manifest-a.json").read_text())
    assert stopped["status"] == "stopped" and failed in stopped["reason"] and "code defect" in stopped["reason"]
    assert children.stopped == [sibling]  # the run that is still training is stopped
    order = [row["name"] for row in run_tsm.planned_runs()]
    assert children.started == order[:order.index(failed) + 3]  # nothing is started once the study has stopped
    for extra in ([], ["--writers", "joint_fact", "--arms", "T0", "--seeds", "43"]):  # any selection, with or without
        with pytest.raises(ValueError, match="the study stops"):  # the flag: the defect is the code's, not a run's
            run_tsm.main(_train_argv(locked, commit, root, f"b{len(extra)}", "--rerun-incomplete", *extra))
        assert not (root / f"train_manifest-b{len(extra)}.json").exists()
    with pytest.raises(ValueError, match="code defect.*the study stops"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    assert not (root / "evaluation").exists()  # the confirmation set was never touched
    assert children.started == order[:order.index(failed) + 3]


def test_a_code_defect_is_reported_before_a_t0_failure_and_an_unfinished_run(tiny):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    root = tmp_path / "run"
    numerical = "FloatingPointError: non-finite train loss at step 4"
    for name, record in (("tsm-oracle_random-T0-s17", dict(status="running")),
                         ("tsm-oracle_random-T0-s29", dict(status="failed", failure=numerical, failure_kind="numerical")),
                         ("tsm-joint_fact-T2-s43", dict(status="failed", failure="KeyError: 'hop'",
                                                        failure_kind="code_defect"))):
        (root / "runs" / name).mkdir(parents=True)
        (root / "runs" / name / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match=r"run tsm-joint_fact-T2-s43 ended with a code defect \(KeyError: 'hop'\)"):
        run_tsm.final_states(design, root)
    assert run_tsm.stops_study(root, "tsm-joint_fact-T2-s43") and run_tsm.stops_study(root, "tsm-oracle_random-T0-s29")
    assert not run_tsm.stops_study(root, "tsm-oracle_random-T0-s17")  # unfinished: only a rerun is needed


def test_cuda_out_of_memory_and_io_errors_are_infrastructure_failures_that_do_not_stop_the_study(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    memory, fault = "tsm-oracle_random-T0-s17", "tsm-joint_fact-T2-s43"  # even a T0 run is only rerun
    children.outcomes[memory], children.outcomes[fault] = ["oom", "complete"], ["fault", "oom", "complete"]
    with pytest.raises(ValueError, match="did not finish"):
        run_tsm.main(_train_argv(locked, commit, root, "a"))
    first = json.loads((root / "train_manifest-a.json").read_text())
    assert first["status"] == "incomplete" and sorted(first["unfinished"]) == sorted([memory, fault])
    assert len(list((root / "runs").glob("*/checkpoint.pt"))) == 16  # the other runs went on
    for name, text in ((memory, "OutOfMemoryError: CUDA out of memory"), (fault, "OSError: [Errno 5]")):
        record = run_tsm.read_record(root, name)
        assert record["failure_kind"] == "infrastructure" and record["failure"].startswith(text)
        assert run_tsm.run_state(root, name) == "incomplete"
    assert run_tsm.study_stop(root) is None
    with pytest.raises(ValueError, match=f"run {memory} is incomplete"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    with pytest.raises(ValueError, match="did not finish"):  # attempt 2: the second run runs out of memory again
        run_tsm.main(_train_argv(locked, commit, root, "b", "--rerun-incomplete"))
    assert run_tsm.read_record(root, memory)["attempt"] == 2 and run_tsm.run_state(root, memory) == "complete"
    assert run_tsm.run_state(root, fault) == "incomplete" and run_tsm.study_stop(root) is None
    assert run_tsm.main(_train_argv(locked, commit, root, "c", "--rerun-incomplete")) == 0  # attempt 3
    assert run_tsm.read_record(root, fault)["attempt"] == 3
    assert run_tsm.main(_evaluate_argv(locked, commit, root)) == 0


def test_evaluate_needs_every_run_complete_or_a_registered_failure(tiny, children):
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    killed = "tsm-oracle_random-T2-s43"
    children.outcomes[killed] = ["killed"]
    with pytest.raises(ValueError, match="did not finish"):
        run_tsm.main(_train_argv(locked, commit, root, "a"))
    with pytest.raises(ValueError, match=f"run {killed} is incomplete"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    assert not (root / "evaluation").exists()  # the confirmation set is still closed
    assert run_tsm.main(_train_argv(locked, commit, root, "b", "--rerun-incomplete")) == 0
    assert children.started.count(killed) == 2 and len(children.started) == 19  # only the unfinished run again
    assert run_tsm.read_record(root, killed)["attempt"] == 2
    record = root / "runs" / killed / "record.json"
    for change, message in (  # each is refused before the confirmation set is opened
            (lambda r: dict(r, steps=99), "not a registered run"),
            (lambda r: dict(r, config_hash="0" * 32), "not a registered run"),
            (lambda r: dict(r, source=dict(r["source"], source_hash="another-code-version")), "same code")):
        with _edited(record, change):
            with pytest.raises(ValueError, match=message):
                run_tsm.main(_evaluate_argv(locked, commit, root))
        assert not (root / "evaluation").exists()
    started = list(children.started)  # the training stage applies the same test to a final record, at its start
    with _edited(record, lambda r: dict(r, steps=99)):
        with pytest.raises(ValueError, match="not a registered run"):
            run_tsm.main(_train_argv(locked, commit, root, "c", "--writers", "oracle_random", "--arms", "T2",
                                     "--seeds", "43"))
    assert children.started == started and not (root / "train_manifest-c.json").exists()
    checkpoint = root / "runs" / killed / "checkpoint.pt"
    sealed = checkpoint.read_bytes()
    checkpoint.write_bytes(sealed + b"x")
    with pytest.raises(ValueError, match="checkpoint missing or changed"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    checkpoint.unlink()
    with pytest.raises(ValueError, match="checkpoint missing or changed"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    checkpoint.write_bytes(sealed)
    assert not (root / "evaluation").exists()
    assert run_tsm.main(_evaluate_argv(locked, commit, root)) == 0
    evaluation = json.loads((root / "evaluation" / "manifest.json").read_text())
    assert len(evaluation["evaluations"]) == 24 and evaluation["training_failures"] == []


def test_an_unfinished_run_earlier_in_the_plan_does_not_hide_a_t0_failure(tiny):
    tmp_path, p1s, f2, _ = tiny
    design = run_tsm.build_design(p1s, f2)
    root = tmp_path / "run"
    failure = "FloatingPointError: non-finite train loss at step 4"
    for name, record in (("tsm-oracle_random-T0-s17", dict(status="running")),
                         ("tsm-oracle_random-T0-s29", dict(status="failed", failure=failure, failure_kind="numerical"))):
        (root / "runs" / name).mkdir(parents=True)
        (root / "runs" / name / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="the study stops"):
        run_tsm.final_states(design, root)


def _trained(tiny):
    """A locked design and a root with all 18 runs trained (three steps each); returns the design, its commit, the
    root and the repository that holds the lock."""
    tmp_path, p1s, f2, repo = tiny
    locked, commit = _locked_design(tmp_path, p1s, f2, repo)
    root = tmp_path / "run"
    assert run_tsm.main(_train_argv(locked, commit, root, "a")) == 0
    return locked, commit, root, repo


def _interrupting(monkeypatch):
    """Make the evaluations that follow write the samples of one writer and seed and then end as a job with an I/O
    fault ends. Returns the real ``evaluate_runs`` for the attempt that is to complete."""
    real = run_tsm.evaluate_runs

    def partial(design, root, **kwargs):
        real(design, root, **dict(kwargs, writers=kwargs["writers"][:1], seeds=kwargs["seeds"][:1]))
        raise OSError("[Errno 5] Input/output error")

    monkeypatch.setattr(run_tsm, "evaluate_runs", partial)
    return real


def _killed(root):
    """Leave the evaluation as a job killed by its walltime leaves it: the manifest still says running."""
    path = root / "evaluation" / "manifest.json"
    manifest = json.loads(path.read_text())
    for key in ("failure", "finished_at"):
        manifest.pop(key, None)
    path.write_text(json.dumps(dict(manifest, status="running")))


def _age_evaluation(root, seconds):
    """Make everything the evaluation wrote look ``seconds`` older."""
    out = root / "evaluation"
    for path in [out, *out.rglob("*")]:
        stat = path.stat()
        os.utime(path, (stat.st_atime - seconds, stat.st_mtime - seconds))


def _seal(folder):
    """The bytes of every sample file in ``folder``, which are then made unreadable: a stage that opens one fails."""
    sealed = {path.name: path.read_bytes() for path in sorted(folder.glob("*.npz"))}
    for path in folder.glob("*.npz"):
        path.chmod(0o000)
    with pytest.raises(PermissionError):  # the precondition: this account cannot read them (root could)
        next(folder.glob("*.npz")).read_bytes()
    return sealed


def test_an_interrupted_evaluation_is_restarted_from_scratch_with_its_output_moved_aside_unread(tiny, monkeypatch):
    locked, commit, root, _ = _trained(tiny)
    real = _interrupting(monkeypatch)
    with pytest.raises(OSError, match="Input/output"):  # attempt 1: the job ends in the middle of the evaluation
        run_tsm.main(_evaluate_argv(locked, commit, root))
    out = root / "evaluation"
    first = json.loads((out / "manifest.json").read_text())
    assert first["status"] == "failed" and first["failure"].startswith("OSError")
    assert first["attempt"] == 1 and first["restarts"] == [] and len(list((out / "samples").glob("*.npz"))) == 4
    with pytest.raises(ValueError, match="did not complete.*--restart-interrupted"):  # a restart is deliberate
        run_tsm.main(_evaluate_argv(locked, commit, root))
    assert not (root / "attempts").exists() and len(list((out / "samples").glob("*.npz"))) == 4
    sealed = _seal(out / "samples")  # the partial samples are never opened
    monkeypatch.setattr(run_tsm, "evaluate_runs", real)
    assert run_tsm.main(_evaluate_argv(locked, commit, root, "--restart-interrupted")) == 0
    moved = root / "attempts" / "evaluation-attempt1"
    assert run_tsm.moved_evaluations(root) == [moved]
    for path in (moved / "samples").glob("*.npz"):
        path.chmod(0o600)
    assert {path.name: path.read_bytes() for path in (moved / "samples").glob("*.npz")} == sealed  # as they were
    assert json.loads((moved / "manifest.json").read_text()) == first
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["attempt"] == 2
    assert manifest["restarts"] == [dict(attempt=1, moved_to="attempts/evaluation-attempt1")]
    assert len(manifest["evaluations"]) == 24 and len(list((out / "samples").glob("*.npz"))) == 24  # from scratch
    assert manifest["design_hash"] == first["design_hash"] and manifest["checkpoints"] == first["checkpoints"]
    assert manifest["source"]["source_hash"] == first["source"]["source_hash"]  # identical inputs
    with pytest.raises(ValueError, match="already started and is complete"):  # a complete evaluation is final
        run_tsm.main(_evaluate_argv(locked, commit, root, "--restart-interrupted"))
    assert run_tsm.moved_evaluations(root) == [moved]

    analysis = root.parent / "analysis"  # the analysis reports the restart and audits it
    assert analyze_tsm.main(["--root", str(root), "--out", str(analysis)]) == 0
    summary = json.loads((analysis / "summary.json").read_text())
    assert summary["evaluation_attempt"] == 2 and summary["evaluation_restarts"] == manifest["restarts"]
    assert "attempts/evaluation-attempt1" in (analysis / "summary.md").read_text()
    with _edited(out / "manifest.json", lambda m: dict(m, attempt=1, restarts=[])):  # a restart that is not recorded
        with pytest.raises(ValueError, match="evaluation attempts differ"):
            analyze_tsm.main(["--root", str(root), "--out", str(root.parent / "again")])
    moved.rename(root / "attempts" / "elsewhere")  # a recorded restart whose output is not where it was moved
    with pytest.raises(ValueError, match="evaluation attempts differ"):
        analyze_tsm.main(["--root", str(root), "--out", str(root.parent / "again")])
    (root / "attempts" / "elsewhere").rename(moved)
    with _edited(out / "manifest.json", lambda m: dict(m, checkpoints={})):
        with pytest.raises(ValueError, match="checkpoint seals differ"):
            analyze_tsm.main(["--root", str(root), "--out", str(root.parent / "again")])


def test_an_evaluation_is_restarted_at_most_twice(tiny, monkeypatch):
    locked, commit, root, _ = _trained(tiny)
    _interrupting(monkeypatch)
    for attempt in (1, 2, 3):  # the evaluation and its two restarts all end before they complete
        with pytest.raises(OSError, match="Input/output"):
            run_tsm.main(_evaluate_argv(locked, commit, root, *(["--restart-interrupted"] if attempt > 1 else [])))
        assert json.loads((root / "evaluation" / "manifest.json").read_text())["attempt"] == attempt
    third = (root / "evaluation" / "manifest.json").read_text()
    with pytest.raises(ValueError, match="used 3 of 3 attempts"):
        run_tsm.main(_evaluate_argv(locked, commit, root, "--restart-interrupted"))
    assert [path.name for path in run_tsm.moved_evaluations(root)] == ["evaluation-attempt1", "evaluation-attempt2"]
    assert (root / "evaluation" / "manifest.json").read_text() == third  # the third attempt is where it was


def test_a_restart_waits_until_the_interrupted_job_has_ended(tiny, monkeypatch):
    locked, commit, root, _ = _trained(tiny)
    monkeypatch.setattr(run_tsm, "QUIET_SECONDS", 3600)  # a live evaluation writes a sample file for every checkpoint
    real = _interrupting(monkeypatch)
    flag = "--restart-interrupted"
    with pytest.raises(OSError):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    with pytest.raises(OSError):  # the job recorded its own end (failed): no need to wait
        run_tsm.main(_evaluate_argv(locked, commit, root, flag))
    _killed(root)  # attempt 2 was killed instead: its manifest says running, and was written just now
    with pytest.raises(ValueError, match="written to .* its job may still be running"):
        run_tsm.main(_evaluate_argv(locked, commit, root, flag))
    assert [path.name for path in run_tsm.moved_evaluations(root)] == ["evaluation-attempt1"]  # nothing moved
    _age_evaluation(root, 7200)
    samples = root / "evaluation" / "samples"
    stamp = samples.stat().st_mtime
    late = samples / "late.npz"  # a live job appends to a file, which leaves the directory old
    late.touch()
    os.utime(samples, (stamp, stamp))
    with pytest.raises(ValueError, match="written to .* its job may still be running"):
        run_tsm.main(_evaluate_argv(locked, commit, root, flag))
    late.unlink()
    _age_evaluation(root, 7200)
    monkeypatch.setattr(run_tsm, "evaluate_runs", real)
    assert run_tsm.main(_evaluate_argv(locked, commit, root, flag)) == 0  # silent for two hours: its job has ended
    manifest = json.loads((root / "evaluation" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["attempt"] == 3 and len(manifest["restarts"]) == 2


def test_a_restart_needs_identical_inputs_also_after_an_aborted_restart(tiny, monkeypatch):
    locked, commit, root, repo = _trained(tiny)
    real = _interrupting(monkeypatch)
    with pytest.raises(OSError):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    monkeypatch.setattr(run_tsm, "evaluate_runs", real)
    manifest_path = root / "evaluation" / "manifest.json"
    interrupted = manifest_path.read_text()

    def refused(match, *, commit=commit):
        with pytest.raises(ValueError, match=match):
            run_tsm.main(_evaluate_argv(locked, commit, root, "--restart-interrupted"))
        assert manifest_path.read_text() == interrupted and not (root / "attempts").exists()  # nothing was moved

    real_source = run_tsm.source_metadata
    monkeypatch.setattr(run_tsm, "source_metadata", lambda: dict(real_source(), source_hash="another-code-version"))
    refused("differs from the interrupted evaluation in code")
    monkeypatch.setattr(run_tsm, "source_metadata", real_source)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "later")  # the same design file under another commit
    refused("differs from the interrupted evaluation in design_lock_commit", commit=_git(repo, "rev-parse", "HEAD"))
    checkpoint, record = root / "runs" / NAME / "checkpoint.pt", root / "runs" / NAME / "record.json"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"x")  # replaced and sealed again: only the earlier attempt shows the difference
    with _edited(record, lambda r: dict(r, checkpoint_blake2b=run_tsm.digest_checkpoint(checkpoint))):
        refused("differs from the interrupted evaluation in checkpoints")
    checkpoint.write_bytes(original)

    # a restart that stopped after it had moved the interrupted evaluation aside and before it wrote a manifest
    run_tsm.move_evaluation_aside(root)
    assert not (root / "evaluation").exists()
    monkeypatch.setattr(run_tsm, "source_metadata", lambda: dict(real_source(), source_hash="another-code-version"))
    with pytest.raises(ValueError, match="differs from the interrupted evaluation in code"):  # still compared
        run_tsm.main(_evaluate_argv(locked, commit, root))
    monkeypatch.setattr(run_tsm, "source_metadata", real_source)
    assert run_tsm.main(_evaluate_argv(locked, commit, root)) == 0  # nothing is in place, so no flag is needed
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete" and manifest["attempt"] == 2
    assert manifest["restarts"] == [dict(attempt=1, moved_to="attempts/evaluation-attempt1")]


@pytest.mark.parametrize("manifest_text", [None, '{"status": "runn'])
def test_an_evaluation_directory_without_a_readable_manifest_is_an_interrupted_evaluation(tiny, monkeypatch,
                                                                                         manifest_text):
    locked, commit, root, _ = _trained(tiny)
    monkeypatch.setattr(run_tsm, "QUIET_SECONDS", 3600)
    samples = root / "evaluation" / "samples"
    samples.mkdir(parents=True)  # a job killed while it started: before its manifest was written, or in the middle of it
    (samples / "partial.npz").write_bytes(b"cut off")
    if manifest_text is not None:
        (root / "evaluation" / "manifest.json").write_text(manifest_text)
    with pytest.raises(ValueError, match="did not complete.*--restart-interrupted"):
        run_tsm.main(_evaluate_argv(locked, commit, root))
    with pytest.raises(ValueError, match="its job may still be running"):  # it was written to just now
        run_tsm.main(_evaluate_argv(locked, commit, root, "--restart-interrupted"))
    assert not (root / "attempts").exists() and (samples / "partial.npz").read_bytes() == b"cut off"
    _age_evaluation(root, 7200)
    assert run_tsm.main(_evaluate_argv(locked, commit, root, "--restart-interrupted")) == 0
    assert (root / "attempts" / "evaluation-attempt1" / "samples" / "partial.npz").read_bytes() == b"cut off"
    assert not (samples / "partial.npz").exists()
    manifest = json.loads((root / "evaluation" / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["attempt"] == 2
    assert manifest["restarts"] == [dict(attempt=1, moved_to="attempts/evaluation-attempt1")]


def _endpoint(writer, arm, *, passes, failed=False):
    row = dict(writer=writer, arm=arm, confidence=0.9875, training_failed=failed, passes=passes and not failed,
               completed_seeds="" if failed else "17;29;43", failed_seeds="17;29;43" if failed else "")
    for name in analyze_tsm.ENDPOINTS:
        row.update({f"{name}_{field}": None if failed else 0.01 for field in analyze_tsm.FIELDS},
                   **{f"{name}_passes": None if failed else passes})
    return row


def test_a_single_writer_pass_is_worded_as_a_writer_specific_result():
    assert analyze_tsm.arm_outcome(["oracle_random", "joint_fact"]) == "both"
    assert analyze_tsm.arm_outcome(["joint_fact"]) == "writer_specific"
    assert analyze_tsm.arm_outcome([]) == "none"
    endpoints = [_endpoint("oracle_random", "T1", passes=False), _endpoint("joint_fact", "T1", passes=True),
                 _endpoint("oracle_random", "T2", passes=True), _endpoint("joint_fact", "T2", passes=True, failed=True)]
    result = dict(endpoints=endpoints, descriptive=[], accuracy=[], **analyze_tsm.reading(endpoints))
    assert result["arms"] == {"T1": False, "T2": False}
    assert result["outcome"] == {"T1": "writer_specific", "T2": "writer_specific"}
    assert result["passing_writers"] == {"T1": ["joint_fact"], "T2": ["oracle_random"]}
    design = dict(decision=dict(confidence=0.9875), evaluation=dict(split="test", seed=999007, episodes=10))
    failure = dict(name="tsm-joint_fact-T2-s17", writer="joint_fact", arm="T2", seed=17, attempt=1,
                   failure="FloatingPointError: non-finite train loss | at step 3")
    manifest = dict(training_failures=[failure], node_hours=0.1, design_lock_commit="a" * 40, attempt=1, restarts=[])
    text = "\n".join(analyze_tsm.render(result, design, manifest))
    assert "| T1 | fails | passes | writer-specific result, not a remedy |" in text
    assert "| T2 | passes | training failure | writer-specific result, not a remedy |" in text
    assert {"- oracle_random: T2 passes", "- joint_fact: T1 passes; training failure: T2"} <= set(text.splitlines())
    assert "## Training failures" in text and "tsm-joint_fact-T2-s17" in text and "non-finite train loss / at step 3" in text
