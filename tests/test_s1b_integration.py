"""The real S1B smoke → JSON builder → candidate → reload CLI boundary, synthetic parent only."""
from test_s1b_artifacts import bundle
from test_s1b_model import backbone, tokenizer
import slot_golden

from looplora.llm import s1b_model as model, s1b_run as run, s1b_artifacts as artifacts
from looplora.llm.slot_experiment import adopt


def test_smoke_builder_and_cli_reload_share_complete_json_metadata(bundle, backbone, tokenizer, tmp_path, monkeypatch):
    def synthetic_parent(branch, **kwargs):
        return model.configure(adopt(slot_golden.tiny_experiment(backbone, tokenizer)), branch)
    original_load = model.load_export
    monkeypatch.setattr(model, "load_parent", synthetic_parent)
    monkeypatch.setattr(model, "load_export", lambda path: original_load(path, model=backbone, tokenizer=tokenizer))
    monkeypatch.setattr(run, "source_metadata", lambda: dict(git_commit="a" * 40))
    monkeypatch.setattr(run, "RUN_ROOT", tmp_path)
    smoke_dir = tmp_path / "actual-smoke"
    smoke_dir.mkdir()
    p = bundle.paths
    run.smoke(p["lock"], p["branch"], p["cal"], smoke_dir)
    candidate = artifacts.prepare_release(p["lock"], p["branch"], p["cal"], p["pilot"], smoke_dir / "smoke.json", "a" * 40)
    assert candidate["status"] == "pending_reload"
    release_path = bundle.write("real-candidate.json", candidate)
    reloads = {}
    for arm in candidate["arms"]:
        out = tmp_path / f"cli-reload-{arm}"
        assert run.main(["--mode", "reload-check", "--manifest", str(p["lock"]), "--release", str(release_path),
                         "--export", str(smoke_dir / arm / "selected.pt"), "--predictions", str(smoke_dir / arm / "reload_samples.npz"),
                         "--arm", arm, "--out", str(out)]) == 0
        reloads[arm] = out / "reload.json"
    ready = artifacts.prepare_release(p["lock"], p["branch"], p["cal"], p["pilot"], smoke_dir / "smoke.json", "a" * 40, reloads=reloads)
    assert ready["status"] == "ready" and len(ready["config_hashes"]) == 12
