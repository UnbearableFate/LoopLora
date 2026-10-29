"""Real-parent S1B gate. Invoked only after the core-code review."""
from __future__ import annotations

from dataclasses import asdict

import torch
from torch.nn import functional as F

from . import s1b_data as data, s1b_rules as rules, s1b_artifacts as artifacts, s1b_model as model
from . import slot_preflight as checks, slot_experiment as legacy
from .evaluate import _slice
from .run import source_metadata
from .slot_read import bind_batch


def matrix_identity(experiment, arrays):
    """Compare full forward and Q/R/U/B gradients to the existing native tensor loop at L4."""
    names = rules.trainable(experiment.s1b_branch)
    legacy.freeze_for_s1a(experiment, names)
    logits, gradients = {}, {}
    for label, bound in (("native", False), ("matrix", True)):
        experiment.adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            batch = experiment.prepare(arrays)
        scored = bind_batch(experiment, batch, rules.read_config("M4")) if bound else batch
        logits[label], _ = experiment.score(scored, 4)
        F.cross_entropy(logits[label].float().flatten(0, 1), batch.answer_ids.flatten()).backward()
        gradients[label] = {n: getattr(experiment.adapter, n).grad.detach().clone() for n in rules.TRAINABLE}
    gaps = {n: float((gradients["native"][n] - gradients["matrix"][n]).abs().max()) for n in rules.TRAINABLE}
    return dict(passed=torch.equal(logits["native"], logits["matrix"]) and all(v == 0 for v in gaps.values()),
                max_abs_difference=float((logits["native"] - logits["matrix"]).abs().max()), gradients=gaps)


def trajectory_check(experiment, arrays):
    with torch.no_grad():
        batch = experiment.prepare(arrays)
        spec = rules.read_config("S4")
        bound = bind_batch(experiment, batch, spec)
        for depth in (1, 2, 4, 8):
            experiment.adapter.reset_observation()
            plain = experiment.score(bound, depth)[0]
            experiment.adapter.reset_observation(capture=True)
            captured = experiment.score(bound, depth)[0]
            if not torch.equal(plain, captured):
                return dict(passed=False, cause="capture changed logits")
            model.verify_trace(experiment.adapter)
        experiment.adapter.reset_observation()
    return dict(passed=True, depths=[1, 2, 4, 8])


def run(out):
    """No calibration/dev, only registered S1B smoke and recorded legacy replay."""
    from .s1b_train import fit
    artifacts.registered()
    manifest, inventory = data.manifest(), artifacts.inventory()
    result = {name: dict(passed=True, evidence=[]) for name in artifacts.REQUIRED}

    def take(name, report):
        result[name]["passed"] &= report["passed"] is True
        result[name]["evidence"].append(report)

    result["registry"]["evidence"].append(dict(task=asdict(data.task_spec()), ids=manifest["ids"]))
    raw = data.raw_episodes(data.eval_set("smoke"))
    tiny = _slice(raw, slice(0, 4))
    evaluation = _slice(data.episodes(data.eval_set("smoke")), slice(0, 4))
    original = legacy.load_parent("S-RE", expected=inventory)
    try:
        take("legacy", checks.native_replay(original, "S-RE", data.old.PARENT_FOLDERS["S-RE"]))
    finally:
        original.close()
    for branch in ("advance", "query_copy"):
        exp = model.load_parent(branch, expected=inventory)
        try:
            take("matrix_L4", matrix_identity(exp, tiny))
            take("trajectory", trajectory_check(exp, evaluation))
            take("owner_mask", checks.owner_mask_check(exp.adapter, device=exp.device, questions=8, seed=data.TRAIN_BASE[17]))
            take("isolation", checks.isolation_check(exp, evaluation))
            take("features", checks.feature_check(exp, evaluation))
            take("bytes", checks.bytes_check(exp, tiny, capacity=12))
            take("capacity", checks.capacity_check(exp, tiny, capacity=12))
            with torch.no_grad():
                batch = exp.prepare(tiny)
                before = exp.score(bind_batch(exp, batch, rules.read_config("M4")), 4)[0]
                exp.adapter.untie()
                after = exp.score(bind_batch(exp, batch, rules.read_config("M4")), 4)[0]
                take("trajectory", dict(passed=torch.equal(before, after), untied_warm_start=True))
        finally:
            exp.close()
        for arm in rules.branch_arms(branch):
            exp = model.load_parent(branch, expected=inventory)
            try:
                source = iter([(_slice(raw, slice((i % 2) * 32, (i % 2 + 1) * 32)), None) for i in range(20)])
                report = fit(exp, rules.read_config(arm), arm=arm, seed=17, lr=1e-3, steps=20,
                             out=out / branch / arm, dev=_slice(data.episodes(data.eval_set("smoke"), [1, 2, 3]), slice(0, 8)),
                             dev_steps=[20], support=tiny, parent_digest=inventory["writer.pt"],
                             science_hash="preflight", source=source)
                reload_arrays = _slice(data.episodes(data.eval_set("smoke"), [1, 2, 3]), slice(0, 8))
                expected_samples = model.evaluate(exp, reload_arrays, rules.read_config(arm), rules.ARM[arm][1])["samples"]
                take("frozen", dict(passed=report["freeze"]["digests"]["frozen_unchanged"] and report["freeze"]["support"]["equal"],
                                    branch=branch, arm=arm, freeze=report["freeze"]))
            finally:
                exp.close()
            restored = model.load_export(out / branch / arm / "selected.pt")
            try:
                if restored.s1b["task_digest"] != data.LAYOUT_DIGEST:
                    raise ValueError("reload task changed")
                actual_samples = model.evaluate(restored, reload_arrays, rules.read_config(arm), rules.ARM[arm][1])["samples"]
                import numpy as np
                equal = all(np.array_equal(actual_samples[k], expected_samples[k]) for k in expected_samples)
                take("reload", dict(passed=equal, branch=branch, arm=arm, metadata=restored.s1b))
            finally:
                restored.close()
    record = dict(stage="S1B", status="complete", passed=all(r["passed"] for r in result.values()), checks=result,
                  manifest=manifest, inventory=inventory, source_blake2b_128=data.old.source_digest(), **source_metadata())
    artifacts.dump(out / "preflight.json", record)
    if not record["passed"]:
        raise rules.StopForOrchestrator("S1B engineering preflight failed")
    return record
