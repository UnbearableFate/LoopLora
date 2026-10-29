"""B8W (docs/prereg/B8W.md): exact M1-L4 to B8 conversion, the equality gate, the continuation stream and the decision rule."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_b8w
from looplora.core.adapters import make_adapter
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import derive_seed, stable_hash
from looplora.llm import train as train_module
from looplora.llm.b8w import (COPIED, NONINFERIORITY, SHARED, compare_signatures, convert_m1_to_b8, decision_text,
                              gate_signature, keeps_solution)
from looplora.llm.config import CONFIRMATION_WINDOWS, DEFAULTS, resolve_config
from looplora.llm.inject import load_tokenizer
from looplora.llm.train import LLMExperiment, continue_training, load_checkpoint, train

P1L_R128 = ROOT / "configs/p1l/main-r128"
# config_hash recorded by the finished P1-L r = 128 trainings (same literals as tests/test_p5l.py)
RECORDED = {"M1-L4-s17": "8b56ec347c3f0be8f1c2f95a5237ebfa", "M1-L4-s29": "a4adcccd53043d1af222c41a69a4a1c1",
            "M1-L4-s43": "2a6e14e563901dded61c510a81004810", "B8-s17": "97174328c2defbc8708330296fe21113"}


# ------------------------------------------------------------------ windows and the optional stream key (off path)

def test_windows_are_asserted_one_by_one_and_stay_disjoint():
    for name, start in {"p1l": 100000, "p1l-s": 110000, "p5l": 104000, "b8w": 108000}.items():
        assert CONFIRMATION_WINDOWS[name] == start
    taken = [(start, start + 2000) for start in CONFIRMATION_WINDOWS.values()] + [(102000, 104000)]
    taken.sort()
    assert all(left[1] <= right[0] for left, right in zip(taken, taken[1:]))
    assert resolve_config({"eval": {"confirmation": "b8w", "splits": ["confirm"], "allow_confirmation": True}})["eval"]["confirmation"] == "b8w"


def test_data_start_step_is_optional_validated_and_outside_the_defaults():
    assert "data_start_step" not in DEFAULTS["train"]
    for name, recorded in RECORDED.items():
        label, seed = name.rsplit("-s", 1)
        resolved = resolve_config(json.loads((P1L_R128 / f"{name}.json").read_text()))
        assert "data_start_step" not in resolved["train"] and stable_hash(resolved) == recorded, name
    base = json.loads((P1L_R128 / "M1-L4-s17.json").read_text())
    resolved = resolve_config({**base, "train": {**base["train"], "data_start_step": 17000}})
    assert resolved["train"]["data_start_step"] == 17000
    for bad in (-1, True, 1.5, "7"):
        with pytest.raises(ValueError, match="data_start_step"):
            resolve_config({**base, "train": {**base["train"], "data_start_step": bad}})
    with pytest.raises(ValueError, match="unknown train fields"):
        resolve_config({**base, "train": {**base["train"], "stream_start": 1}})


def test_runner_constants_are_the_registered_ones():
    lock = json.loads((ROOT / "reports/p1l/r128/lr-S-lock.json").read_text())
    assert run_b8w.STEPS == 2000 and run_b8w.LR == lock["lrs"]["M1-L4"] == lock["lrs"]["B8"] and run_b8w.DATA_START == lock["S"]
    assert NONINFERIORITY == 0.02


# ------------------------------------------------------------------ containment: identical forward, consistent gradient

def adapters(r=8, d=12):
    cfg = dict(d_in=d, d_out=d, r=r, rho=.25, alpha=1.)
    shared = make_adapter(AdapterConfig(variant="iterq", **cfg))
    unshared = make_adapter(AdapterConfig(variant="unshared", unshared_steps=4, **cfg))
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in shared.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .3)
    state = {key: value.clone() for key, value in shared.state_dict().items()}
    copied = {key: state[key] for key in COPIED}
    for step in range(4):
        for key in SHARED:
            copied[f"{key}.{step}"] = state[key].clone()
    unshared.load_state_dict(copied, strict=True)
    return shared, unshared


def test_copied_unshared_reader_is_bitwise_the_shared_reader():
    shared, unshared = adapters()
    generator = torch.Generator().manual_seed(9)
    u = torch.randn(6, 8, generator=generator)
    memory = torch.randn(6, 8, 8, generator=generator)
    assert torch.equal(shared.read(u, memory, 4), unshared.read(u, memory, 4))
    x = torch.randn(3, 5, 12, generator=generator)
    a, _, _ = shared.process_sequence_fast(x, loop_steps=4)
    b, _, _ = unshared.process_sequence_fast(x, loop_steps=4)
    assert torch.equal(a, b)


def test_untied_gradients_sum_to_the_shared_gradient():
    shared, unshared = adapters()
    generator = torch.Generator().manual_seed(11)
    u = torch.randn(6, 8, generator=generator)
    memory = torch.randn(6, 8, 8, generator=generator)
    target = torch.randn(6, 8, generator=generator)
    for adapter in (shared, unshared):
        ((adapter.read(u, memory, 4) - target) ** 2).sum().backward()
    for key in SHARED:
        total = sum(getattr(unshared, key)[step].grad for step in range(4))
        torch.testing.assert_close(total, getattr(shared, key).grad, rtol=1e-5, atol=1e-6)
    assert not any(torch.equal(unshared.R[0].grad, unshared.R[step].grad) for step in (1, 2, 3))


# ------------------------------------------------------------------ the conversion function

def synthetic_record(r=8):
    cfg = resolve_config(dict(name="p1l-r128-M1-L4-s17", writer="oracle_random", stage="reader", variant="M1",
                              model=dict(r=32, layer=1), task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
                              train=dict(depths=[4]), eval=dict(depths=[4])))
    shared, _ = adapters(r=32, d=32)
    return {"format_version": 1, "config": cfg, "adapter": {k: v.clone() for k, v in shared.state_dict().items()},
            "writer": {"codes": torch.arange(3.)}, "signature": {"hash": "x"}, "summary": {"steps": 17000}}


def test_conversion_maps_keys_copies_values_and_leaves_the_input_alone():
    record = synthetic_record()
    before = copy.deepcopy(record)
    converted = convert_m1_to_b8(record, name="b8w-converted-s17")
    assert converted["config"]["variant"] == "B8" and converted["config"]["name"] == "b8w-converted-s17"
    assert converted["config"]["train"]["depths"] == converted["config"]["eval"]["depths"] == [4]
    assert {k: v for k, v in converted["config"].items() if k not in {"variant", "name"}} == \
           {k: v for k, v in record["config"].items() if k not in {"variant", "name"}}
    state = converted["adapter"]
    assert set(state) == set(COPIED) | {f"{key}.{step}" for key in SHARED for step in range(4)}
    for key in COPIED:
        assert torch.equal(state[key], record["adapter"][key])
    for key in SHARED:
        for step in range(4):
            assert torch.equal(state[f"{key}.{step}"], record["adapter"][key])
            assert state[f"{key}.{step}"].data_ptr() != record["adapter"][key].data_ptr()
    assert converted["writer"].keys() == record["writer"].keys()
    assert all(torch.equal(converted["writer"][k], record["writer"][k]) for k in record["writer"])
    assert converted["writer"]["codes"].data_ptr() != record["writer"]["codes"].data_ptr()      # a copy, not a shared tensor
    assert converted["signature"] == record["signature"]
    assert converted["summary"]["converted_from"] == dict(variant="M1", name="p1l-r128-M1-L4-s17")
    assert torch.equal(record["adapter"]["R"], before["adapter"]["R"]) and record["config"] == before["config"]
    # the converted state dict loads into a real UnsharedAdapter strictly
    adapter = make_adapter(AdapterConfig(d_in=32, d_out=32, r=32, variant="unshared", unshared_steps=4))
    adapter.load_state_dict(state, strict=True)


@pytest.mark.parametrize("damage", ["variant", "writer", "format", "extra_key", "missing_key", "shape"])
def test_conversion_rejects_what_is_not_an_oracle_m1_reader(damage):
    record = synthetic_record()
    if damage == "variant":
        record["config"]["variant"] = "B4"
    elif damage == "writer":
        record["config"]["writer"] = "learned"
    elif damage == "format":
        record["format_version"] = 2
    elif damage == "extra_key":
        record["adapter"]["w_lambda"] = torch.zeros(32)
    elif damage == "missing_key":
        del record["adapter"]["Q"]
    else:
        record["adapter"]["R"] = torch.zeros(32, 31)
    with pytest.raises(ValueError):
        convert_m1_to_b8(record, name="x")


# ------------------------------------------------------------------ the gate comparison and the decision rule

def signature(value=0.):
    return {"logits": [torch.full((2, 3, 5), value, dtype=torch.bfloat16)], "read": torch.full((4, 3), value)}


def test_gate_comparison_has_tolerance_zero_and_needs_a_deterministic_control():
    ok = compare_signatures(signature(1.), signature(1.), signature(1.))
    assert ok["passed"] and ok["max_abs_difference"] == 0.0 and ok["tolerance"] == 0.0 and ok["logits_dtype"] == "torch.bfloat16"
    tiny = signature(1.)
    tiny["logits"][0][0, 0, 0] = torch.tensor(1.0078125, dtype=torch.bfloat16)           # one bf16 step away
    assert not compare_signatures(signature(1.), signature(1.), tiny)["passed"]
    read = signature(1.)
    read["read"][0, 0] += 1e-6
    assert not compare_signatures(signature(1.), signature(1.), read)["passed"]
    assert not compare_signatures(signature(1.), tiny, signature(1.))["passed"]           # control not reproducible
    wrong_dtype = {"logits": [signature(1.)["logits"][0].float()], "read": signature(1.)["read"]}
    assert not compare_signatures(signature(1.), signature(1.), wrong_dtype)["passed"]


def contrast(differences, ci_low):
    return {"difference": sum(differences) / len(differences), "ci_low": ci_low, "ci_high": ci_low + .02,
            "per_seed": [{"seed": seed, "difference": value} for seed, value in zip((17, 29, 43), differences)]}


def test_decision_rule_boundaries():
    assert keeps_solution(contrast([0., .01, -.01], -.005))["keeps"]
    assert keeps_solution(contrast([-.02, 0., 0.], -.0199))["keeps"]                       # exactly 2 pp is allowed
    assert not keeps_solution(contrast([-.0201, .02, .02], -.0150))["keeps"]              # one seed worse than 2 pp
    assert not keeps_solution(contrast([-.01, -.01, -.01], -.02))["keeps"]                # interval does not exclude -2 pp
    assert not keeps_solution(contrast([-.01, -.01, -.01], -.03))["keeps"]
    assert keeps_solution(contrast([.05, .05, .05], .04))["keeps"]                        # gaining is not a loss
    result = keeps_solution(contrast([-.03, -.03, -.03], -.035))
    assert not result["keeps"] and not result["every_seed_within_2pp"] and not result["interval_excludes_worse_than_2pp"]
    assert "fragility" in decision_text(result) and "not lack of expressivity" in decision_text(result)
    assert "equal-parameter" in decision_text(keeps_solution(contrast([0., 0., 0.], 0.)))


# ------------------------------------------------------------------ the runner's lock check

def write_lock(tmp_path, **changes):
    entries = {}
    for label in ("M1-L4", "B8", "B4"):
        path = tmp_path / f"{label}-s17.pt"
        path.write_bytes(label.encode())
        entries[f"{label}-s17"] = dict(path=str(path), hash=run_b8w.digest(path))
    lock = dict(status="locked", study="b8w", prereg_hash=run_b8w.digest(ROOT / run_b8w.PREREG), steps=2000, lr=0.003,
                data_start_step=17000, checkpoints=entries)
    return {**lock, **changes}


def test_main_runs_only_against_a_locked_lock_that_agrees_with_the_files(tmp_path):
    run_b8w.check_lock(write_lock(tmp_path), 17)
    for changes in (dict(status="draft"), dict(study="p5l"), dict(prereg_hash="0" * 32), dict(steps=3000), dict(lr=0.001),
                    dict(data_start_step=0)):
        with pytest.raises(ValueError):
            run_b8w.check_lock(write_lock(tmp_path, **changes), 17)
    lock = write_lock(tmp_path)
    Path(lock["checkpoints"]["B4-s17"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="B4-s17"):
        run_b8w.check_lock(lock, 17)


# ------------------------------------------------------------------ end to end on a tiny backbone (compute)

@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(derive_seed(17, "b8w-tiny-qwen"))
        value = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            attention_dropout=0.)).cuda().to(torch.bfloat16).eval()
    yield value
    assert not hasattr(value, "looplora_adapters")


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer()


def config(steps=2, **train):
    return resolve_config(dict(name="b8w-unit", writer="oracle_random", stage="reader", variant="M1",
        model=dict(r=32, layer=1), task=dict(E=4, Q=2, chain_lengths=[4], n_rel=4, head_sharing=.3),
        train=dict(steps=steps, batch_size=2, depths=[4], log_every=1, warmup_steps=1, **train),
        eval=dict(episodes=2, batch_size=2, hops=[1, 2, 3, 4], counterfactuals=False)))


@pytest.fixture(scope="module")
def start(model, tokenizer, tmp_path_factory):
    """A trained M1-L4 checkpoint whose adapter weights are then made large and distinct, so reads are not vacuous."""
    directory = tmp_path_factory.mktemp("b8w-start")
    experiment, _ = train(config(), directory / "m1", model=model, tokenizer=tokenizer)
    experiment.close()
    record = torch.load(directory / "m1/checkpoint.pt", map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(3)
    for key in ("A", "B", "R", "U", "Q"):
        record["adapter"][key] = torch.randn(record["adapter"][key].shape, generator=generator) * (.5 if key != "A" else 1.)
    torch.save(record, directory / "start.pt")
    converted = convert_m1_to_b8(record, name="b8w-converted-s17")
    torch.save(converted, directory / "converted.pt")
    return directory


def signature_of(path, model, tokenizer):
    experiment = load_checkpoint(path, model=model, tokenizer=tokenizer)
    try:
        return gate_signature(experiment)
    finally:
        experiment.close()


def test_the_converted_checkpoint_passes_the_equality_gate_exactly(start, model, tokenizer):
    reference = signature_of(start / "start.pt", model, tokenizer)
    control = signature_of(start / "start.pt", model, tokenizer)
    converted = signature_of(start / "converted.pt", model, tokenizer)
    result = compare_signatures(reference, control, converted)
    assert result["passed"] and result["max_abs_difference"] == 0.0 and result["control_equal"]
    assert float(reference["read"].abs().max()) > 0 and len(reference["logits"]) == 4          # 64 episodes in batches of 16
    assert not torch.equal(reference["logits"][0], torch.zeros_like(reference["logits"][0]))


def test_the_gate_rejects_a_perturbed_copy(start, model, tokenizer, tmp_path):
    record = torch.load(start / "converted.pt", map_location="cpu", weights_only=True)
    record["adapter"]["R.2"] = record["adapter"]["R.2"] + torch.randn(32, 32, generator=torch.Generator().manual_seed(1))
    torch.save(record, tmp_path / "perturbed.pt")
    reference = signature_of(start / "start.pt", model, tokenizer)
    result = compare_signatures(reference, signature_of(start / "start.pt", model, tokenizer),
                                signature_of(tmp_path / "perturbed.pt", model, tokenizer))
    assert not result["passed"] and result["max_abs_difference"] > 0


def recorded_batches(monkeypatch):
    seen = []
    original = LLMExperiment.prepare

    def spy(self, arrays, *args, **kwargs):
        seen.append(arrays.episode_seed.copy())
        return original(self, arrays, *args, **kwargs)

    monkeypatch.setattr(LLMExperiment, "prepare", spy)
    return seen


def test_off_path_stream_is_unchanged_and_the_continuation_reads_the_later_batches(start, model, tokenizer, tmp_path, monkeypatch):
    calls = []
    original = train_module.BatchStream

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "BatchStream", spy)
    seen = recorded_batches(monkeypatch)
    experiment, _ = train(config(steps=4), tmp_path / "whole", model=model, tokenizer=tokenizer)
    experiment.close()
    (args, kwargs), = calls
    assert args[4] == 4 and kwargs["start_step"] == 0                     # steps unchanged; the stream starts where it always did
    whole = [item.tolist() for item in seen]
    seen.clear()
    resumed = load_checkpoint(start / "start.pt", model=model, tokenizer=tokenizer)
    try:
        summary = continue_training(resumed, tmp_path / "continued", dict(steps=2, data_start_step=2, lr=0.003),
                                    name="b8w-shared-s17", continued_from=dict(arm="shared"))
    finally:
        resumed.close()
    assert [item.tolist() for item in seen] == whole[2:]                  # the same episodes as steps 3 and 4 of the original stream
    assert summary["data_start_step"] == 2 and summary["steps"] == 2 and summary["continued_from"] == dict(arm="shared")
    record = torch.load(tmp_path / "continued/checkpoint.pt", map_location="cpu", weights_only=True)
    assert record["config"]["name"] == "b8w-shared-s17" and record["config"]["train"]["lr"] == 0.003
    assert record["config"]["train"]["data_start_step"] == 2 and record["config"]["train"]["steps"] == 2
    log = [json.loads(line) for line in (tmp_path / "continued/train_log.jsonl").read_text().splitlines()]
    assert [row["step"] for row in log] == [1, 2] and log[0]["lr"] == pytest.approx(0.003)   # warmup of 1 step restarts
    with pytest.raises(FileExistsError):
        again = load_checkpoint(start / "start.pt", model=model, tokenizer=tokenizer)
        try:
            continue_training(again, tmp_path / "continued", dict(steps=2), name="x", continued_from={})
        finally:
            again.close()


def test_the_untied_arm_trains_each_step_separately_from_the_copy(start, model, tokenizer, tmp_path):
    experiment = load_checkpoint(start / "converted.pt", model=model, tokenizer=tokenizer)
    try:
        before = {k: v.detach().cpu().clone() for k, v in experiment.adapter.state_dict().items()}
        summary = continue_training(experiment, tmp_path / "untied", dict(steps=3, data_start_step=7, lr=0.003),
                                    name="b8w-untied-s17", continued_from=dict(arm="untied"))
    finally:
        experiment.close()
    # B8 trains A, B and the 12 unshared tensors; K, V and the gates stay frozen under the oracle writer
    assert summary["trainable_parameters"] == sum(before[k].numel() for k in before if k not in {"K", "V", "a_lambda", "a_beta"})
    after = torch.load(tmp_path / "untied/checkpoint.pt", map_location="cpu", weights_only=True)["adapter"]
    assert all(torch.equal(after[k], before[k]) for k in ("K", "V", "a_lambda", "a_beta"))
    moved = [not torch.equal(after[f"R.{step}"], before[f"R.{step}"]) for step in range(4)]
    assert all(moved) and not any(torch.equal(after[f"R.{a}"], after[f"R.{b}"]) for a in range(4) for b in range(a + 1, 4))
