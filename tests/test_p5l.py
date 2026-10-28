"""P5L step 1: the backbone option (off path unchanged), shapes from the loaded model, configs and runner guards."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p5l
from make_p5l_configs import BACKBONE, LABELS, SEEDS, make
from looplora.core.config import AdapterConfig
from looplora.infra.seeds import stable_hash
from looplora.llm import inject
from looplora.llm.config import DEFAULTS, resolve_config
from looplora.llm.inject import MODEL_ID, MODEL_REVISION, attach_adapters, backbone_pin, load_model
from looplora.llm.train import LLMExperiment

P1L_R128 = ROOT / "configs/p1l/main-r128"
RECORDED = {
    "B3-L4-s17": "b7931c9835a6a01a2c541c943dcb9e18",
    "B3-L4-s29": "c11d70f5a28e73eeddd69d1ab037f46d",
    "B3-L4-s43": "9503532e1e373ead94356bba3cb044f8",
    "B4-s17": "be4343ffeedc1474f63dcd2354fa4cc7",
    "B4-s29": "c215ddc72e6107f89fd3d9635304df9a",
    "B4-s43": "72cc3bbbb5773edd749934d171bc36ed",
    "B5-L4-s17": "fb3121a540a8603a3b111a001ce33c5a",
    "B5-L4-s29": "05af0dad7d6bf41155a3b31aff2962da",
    "B5-L4-s43": "abf18f7b02ef413d8d0c006364d6ac6a",
    "B8-s17": "97174328c2defbc8708330296fe21113",
    "B8-s29": "2f5acd6bd4f6e2d67d723b4aa0d6d1cc",
    "B8-s43": "7e60777e32e36d74a4844f7960dadaff",
    "M1-L4-s17": "8b56ec347c3f0be8f1c2f95a5237ebfa",
    "M1-L4-s29": "a4adcccd53043d1af222c41a69a4a1c1",
    "M1-L4-s43": "2a6e14e563901dded61c510a81004810",
}


def read(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------- the off path is unchanged

def test_resolved_p1l_r128_configs_keep_their_recorded_hashes():
    """Every committed P1-L r = 128 training config resolves to the config_hash its finished training run recorded
    (record.json of runs/p1l/main/r128/main, read before the backbone option existed)."""
    names = sorted(path.stem for path in P1L_R128.glob("*.json"))
    assert names == sorted(RECORDED)
    for name in names:
        resolved = resolve_config(read(P1L_R128 / f"{name}.json"))
        assert "backbone" not in resolved
        assert stable_hash(resolved) == RECORDED[name], name


def test_backbone_is_an_optional_key_outside_the_defaults():
    assert "backbone" not in DEFAULTS
    base = read(P1L_R128 / "M1-L4-s17.json")
    pinned = resolve_config({**base, "backbone": BACKBONE})
    assert pinned["backbone"] == BACKBONE
    assert {key: value for key, value in pinned.items() if key != "backbone"} == resolve_config(base)
    for bad in ({"id": "x"}, {"id": "x", "revision": ""}, {"id": "x", "revision": "y", "extra": "z"}, "x", None, {"id": 1, "revision": "y"}):
        with pytest.raises(ValueError, match="backbone"):
            resolve_config({**base, "backbone": bad})


class Recorder:
    """Stands in for from_pretrained: records the call and returns an object accepted by load_model/load_tokenizer."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self

    def to(self, device):
        return self

    def requires_grad_(self, flag):
        return self

    def eval(self):
        return self


def test_default_loading_makes_exactly_the_original_calls(monkeypatch):
    model, tokenizer = Recorder(), Recorder()
    monkeypatch.setattr(inject.AutoModelForCausalLM, "from_pretrained", model)
    monkeypatch.setattr(inject.AutoTokenizer, "from_pretrained", tokenizer)
    assert backbone_pin() == (MODEL_ID, MODEL_REVISION) == ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
    load_model(device="cpu")
    inject.load_tokenizer()
    assert model.calls == [((MODEL_ID,), dict(revision=MODEL_REVISION, cache_dir=inject.CACHE_DIR, local_files_only=True,
                                              torch_dtype=torch.bfloat16, attn_implementation="sdpa"))]
    assert tokenizer.calls == [((MODEL_ID,), dict(revision=MODEL_REVISION, cache_dir=inject.CACHE_DIR, local_files_only=True))]
    # an explicit backbone changes the id and revision and nothing else
    model.calls.clear()
    load_model(device="cpu", backbone=BACKBONE)
    ((args, kwargs),) = model.calls
    assert args == (BACKBONE["id"],) and kwargs["revision"] == BACKBONE["revision"]
    assert {k: v for k, v in kwargs.items() if k != "revision"} == dict(cache_dir=inject.CACHE_DIR, local_files_only=True,
                                                                        torch_dtype=torch.bfloat16, attn_implementation="sdpa")


# ------------------------------------------------------------- shapes and options follow the loaded model (tiny model)

REVISION = "0123456789abcdef0123456789abcdef01234567"
TINY = dict(vocab_size=300, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=32, attention_dropout=0.)


@pytest.fixture()
def cache(tmp_path, monkeypatch):
    """A cache directory with a tiny Qwen3 whose o_proj is 128 -> 64, found through the backbone option only."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        Qwen3ForCausalLM(Qwen3Config(**TINY)).save_pretrained(tmp_path / "models--tiny--qwen" / "snapshots" / REVISION)
    monkeypatch.setattr(inject, "CACHE_DIR", str(tmp_path))
    return {"id": "tiny/qwen", "revision": REVISION}


def test_option_loads_another_backbone_and_adapter_shapes_follow_it(cache):
    model = load_model(dtype=torch.float32, device="cpu", backbone=cache)
    assert model.config.hidden_size == 64
    bank = attach_adapters(model, AdapterConfig(d_in=1, d_out=1, r=128, variant="iterq"), layers=(1,), seed=3)
    cfg = bank.modules[1].adapter.cfg
    assert (cfg.d_in, cfg.d_out, cfg.r) == (128, 64, 128)          # heads * head_dim in, hidden_size out
    with pytest.raises(Exception):                                   # without the option the pinned 0.6B is looked up
        load_model(dtype=torch.float32, device="cpu")


def test_experiment_loads_the_configured_backbone(cache):
    cfg = resolve_config(dict(writer="oracle_random", stage="reader", variant="M1", device="cpu", backbone=cache,
                              model=dict(r=128, layer=1), task=dict(E=6, Q=2, chain_lengths=[4, 2], n_rel=4, head_sharing=.3),
                              train=dict(batch_size=2, depths=[4]), eval=dict(episodes=6, batch_size=3, hops=[1, 2, 3, 4])))
    experiment = LLMExperiment(cfg, tokenizer=object())
    try:
        assert experiment.model.config.hidden_size == 64
        assert (experiment.adapter.cfg.d_in, experiment.adapter.cfg.d_out) == (128, 64)
        assert experiment.writer_signature()["r_mem"] == 128
    finally:
        experiment.close()


# --------------------------------------------------------------- the 1.7B backbone and the registered configs

def snapshot(model_id, revision):
    return Path(inject.CACHE_DIR) / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision


def test_the_pinned_17b_backbone_has_the_registered_shape_and_the_same_tokenizer():
    small, large = snapshot(MODEL_ID, MODEL_REVISION), snapshot(**BACKBONE)
    if not large.is_dir():
        pytest.skip("Qwen3-1.7B-Base is not cached here")
    config = read(large / "config.json")
    assert (config["num_hidden_layers"], config["hidden_size"], config["num_attention_heads"], config["head_dim"],
            config["vocab_size"]) == (28, 2048, 16, 128, 151936)
    assert read(small / "config.json")["num_hidden_layers"] == config["num_hidden_layers"]
    for name in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
        digest = [hashlib.blake2b((folder / name).read_bytes(), digest_size=16).hexdigest() for folder in (small, large)]
        assert digest[0] == digest[1], name


def test_p5l_configs_are_p1l_r128_with_the_backbone_added():
    configs = make()
    assert sorted(configs) == sorted(f"{label}-s{seed}" for label in LABELS for seed in SEEDS) and len(configs) == 15
    for name, value in configs.items():
        base = read(P1L_R128 / f"{name}.json")
        assert read(ROOT / run_p5l.CONFIGS / f"{name}.json") == value           # committed files are what the generator writes
        assert value == {**base, "name": f"p5l-r128-{name}", "backbone": BACKBONE}
        resolved = resolve_config(value)
        assert resolved["model"]["r"] == 128 and resolved["train"]["steps"] == 17000
        assert resolved["train"]["lr"] == (0.001 if name.startswith("B3-L4") else 0.003)


# ---------------------------------------------------------------------------------------------- the runner guards

def lock():
    p1l = read(ROOT / "reports/p1l/r128/lr-S-lock.json")
    return {"status": "locked", "rank": 128, "S": p1l["S"], "lrs": p1l["lrs"]}


def test_runner_names_one_backbone_and_the_lock_must_agree_with_the_configs(monkeypatch):
    assert run_p5l.backbone_of(run_p5l.LABELS, 17) == BACKBONE
    run_p5l.check_lock(lock(), run_p5l.LABELS, 29)                                  # the carried P1-L values
    for change in ({"status": "draft"}, {"rank": 32}, {"S": 16000}, {"lrs": {**lock()["lrs"], "B8": 0.001}}):
        with pytest.raises(ValueError):
            run_p5l.check_lock({**lock(), **change}, run_p5l.LABELS, 17)
    real = run_p5l.config
    monkeypatch.setattr(run_p5l, "config", lambda label, seed: {**real(label, seed), "backbone": BACKBONE if label != "B4" else
                                                                {"id": "Qwen/Qwen3-0.6B-Base", "revision": MODEL_REVISION}})
    with pytest.raises(ValueError, match="different backbones"):
        run_p5l.backbone_of(run_p5l.LABELS, 17)
