"""S1B registry and seals; independent of S1A's registered T1/L1 globals."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from looplora.data.relations import GENERATOR_VERSION, TaskSpec, generate
from looplora.infra.seeds import derive_seed, stable_hash
from . import slot_data as old

PREREG = "docs/prereg/SLOT-LOOP-v1-S1B.md"
PREREG_DIGEST = "6f2e776f2c6cfde131380fb343854784"
LAYOUT_DIGEST = "f556c1b91252b0b74d3ff674717be37c"
ANALYSIS = "scripts/analyze_s1b.py"
TASK = dict(name="t2_chain_end", E=12, Q=4, V=1024, n_rel=4, head_sharing=.3,
            seen=(0, 768), unseen=(768, 1024), chain_lengths=(8, 4), hops=(1, 2, 3))
RUN_SEEDS = (17, 29, 43)
BATCH, STEPS, PILOT_STEPS = 32, 6000, 200
TRAIN_BASE = {17: 693232309558345634, 29: 373104883948690398,
              43: 6045459755310898920, "pilot": 907865416675862555}
BOOTSTRAP = 5799536610111230359


def task_spec():
    return TaskSpec(**TASK)


@dataclass(frozen=True)
class EvalSet:
    purpose: str
    split: str
    count: int
    base_seed: int
    sealed: bool = False
    opened: bool = False

    def ids(self):
        return [episode_id(self.base_seed, self.split, i) for i in range(self.count)]


SETS = {s.purpose: s for s in (
    EvalSet("smoke", "train", 64, 6584302131768675481),
    EvalSet("calibration", "val", 2048, 254983947625000635),
    EvalSet("pilot", "val", 1000, 2437977920860386590),
    EvalSet("dev", "val", 1000, 5100482262494855894),
    EvalSet("confirmation", "test", 2000, 3460300042162207747, True),
    EvalSet("unseen", "test_unseen", 2000, 4377963709756993594, True),
)}


def episode_id(base, split, index, layout=LAYOUT_DIGEST):
    return derive_seed(base, "relations", GENERATOR_VERSION, layout, split, index)


def verify_registry():
    if stable_hash(asdict(task_spec())) != LAYOUT_DIGEST:
        raise ValueError("S1B TaskSpec digest differs from registration")
    for purpose, s in SETS.items():
        if s.base_seed != derive_seed(old.BASE_SEED, old.STUDY, purpose, "S1B"):
            raise ValueError("S1B purpose seed differs from registration")
    for run, seed in TRAIN_BASE.items():
        if seed != derive_seed(old.BASE_SEED, old.STUDY, "train", "S1B", run):
            raise ValueError("S1B training seed differs from registration")
    if BOOTSTRAP != derive_seed(old.BASE_SEED, old.STUDY, "bootstrap", "S1B"):
        raise ValueError("S1B bootstrap seed differs from registration")
    seeds = [s.base_seed for s in SETS.values()] + list(TRAIN_BASE.values()) + [BOOTSTRAP]
    previous = [s.base_seed for s in old.SETS.values()] + list(old.TRAIN_BASE.values()) + list(old.BOOTSTRAP.values())
    if len(set(seeds)) != len(seeds) or set(seeds) & set(previous + list(old.OLD_BASE_SEEDS)):
        raise ValueError("S1B base seed collision")


def check_disjoint():
    """Only derive IDs, including sealed IDs. Old identities retain their own T1 digest."""
    verify_registry()
    seen = old.old_ids()
    rows = {}

    def take(name, ids):
        if len(set(ids)) != len(ids) or not seen.isdisjoint(ids):
            raise ValueError(f"episode identity overlap: {name}")
        seen.update(ids)
        rows[name] = dict(count=len(ids), ids_blake2b_128=old.ids_digest(ids))

    for s in old.SETS.values():
        take(f"old/{s.name}", s.episode_ids())
    for (stage, run) in old.TRAIN_BASE:
        take(f"old/train/{stage}/{run}", old.train_ids(stage, run, old.PILOT_STEPS if run == "pilot" else old.MAX_STEPS))
    for purpose, s in SETS.items():
        take(purpose, s.ids())
    for run, base in TRAIN_BASE.items():
        count = BATCH * (PILOT_STEPS if run == "pilot" else STEPS)
        take(f"train/{run}", [episode_id(base, "train", i) for i in range(count)])
    return rows


def manifest():
    verify_registry()
    return dict(stage="S1B", task=json.loads(json.dumps(asdict(task_spec()))), layout_digest=LAYOUT_DIGEST,
                generator_version=GENERATOR_VERSION, prereg_blake2b_128=PREREG_DIGEST,
                sets={p: dict(split=s.split, count=s.count, sealed=s.sealed,
                              hops=[1, 2, 3] if p in {"calibration", "pilot", "dev"} else list(range(1, 9)),
                              **({} if s.sealed else dict(base_seed=s.base_seed, query_seed=s.base_seed)))
                      for p, s in SETS.items()}, ids=check_disjoint())


@dataclass(frozen=True)
class OpenToken:
    selected: dict
    release_digest: str


def load_open_token(root, release_path, lock, selection):
    """Bind the complete selected mapping, release bytes, scripts and registered documents."""
    from .s1b_rules import branch_arms, contrasts
    root = Path(root)
    if not root.is_absolute():
        raise old.SealedSetError("S1B opening root must be absolute")
    record = json.loads((root / "reports/slot-loop-v1/s1b/open.json").read_text())
    release = json.loads(Path(release_path).read_text())
    arms = branch_arms(release["branch"])
    selected = record.get("selected_checkpoints", {})
    expected = {f"{arm}/{seed}" for arm in arms for seed in RUN_SEEDS}
    checks = (record.get("stage") == "S1B", set(selected) == expected,
              selected == selection["selected_checkpoints"], len(set(selected.values())) == len(expected),
              all(isinstance(v, str) and len(v) == 32 and all(c in "0123456789abcdef" for c in v) for v in selected.values()),
              record.get("release_blake2b_128") == old.file_digest(release_path),
              record.get("release_blake2b_128") == selection["release_blake2b_128"],
              record.get("analysis_script_blake2b_128") == release["analysis_script_blake2b_128"] == old.file_digest(root / ANALYSIS),
              record.get("branch") == release["branch"], record.get("contrasts") == contrasts(release["branch"]),
              record.get("lock_blake2b_128") == release["lock_blake2b_128"],
              record.get("inventory_blake2b_128") == stable_hash(lock["inventory"]),
              release["status"] == "ready", selection.get("status") == "selected")
    for name, digest in ((PREREG, PREREG_DIGEST), (old.PREREG, old.PREREG_BLAKE2B_128)):
        checks += (old.file_digest(root / name) == digest, record.get(name) == digest)
    if not all(checks):
        raise old.SealedSetError("S1B opening record provenance/mapping mismatch")
    return OpenToken(dict(selected), old.file_digest(release_path))


def eval_set(purpose, *, token=None, checkpoint_digest=None):
    s = SETS[purpose]
    if s.sealed:
        if not isinstance(token, OpenToken) or checkpoint_digest not in token.selected.values():
            raise old.SealedSetError(f"S1B/{purpose} is sealed")
        return replace(s, opened=True)
    return s


def raw_episodes(s):
    if s.sealed and not s.opened:
        raise old.SealedSetError("unopened S1B set cannot generate episodes")
    return generate(task_spec(), s.split, 0, s.count, s.base_seed)


def episodes(s, hops=None):
    # Importing the stratifier imports torch; this function is compute-only.
    from looplora.sym.evaluate import stratify_queries
    allowed = list(range(1, 9)) if s.purpose in {"smoke", "confirmation", "unseen"} else [1, 2, 3]
    if hops is None:
        hops = allowed
    permitted = list(range(1, 9)) if s.purpose == "dev" else allowed
    if not hops or len(set(hops)) != len(hops) or any(h not in permitted for h in hops):
        raise ValueError("unregistered hop selection")
    return stratify_queries(raw_episodes(s), task_spec(), hops, s.base_seed)
