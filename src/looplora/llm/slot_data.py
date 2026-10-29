"""SLOT-LOOP-v1 data registry, purpose-seed guard and episode-id checks (docs/prereg/SLOT-LOOP-v1.md, "Data and manifest").

Torch-free: the manifest script and the login-node checks import it. Every evaluation set is defined by its own base seed
(`eval_base`), so an index window under another seed cannot protect it; the confirmation and unseen-name sets of S1A are
sealed here and open only with an `OpenToken` built from the orchestrator's `reports/slot-loop-v1/s1a-open.json`. The sealed
base seeds occur in this file only (tests scan `src/` and `scripts/`).
"""
from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass, replace
from pathlib import Path

from looplora.data.relations import GENERATOR_VERSION, TaskSpec
from looplora.infra.seeds import derive_seed
from .sq2_release import PARENT as D128_3

STUDY = "slot-loop-v1"
BASE_SEED = 20261029
LAYOUT_DIGEST = "b7f93c1ca56e5dd1f79a9dccfe2f6054"
PREREG = "docs/prereg/SLOT-LOOP-v1.md"
PREREG_BLAKE2B_128 = "56e8c611c01e0ef9b7ae32db97b0e334"  # the file as registered on master (5f04a5c)
ANALYSIS_SCRIPT = "scripts/analyze_slot_study.py"
OPEN_RECORD = "reports/slot-loop-v1/s1a-open.json"
RUN_SEEDS = (17, 29, 43)
BATCH = 32
MAX_STEPS = 6000
PILOT_STEPS = 200

# The parents: S-RE (the SQ2 arm whose weights both S1A arms start from) and D128-3 (the delta-store recheck of S0).
PARENT_FOLDERS = {"S-RE": "/work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE", "D128-3": D128_3["folder"]}
PARENT_RECORD_FILES = {"S-RE": ("writer.pt", "checkpoint.pt", "record.json", "training.json", "val-6000/samples.npz"),
                       "D128-3": ("writer.pt", "checkpoint.pt", "record.json", "training.json", "val-22370/samples.npz")}

# S-RE's task (its record.json task block: T1, E 16, Q 4, V 1024, four relations, head sharing 0.3, hop 1 only).
TASK = dict(name="t1", E=16, Q=4, V=1024, n_rel=4, seen=(0, 768), unseen=(768, 1024), head_sharing=0.3, hops=(1,))


def task_spec() -> TaskSpec:
    return TaskSpec(**TASK)


@dataclass(frozen=True)
class EvalSet:
    stage: str
    purpose: str
    split: str
    start: int
    count: int
    base_seed: int
    sealed: bool = False
    opened: bool = False

    @property
    def name(self) -> str:
        return f"{self.stage}/{self.purpose}"

    def episode_ids(self) -> list[int]:
        return [episode_seed(self.base_seed, self.split, i) for i in range(self.start, self.start + self.count)]


# (stage, purpose) -> set; the base seeds are the integers derive_seed(BASE_SEED, STUDY, purpose, stage) resolves to.
_SETS = (
    EvalSet("S0", "smoke", "train", 0, 64, 8726904610828922277),
    EvalSet("S0", "calibration", "val", 0, 2048, 7872142726734868149),
    EvalSet("S0", "dev", "val", 0, 1000, 4206094924762162946),
    EvalSet("S1A", "smoke", "train", 0, 64, 5181101083092932984),
    EvalSet("S1A", "pilot", "val", 0, 1000, 2691891574861457280),
    EvalSet("S1A", "dev", "val", 0, 1000, 6518772247750212169),
    EvalSet("S1A", "confirmation", "test", 0, 2000, 4685753055482933683, sealed=True),
    EvalSet("S1A", "unseen", "test_unseen", 0, 2000, 5570148062033127911, sealed=True),
)
SETS = {(s.stage, s.purpose): s for s in _SETS}
# (stage, run) -> training base seed: derive_seed(BASE_SEED, STUDY, "train", stage, run); the LR-pilot stream's run is "pilot".
TRAIN_BASE = {
    ("S1A", 17): 5113975105655166458,
    ("S1A", 29): 9197923827134630300,
    ("S1A", 43): 5009954469258593308,
    ("S1A", "pilot"): 2442741374644229280,
}
BOOTSTRAP = {"S0": 2079281455337892545, "S1A": 9056755664197379988}  # numpy default_rng seeds: derive_seed(.., "bootstrap", stage)

# Old protections that must not be touched or reused (docs/prereg/SLOT-LOOP-v1.md, "No overlap"): (split, base seed, start, count).
OLD_WINDOWS = (("val", 4242, 0, 3000), ("val", 4242, 200000, 4000), ("test", 4242, 100000, 12000))
OLD_TRAIN = (  # (base seed, episodes): the D128-3 and S-RE training streams
    (derive_seed(17, "llm", "writer-data"), 22370 * BATCH),
    (derive_seed(17, "llm", "sq2-continuation"), 6000 * BATCH),
)
OLD_BASE_SEEDS = (4242, derive_seed(17, "llm", "writer-data"), derive_seed(17, "llm", "sq2-continuation"),
                  derive_seed(20261027, "p1l-smoke-eval"))


def eval_base(stage: str, purpose: str) -> int:
    return derive_seed(BASE_SEED, STUDY, purpose, stage)


def train_base(stage: str, run) -> int:
    return derive_seed(BASE_SEED, STUDY, "train", stage, run)


def bootstrap_seed(stage: str) -> int:
    return derive_seed(BASE_SEED, STUDY, "bootstrap", stage)


def verify_registry() -> None:
    """Re-derive every registered integer; a mismatch is a stop (the manifest script and the tests call this)."""
    for (stage, purpose), s in SETS.items():
        if s.base_seed != eval_base(stage, purpose):
            raise ValueError(f"{stage}/{purpose}: the registered base seed is not derive_seed(...)")
    for (stage, run), seed in TRAIN_BASE.items():
        if seed != train_base(stage, run):
            raise ValueError(f"train {stage}/{run}: the registered base seed is not derive_seed(...)")
    for stage, seed in BOOTSTRAP.items():
        if seed != bootstrap_seed(stage):
            raise ValueError(f"bootstrap {stage}: the registered seed is not derive_seed(...)")
    seeds = [s.base_seed for s in _SETS] + list(TRAIN_BASE.values()) + list(BOOTSTRAP.values())
    if len(set(seeds)) != len(seeds) or set(seeds) & set(OLD_BASE_SEEDS):
        raise ValueError("registered base seeds must differ from each other and from the old ones")


def episode_seed(base_seed: int, split: str, index: int) -> int:
    """The episode id (`EpisodeArrays.episode_seed`): derive_seed(base, "relations", generator version, layout digest, split, index)."""
    return derive_seed(base_seed, "relations", GENERATOR_VERSION, LAYOUT_DIGEST, split, index)


def train_ids(stage: str, run, steps: int) -> list[int]:
    return [episode_seed(TRAIN_BASE[stage, run], "train", i) for i in range(steps * BATCH)]


def ids_digest(ids) -> str:
    """blake2b-128 of the episode ids as little-endian int64."""
    packed = array("q", ids)
    if packed.itemsize != 8:
        raise RuntimeError("expected an 8-byte array('q')")
    return hashlib.blake2b(packed.tobytes() if _LITTLE else _swapped(packed), digest_size=16).hexdigest()


_LITTLE = array("q", [1]).tobytes()[0] == 1


def _swapped(packed: array) -> bytes:
    swapped = array("q", packed)
    swapped.byteswap()
    return swapped.tobytes()


def old_ids() -> set[int]:
    ids: set[int] = set()
    for split, base, start, count in OLD_WINDOWS:
        ids.update(episode_seed(base, split, i) for i in range(start, start + count))
    for base, count in OLD_TRAIN:
        ids.update(episode_seed(base, "train", i) for i in range(count))
    return ids


def check_disjoint(max_steps: int = MAX_STEPS) -> dict:
    """The ids of every registered set, of the first `32 * max_steps` episodes of each S1A train stream and of the pilot
    stream's first PILOT_STEPS * 32 = 6,400 are pairwise disjoint and disjoint from the old protected ones. Returns the
    per-set counts and digests; raises on any overlap."""
    seen: set[int] = old_ids()
    rows = {}

    def take(name: str, ids: list[int]) -> None:
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: repeated episode ids")
        if not seen.isdisjoint(ids):
            raise ValueError(f"{name}: episode ids overlap an old or an earlier registered set")
        seen.update(ids)
        rows[name] = dict(count=len(ids), ids_blake2b_128=ids_digest(ids))

    for s in _SETS:
        take(f"eval/{s.name}", s.episode_ids())
    for (stage, run), _ in TRAIN_BASE.items():
        steps = PILOT_STEPS if run == "pilot" else max_steps
        take(f"train/{stage}/{run}", train_ids(stage, run, steps))
    return rows


def manifest() -> dict:
    """The record `manifest.lock.json` freezes: identity fields, every set and its id digest, the old windows checked."""
    verify_registry()
    return dict(study=STUDY, base_seed=BASE_SEED, generator_version=GENERATOR_VERSION, layout_digest=LAYOUT_DIGEST,
                task=dict(TASK), prereg=PREREG, prereg_blake2b_128=PREREG_BLAKE2B_128, batch=BATCH,
                sets={s.name: dict(split=s.split, start=s.start, count=s.count, sealed=s.sealed,
                                   **({} if s.sealed else dict(base_seed=s.base_seed, query_seed=s.base_seed))) for s in _SETS},
                train_base={f"{stage}/{run}": seed for (stage, run), seed in TRAIN_BASE.items()},
                bootstrap=dict(BOOTSTRAP), old_windows=[list(w) for w in OLD_WINDOWS],
                old_train=[list(t) for t in OLD_TRAIN], ids=check_disjoint())


# ----------------------------------------------------------------------------------------------------- purpose-seed guard

def file_digest(path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(root=None) -> str:
    """blake2b-128 of sorted source paths and bytes under src; generated bytecode is excluded."""
    source = (Path(root) if root is not None else Path(__file__).resolve().parents[3]) / "src"
    digest = hashlib.blake2b(digest_size=16)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(source).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def inventory() -> dict:
    """blake2b-128 of the real parents' files. D128-3's writer.pt and checkpoint.pt are registered and must match."""
    out = {name: {file: file_digest(Path(PARENT_FOLDERS[name]) / file) for file in files}
           for name, files in PARENT_RECORD_FILES.items()}
    if any(out["D128-3"][file] != digest for file, digest in D128_3["digests"].items()):
        raise ValueError("D128-3 differs from its registered digests")
    return out


class SealedSetError(PermissionError):
    """A sealed evaluation set was requested without an open token (or for a checkpoint the token does not name)."""


@dataclass(frozen=True)
class OpenToken:
    prereg: str
    analysis_script: str
    selected_checkpoints: dict[str, str]
    reference_checkpoint: str
    reference_reads: dict

    @property
    def checkpoints(self) -> frozenset:
        return frozenset([*self.selected_checkpoints.values(), self.reference_checkpoint])


def load_open_token(root, inventory: dict, release: dict) -> OpenToken:
    """The token of the orchestrator's open record: valid only if the registered prereg, the frozen analysis script and the six
    selected checkpoints of the record are what is on disk and in the registry."""
    root = Path(root)
    if not root.is_absolute():
        raise SealedSetError("the open record root must be absolute")
    record = json.loads((root / OPEN_RECORD).read_text())
    selected = record.get("selected_checkpoints", {})
    if (record.get("prereg_blake2b_128") != PREREG_BLAKE2B_128 or file_digest(root / PREREG) != PREREG_BLAKE2B_128
            or set(selected) != {f"{arm}/{seed}" for arm in ("matrix_L1", "soft_slot_L1") for seed in RUN_SEEDS}
            or any(not isinstance(v, str) or len(v) != 32 or any(c not in "0123456789abcdef" for c in v) for v in selected.values())
            or len(set(selected.values())) != 6
            or release.get("status") != "ready" or release.get("prereg_blake2b_128") != PREREG_BLAKE2B_128
            or record.get("analysis_script_blake2b_128") != release.get("analysis_script_blake2b_128")
            or record.get("analysis_script_blake2b_128") != file_digest(root / ANALYSIS_SCRIPT)):
        raise SealedSetError("the open record does not match the registered prereg, the analysis script or six checkpoints")
    reference = inventory["S-RE"]["writer.pt"]
    if reference in selected.values() or release.get("parent", {}).get("writer_blake2b_128") != reference:
        raise SealedSetError("the ready release reference differs from the locked inventory")
    reads = {"s_re_native": None, "matrix_step0": release["arms"]["matrix_L1"]["read"],
             "soft_step0": release["arms"]["soft_slot_L1"]["read"]}
    return OpenToken(PREREG_BLAKE2B_128, record["analysis_script_blake2b_128"], dict(selected), reference, reads)


def eval_set(stage: str, purpose: str, *, token: OpenToken | None = None, checkpoint_digest: str | None = None) -> EvalSet:
    """The registered set; a sealed set needs an open token that names the checkpoint about to be evaluated."""
    try:
        s = SETS[stage, purpose]
    except KeyError:
        raise ValueError(f"no registered set {stage}/{purpose}") from None
    if s.sealed and (token is None or token.prereg != PREREG_BLAKE2B_128 or checkpoint_digest not in token.checkpoints):
        raise SealedSetError(f"{s.name} is sealed: it opens only with the orchestrator's open record and a named checkpoint")
    return replace(s, opened=True) if s.sealed else s
