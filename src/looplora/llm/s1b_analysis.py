"""Frozen S1B statistics. Episode axes remain intact across hops and continuation seeds."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import numpy as np

from . import s1b_data as data, s1b_rules as rules


def audit_checkpoints(record):
    """Retained exports and original dev samples bind every row and the selected bytes."""
    root = Path(record["artifact_root"])
    for row in record["dev"]:
        checkpoint = row["checkpoint"]
        if (checkpoint["step"] != row["step"]
                or data.old.file_digest(root / checkpoint["file"]) != checkpoint["blake2b_128"]
                or data.old.file_digest(root / checkpoint["samples"]) != checkpoint["samples_blake2b_128"]):
            raise ValueError("dev checkpoint step/digest does not match retained artifact")
        with np.load(root / checkpoint["samples"], allow_pickle=False) as samples:
            if (int(samples["step"].item()) != row["step"]
                    or samples["checkpoint_blake2b_128"].item() != checkpoint["blake2b_128"]
                    or len(samples["episode_id"]) != row["n"]
                    or float(samples["nll"].astype(np.float64).mean()) != row["nll"]):
                raise ValueError("dev row differs from original checkpoint samples")
    chosen = next(row for row in record["dev"] if row["step"] == record["selected_step"])
    if (data.old.file_digest(root / "selected.pt") != chosen["checkpoint"]["blake2b_128"]
            or record["selected_checkpoint_blake2b_128"] != chosen["checkpoint"]["blake2b_128"]):
        raise ValueError("selected file does not match winning dev checkpoint")
    return True


def select(records, release, lock, release_digest):
    expected = {f"{arm}/{seed}" for arm in rules.branch_arms(release["branch"]) for seed in data.RUN_SEEDS}
    if set(records) != expected:
        raise ValueError("incomplete S1B main records")
    selected = {}
    for label, r in records.items():
        arm, seed = label.split("/")
        if (r.get("status") != "complete" or r.get("all_finite") is not True or r.get("stage") != "S1B"
                or r.get("arm") != arm or r.get("seed") != int(seed) or r.get("steps") != data.STEPS
                or r.get("branch") != release["branch"] or r.get("control") != "shared"
                or r.get("trained_depth") != rules.ARM[arm][1]
                or r.get("release_blake2b_128") != release_digest
                or r.get("science_hash") != release["config_hashes"][label]
                or r.get("dev_ids_blake2b_128") != lock["manifest"]["ids"]["dev"]["ids_blake2b_128"]):
            raise ValueError(f"invalid S1B record provenance: {label}")
        rows = r["dev"]
        if [row["step"] for row in rows] != [0, *rules.checkpoint_steps(data.STEPS)]:
            raise ValueError("incomplete registered dev schedule")
        if any(row["n"] != data.SETS["dev"].count or not np.isfinite(row["nll"]) for row in rows):
            raise ValueError("nonfinite/mismatched dev record")
        step = rules.select_checkpoint([(row["step"], row["nll"]) for row in rows[1:]])
        chosen = next(row for row in rows if row["step"] == step)
        if step != r["selected_step"] or chosen["nll"] > rows[0]["nll"]:
            raise rules.StopForOrchestrator("unstable or mismatched S1B checkpoint selection")
        if not r["freeze"]["digests"]["frozen_unchanged"] or not r["freeze"]["support"]["equal"]:
            raise ValueError("failed frozen-writer evidence")
        audit_checkpoints(r)
        selected[label] = r["selected_checkpoint_blake2b_128"]
    if len(set(selected.values())) != len(selected) or any(not isinstance(v, str) or len(v) != 32 for v in selected.values()):
        raise ValueError("missing/duplicate selected S1B digest")
    return dict(status="selected", stage="S1B", branch=release["branch"], selected_checkpoints=selected,
                release_blake2b_128=release_digest, contrasts=rules.contrasts(release["branch"]))


def validate_samples(samples, identity, *, hops=tuple(range(1, 9))):
    ids, correct, nll = samples["episode_id"], samples["correct"], samples["nll"]
    if (ids.ndim != 1 or len(ids) != identity["count"] or len(set(ids.tolist())) != len(ids)
            or data.old.ids_digest(ids) != identity["ids_blake2b_128"]
            or correct.shape != (len(ids), len(hops)) or nll.shape != correct.shape
            or not np.array_equal(samples["hops"], hops)
            or not np.isin(correct, [0, 1]).all() or not np.isfinite(nll).all()):
        raise ValueError("S1B sample count/IDs/hop coverage/values differ from manifest")


def validate_trajectories(samples, depth):
    n, hops = samples["correct"].shape
    if not np.array_equal(samples.get("trajectory_step"), np.arange(1, depth + 1)):
        raise ValueError("saved trajectory has missing or unordered loop steps")
    for name, width in (("q", 768), ("m", 128), ("z", 128)):
        value = samples.get("trajectory_" + name)
        if value is None or value.shape != (n, hops, depth, width) or not np.isfinite(value).all():
            raise ValueError("saved final-token trajectory shape/values differ from episode/hop/depth order")


def exact_percentile(values, probability):
    """Linear percentile of integer bootstrap sums, including interpolation, as a rational."""
    ordered = np.sort(np.asarray(values, dtype=np.int64))
    position = probability * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    return Fraction(int(ordered[lower])) + (position - lower) * int(ordered[upper] - ordered[lower])


def descriptive_tables(main, sweeps, branch):
    """Registered confirmation report, with no role in any success/continuation decision."""
    def summary(sample):
        c, nll = sample["correct"], np.asarray(sample["nll"], dtype=np.float64)
        if nll.shape != c.shape or not np.isfinite(nll).all():
            raise ValueError("missing/nonfinite descriptive NLL")
        return dict(episodes=len(c), questions=int(c.size), correct=int(c.sum()), em=float(c.mean()), nll=float(nll.mean()),
                    per_hop={str(h): dict(n=len(c), correct=int(c[:, h - 1].sum()), em=float(c[:, h - 1].mean()),
                                          nll=float(nll[:, h - 1].mean())) for h in range(1, 9)})

    def contrast(vector):
        return dict(primary_hops=[2, 3], delta_pp=100 * int(vector[:, :, 1:3].sum()) / vector[:, :, 1:3].size,
                    per_hop_pp={str(h): 100 * int(vector[:, :, h - 1].sum()) / vector[:, :, h - 1].size for h in range(1, 9)},
                    per_seed_pp={str(seed): 100 * int(vector[j, :, 1:3].sum()) / vector[j, :, 1:3].size
                                 for j, seed in enumerate(data.RUN_SEEDS)})

    points = {}
    if branch != "matrix_only":
        counts = {a: np.stack([main[a, s]["correct"].astype(np.int64) for s in data.RUN_SEEDS])
                  for a in rules.branch_arms(branch)}
        points["S1-M1"] = contrast(counts["S1"] - counts["M1"])
        points["Memory_x_Depth"] = contrast((counts["S4"] - counts["S1"]) - (counts["M4"] - counts["M1"]))
    return dict(label="descriptive; does not change registered decisions", episodes=len(next(iter(main.values()))["episode_id"]),
                arms={f"{a}/{s}": summary(sample) for (a, s), sample in main.items()},
                depth_sweeps={f"{a}/{s}@{depth}": summary(sample) for (a, s, depth), sample in sweeps.items()},
                contrasts=points)


def confirmation_markdown(result):
    """Human-readable frozen report; descriptive tables stay separate from decisions."""
    lines = ["# S1B confirmation", "", f"Decision: {result['verdict']}; S2 primary: {result['s2_primary']}.", "",
             "## Registered decisions", "", "| contrast | EM difference (pp) | interval (pp) | passed |",
             "| --- | ---: | --- | --- |"]
    for name, row in result["contrasts"].items():
        lines.append(f"| {name} | {row['delta_pp']:.6f} | {row['interval_pp']} | {row['passed']} |")
    table = result["descriptive"]
    for title, rows in (("Descriptive arm/seed/hop counts, EM and NLL", table["arms"]),
                        ("Descriptive same-checkpoint depth sweeps", table["depth_sweeps"])):
        lines += ["", f"## {title}", "", "| arm/seed[@depth] | hop | correct | n | EM | NLL |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for label, row in rows.items():
            for hop, metrics in row["per_hop"].items():
                lines.append(f"| {label} | {hop} | {metrics['correct']} | {metrics['n']} | {metrics['em']:.6f} | {metrics['nll']:.9f} |")
    lines += ["", "## Descriptive contrasts (no decision role)", "", "| contrast | hops 2–3 EM difference (pp) |",
              "| --- | ---: |"]
    for name, row in table["contrasts"].items():
        lines.append(f"| {name} | {row['delta_pp']:.6f} |")
    if not table["contrasts"]:
        lines.append("| not applicable in matrix-only branch | — |")
    return "\n".join(lines) + "\n"


def paired(main, sweeps, branch, *, seed=data.BOOTSTRAP, draws=10000):
    """10,000 resamples, 500 rows at a time; one shared episode draw for the whole family."""
    arms = rules.branch_arms(branch)
    expected = {(arm, s) for arm in arms for s in data.RUN_SEEDS}
    if set(main) != expected:
        raise ValueError("missing or extra primary arm/seed")
    reference = next(iter(main.values()))
    n = len(reference["episode_id"])
    if not n:
        raise ValueError("empty S1B paired sample")
    for sample in [*main.values(), *sweeps.values()]:
        if (sample["correct"].shape != (n, 8) or not np.array_equal(sample["episode_id"], reference["episode_id"])
                or not np.array_equal(sample["hops"], np.arange(1, 9))
                or not np.isin(sample["correct"], [0, 1]).all()):
            raise ValueError("unpaired episode/hop/seed arrays")
    counts = {arm: np.stack([main[arm, s]["correct"].astype(np.int64) for s in data.RUN_SEEDS]) for arm in arms}
    names = rules.contrasts(branch)
    diffs = {name: counts[name.split("-")[0]] - counts[name.split("-")[1]] for name in names}
    vectors = {name: diff[:, :, 1:3].sum(axis=(0, 2), dtype=np.int64) for name, diff in diffs.items()}
    distribution = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for start in range(0, draws, 500):
        indices = rng.integers(0, n, size=(min(500, draws - start), n))
        for name, vector in vectors.items():
            distribution[name].append(vector[indices].sum(axis=1, dtype=np.int64))
    alpha = Fraction(1, 20 * len(names))
    result = {}
    for name, diff in diffs.items():
        draws_for_contrast = np.concatenate(distribution[name])
        interval = [exact_percentile(draws_for_contrast, probability) for probability in (alpha / 2, 1 - alpha / 2)]
        per_seed_counts = diff[:, :, 1:3].sum(axis=(1, 2))
        total, denominator = int(per_seed_counts.sum()), 3 * n * 2
        hop1_count = int(diff[:, :, 0].sum())
        tests = dict(lower_positive=bool(interval[0] > 0), seeds_positive=bool((per_seed_counts > 0).all()))
        if name == "S4-M4":
            tests["positive_point"] = total > 0
        else:
            tests.update(at_least_three_pp=100 * total >= 3 * denominator,
                         hop1_drop_at_most_two_pp=100 * hop1_count >= -2 * 3 * n)
        result[name] = dict(delta_pp=100 * total / denominator,
                            interval_pp=[float(100 * bound / denominator) for bound in interval],
                            interval_integer_sum=[dict(numerator=bound.numerator, denominator=bound.denominator) for bound in interval],
                            confidence=float(1 - alpha), per_seed_pp=(100 * per_seed_counts / (2 * n)).tolist(),
                            hop1_pp=100 * hop1_count / (3 * n), tests=tests, passed=all(tests.values()))
    continuation = {}
    for arm in (a for a in arms if a.endswith("4")):
        contrasts = f"{arm}-{arm[0]}1"
        keys = {(arm, s, depth) for s in data.RUN_SEEDS for depth in (1, 2, 4, 8)}
        if not keys.issubset(sweeps):
            raise ValueError("missing descriptive trained-L4 depth sweep")
        if any(not np.array_equal(sweeps[arm, s, 4]["correct"], main[arm, s]["correct"]) for s in data.RUN_SEEDS):
            raise ValueError("sweep L4 differs from primary selected checkpoint")
        delta = np.stack([sweeps[arm, s, 4]["correct"].astype(np.int64) - sweeps[arm, s, 1]["correct"].astype(np.int64)
                          for s in data.RUN_SEEDS])[:, :, 1:3].sum(axis=(1, 2))
        screen = bool(delta.sum() > 0 and (delta >= 0).all())
        continuation[arm[0]] = dict(confirmed_loop=result[contrasts]["passed"], sweep_sign_screen=screen,
                                    qualified=result[contrasts]["passed"] and screen,
                                    sweep_per_seed_pp=(100 * delta / (2 * n)).tolist())
    matrix, slot = continuation["M"]["qualified"], continuation.get("S", {}).get("qualified", False)
    storage = result.get("S4-M4", {}).get("passed", False)
    primary = "soft" if slot and (storage or not matrix) else "matrix" if matrix else None
    return dict(stage="S1B", branch=branch, contrasts=result, continuation=continuation,
                s2_primary=primary, verdict="LOOP SUPPORTED" if primary else "NOT ADVANCED",
                bootstrap_seed=seed, resamples=draws, conditional_on="S-RE, continuation seeds 17/29/43, E12/(8,4)",
                descriptive=descriptive_tables(main, sweeps, branch))
