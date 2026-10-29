#!/usr/bin/env python3
"""WP-T11 (G1): tables for the paper revision, from committed CSV/JSON files only (stdlib).

Reads, for the two P1-L-family confirmations at rank 128 (primary setting, memory-only protocol):
  reports/p1l/r128/{accuracy.csv,inference.csv,endpoints.json}   Qwen3-0.6B-Base
  reports/p5l/r128/{accuracy.csv,inference.csv,endpoints.json}   Qwen3-1.7B-Base
and prints (1) the M1-L4 matrix of seed-mean EM by evaluation depth L and hop with the oracle-stop row,
(2) the same-rank main table (B4@1, M1-L4@4, B8@4: hop-wise EM, adapter parameters, persistent state bytes,
measured time per question), and (3) the seed-level statistics of the headline contrasts beside the registered
episode-bootstrap intervals. Every printed contrast is cross-checked against `endpoints.json` (assertions).

Usage (no torch, no third-party packages):
  python3 scripts/paper_tables_g1.py [> docs/paper/g1_tables.txt]
  python3 scripts/paper_tables_g1.py --tex docs/paper/tex/g1_tables.tex   (LaTeX table fragments for the manuscript)
"""
import csv
import json
import math
import os
import sys
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKBONES = (("Qwen3-0.6B-Base", "reports/p1l/r128"), ("Qwen3-1.7B-Base", "reports/p5l/r128"))
SEEDS = (17, 29, 43)
HOPS = tuple(range(1, 9))
RANK = 128
CONF = 0.975  # two-sided level of the registered episode intervals; reused for the seed-level interval


def t_quantile_df2(two_sided):
    """Exact Student-t quantile for 2 degrees of freedom: t = (2p-1) / sqrt(2 p (1-p)), p = (1 + two_sided) / 2."""
    p = (1.0 + two_sided) / 2.0
    return (2.0 * p - 1.0) / math.sqrt(2.0 * p * (1.0 - p))


def load(rel):
    with open(os.path.join(ROOT, rel, "accuracy.csv"), newline="") as f:
        acc = list(csv.DictReader(f))
    with open(os.path.join(ROOT, rel, "inference.csv"), newline="") as f:
        inf = list(csv.DictReader(f))
    with open(os.path.join(ROOT, rel, "endpoints.json")) as f:
        end = json.load(f)
    return acc, inf, end


def cell(acc, reader, L, setting, hop, seed):
    rows = [r for r in acc if r["reader"] == reader and int(r["L"]) == L and r["setting"] == setting
            and r["protocol"] == "memory_only" and int(r["hop"]) == hop and int(r["seed"]) == seed
            and int(r["rank"]) == RANK]
    assert len(rows) == 1, (reader, L, setting, hop, seed, len(rows))
    r = rows[0]
    value = Fraction(int(r["correct"]), int(r["n"]))  # exact; the csv `accuracy` column is checked against it
    assert abs(float(value) - float(r["accuracy"])) < 1e-12, (reader, L, setting, hop, seed)
    return value


def seed_mean(acc, reader, L, setting, hop):
    return sum(cell(acc, reader, L, setting, hop, s) for s in SEEDS) / len(SEEDS)


def pooled_seed(acc, reader, L, setting, hops, seed):
    return sum(float(cell(acc, reader, L, setting, h, seed)) for h in hops) / len(hops)


def pct(x, nd=1):
    """Percent of an exact fraction, rounded half up (not binary-float rounding)."""
    scaled = Fraction(x) * 100 * 10 ** nd
    q = int(scaled + Fraction(1, 2)) if scaled >= 0 else -int(-scaled + Fraction(1, 2))
    s = f"{abs(q):0{nd + 1}d}"
    return ("-" if q < 0 else "") + (s[:-nd] + "." + s[-nd:] if nd else s)


def matrix(acc):
    m = {L: {h: seed_mean(acc, "M1-L4", L, "primary", h) for h in HOPS} for L in range(1, 9)}
    best = {h: max(range(1, 9), key=lambda L: m[L][h]) for h in HOPS}
    return m, best


def print_matrix(name, acc):
    m, best = matrix(acc)
    print(f"### {name}: M1-L4 seed-mean EM (%) by evaluation depth L and hop (primary, memory-only, rank 128, "
          f"3 seeds, 2,000 episodes each; trained depth L=4, trained hops 1-3)\n")
    print("| L | " + " | ".join(f"h{h}" for h in HOPS) + " |")
    print("| --- | " + " | ".join("---" for _ in HOPS) + " |")
    for L in range(1, 9):
        cells = []
        for h in HOPS:
            s = pct(m[L][h])
            cells.append(f"**{s}**" if best[h] == L else s)
        print(f"| {L}{' (trained)' if L == 4 else ''} | " + " | ".join(cells) + " |")
    print("| oracle stop: best L per hop (uses labels; upper bound; descriptive) | "
          + " | ".join(f"{pct(m[best[h]][h])} (L={best[h]})" for h in HOPS) + " |")
    print()
    hi = [h for h in HOPS if h >= 4]
    print(f"{name} summary values:")
    print(f"- at the trained depth L=4: hops 1-4 = " + " / ".join(pct(m[4][h]) for h in (1, 2, 3, 4))
          + "; hops 5-8 = " + " / ".join(pct(m[4][h]) for h in (5, 6, 7, 8)))
    print(f"- oracle stop, hops 4-8: " + " / ".join(f"h{h} {pct(m[best[h]][h])} at L={best[h]}" for h in hi)
          + f"; range {pct(min(m[best[h]][h] for h in hi))}-{pct(max(m[best[h]][h] for h in hi))}")
    print(f"- L=4 hops 4-8 range: {pct(min(m[4][h] for h in hi))}-{pct(max(m[4][h] for h in hi))}")
    print(f"- hop 1: L=1 {pct(m[1][1])}, best {pct(m[best[1]][1])} at L={best[1]}, L=4 {pct(m[4][1])}, L=8 {pct(m[8][1])}")
    print(f"- at L=8: hops 1-8 = " + " / ".join(pct(m[8][h]) for h in HOPS))
    print(f"- oracle stop, all hops 1-8: " + " / ".join(pct(m[best[h]][h]) for h in HOPS))
    print(f"- per-hop ratio of best cells (h+1 / h): "
          + " / ".join(f"{float(m[best[h + 1]][h + 1] / m[best[h]][h]):.2f}" for h in range(1, 8)))
    # every hop's best depth, for the 'peaks near h+1 or h+2' statement
    print(f"- best L per hop: " + " / ".join(f"h{h}:L{best[h]}" for h in HOPS))
    print()
    return m, best


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


READERS = (("B4@1", "B4", 1), ("M1-L4@4", "M1-L4", 4), ("B8@4", "B8", 4))


def main_rows(acc, inf):
    """Rows of the same-rank main table: hop-wise EM (exact fractions), adapter parameters, state bytes, time."""
    rows = []
    for label, reader, L in READERS:
        per = {h: seed_mean(acc, reader, L, "primary", h) for h in HOPS}
        sel = [r for r in inf if r["reader"] == reader and int(r["L"]) == L and r["protocol"] == "memory_only"
               and int(r["rank"]) == RANK]
        assert sorted(int(r["seed"]) for r in sel) == list(SEEDS), (reader, len(sel))
        params = {int(r["adapter_parameters"]) for r in sel}
        assert len(params) == 1
        rows.append(dict(label=label, per=per, h23=(per[2] + per[3]) / 2, params=params.pop(),
                         state=4 * RANK * RANK,  # FP32 delta memory r_mem x r_mem per episode (llm/run.py state_bytes)
                         ms=1000.0 * mean(float(r["mean_seconds_per_question"]) for r in sel)))
    return rows


def other_hop_ranges(acc, lo=4, hi=8):
    """Range over hops lo..hi of the seed-mean EM of the comparators at their own depth (B4@1, B8@4, B3-L4@4)."""
    out = {}
    for label, reader, L in (("B4@1", "B4", 1), ("B8@4", "B8", 4), ("B3-L4@4", "B3-L4", 4)):
        vals = [seed_mean(acc, reader, L, "primary", h) for h in range(lo, hi + 1)]
        out[label] = (min(vals), max(vals))
    return out


def print_main_table(name, acc, inf):
    print(f"### {name}: same-rank (rank 128) main-table values, primary setting, memory-only, seed means\n")
    rows = main_rows(acc, inf)
    print("| reader | " + " | ".join(f"h{h}" for h in HOPS) + " | hops 2-3 | adapter parameters | "
          "persistent state bytes per episode | time per question (ms) |")
    print("| --- | " + " | ".join("---" for _ in HOPS) + " | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['label']} | " + " | ".join(pct(r['per'][h], 2) for h in HOPS) + f" | {pct(r['h23'], 2)} | "
              f"{r['params']:,} | {r['state']:,} | {r['ms']:.3f} |")
    base = {}
    for reader, L in (("B4", 1), ("M1-L4", 4), ("B8", 4), ("B5-L4", 4), ("B3-L4", 4)):
        sel = [r for r in inf if r["reader"] == reader and int(r["L"]) == L and r["protocol"] == "memory_only"
               and int(r["rank"]) == RANK]
        base[reader] = 1000.0 * mean(float(r["mean_seconds_per_question"]) for r in sel)
    print(f"\nTime ratios to B4@1 ({base['B4']:.3f} ms): M1-L4@4 {base['M1-L4'] / base['B4']:.2f}, "
          f"B8@4 {base['B8'] / base['B4']:.2f}; B5-L4@4 {base['B5-L4']:.3f} ms, B3-L4@4 {base['B3-L4']:.3f} ms.")
    pa = {r["label"]: r["params"] for r in rows}
    print(f"B8 minus B4 adapter parameters: {pa['B8@4'] - pa['B4@1']:,} "
          f"({100.0 * (pa['B8@4'] - pa['B4@1']) / pa['B4@1']:.1f}%).")
    rg = other_hop_ranges(acc)
    print("Hops 4-8 at the comparators' own depth (seed-mean EM %, min-max): "
          + "; ".join(f"{k} {pct(v[0])}-{pct(v[1])}" for k, v in rg.items()) + ".\n")


# headline contrasts: (label, endpoints key, reader a, L_a, reader b, L_b, setting, hops)
CONTRASTS = (
    ("B6 contrast M1-L4@4 - B4@1, hops 2-3", "M1_vs_B4", "M1-L4", 4, "B4", 1, "primary", (2, 3)),
    ("G1-L reach B8@4 - B4@1, hops 2-3", "G1_reach", "B8", 4, "B4", 1, "primary", (2, 3)),
    ("G1-L hop 1 B8@4 - B4@1", "G1_hop1", "B8", 4, "B4", 1, "primary", (1,)),
    ("G1-L typed replication B8@4 - B4@1, hops 2-3", "G1_typed", "B8", 4, "B4", 1, "typed", (2, 3)),
    ("G2-L B8@4 - B5-L4@4 (fixed query), hops 2-3", "G2_fixed_query", "B8", 4, "B5-L4", 4, "primary", (2, 3)),
    ("G2-L B8@4 - B3-L4@4 (memory-free), hops 2-3", "G2_no_memory", "B8", 4, "B3-L4", 4, "primary", (2, 3)),
)


def seed_level(effects, conf=CONF):
    n = len(effects)
    mu = mean(effects)
    var = sum((e - mu) ** 2 for e in effects) / (n - 1)
    se = math.sqrt(var / n)
    t = t_quantile_df2(conf)
    return mu, se, t, mu - t * se, mu + t * se


def print_seed_stats(name, acc, end):
    print(f"### {name}: headline contrasts at rank 128, per-seed effects and seed-level intervals (pp)\n")
    print("| contrast | seed 17 | seed 29 | seed 43 | seed mean | seed-level 97.5% t-interval (df=2) | "
          "episode bootstrap 97.5% (registered) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    out = []
    for label, key, ra, La, rb, Lb, setting, hops in CONTRASTS:
        eff = [100.0 * (pooled_seed(acc, ra, La, setting, hops, s) - pooled_seed(acc, rb, Lb, setting, hops, s))
               for s in SEEDS]
        c = end["contrasts"][key]
        # cross-check against the committed endpoint record (per-seed effects and pooled difference)
        for e, ps in zip(eff, c["per_seed"]):
            assert abs(e - 100.0 * ps["difference"]) < 1e-9, (key, e, ps)
        assert abs(mean(eff) - 100.0 * c["difference"]) < 1e-9, key
        mu, se, t, lo, hi = seed_level(eff)
        print(f"| {label} | " + " | ".join(f"{e:+.3f}" for e in eff)
              + f" | {mu:+.3f} | [{lo:+.2f}, {hi:+.2f}] (se {se:.3f}, t {t:.3f}) | "
              f"[{100.0 * c['ci_low']:+.3f}, {100.0 * c['ci_high']:+.3f}] |")
        out.append((key, eff, mu, lo, hi, 100.0 * c["ci_low"], 100.0 * c["ci_high"]))
    print("\nSeed-level interval: mean of the three seed effects +/- t(0.9875; df=2) x sample standard error "
          f"(t = {t_quantile_df2(CONF):.3f}); it treats the three fitted seeds as exchangeable draws and is conditional on "
          "one oracle code table and one set of 2,000 confirmation episodes. Same two-sided level as the registered "
          "97.5% episode intervals.\n")
    return out


def tex_num(s):
    return s.replace("-", r"\mi{}") if s.startswith("-") else s


def write_tex(path):
    """LaTeX fragments (booktabs tabulars) with the same values as the text output; tags as comments."""
    out = ["% Generated by scripts/paper_tables_g1.py --tex; do not edit by hand."]
    short = {"Qwen3-0.6B-Base": "Small", "Qwen3-1.7B-Base": "Large"}
    data = {n: load(rel) for n, rel in BACKBONES}
    names = [n for n, _ in BACKBONES]
    out += [r"\newcommand{\gtMainEM}{%", r"\begin{tabular}{@{}ll" + "r" * 8 + r"@{}}", r"\toprule",
            "Backbone & Reader & " + " & ".join(f"hop {h}" for h in HOPS) + r" \\ \midrule"]
    for n in names:
        acc, inf, end = data[n]
        for r in main_rows(acc, inf):
            out.append(f"{n.replace('Qwen3-', '')} & {r['label']} & " + " & ".join(pct(r['per'][h], 2) for h in HOPS) + r" \\")
        if n != names[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}}  % LP23", ""]
    out += [r"\newcommand{\gtMainCost}{%", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
            r"Backbone & Reader & Adapter parameters & State bytes per episode & Time per question (ms) & Time relative to B4@1 \\ \midrule"]
    for n in names:
        acc, inf, end = data[n]
        rows = main_rows(acc, inf)
        b4 = rows[0]["ms"]
        for r in rows:
            out.append(f"{n.replace('Qwen3-', '')} & {r['label']} & {r['params']:,} & {r['state']:,} & {r['ms']:.3f} & {r['ms'] / b4:.2f} " + r"\\")
        if n != names[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}}  % LP23", ""]
    for n in names:
        acc, inf, end = data[n]
        m, best = matrix(acc)
        k = short[n]
        out += [r"\newcommand{\gtMatrix" + k + r"}{%", r"\begin{tabular}{@{}l" + "r" * 8 + r"@{}}", r"\toprule",
                "$L$ & " + " & ".join(f"hop {h}" for h in HOPS) + r" \\ \midrule"]
        for L in range(1, 9):
            cells = [(r"\textbf{" + pct(m[L][h]) + "}") if best[h] == L else pct(m[L][h]) for h in HOPS]
            out.append(f"{L}{' (trained)' if L == 4 else ''} & " + " & ".join(cells) + r" \\")
        out += [r"\midrule", "Oracle stop ($L$) & " + " & ".join(f"{pct(m[best[h]][h])} ({best[h]})" for h in HOPS) + r" \\",
                r"\bottomrule", r"\end{tabular}}  % LP21", ""]
        out += [r"\newcommand{\gtSeeds" + k + r"}{%", r"\begin{tabular}{@{}p{4.4cm}rrrrll@{}}", r"\toprule",
                r"Contrast (pp) & Seed 17 & Seed 29 & Seed 43 & Mean & Seed-level 97.5\% & Episode bootstrap 97.5\% \\ \midrule"]
        for label, key, ra, La, rb, Lb, setting, hops in CONTRASTS:
            eff = [100.0 * (pooled_seed(acc, ra, La, setting, hops, s) - pooled_seed(acc, rb, Lb, setting, hops, s)) for s in SEEDS]
            c = end["contrasts"][key]
            mu, se, tq, lo, hi = seed_level(eff)

            def fm(x, nd=3):
                return tex_num(f"{x:+.{nd}f}")
            lab = label.replace(" - ", r" $-$ ").replace("hops 2-3", "hops 2 to 3")
            out.append(f"{lab} & " + " & ".join(fm(e) for e in eff) + f" & {fm(mu)} & [{fm(lo, 2)}, {fm(hi, 2)}] & "
                       f"[{fm(100.0 * c['ci_low'])}, {fm(100.0 * c['ci_high'])}] " + r"\\")
        out += [r"\bottomrule", r"\end{tabular}}  % LP22", ""]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--tex":
        write_tex(sys.argv[2])
        return 0
    print("# WP-T11 G1 tables (scripts/paper_tables_g1.py; sources: reports/p1l/r128 and reports/p5l/r128)\n")
    for name, rel in BACKBONES:
        acc, inf, end = load(rel)
        assert end["rank"] == RANK and abs(end["confidence"] - CONF) < 1e-12
        print_matrix(name, acc)
        print_main_table(name, acc, inf)
        print_seed_stats(name, acc, end)


if __name__ == "__main__":
    sys.exit(main())
