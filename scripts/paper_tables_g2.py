#!/usr/bin/env python3
"""WP-T12 part 1: the numbers of G2 (LSR), G3 (B8W) and G4 (CTC) as the paper prints them (stdlib only, no torch).

Reads committed files only:
  reports/lsr/{accuracy.csv,results.json,thresholds.json}      (G2, label-free stop rules; WP-L9)
  reports/b8w/results/{endpoints.json,accuracy.csv,drift.csv}   (G3, B8 warm start; WP-L10)
  reports/ctc/{decision.json,eval-metrics.json}                 (G4, classify-then-code writer; WP-L11)
Every number is checked against the value the study's own report states (EXPECT below), so a drift between the
reports and the paper fails loudly.

    python3 scripts/paper_tables_g2.py                              # markdown/text to stdout
    python3 scripts/paper_tables_g2.py --tex docs/paper/tex/g2_tables.tex   # the LaTeX table of the LSR matrix
Rounding: exact decimal arithmetic on the stored value, half up (so 54.25 prints as 54.3; LSR.md rounds that cell to 54.2).
"""
import argparse
import csv
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rd(*p):
    return os.path.join(ROOT, *p)


def jload(*p):
    with open(rd(*p)) as f:
        return json.load(f)


def q(x, nd=1, scale=1):
    """Half-up rounding of x*scale to nd decimals, exact on repr(x)."""
    v = (Decimal(repr(float(x))) * scale).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return v


def pct(x, nd=1):
    return str(q(x, nd, 100))


def pp(x, nd=2, plus=True):
    v = q(x, nd)
    s = str(v)
    if plus and v > 0:
        s = "+" + s
    return s


def iv(lo, hi, nd=2):
    return f"[{pp(lo, nd, False)}, {pp(hi, nd, False)}]"


# values the studies' own reports state (the paper must agree with them)
EXPECT = {
    "lsr_gain48": {"0.6b": "3.90", "1.7b": "5.68"},
    "lsr_gain48_iv": {"0.6b": "[3.35, 4.43]", "1.7b": "[5.09, 6.25]"},
    "lsr_hop1": {"0.6b": "6.28", "1.7b": "4.70"},
    "lsr_hop1_iv": {"0.6b": "[5.09, 7.51]", "1.7b": "[3.48, 5.92]"},
    "lsr_hop2": {"0.6b": "-10.83", "1.7b": "-12.27"},
    "lsr_hop2_iv": {"0.6b": "[-11.85, -9.87]", "1.7b": "[-13.32, -11.23]"},
    "lsr_reads": {"0.6b": "5.60", "1.7b": "5.66"},
    "lsr_vsL8_48": {"0.6b": "-3.54", "1.7b": "-5.05"},
    "b8w_delta": "+0.90 [0.09, 1.71]",
    "ctc_gain": "+24.05 [21.80, 26.30]",
}


def check(label, got, want):
    if got != want:
        sys.exit(f"MISMATCH {label}: script {got!r}, report {want!r}")


# ---------------------------------------------------------------- LSR (G2)
BK = {"0.6b": "Qwen3-0.6B-Base", "1.7b": "Qwen3-1.7B-Base"}
ROWS = [("L2", "L = 2"), ("L4", "L = 4 (trained)"), ("L6", "L = 6"), ("L8", "L = 8"),
        ("R2", "R2 (label-free stop)"), ("O1", "O1 (best fixed L per hop, chosen on validation)"),
        ("O2", "O2 (best test cell per hop, labels used)")]


def lsr():
    acc = {}
    with open(rd("reports/lsr/accuracy.csv")) as f:
        for r in csv.DictReader(f):
            if r["seed"] == "pooled":
                acc[(r["backbone"], r["series"], int(r["hop"]))] = float(r["em"])
    res = jload("reports/lsr/results.json")
    thr = jload("reports/lsr/thresholds.json")
    out = {"acc": acc, "res": res, "thr": thr}
    return out


def lsr_hops48(acc, bk, series):
    # pooled over hops 4-8: the mean of five equal-sized hop cells (2,000 questions each, three seeds)
    return sum(acc[(bk, series, h)] for h in range(4, 9)) / 5


def print_lsr(L):
    acc, res, thr = L["acc"], L["res"], L["thr"]
    print("Rounding: half up on the stored value. A cell that ends in an exact 5 (for example 64.55) therefore prints 0.1 higher than in the study reports, which round such cells by float arithmetic.")
    print()
    print("## LSR (G2): label-free per-question stop rules at r = 128, evaluation only; test [106000, 108000), 2,000 episodes per checkpoint, seeds 17/29/43")
    print()
    print("Registered decision: R1 and R3 infeasible at both backbones; R2 passes at both (99.17% intervals, six comparisons).")
    for bk in BK:
        for r in ("R1", "R3"):
            g = thr["backbones"][bk]["rules"][r]
            assert g["feasible"] is False and len(g["grid"]) == 25 and not any(x["feasible"] for x in g["grid"])
        assert res["passing"][bk] == {"R1": False, "R2": True, "R3": False}
    print("R1 and R3: 0 of 25 validation grid values feasible at each backbone (hop-1 loss against fixed L = 4 beyond 8 pp pooled / 10 pp per seed).")
    print()
    print("### EM % by hop (seed-pooled), same range for every row")
    print()
    for bk, name in BK.items():
        print(f"**{name}**")
        print()
        print("| Row | " + " | ".join(str(h) for h in range(1, 9)) + " | hops 4-8 |")
        print("| --- |" + " --- |" * 9)
        for s, lab in ROWS:
            cells = [pct(acc[(bk, s, h)]) for h in range(1, 9)]
            print(f"| {lab} | " + " | ".join(cells) + f" | {pct(lsr_hops48(acc, bk, s))} |")
        print()
    print("### R2 against fixed L = 4 (= L_c), pp [intervals: registered 99.17%, descriptive 95%]")
    print()
    print("| | 0.6B | 1.7B |")
    print("| --- | --- | --- |")
    rows = {}
    for bk in BK:
        r2 = res["backbones"][bk]["rules"]["R2"]
        d = r2["descriptive"]
        rows[bk] = {
            "gain48": pp(r2["gain48"][0]), "gain48_iv": iv(r2["gain48"][1], r2["gain48"][2]),
            "hop1": pp(r2["hop1"][0]), "hop1_iv": iv(r2["hop1"][1], r2["hop1"][2]),
            "hop2": pp(d["hop2"][0]), "hop2_iv": iv(d["hop2"][1], d["hop2"][2]),
            "hop3": pp(d["hop3"][0]), "hops13": pp(d["hops13"][0]), "hops13_iv": iv(d["hops13"][1], d["hops13"][2]),
            "vsL8_48": pp(d["vsL8|gain48"][0]), "vsL8_48_iv": iv(d["vsL8|gain48"][1], d["vsL8|gain48"][2]),
            "vsL8_hop1": pp(d["vsL8|hop1"][0], 1),
            "seed_gain": "/".join(pp(r2["per_seed"][s]["gain48"][0]) for s in ("17", "29", "43")),
            "seed_hop1": "/".join(pp(r2["per_seed"][s]["hop1"][0]) for s in ("17", "29", "43")),
            "reads": str(q(r2["mean_reads"], 2)), "passes": str(q(r2["upper_layer_passes"], 2)),
            "reads_by_hop": ", ".join(str(q(x, 2)) for x in r2["mean_reads_by_hop"]),
            "by_hop_vsL4": ", ".join(pp(d[f"vsL4|hop{h}"][0]) for h in range(1, 9)),
        }
        assert r2["passes"] is True and all(r2["conditions"].values()) and r2["tau"] is None
        assert res["backbones"][bk]["l_c"] == 4
        check("gain48 " + bk, rows[bk]["gain48"].lstrip("+"), EXPECT["lsr_gain48"][bk])
        check("gain48_iv " + bk, rows[bk]["gain48_iv"], EXPECT["lsr_gain48_iv"][bk])
        check("hop1 " + bk, rows[bk]["hop1"].lstrip("+"), EXPECT["lsr_hop1"][bk])
        check("hop1_iv " + bk, rows[bk]["hop1_iv"], EXPECT["lsr_hop1_iv"][bk])
        check("hop2 " + bk, rows[bk]["hop2"], EXPECT["lsr_hop2"][bk])
        check("hop2_iv " + bk, rows[bk]["hop2_iv"], EXPECT["lsr_hop2_iv"][bk])
        check("reads " + bk, rows[bk]["reads"], EXPECT["lsr_reads"][bk])
        check("vsL8 " + bk, rows[bk]["vsL8_48"], EXPECT["lsr_vsL8_48"][bk])
    for key, lab in [("gain48", "Hops 4-8 gain"), ("gain48_iv", "  interval (99.17%)"), ("hop1", "Hop 1 change"), ("hop1_iv", "  interval (99.17%)"),
                     ("seed_gain", "Hops 4-8 gain per seed (17/29/43)"), ("seed_hop1", "Hop 1 change per seed"),
                     ("hop2", "Hop 2 change (descriptive)"), ("hop2_iv", "  interval (95%)"), ("hop3", "Hop 3 change"),
                     ("hops13", "Hops 1-3 pooled change"), ("hops13_iv", "  interval (95%)"),
                     ("vsL8_48", "R2 minus L = 8, hops 4-8"), ("vsL8_48_iv", "  interval (95%)"), ("vsL8_hop1", "R2 minus L = 8, hop 1"),
                     ("reads", "Mean reads per question"), ("passes", "Upper-layer passes per question (s* + 1)"),
                     ("reads_by_hop", "Mean reads by hop 1..8"), ("by_hop_vsL4", "R2 minus L = 4 by hop 1..8")]:
        print(f"| {lab} | {rows['0.6b'][key]} | {rows['1.7b'][key]} |")
    print()
    parts = []
    for bk in BK:
        lat = res["provenance"][bk]["latency"]
        m = {k: sum(v["seconds_per_question"][k]["mean"] for v in lat.values()) / len(lat) for k in ("fixed_L4", "fixed_L8", "phase1_trace_L8")}
        parts.append(f"{BK[bk]}: L=4 {q(m['fixed_L4'], 2, 1000)}, L=8 {q(m['fixed_L8'], 2, 1000)}, L=8 trace pass {q(m['phase1_trace_L8'], 2, 1000)}")
    print("Latency (descriptive, within one job, validation, ms per question, mean of the three seeds; R2 not timed): " + "; ".join(parts))
    print()
    return rows


def lsr_tex(L, path):
    acc = L["acc"]
    lines = ["% generated by scripts/paper_tables_g2.py; do not edit by hand",
             "\\newcommand{\\lsrtable}{%",
             "\\begin{table}[htbp]", "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Exact match (\\%) of the L=4-trained M1-L4 checkpoints on hops 1 to 8 at rank 128 on the LSR test range "
             "(episodes [106000, 108000), seeds 17, 29, 43 pooled, oracle memory). R2 is the label-free per-question stop; "
             "O1 and O2 choose a depth per hop (O1 on validation, O2 on the test cell with the label), so they are upper references and not stopping rules. "
             "The LSR range differs from the range of the depth matrix of Tables~\\ref{tab:matrixA} and~\\ref{tab:matrixB}, so its rows are compared only with each other.}",
             "\\label{tab:lsr}",
             "\\begin{tabular}{@{}p{4.6cm}rrrrrrrrr@{}}", "\\toprule",
             "Row & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 4 to 8 \\\\"]
    for bk, name in BK.items():
        lines += ["\\midrule", f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\"]
        for s, lab in ROWS:
            lab = lab.replace("(best fixed L per hop, chosen on validation)", "(validation depth per hop)")
            lab = lab.replace("(best test cell per hop, labels used)", "(best test cell, labels)")
            lab = lab.replace("(label-free stop)", "(label-free stop)")
            cells = [pct(acc[(bk, s, h)]) for h in range(1, 9)] + [pct(lsr_hops48(acc, bk, s))]
            bold = s == "R2"
            row = " & ".join(("\\textbf{%s}" % c) if bold else c for c in cells)
            lab = ("\\textbf{%s}" % lab) if bold else lab
            lines.append(f"{lab} & {row} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}%", "}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- B8W (G3)
def b8w():
    e = jload("reports/b8w/results/endpoints.json")
    c, c975 = e["contrasts"], e["contrasts_975"]
    out = {}

    def one(key, nd=2):
        x = c[key]
        return {"diff": pp(x["difference"] * 100, nd), "iv": iv(x["ci_low"] * 100, x["ci_high"] * 100, nd),
                "a": pct(x["accuracy_a"], 2), "b": pct(x["accuracy_b"], 2),
                "seeds": "/".join(pp(s["difference"] * 100, nd) for s in x["per_seed"])}

    for k in c:
        out[k] = one(k)
    out["DELTA_975"] = iv(c975["DELTA_untied_vs_shared"]["ci_low"] * 100, c975["DELTA_untied_vs_shared"]["ci_high"] * 100)
    check("b8w delta", out["DELTA_untied_vs_shared"]["diff"] + " " + out["DELTA_untied_vs_shared"]["iv"], EXPECT["b8w_delta"])
    k = e["keeps"]
    assert k["keeps"] and k["every_seed_within_2pp"] and k["interval_excludes_worse_than_2pp"]
    assert e["protocol_check_shared_below_start_by_more_than_2pp"] is False
    out["decision"] = e["decision"]
    # accuracy by hop (seed means) and drift ranges
    acc = {}
    with open(rd("reports/b8w/results/accuracy.csv")) as f:
        for r in csv.DictReader(f):
            acc.setdefault((r["reader"], int(r["hop"])), []).append(int(r["correct"]) / int(r["n"]))
    out["by_hop"] = {rd_: [pct(sum(acc[(rd_, h)]) / len(acc[(rd_, h)])) for h in range(1, 9)] for rd_ in ("start", "shared", "untied", "B8", "B4")}
    dr = list(csv.DictReader(open(rd("reports/b8w/results/drift.csv"))))
    out["drift"] = {}
    for t in ("R", "U", "Q"):
        rows = [r for r in dr if r["tensor"] == t]
        sh = [float(r["shared_relative_change"]) for r in rows]
        un = [float(r[f"untied_copy{i}_relative_change"]) for r in rows for i in range(4)]
        cd = [float(r["untied_mean_pairwise_cosine_distance"]) for r in rows]
        out["drift"][t] = (min(sh), max(sh), min(un), max(un), min(cd), max(cd))
    return out


def print_b8w(B):
    print("## B8W (G3): B8 warm start from the M1-L4 solution; Qwen3-0.6B-Base, r = 128, test [108000, 110000), 2,000 episodes, L = 4, seeds 17/29/43, 95% intervals")
    print()
    print(f"Registered decision: {B['decision']}")
    print()
    print("| Contrast (hops 2-3 unless stated) | Difference pp [95% interval] | Accuracies % | Per seed 17/29/43 |")
    print("| --- | --- | --- | --- |")
    names = [("DELTA_untied_vs_shared", "Delta = untied@4 - shared@4 (decision)"), ("hop1_untied_vs_shared", "untied - shared, hop 1"),
             ("untied_vs_start", "untied - start (M1-L4@4)"), ("shared_vs_start", "shared - start"), ("untied_vs_start_hop1", "untied - start, hop 1"),
             ("shared_vs_start_hop1", "shared - start, hop 1"), ("untied_vs_B8_original", "untied - original B8@4 (P1-L)"),
             ("untied_vs_B4", "untied - B4@1"), ("shared_vs_B4", "shared - B4@1"), ("start_vs_B4", "start M1-L4@4 - B4@1")]
    for k, lab in names:
        o = B[k]
        print(f"| {lab} | {o['diff']} {o['iv']} | {o['a']} vs {o['b']} | {o['seeds']} |")
    print()
    print(f"Delta, 97.5% interval: {B['DELTA_975']} (touches 0; exploratory only).")
    print()
    print("Accuracy by hop 1..8, seed means (%): " + "; ".join(f"{k} " + "/".join(v) for k, v in B["by_hop"].items()))
    print()
    for t, (a, b, c, d, e, f) in B["drift"].items():
        print(f"Drift {t}: shared {q(a, 2)}-{q(b, 2)}, untied copies {q(c, 2)}-{q(d, 2)}, pairwise cosine distance {q(e, 3)}-{q(f, 3)}")
    print()


# ---------------------------------------------------------------- CTC (G4)
CONDS = [("oracle/hard", "oracle memory (reference)"), ("ce/hard", "CE, hard (selected)"), ("ce/soft", "CE, soft"),
         ("ceans/hard", "CE+ans, hard"), ("ceans/soft", "CE+ans, soft"), ("zero", "zero memory"), ("ce/shuffled", "shuffled memory (CE)")]


def ctc():
    dec = jload("reports/ctc/decision.json")
    ev = jload("reports/ctc/eval-metrics.json")
    assert dec["outcome"] == "candidate" and dec["selected_arm"] == "ce" and ev["final"] is True and ev["split"] == "val"
    t = ev["table"]
    em = {}
    for r in t:
        em[(r["reader"], r["condition"], r["L"], r["hop"])] = r["em"]
    return dec, ev, em


def print_ctc(C):
    dec, ev, em = C
    n = dec["numbers"]
    print("## CTC (G4): classify-then-code writer into the frozen P1-L r = 128 readers; validation [0, 1,000), seed 17 readers, eval seed 4242")
    print()
    print(f"Decision: outcome {dec['outcome']}, selected arm {dec['selected_arm']}; hop-1 EM of M1-L4@4 {pct(n['ce']['hop1_em'])}%; "
          f"gain over B4@1 on hops 2-3 {pp(n['ce']['gain_pp'])} pp [{q(n['ce']['gain_ci95_pp'][0], 2)}, {q(n['ce']['gain_ci95_pp'][1], 2)}]; "
          f"oracle reference gain {pp(dec['oracle_reference']['gain_pp'])} pp (hop-1 {pct(dec['oracle_reference']['hop1_em'])}%); "
          f"joint classification accuracy p = {n['ce']['joint']:.5f} (CE), {n['ceans']['joint']:.5f} (CE+ans); "
          f"zero / shuffled memory hop-1 EM {pct(n['ce']['zero_hop1_em'])}% / {pct(n['ce']['shuffled_hop1_em'])}%")
    check("ctc gain", f"{pp(n['ce']['gain_pp'])} [{q(n['ce']['gain_ci95_pp'][0], 2)}, {q(n['ce']['gain_ci95_pp'][1], 2)}]", EXPECT["ctc_gain"])
    cl = ev["classification"]["ce"]
    assert cl["joint"] == 1.0 and cl["episodes_all_facts_correct"] == 1.0 and cl["perfect_lookup_equals_oracle"] is True
    print(f"Writer parameters {cl['parameters']:,} (CE); perfect lookup equals the oracle memory: {cl['perfect_lookup_equals_oracle']}")
    print()
    print("| M1-L4 at L = 4, EM % by hop | " + " | ".join(str(h) for h in range(1, 9)) + " |")
    print("| --- |" + " --- |" * 8)
    maxdiff = 0.0
    for c, lab in CONDS:
        print(f"| {lab} | " + " | ".join(pct(em[("M1-L4", c, 4, h)]) for h in range(1, 9)) + " |")
    for h in range(1, 9):
        maxdiff = max(maxdiff, abs(em[("M1-L4", "ce/hard", 4, h)] - em[("M1-L4", "oracle/hard", 4, h)]))
    print()
    print(f"Largest hop-wise difference between CE hard and the oracle reference: {q(maxdiff, 1, 100)} pp")
    assert maxdiff <= 0.002 + 1e-12
    print()
    print("| B4 at L = 1, EM % by hop | 1 | 2 | 3 | 4 to 8 |")
    print("| --- | --- | --- | --- | --- |")
    for c, lab in [("oracle/hard", "oracle memory"), ("ce/hard", "CE, hard"), ("ceans/hard", "CE+ans, hard"), ("zero", "zero memory")]:
        v = [em[("B4", c, 1, h)] for h in range(1, 9)]
        print(f"| {lab} | {pct(v[0])} | {pct(v[1])} | {pct(v[2])} | {pct(min(v[3:]))} to {pct(max(v[3:]))} |")
    print()
    print("B3-L4 at L = 4, EM % at every hop: oracle " + pct(max(em[("B3-L4", "oracle/hard", 4, h)] for h in range(1, 9))) + " (max), zero " +
          pct(max(em[("B3-L4", "zero", 4, h)] for h in range(1, 9))) + " (max); predictions identical under both:", ev["b3_memory_independent"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", help="write the LaTeX table macros of the LSR matrix to this file")
    a = ap.parse_args()
    L = lsr()
    print_lsr(L)
    B = b8w()
    print_b8w(B)
    C = ctc()
    print_ctc(C)
    if a.tex:
        lsr_tex(L, a.tex)


if __name__ == "__main__":
    main()
