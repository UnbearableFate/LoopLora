#!/usr/bin/env python3
"""WP-T12 part 2: the numbers of G5 (SQ2 and LSQT) as the paper prints them (stdlib only, no torch).

Reads committed files only:
  reports/p1l-sq2/{decision.json,audit.json,arms/*/{record.json,train_log.jsonl,val-*/metrics.json}}   (SQ2, WP-L12)
  reports/p1l-lsqt/{decision.json,audit.json,arms/*/{record.json,val-*/metrics.json}}                   (LSQT, WP-L12)
  reports/p1l-dpfp/arms/D128-3/val-22370/metrics.json and refs/R-r128/val-24645/metrics.json            (references)
Every key number is checked against the value the two study reports state, so a drift fails loudly.

    python3 scripts/paper_tables_g5.py                                   # markdown/text to stdout
    python3 scripts/paper_tables_g5.py --tex docs/paper/tex/g5_tables.tex # the two LaTeX table macros
Rounding: exact decimal arithmetic on the stored value, half up. Differences in pp come from the integer correct counts
(1,000 validation episodes, so one episode is 0.1 pp).
"""
import argparse
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jload(*p):
    with open(os.path.join(ROOT, *p)) as f:
        return json.load(f)


def q(x, nd=1, scale=1):
    return (Decimal(repr(float(x))) * scale).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)


def pct(x, nd=1):
    return str(q(x, nd, 100))


def pp(n_correct_a, n_correct_b):
    """Difference of two correct counts out of 1,000, in pp, with an explicit sign."""
    v = Decimal(n_correct_a - n_correct_b) / 10
    return ("+" if v > 0 else "") + str(v.quantize(Decimal("0.1")))


def metrics(*p):
    return jload(*p, "metrics.json")


def check(label, got, want):
    if got != want:
        sys.exit(f"MISMATCH {label}: script {got!r}, report {want!r}")


SQ2_ARMS = [("S-DE", "delta", "no"), ("S-DC", "delta", "yes"), ("S-RE", "ridge (DPFP features)", "no"), ("S-RC", "ridge (DPFP features)", "yes")]
LT_ARMS = [("LT-K4", "four Kaczmarz sweeps"), ("LT-RG001", "ridge, eta 0.01"), ("LT-RG1", "ridge, eta 1")]
SQ2_STEPS = [0, 1200, 2400, 3600, 4800, 6000]


def load():
    d = {}
    d["sq2_dec"] = jload("reports/p1l-sq2/decision.json")
    d["lt_dec"] = jload("reports/p1l-lsqt/decision.json")
    d["sq2"] = {}
    for a, _, _ in SQ2_ARMS:
        d["sq2"][a] = {s: metrics("reports/p1l-sq2/arms", a, f"val-{s}") for s in SQ2_STEPS}
    d["lt"] = {}
    for a, _ in LT_ARMS:
        rec = jload("reports/p1l-lsqt/arms", a, "record.json")
        steps = rec["training"]["eval_steps"]
        d["lt"][a] = {"rec": rec, "steps": steps, "m": {s: metrics("reports/p1l-lsqt/arms", a, f"val-{s}") for s in steps}}
    d["parent"] = metrics("reports/p1l-dpfp/arms/D128-3/val-22370")
    d["r128"] = metrics("reports/p1l-dpfp/refs/R-r128/val-24645")
    d["clip"] = {}
    for a, _, _ in SQ2_ARMS:
        rows = [json.loads(line) for line in open(os.path.join(ROOT, "reports/p1l-sq2/arms", a, "train_log.jsonl"))]
        d["clip"][a] = (sum(1 for r in rows if r["clipped"]), len(rows))
    d["sq2_audit"] = jload("reports/p1l-sq2/audit.json")
    d["lt_audit"] = jload("reports/p1l-lsqt/audit.json")
    return d


def audit_rows(audit):
    return {r["arm"]: r for r in audit["runs"].values()}


def print_sq2(d):
    s, dec = d["sq2"], d["sq2_dec"]
    fin = {a: s[a][6000] for a, _, _ in SQ2_ARMS}
    corr = {a: fin[a]["correct"] for a in fin}
    assert corr == dec["correct"] and dec["advances"] is False and dec["prediction_holds"] is False and dec["candidate"] is None
    assert dec["conditions"] == {"em_gain": False, "read_error_halved": False, "stable_decoding": True}
    assert all(m["n"] == 1000 for a in fin for m in [fin[a]])
    ratio = fin["S-RC"]["relative_read_error"] / fin["S-DE"]["relative_read_error"]
    gap = pp(corr["S-RC"], corr["S-DE"])
    check("sq2 gap", gap, "+4.1")
    check("sq2 ratio", str(q(ratio, 3)), "0.554")
    inter = Decimal((corr["S-RC"] - corr["S-RE"]) - (corr["S-DC"] - corr["S-DE"])) / 10
    check("sq2 interaction", str(inter), "-0.4")
    check("sq2 S-DC", pp(corr["S-DC"], corr["S-DE"]), "-2.4")
    check("sq2 S-RE", pp(corr["S-RE"], corr["S-DE"]), "+6.9")
    check("sq2 S-RE ratio", str(q(fin["S-RE"]["relative_read_error"] / fin["S-DE"]["relative_read_error"], 3)), "0.546")
    for a, em, err in [("S-DE", "65.4", "0.359"), ("S-DC", "63.0", "0.353"), ("S-RE", "72.3", "0.196"), ("S-RC", "69.5", "0.199")]:
        check("sq2 " + a, (pct(fin[a]["em"]), str(q(fin[a]["relative_read_error"], 3))), (em, err))
    print("## SQ2 (G5): D128-3 continued 6,000 steps with ridge storage on the DPFP features and the coefficient loss; validation [0,1000), eval seed 4242, one seed (17)")
    print()
    print("Registered decision: the primary contrast S-RC against S-DE does not advance; the learned-writer line stays stopped; no cell reaches 0.80.")
    print(f"Primary contrast: EM gap {gap} pp (needs >= +10.0), read error ratio {q(ratio, 3)} (needs <= 0.500), stable decoding met "
          f"(S-RC clean-value head minimum {pct(min(s['S-RC'][k]['clean_value_correct'] for k in SQ2_STEPS[1:]))}%, floor {pct(dec['clean_floor'])}%).")
    print()
    print("| arm | storage | coefficient loss | correct of 1,000 | val EM % | normalized read error | clean-value head % | read-value head % |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for a, st, cl in SQ2_ARMS:
        m = fin[a]
        print(f"| {a} | {st} | {cl} | {m['correct']} | {pct(m['em'])} | {q(m['relative_read_error'], 3)} | {pct(m['clean_value_correct'])} | {pct(m['actual_read_correct'])} |")
    p = d["parent"]
    print(f"| D128-3 (parent, 22,370 steps) | delta | no | {p['correct']} | {pct(p['em'])} | {q(p['relative_read_error'], 3)} | {pct(p['clean_value_correct'])} | {pct(p['actual_read_correct'])} |")
    assert p["correct"] == 608
    print()
    print(f"Single changes against S-DE: S-DC {pp(corr['S-DC'], corr['S-DE'])} pp (read error ratio {q(fin['S-DC']['relative_read_error'] / fin['S-DE']['relative_read_error'], 3)}); "
          f"S-RE {pp(corr['S-RE'], corr['S-DE'])} pp (ratio {q(fin['S-RE']['relative_read_error'] / fin['S-DE']['relative_read_error'], 3)}); S-RC against S-RE {pp(corr['S-RC'], corr['S-RE'])} pp.")
    print(f"Interaction (S-RC - S-RE) - (S-DC - S-DE): {inter} pp (descriptive; standard error of the double difference about 4.4 pp).")
    z = s["S-RE"][0]
    for a in ("S-DE", "S-DC", "S-RE", "S-RC"):
        print(f"Step 0 {a}: {s[a][0]['correct']} correct, {pct(s[a][0]['em'])}%, read error {q(s[a][0]['relative_read_error'], 3)}; gain over 6,000 steps {pp(corr[a], s[a][0]['correct'])} pp")
    check("sq2 step0 ridge", (z["correct"], str(q(z["relative_read_error"], 3))), (687, "0.312"))
    check("sq2 S-DE step0", s["S-DE"][0]["correct"], 608)
    check("sq2 S-DE gain", pp(corr["S-DE"], s["S-DE"][0]["correct"]), "+4.6")
    print()
    print("| arm | " + " | ".join(str(k) for k in SQ2_STEPS) + " |")
    print("| --- |" + " ---: |" * len(SQ2_STEPS))
    for a, _, _ in SQ2_ARMS:
        print(f"| {a} EM % | " + " | ".join(pct(s[a][k]["em"]) for k in SQ2_STEPS) + " |")
    for a, _, _ in SQ2_ARMS:
        print(f"| {a} read error | " + " | ".join(str(q(s[a][k]["relative_read_error"], 3)) for k in SQ2_STEPS) + " |")
    print()
    print("Gradient clip active at logged steps: " + ", ".join(f"{a} {d['clip'][a][0]} of {d['clip'][a][1]}" for a, _, _ in SQ2_ARMS))
    assert all(c == n == 121 for c, n in d["clip"].values())
    au = audit_rows(d["sq2_audit"])
    print("Zero / shuffled memory correct of 1,000 (original, zero, shuffled): " + "; ".join(
        f"{a} {au[a]['original']['correct']}/{au[a]['zero']['correct']}/{au[a]['shuffle']['correct']}" for a, _, _ in SQ2_ARMS))
    check("sq2 audit", [(au[a]["zero"]["correct"], au[a]["shuffle"]["correct"]) for a, _, _ in SQ2_ARMS], [(0, 1), (0, 2), (1, 0), (0, 1)])
    print("State bytes per episode: ridge arms " + str(jload("reports/p1l-sq2/arms/S-RC/record.json")["memory"]["state_bytes_per_episode"]) +
          ", delta arms " + str(jload("reports/p1l-sq2/arms/S-DE/record.json")["memory"]["state_bytes_per_episode"]) + "; inference parameters 475,138 in every arm")
    print()


def print_lsqt(d):
    dec, m, r, par = d["lt_dec"], d["lt"], d["r128"], d["parent"]
    corr = {a: m[a]["m"][m[a]["steps"][-1]]["correct"] for a, _ in LT_ARMS}
    assert corr == dec["correct"] and dec["outcome"] == "no_candidate" and dec["best_rg_band"] == "open" and dec["trained_lsq_line"] == "open"
    assert dec["candidate"] is None and dec["best_arm"] == "LT-RG001" and dec["e8_retrain"] is False and dec["u_control_proposed"] is True
    assert r["correct"] == 335 and par["correct"] == 608
    print("## LSQT (G5): trained least-squares stores at r = 128 from scratch (unit query); validation [0,1000), eval seed 4242, one seed (17)")
    print()
    print(f"Registered decision: no arm reaches 0.80; best ridge arm {dec['best_rg']} {pct(corr[dec['best_rg']] / 1000)}%, band {dec['best_rg_band']} (40% to < 48.5%); trained-LSQ line {dec['trained_lsq_line']}; U control proposed.")
    print()
    print("| arm | store | steps | correct of 1,000 | val EM % | normalized read error | clean-value head % | read-value head % | minus R-r128 (pp) | minus D128-3 (pp) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for a, st in LT_ARMS:
        s_last = m[a]["steps"][-1]
        x = m[a]["m"][s_last]
        print(f"| {a} | {st} | {s_last} | {x['correct']} | {pct(x['em'])} | {q(x['relative_read_error'], 3)} | {pct(x['clean_value_correct'])} | {pct(x['actual_read_correct'])} | {pp(x['correct'], r['correct'])} | {pp(x['correct'], par['correct'])} |")
    print(f"| R-r128 (WP-L4, not rerun) | delta | 24645 | {r['correct']} | {pct(r['em'])} | {q(r['relative_read_error'], 3)} | {pct(r['clean_value_correct'])} | {pct(r['actual_read_correct'])} |  |  |")
    print(f"| D128-3 (DPFP) | delta | 22370 | {par['correct']} | {pct(par['em'])} | {q(par['relative_read_error'], 3)} | {pct(par['clean_value_correct'])} | {pct(par['actual_read_correct'])} |  |  |")
    for a, em, err, cl, rv, dr, dp in [("LT-K4", "38.3", "0.554", "92.3", "27.4", "+4.8", "-22.5"), ("LT-RG001", "46.1", "0.512", "93.9", "26.5", "+12.6", "-14.7"),
                                       ("LT-RG1", "42.6", "0.499", "95.9", "22.3", "+9.1", "-18.2")]:
        x = m[a]["m"][m[a]["steps"][-1]]
        check("lsqt " + a, (pct(x["em"]), str(q(x["relative_read_error"], 3)), pct(x["clean_value_correct"]), pct(x["actual_read_correct"]), pp(x["correct"], 335), pp(x["correct"], 608)), (em, err, cl, rv, dr, dp))
    best = m["LT-RG001"]["m"][m["LT-RG001"]["steps"][-1]]
    held = best["em"] >= 0.50 and best["relative_read_error"] <= 0.35
    print()
    print(f"Registered prediction (best ridge arm at least 50% with read error at most 0.35): {'held' if held else 'did not hold'} ({pct(best['em'])}%, {q(best['relative_read_error'], 3)}).")
    assert not held
    print("Ridge minus K4: " + ", ".join(f"{a} {pp(corr[a], corr['LT-K4'])} pp" for a in ("LT-RG001", "LT-RG1")))
    print()
    print("| arm | " + " | ".join(["20%", "40%", "60%", "80%", "100%"]) + " |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for a, _ in LT_ARMS:
        print(f"| {a} EM % | " + " | ".join(pct(m[a]["m"][s]["em"]) for s in m[a]["steps"]) + " |")
    for a, _ in LT_ARMS:
        print(f"| {a} read error | " + " | ".join(str(q(m[a]["m"][s]["relative_read_error"], 3)) for s in m[a]["steps"]) + " |")
    print()
    print("Gram reconstruction relative error at the final checkpoint: " + ", ".join(f"{a} {m[a]['m'][m[a]['steps'][-1]]['gram_relative_error']:.1e}" for a, _ in LT_ARMS))
    assert all(m[a]["m"][m[a]["steps"][-1]]["gram_relative_error"] < 1e-6 for a, _ in LT_ARMS)
    au = audit_rows(d["lt_audit"])
    print("Zero / shuffled memory correct of 1,000 (original, zero, shuffled): " + "; ".join(
        f"{a} {au[a]['original']['correct']}/{au[a]['zero']['correct']}/{au[a]['shuffle']['correct']}" for a, _ in LT_ARMS))
    check("lsqt audit", [(au[a]["zero"]["correct"], au[a]["shuffle"]["correct"]) for a, _ in LT_ARMS], [(1, 2), (2, 4), (1, 4)])
    clip = {a: [json.loads(line) for line in open(os.path.join(ROOT, "reports/p1l-lsqt/arms", a, "train_log.jsonl"))] for a, _ in LT_ARMS}
    print("Gradient clip active at logged steps: " + ", ".join(f"{a} {sum(1 for r_ in clip[a] if r_['clipped'])} of {len(clip[a])}" for a, _ in LT_ARMS))
    assert all(all(r_["clipped"] for r_ in clip[a]) for a, _ in LT_ARMS)
    print("State bytes per episode: " + str(m["LT-K4"]["rec"]["memory"]["state_bytes_per_episode"]) + "; inference parameters 475,138 in every arm")


def write_tex(d, path):
    s, p, r, m = d["sq2"], d["parent"], d["r128"], d["lt"]
    L = ["% generated by scripts/paper_tables_g5.py; do not edit by hand",
         "\\newcommand{\\sqtwotable}{%", "\\begin{table}[htbp]", "\\centering\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{SQ2: four continuations of the DPFP learned writer D128-3 for 6,000 steps (validation episodes [0, 1000), one seed). "
         "Exact match (EM), normalized read error and clean-value head accuracy at the last checkpoint. "
         "No cell reaches 0.80, and all cells are descriptive; the registered primary contrast S-RC against S-DE does not advance.}",
         "\\label{tab:sq2}", "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
         "Arm & Storage and coefficient loss & EM (\\%) & Read error & Clean-value head (\\%) & Step 0 EM (\\%) \\\\", "\\midrule"]
    names = {"S-DE": "delta, no coefficient loss", "S-DC": "delta, coefficient loss", "S-RE": "ridge, no coefficient loss", "S-RC": "ridge, coefficient loss"}
    for a, _, _ in SQ2_ARMS:
        f = s[a][6000]
        L.append(f"{a} & {names[a]} & {pct(f['em'])} & {q(f['relative_read_error'], 3)} & {pct(f['clean_value_correct'])} & {pct(s[a][0]['em'])} \\\\")
    L += ["\\midrule", f"D128-3 (parent) & delta, 22,370 steps & {pct(p['em'])} & {q(p['relative_read_error'], 3)} & {pct(p['clean_value_correct'])} & \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}%", "}",
          "\\newcommand{\\lsqttable}{%", "\\begin{table}[htbp]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
          "\\caption{LSQT: trained least-squares stores at rank 128, trained from scratch with the unit query (validation episodes [0, 1000), one seed), "
          "against the delta-store references. The registered decision is that no arm reaches 0.80, the best ridge arm lies in the open band 40\\% to below 48.5\\%, "
          "and the trained-LSQ line stays open. The unit query is in every LSQT arm and not in R-r128. Clean and read head: accuracy of the value decoded from the stored and from the read value.}",
          "\\label{tab:lsqt}", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
          "Arm & Store & EM (\\%) & Read error & Clean head (\\%) & Read head (\\%) & vs R-r128 (pp) \\\\", "\\midrule"]
    lt_names = {"LT-K4": "four Kaczmarz sweeps", "LT-RG001": "ridge, $\\eta=0.01$", "LT-RG1": "ridge, $\\eta=1$"}
    for a, _ in LT_ARMS:
        x = m[a]["m"][m[a]["steps"][-1]]
        L.append(f"{a} & {lt_names[a]} & {pct(x['em'])} & {q(x['relative_read_error'], 3)} & {pct(x['clean_value_correct'])} & {pct(x['actual_read_correct'])} & {pp(x['correct'], r['correct'])} \\\\")
    L += ["\\midrule",
          f"R-r128 (WP-L4) & delta & {pct(r['em'])} & {q(r['relative_read_error'], 3)} & {pct(r['clean_value_correct'])} & {pct(r['actual_read_correct'])} & \\\\",
          f"D128-3 (DPFP) & delta & {pct(p['em'])} & {q(p['relative_read_error'], 3)} & {pct(p['clean_value_correct'])} & {pct(p['actual_read_correct'])} & \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}%", "}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", help="write the two LaTeX table macros to this file")
    a = ap.parse_args()
    d = load()
    print_sq2(d)
    print_lsqt(d)
    if a.tex:
        write_tex(d, a.tex)


if __name__ == "__main__":
    main()
