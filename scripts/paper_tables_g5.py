#!/usr/bin/env python3
"""WP-T12 parts 2 and 3: the numbers of G5 (SQ2, LSQT and the LSQT U control) as the paper prints them (stdlib only, no torch).

Reads committed files only:
  reports/p1l-sq2/{decision.json,audit.json,arms/*/{record.json,train_log.jsonl,val-*/metrics.json}}   (SQ2, WP-L12)
  reports/p1l-lsqt/{decision.json,audit.json,arms/*/{record.json,val-*/metrics.json}}                   (LSQT, WP-L12)
  reports/p1l-lsqt-u/{split.json,audit.json,arms/LT-U/{record.json,train_log.jsonl,val-*/metrics.json}} (U control, WP-L12)
  reports/p1l-dpfp/arms/D128-3/val-22370/metrics.json and refs/R-r128/{record.json,val-*/metrics.json}  (references)
Every key number is checked against the value the study reports state (and the U split against its split.json), so a drift fails loudly.

    python3 scripts/paper_tables_g5.py                                   # markdown/text to stdout
    python3 scripts/paper_tables_g5.py --tex docs/paper/tex/g5_tables.tex # the LaTeX table macros
Rounding: exact decimal arithmetic on the stored value, half up. Differences in pp come from the integer correct counts
(1,000 validation episodes, so one episode is 0.1 pp). The standard error of a difference is the unpaired binomial one,
100 * sqrt(p1 (1 - p1) / n + p2 (1 - p2) / n): it covers the sampling of the 1,000 episodes only (no training-seed variation, no pairing).
"""
import argparse
import json
import math
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
U_ARM = "LT-U"
U_DIR = "reports/p1l-lsqt-u"
R128_DIR = "reports/p1l-dpfp/refs/R-r128"


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
    urec = jload(U_DIR, "arms", U_ARM, "record.json")
    ust = urec["training"]["eval_steps"]
    d["u"] = {"rec": urec, "steps": ust, "m": {s: metrics(U_DIR, "arms", U_ARM, f"val-{s}") for s in ust}}
    d["u_log"] = [json.loads(line) for line in open(os.path.join(ROOT, U_DIR, "arms", U_ARM, "train_log.jsonl"))]
    d["u_split"] = jload(U_DIR, "split.json")
    d["u_audit"] = jload(U_DIR, "audit.json")
    rrec = jload(R128_DIR, "record.json")
    rst = rrec["training"]["eval_steps"]
    d["r128_traj"] = {"rec": rrec, "steps": rst, "m": {s: metrics(R128_DIR, f"val-{s}") for s in rst}}
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
    print(f"Registered decision: no arm reaches 0.80; best ridge arm {dec['best_rg']} {pct(corr[dec['best_rg']] / 1000)}%, band {dec['best_rg_band']} (40% to < 48.5%); trained-LSQ line {dec['trained_lsq_line']}; the decision proposes the conditional U control (reported below).")
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


def se_pp(a, b, n=1000):
    """Unpaired binomial standard error of a difference of two EMs (correct counts a and b out of n episodes), in pp.
    It covers the sampling of the episodes only: no training-seed variation, and the pairing of the two runs on the same episodes is not used."""
    p, r = a / n, b / n
    return 100 * math.sqrt(p * (1 - p) / n + r * (1 - r) / n)


def signed(a, b):
    """'+2.5 (2.1)': a difference of two correct counts in pp and its unpaired standard error (both rounded half up to one decimal)."""
    return f"{pp(a, b)} ({q(se_pp(a, b), 1)})"


def flat(o, path=""):
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            out.update(flat(v, f"{path}.{k}" if path else k))
        return out
    return {path: o}


def final_counts(d):
    """Final-checkpoint correct counts: the three LSQT arms, LT-U and R-r128 (out of 1,000 validation episodes)."""
    corr = {a: d["lt"][a]["m"][d["lt"][a]["steps"][-1]]["correct"] for a, _ in LT_ARMS}
    return corr, d["u"]["m"][d["u"]["steps"][-1]]["correct"], d["r128"]["correct"]


def matched(d):
    """Differences in pp at the five matched fractions of training (each arm at 20, 40, 60, 80 and 100% of its own steps)."""
    u, rt, m = d["u"], d["r128_traj"], d["lt"]
    um, rm = u["m"], rt["m"]
    rows = {"LT-U - R-r128": [pp(um[a]["correct"], rm[b]["correct"]) for a, b in zip(u["steps"], rt["steps"])]}
    for a, _ in LT_ARMS:
        rows[f"{a} - LT-U"] = [pp(m[a]["m"][x]["correct"], um[y]["correct"]) for x, y in zip(m[a]["steps"], u["steps"])]
    return rows


def print_lsqt_u(d):
    u, ust, sp, rt, m = d["u"], d["u"]["steps"], d["u_split"], d["r128_traj"], d["lt"]
    uf, r = u["m"][ust[-1]], d["r128"]
    corr, uc, rc = final_counts(d)
    rec, dec = u["rec"], d["lt_dec"]
    assert rec["status"] == "complete" and rec["seed"] == 17 and rec["eval_seed"] == 4242 and rec["rank"] == 128 and rec["store"] == "delta"
    assert rec["training"]["actual_steps"] == ust[-1] == 24560 and len(ust) == 5 and uf["n"] == 1000 and r["n"] == 1000
    assert rec["git_commit"].startswith("c07452b") and rec["job_id"] == "3725694.opbs" and not rec["smoke"] and not rec["source_dirty"]
    fu, fr = flat(rec["config"]), flat(rt["rec"]["config"])
    differ = {k for k in set(fu) | set(fr) if fu.get(k) != fr.get(k) and not k.startswith("lsqt.") and k != "name"}
    check("U config differs from R-r128 only in", differ, {"model.unit_query", "train.steps"})
    check("U unit query", (fu["model.unit_query"], fr.get("model.unit_query")), (True, None))
    # the registered LSQT decision is not changed by U
    assert dec["outcome"] == "no_candidate" and dec["best_rg_band"] == "open" and dec["trained_lsq_line"] == "open" and dec["candidate"] is None
    check("U split counts", sp["correct"], {**corr, "LT-U": uc})
    check("U split r128", (sp["r_r128_correct"], sp["n"], sp["steps"], sp["arm"], sp["study"]), (rc, 1000, 24560, "LT-U", "LSQT-U"))
    check("U split flags", (sp["enters_registered_outcomes"], sp["u_reaches_0_80"]), (False, False))
    check("U facts", (uc, pct(uf["em"]), str(q(uf["relative_read_error"], 3)), pct(uf["clean_value_correct"]), pct(uf["actual_read_correct"]), str(q(uf["nll"], 3))),
          (360, "36.0", "0.589", "93.7", "21.6", "3.382"))
    em = [u["m"][s]["correct"] for s in ust]
    assert em == sorted(em) and len(set(em)) == 5 and max(em) == uc, "U's EM must rise at every checkpoint, so the final checkpoint is also its highest"
    check("U trajectory", ([pct(u["m"][s]["em"]) for s in ust], [str(q(u["m"][s]["relative_read_error"], 3)) for s in ust]),
          (["15.9", "24.5", "30.3", "32.8", "36.0"], ["0.615", "0.593", "0.596", "0.596", "0.589"]))
    parts = {"unit_query": (uc, rc)}
    for a, _ in LT_ARMS:
        parts[f"total_{a}"], parts[f"store_{a}"] = (corr[a], rc), (corr[a], uc)
    assert set(parts) == set(sp["split"])
    for k, (x, y) in parts.items():
        got = sp["split"][k]
        assert Decimal(x - y) / 10 == Decimal(repr(got["pp"])) and abs(round(se_pp(x, y), 6) - got["se_pp"]) < 1e-9, k
    for a, _ in LT_ARMS:
        assert (corr[a] - rc) == (uc - rc) + (corr[a] - uc), "the parts must add up to the total exactly"
    check("U split rounded", {k: signed(*v) for k, v in parts.items()},
          {"unit_query": "+2.5 (2.1)", "total_LT-K4": "+4.8 (2.1)", "store_LT-K4": "+2.3 (2.2)", "total_LT-RG001": "+12.6 (2.2)", "store_LT-RG001": "+10.1 (2.2)",
           "total_LT-RG1": "+9.1 (2.2)", "store_LT-RG1": "+6.6 (2.2)"})
    steps = {U_ARM: ust, "R-r128": rt["steps"], **{a: m[a]["steps"] for a, _ in LT_ARMS}}
    check("U step counts", {k: v[-1] for k, v in steps.items()}, {U_ARM: 24560, "R-r128": 24645, "LT-K4": 17970, "LT-RG001": 28005, "LT-RG1": 28245})
    print("## LSQT U control (G5): R-r128's delta store trained from scratch with the unit query (LT-U); validation [0,1000), eval seed 4242, one seed (17)")
    print()
    print("Registered role (docs/prereg/LSQT.md, decision 2): conditional on the best registered arm reaching 40% (LT-RG001 46.1%); descriptive only; "
          "it splits a gain over R-r128 between the store and the unit query and enters neither outcome 1 nor outcome 2 (split.json: enters_registered_outcomes false). "
          "The registered LSQT decision is unchanged.")
    print("Config against R-r128: the same except model.unit_query (true) and the step count (24,560 against 24,645).")
    print()
    print("| arm | store | steps | correct of 1,000 | val EM % | normalized read error | clean-value head % | read-value head % | NLL | minus R-r128 (pp) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print(f"| LT-U | delta, unit query | {ust[-1]} | {uc} | {pct(uf['em'])} | {q(uf['relative_read_error'], 3)} | {pct(uf['clean_value_correct'])} | {pct(uf['actual_read_correct'])} | {q(uf['nll'], 3)} | {pp(uc, rc)} |")
    print(f"| R-r128 (WP-L4, not rerun) | delta | 24645 | {rc} | {pct(r['em'])} | {q(r['relative_read_error'], 3)} | {pct(r['clean_value_correct'])} | {pct(r['actual_read_correct'])} | {q(r['nll'], 3)} |  |")
    print()
    print(f"LT-U does not reach 0.80 ({pct(uf['em'])}%). Its EM rises at every checkpoint ({', '.join(pct(u['m'][s]['em']) for s in ust)}), so the final checkpoint, the registered selection, is also its highest.")
    print()
    print("Split of the gain over R-r128 through LT-U (pp of val EM at the final checkpoints; standard error in parentheses, unpaired binomial, episode sampling only):")
    print()
    print("| arm | steps | total: arm minus R-r128 | unit query: LT-U minus R-r128 | store: arm minus LT-U |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for a, _ in LT_ARMS:
        print(f"| {a} | {steps[a][-1]} | {signed(corr[a], rc)} | {signed(uc, rc)} | {signed(corr[a], uc)} |")
    print()
    print("The parts add up to the total exactly (same counts). Descriptive: one seed; the standard errors cover the sampling of the 1,000 episodes only, are unpaired, "
          "and do not include training-seed variation; no ridge store with a learned query was trained, so an interaction of store and query cannot be separated; "
          f"the step counts differ (LT-U {steps[U_ARM][-1]:,}, R-r128 {steps['R-r128'][-1]:,}, LT-K4 {steps['LT-K4'][-1]:,}, LT-RG001 {steps['LT-RG001'][-1]:,}, LT-RG1 {steps['LT-RG1'][-1]:,}).")
    print()
    print("| arm | " + " | ".join(["20%", "40%", "60%", "80%", "100%"]) + " |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, st, mm in [(U_ARM, ust, u["m"]), ("R-r128", rt["steps"], rt["m"])] + [(a, m[a]["steps"], m[a]["m"]) for a, _ in LT_ARMS]:
        print(f"| {name} steps | " + " | ".join(str(x) for x in st) + " |")
    for name, st, mm in [(U_ARM, ust, u["m"]), ("R-r128", rt["steps"], rt["m"])]:
        print(f"| {name} EM % | " + " | ".join(pct(mm[x]["em"]) for x in st) + " |")
    for name, st, mm in [(U_ARM, ust, u["m"]), ("R-r128", rt["steps"], rt["m"])]:
        print(f"| {name} read error | " + " | ".join(str(q(mm[x]["relative_read_error"], 3)) for x in st) + " |")
    print()
    print("Differences at the matched fractions of training (pp of val EM; each cell is a difference of two single evaluations of one seed):")
    print()
    print("| contrast | 20% | 40% | 60% | 80% | 100% |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    mf = matched(d)
    for k, v in mf.items():
        print(f"| {k} | " + " | ".join(v) + " |")
    check("matched fractions", mf, {"LT-U - R-r128": ["-0.4", "+2.1", "+2.7", "+1.8", "+2.5"], "LT-K4 - LT-U": ["-1.0", "+1.4", "+0.1", "+4.8", "+2.3"],
                                    "LT-RG001 - LT-U": ["+7.7", "+11.2", "+11.8", "+14.1", "+10.1"], "LT-RG1 - LT-U": ["+7.1", "+5.6", "+5.4", "+3.8", "+6.6"]})
    rg, rg1 = m["LT-RG001"], m["LT-RG1"]
    assert all(a > b for a, b in zip((rg["m"][x]["correct"] for x in rg["steps"]), (u["m"][y]["correct"] for y in ust)))
    assert all(a > b for a, b in zip((rg1["m"][x]["correct"] for x in rg1["steps"]), (u["m"][y]["correct"] for y in ust)))
    check("RG001 at 22,404 steps", (rg["steps"][3], pct(rg["m"][rg["steps"][3]]["em"]), rg["steps"][3] < ust[-1], rg["m"][rg["steps"][3]]["correct"] > uc), (22404, "46.9", True, True))
    check("RG1 at 22,596 steps", (rg1["steps"][3], pct(rg1["m"][rg1["steps"][3]]["em"])), (22596, "36.6"))
    sizes = {"unit query": (uc, rc), **{f"{a} store": (corr[a], uc) for a, _ in LT_ARMS}}
    size = {k: str(q((x - y) / 10 / se_pp(x, y), 1)) for k, (x, y) in sizes.items()}
    check("size in standard errors", size, {"unit query": "1.2", "LT-K4 store": "1.1", "LT-RG001 store": "4.6", "LT-RG1 store": "3.0"})
    print()
    print("Size of each part in its own standard errors (size references only, not tests): " + ", ".join(f"{k} {v}" for k, v in size.items()) + ".")
    k4 = [m["LT-K4"]["m"][x]["correct"] - u["m"][y]["correct"] for x, y in zip(m["LT-K4"]["steps"], ust)]
    ur = [u["m"][x]["correct"] - rt["m"][y]["correct"] for x, y in zip(ust, rt["steps"])]
    assert max(abs(v) for v in k4[:3]) == 14 and min(k4[3:]) == 23 and max(k4[3:]) == 48, k4
    assert ur[0] == -4 and min(ur[1:]) == 18 and max(ur[1:]) == 27, ur
    print()
    print(f"Matched-fraction fact: LT-RG001 is above LT-U at all five fractions ({mf['LT-RG001 - LT-U'][0]} to {mf['LT-RG001 - LT-U'][3]} pp), and its checkpoint at {rg['steps'][3]:,} steps "
          f"({pct(rg['m'][rg['steps'][3]]['em'])}%) is already above LT-U's final checkpoint at {ust[-1]:,} steps ({pct(uf['em'])}%). "
          f"LT-RG1 is above LT-U at all five fractions ({mf['LT-RG1 - LT-U'][3]} to {mf['LT-RG1 - LT-U'][0]} pp), but its lead at the end rests on its last interval: it has {pct(rg1['m'][rg1['steps'][3]]['em'])}% at {rg1['steps'][3]:,} steps and {pct(rg1['m'][rg1['steps'][4]]['em'])}% at {rg1['steps'][4]:,}. "
          f"LT-K4 ({m['LT-K4']['steps'][-1]:,} steps) is within 1.4 pp of LT-U at 20, 40 and 60% ({', '.join(mf['LT-K4 - LT-U'][:3])} pp) and above it by {mf['LT-K4 - LT-U'][3]} and {mf['LT-K4 - LT-U'][4]} pp at 80 and 100%. "
          f"LT-U is above R-r128 by 1.8 to 2.7 pp at four of the five fractions and below it by 0.4 pp at 20% ({', '.join(mf['LT-U - R-r128'])} pp).")
    au = d["u_audit"]["runs"][f"{U_ARM}-E16"]
    assert au["original"]["correct"] == uc and au["original_matches_record"]["correct"] is True and au["max_repeated_logit_drift"] == 0.0
    check("U audit", (au["zero"]["correct"], au["shuffle"]["correct"]), (2, 0))
    print()
    print(f"Zero / shuffled memory correct of 1,000 (original, zero, shuffled): LT-U {au['original']['correct']}/{au['zero']['correct']}/{au['shuffle']['correct']} (original equals the record; maximum repeated-logit drift {au['max_repeated_logit_drift']})")
    log = d["u_log"]
    assert all(r_["clipped"] for r_ in log) and len(log) == 493
    print(f"Gradient clip active at logged steps: LT-U {sum(1 for r_ in log if r_['clipped'])} of {len(log)}")
    t = rec["training"]
    check("U compute", (str(q(t["steady_step_mean_seconds"], 3)), str(q(t["seconds"] / 60, 1)), rec["memory"]["state_bytes_per_episode"], rec["inference_parameters"]), ("0.180", "74.3", 65536, 475138))
    print(f"State bytes per episode: {rec['memory']['state_bytes_per_episode']}; inference parameters {rec['inference_parameters']:,}; training {q(t['seconds'] / 60, 1)} min at {q(t['steady_step_mean_seconds'], 3)} s per step "
          f"(job {rec['job_id']}, science commit {rec['git_commit'][:7]})")


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
          "and the trained-LSQ line stays open. The unit query is in every LSQT arm and not in R-r128. LT-U is the descriptive control (the delta store of R-r128 trained with the unit query); "
          "it is not a registered test and enters neither registered outcome. Clean and read head: accuracy of the value decoded from the stored and from the read value.}",
          "\\label{tab:lsqt}", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
          "Arm & Store & EM (\\%) & Read error & Clean head (\\%) & Read head (\\%) & vs R-r128 (pp) \\\\", "\\midrule"]
    lt_names = {"LT-K4": "four Kaczmarz sweeps", "LT-RG001": "ridge, $\\eta=0.01$", "LT-RG1": "ridge, $\\eta=1$"}
    for a, _ in LT_ARMS:
        x = m[a]["m"][m[a]["steps"][-1]]
        L.append(f"{a} & {lt_names[a]} & {pct(x['em'])} & {q(x['relative_read_error'], 3)} & {pct(x['clean_value_correct'])} & {pct(x['actual_read_correct'])} & {pp(x['correct'], r['correct'])} \\\\")
    uf = d["u"]["m"][d["u"]["steps"][-1]]
    corr, uc, rc = final_counts(d)
    L += ["\\midrule",
          f"LT-U (control) & delta, unit query & {pct(uf['em'])} & {q(uf['relative_read_error'], 3)} & {pct(uf['clean_value_correct'])} & {pct(uf['actual_read_correct'])} & {pp(uc, rc)} \\\\",
          "\\midrule",
          f"R-r128 (WP-L4) & delta & {pct(r['em'])} & {q(r['relative_read_error'], 3)} & {pct(r['clean_value_correct'])} & {pct(r['actual_read_correct'])} & \\\\",
          f"D128-3 (DPFP) & delta & {pct(p['em'])} & {q(p['relative_read_error'], 3)} & {pct(p['clean_value_correct'])} & {pct(p['actual_read_correct'])} & \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}%", "}"]
    tx = lambda x: x.replace("-", "$-$")
    ust, rt = d["u"]["steps"], d["r128_traj"]["steps"]
    L += ["\\newcommand{\\lsqtsplittable}{%", "\\begin{table}[htbp]", "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
          "\\caption{LSQT-U: the gain of each LSQT arm over R-r128 (percentage points of validation exact match at the final checkpoints), split through the control LT-U "
          "(R-r128's delta store trained with the unit query) into a unit-query part and a store part; the parts add up to the total. One seed. "
          "The standard errors in parentheses are unpaired binomial errors of the 1,000 episodes and cover episode sampling only. "
          "The split is descriptive: no ridge store with a learned query was trained, so an interaction of store and query cannot be separated, and the arms differ in step count "
          f"(LT-U {ust[-1]:,} steps, R-r128 {rt[-1]:,}). LT-U is not a registered test and enters neither registered outcome.}}",
          "\\label{tab:lsqt-split}", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
          " & & Total & Unit query & Store \\\\", "Arm & Steps & arm $-$ R-r128 & LT-U $-$ R-r128 & arm $-$ LT-U \\\\", "\\midrule"]
    for a, _ in LT_ARMS:
        L.append(f"{a} & {m[a]['steps'][-1]:,} & {signed(corr[a], rc)} & {signed(uc, rc)} & {signed(corr[a], uc)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}%", "}",
          "\\newcommand{\\lsqtmatchtable}{%", "\\begin{table}[htbp]", "\\centering\\footnotesize",
          "\\caption{LSQT-U: differences of validation exact match (pp) at 20\\%, 40\\%, 60\\%, 80\\% and 100\\% of each arm's training (one seed, 1,000 episodes; "
          "each cell is a difference of two single evaluations). First and last checkpoint, in steps: "
          + "; ".join(f"{n} {x[0]:,} and {x[-1]:,}" for n, x in [("LT-U", ust), ("R-r128", rt)] + [(a, m[a]["steps"]) for a, _ in LT_ARMS]) + ".}",
          "\\label{tab:i-lsqt-match}", "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Difference & 20\\% & 40\\% & 60\\% & 80\\% & 100\\% \\\\", "\\midrule"]
    for k, v in matched(d).items():
        L.append(k.replace(" - ", " $-$ ") + " & " + " & ".join(tx(x) for x in v) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}%", "}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", help="write the two LaTeX table macros to this file")
    a = ap.parse_args()
    d = load()
    print_sq2(d)
    print_lsqt(d)
    print()
    print_lsqt_u(d)
    if a.tex:
        write_tex(d, a.tex)


if __name__ == "__main__":
    main()
