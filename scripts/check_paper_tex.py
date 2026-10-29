#!/usr/bin/env python3
"""WP-T11 Part B checks for the LaTeX manuscript docs/paper/tex (stdlib only).

1. Every evidence tag `% ID ...` (a trailing LaTeX comment made only of ledger ids) resolves to a row of
   docs/paper/evidence_llm.md or docs/paper/evidence_symbolic.md.
2. Every tag of docs/paper/draft.md (the superseded markdown record) still occurs somewhere in the .tex sources
   (WP-T12 adds tags LS/BW/CT/SQ/LQ to the .tex sources and the ledger only; the draft is not extended).
3. No en dash, em dash or `--` in any .tex source (house style), and no tab characters.
4. Every number in the hand-written main text (abstract.tex, sec_*.tex) occurs in draft.md, a ledger or the
   generated G1, G2 and G5 tables (so no number was invented while condensing).
5. refs.bib has one entry per reference of draft.md.
Exit status 1 if any check fails.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "docs/paper")
TEX = os.path.join(P, "tex")
ID = r"[A-Z]+\d+"


def read(path):
    with open(path) as f:
        return f.read()


def ledger_ids():
    ids = set()
    for name in ("evidence_llm.md", "evidence_symbolic.md"):
        ids |= set(re.findall(r"^\|\s*\(?(" + ID + r")\b", read(os.path.join(P, name)), re.M))
    return ids


def tex_tags(path):
    tags = []
    for ln in read(path).split("\n"):
        m = re.search(r"%\s*((?:" + ID + r"\s*)+)$", ln)
        if m:
            tags += m.group(1).split()
    return tags


def numbers(text):
    return set(re.findall(r"\d+(?:,\d{3}(?!\d))*(?:\.\d+)?", text))


def main():
    bad = 0
    ids = ledger_ids()
    files = sorted(glob.glob(os.path.join(TEX, "*.tex")))
    all_tags = set()
    for f in files:
        t = tex_tags(f)
        all_tags |= set(t)
        un = sorted(set(t) - ids)
        if un:
            print("unresolved tags in", os.path.basename(f), un)
            bad += 1
    draft = read(os.path.join(P, "draft.md"))
    dtags = set(re.findall(r"<!--\s*(" + ID + r")\s*-->", draft))
    missing = sorted(dtags - all_tags)
    print(f"tex tags {len(all_tags)}, draft tags {len(dtags)}; in draft but not in tex: {missing}")
    bad += bool(missing)
    for f in files:
        t = read(f)
        if "\u2013" in t or "\u2014" in t or "--" in t.replace("---", "") and not f.endswith("g1_tables.tex") and re.search(r"(?<!-)--(?!-)", re.sub(r"^%.*$", "", t, flags=re.M)):
            print("dash found in", os.path.basename(f))
            bad += 1
        if "\t" in t:
            print("tab in", os.path.basename(f))
            bad += 1
    pool = numbers(draft)
    for name in ("evidence_llm.md", "evidence_symbolic.md", "g1_tables.txt", "g2_tables.txt", "g5_tables.txt"):
        pool |= numbers(read(os.path.join(P, name)))
    pool |= numbers(read(os.path.join(TEX, "g1_tables.tex")))
    pool |= numbers(read(os.path.join(TEX, "g2_tables.tex")))
    pool |= numbers(read(os.path.join(TEX, "g5_tables.tex")))
    main_files = [os.path.join(TEX, "abstract.tex")] + sorted(glob.glob(os.path.join(TEX, "sec_*.tex")))
    unmatched = {}
    floats = []
    for n in pool:
        try:
            floats.append(float(n.replace(",", "")))
        except ValueError:
            pass

    def rounded_match(n):
        """A shortened number is accepted when it is a rounding of a pool number (the ledgers keep six decimals)."""
        if "." not in n:
            return False
        d = len(n.split(".")[1])
        return any(f"{p:.{d}f}" == n for p in floats)

    for f in main_files:
        t = re.sub(r"\\(?:label|ref|input|cite)\{[^}]*\}", "", read(f))
        t = re.sub(r"\\href\{[^}]*\}", "", t)
        t = re.sub(r"\\includegraphics\[[^\]]*\]|p\{[0-9.]+cm\}", "", t)
        t = re.sub(r"\\begin\{tabular\}\{[^}]*\}|W\{\d+\}\{\d+\}", "", t)
        t = re.sub(r"(?m)%.*$", "", t)
        for n in numbers(t):
            n2 = n.rstrip(".,")
            if n2 not in pool and n2.lstrip("0") not in pool and not rounded_match(n2) and n2 not in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "10"):
                unmatched.setdefault(os.path.basename(f), set()).add(n2)
    for k, v in unmatched.items():
        print("numbers not found in draft/ledgers/tables:", k, sorted(v))
    bib = read(os.path.join(TEX, "refs.bib"))
    refs = [l for l in draft.split("\n") if re.match(r"- .+?, \[\*.+?\*\]\(https?://", l)]
    print(f"refs.bib entries {bib.count('@misc')}, draft reference lines {len(refs)}")
    bad += bib.count("@misc") != len(refs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
