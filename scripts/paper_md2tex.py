#!/usr/bin/env python3
"""WP-T11 Part B: convert sections of docs/paper/draft.md into LaTeX appendix files under docs/paper/tex/ (stdlib only).

This is the one-time conversion that produced docs/paper/tex/app_*.tex from the markdown draft; from then on the .tex files
are the manuscript. It keeps the wording of the draft, escapes LaTeX specials, turns tables into longtables and figures
into figure floats, replaces en dashes by hyphens and the minus sign by a math minus, and moves every evidence tag
`<!-- ID -->` to a LaTeX comment `% ID` at the end of the line it stood on (tags inside a table row go to the row's end).

Usage: python3 scripts/paper_md2tex.py   (rewrites docs/paper/tex/app_*.tex)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFT = os.path.join(ROOT, "docs/paper/draft.md")
OUT = os.path.join(ROOT, "docs/paper/tex")

TAG = re.compile(r"<!--\s*([A-Z]+\d+)\s*-->")
OTHER_COMMENT = re.compile(r"<!--(.*?)-->", re.S)

UNI = {"−": r"\ensuremath{-}", "–": "-", "—": ", ", "≥": r"\ensuremath{\geq}", "≤": r"\ensuremath{\leq}",
       "×": r"\ensuremath{\times}", "→": r"\ensuremath{\rightarrow}", "≈": r"\ensuremath{\approx}",
       "±": r"\ensuremath{\pm}", "ρ": r"\ensuremath{\rho}", "δ": r"\ensuremath{\delta}", "θ": r"\ensuremath{\theta}",
       "ν": r"\ensuremath{\nu}", "Π": r"\ensuremath{\Pi}", "⌊": r"\ensuremath{\lfloor}", "⌋": r"\ensuremath{\rfloor}",
       "²": r"\ensuremath{^2}", "·": r"\ensuremath{\cdot}", "…": r"\ldots{}", "§": r"\S{}"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "$": r"\$", "/": r"/\allowbreak{}"}


def esc_plain(s):
    out = []
    for ch in s:
        out.append(SPECIAL.get(ch) or UNI.get(ch) or ch)
    return "".join(out)


def esc_code(s):
    s = s.replace("−", "-").replace("–", "-")
    s = "".join(SPECIAL.get(ch, ch) for ch in s)
    # long hexadecimal revisions may break across lines
    return re.sub(r"[0-9a-f]{16,}", lambda m: r"\allowbreak{}".join(m.group(0)[i:i + 8] for i in range(0, len(m.group(0)), 8)), s)


def curly(s):
    """Straight double quotes to paired curly quotes (per call); apostrophes between letters to a right single quote."""
    out, opened = [], False
    for ch in s:
        if ch == '"':
            out.append("”" if opened else "“")
            opened = not opened
        else:
            out.append(ch)
    s = "".join(out)
    return re.sub(r"(?<=\w)'(?=\w)", "’", s)


def href(url, text):
    url = url.replace("%", r"\%").replace("#", r"\#")
    return r"\href{" + url + "}{" + text + "}"


def inline(s):
    """Convert one markdown inline string (may contain newlines) to LaTeX."""
    s = curly(s)
    s = re.sub(r"(?<=\d),(?=[\u2212-]?\d+\.\d)", ", ", s)  # interval separators: allow a line break
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "`":
            j = s.find("`", i + 1)
            if j > 0:
                out.append(r"\texttt{" + esc_code(s[i + 1:j]) + "}")
                i = j + 1
                continue
        if s.startswith("\\(", i):
            j = s.find("\\)", i)
            if j > 0:
                out.append(s[i:j + 2])
                i = j + 2
                continue
        if s.startswith("**", i):
            j = s.find("**", i + 2)
            if j > 0:
                out.append(r"\textbf{" + inline(s[i + 2:j]) + "}")
                i = j + 2
                continue
        if c == "*" and (i == 0 or not s[i - 1].isalnum()):
            j = i + 1
            while True:
                j = s.find("*", j)
                if j < 0 or not (j + 1 < n and s[j + 1].isalnum()):
                    break
                j += 1
            if j > 0 and j > i + 1 and not s[i + 1].isspace():
                out.append(r"\emph{" + inline(s[i + 1:j]) + "}")
                i = j + 1
                continue
        if c == "[":
            m = re.match(r"\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)\)", s[i:], re.S)
            if m:
                text, url = inline(m.group(1)), m.group(2)
                if url.startswith("http"):
                    out.append(href(url, text))
                else:
                    out.append(text)  # relative repository links do not resolve in a PDF: keep the text only
                i += m.end()
                continue
        out.append(esc_plain(c))
        i += 1
    return "".join(out)


def split_tags(lines):
    """Remove evidence tags from each line; return (clean lines, tags per line)."""
    clean, tags = [], []
    for ln in lines:
        t = TAG.findall(ln)
        ln = TAG.sub("", ln)
        ln = OTHER_COMMENT.sub("", ln)
        clean.append(re.sub(r"[ ]{2,}", " ", ln).rstrip())
        tags.append(t)
    return clean, tags


def emit_lines(text_lines, tag_lists):
    """Attach `% tags` to the end of the line each tag stood on; tag-only lines become comment lines."""
    out = []
    for ln, t in zip(text_lines, tag_lists):
        if not ln.strip():
            if t:
                out.append("% " + " ".join(t))
            else:
                out.append("")
            continue
        out.append(ln + ("  % " + " ".join(t) if t else ""))
    return out


def paragraph(lines):
    clean, tags = split_tags(lines)
    # a tag-only line is kept as a comment; the remaining lines form the paragraph text
    text = "\n".join(clean)
    conv = inline(text).split("\n")
    assert len(conv) == len(clean), (clean, conv)
    return emit_lines(conv, tags)


def table(rows, wide=False):
    cells = []
    for r in rows:
        clean, tags = split_tags([r])
        parts = [c.strip() for c in re.split(r"(?<!\\)\|", clean[0].strip())[1:-1]]
        cells.append((parts, tags[0]))
    header, body = cells[0], cells[2:]
    ncol = len(header[0])
    allrows = [header] + body
    weights = []
    for j in range(ncol):
        lens = [len(r[0][j]) for r in allrows if j < len(r[0])]
        avg = sum(lens) / max(1, len(lens))
        weights.append(max(12.0, min(70.0, avg)))
    tot = sum(weights)
    frac = [int(round(1000 * w / tot)) for w in weights]
    spec = "".join("W{%d}{%d}" % (f, 2 * ncol) for f in frac)
    size = r"\footnotesize" if ncol >= 4 else r"\small"
    out = [r"{" + size, r"\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{" + spec + "}", r"\toprule"]

    def row(r):
        return " & ".join(inline(c) for c in r[0]) + r" \\ \addlinespace[3pt]" + ("  % " + " ".join(r[1]) if r[1] else "")

    out.append(row(header))
    out.append(r"\midrule \endhead")
    for r in body:
        out.append(row(r))
    out += [r"\bottomrule", r"\end{longtable}", "}"]
    return out


def convert(lines, first_level=2):
    """Convert draft lines to LaTeX lines. `##` headings become \\subsection*, `###` become \\subsubsection*."""
    out, i, n = [], 0, len(lines)
    while i < n:
        ln = lines[i]
        if not ln.strip():
            out.append("")
            i += 1
            continue
        m = re.match(r"(#{2,4})\s+(.*)$", ln)
        if m:
            lvl, title = len(m.group(1)), m.group(2).strip()
            clean, tags = split_tags([title])
            cmd = {2: r"\subsection*", 3: r"\subsubsection*", 4: r"\paragraph*"}[lvl]
            body = inline(clean[0])
            toc = ""
            if lvl == 2:
                toc = r"\addcontentsline{toc}{subsection}{" + body.replace(r"\texttt", r"\textit") + "}"
            out += [cmd + "{" + body + "}" + toc + ("  % " + " ".join(tags[0]) if tags[0] else "")]
            i += 1
            continue
        if ln.startswith("|") and i + 1 < n and re.match(r"^\|[ \-|:]+\|$", lines[i + 1]):
            j = i
            while j < n and lines[j].startswith("|"):
                j += 1
            out += [""] + table(lines[i:j]) + [""]
            i = j
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", ln)
        if m:
            src = m.group(2).replace(".png", ".pdf")
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            cap = []
            while j < n and lines[j].strip():
                cap.append(lines[j])
                j += 1
            captext = "\n".join(cap).strip()
            if captext.startswith("*") and captext.endswith("*"):
                captext = captext[1:-1]
            captext = re.sub(r"\s*\[PDF\]\([^)]*\)\.?", "", captext)
            captext = re.sub(r"^Figure\s+\d+\.\s*", "", captext)
            clean, tags = split_tags(captext.split("\n"))
            conv = inline("\n".join(clean)).split("\n")
            capl = emit_lines(conv, tags)
            out += [r"\begin{figure}[htbp]", r"\centering", r"\includegraphics[width=0.96\linewidth]{../" + src + "}",
                    r"\caption{" + capl[0]] + capl[1:] + ["}", r"\end{figure}"]
            i = j
            continue
        if ln.strip() == r"\[":
            j = i
            while j < n and lines[j].strip() != r"\]":
                j += 1
            out += lines[i:j + 1]
            i = j + 1
            continue
        if ln.startswith("- "):
            items = []
            while i < n and (lines[i].startswith("- ") or (lines[i].startswith("  ") and lines[i].strip())):
                if lines[i].startswith("- "):
                    items.append([lines[i][2:]])
                else:
                    items[-1].append(lines[i].strip())
                i += 1
            out.append(r"\begin{itemize}[leftmargin=1.4em,itemsep=2pt,topsep=3pt]")
            for it in items:
                body = paragraph(it)
                out.append(r"\item " + body[0])
                out += body[1:]
            out += [r"\end{itemize}", ""]
            continue
        j = i
        para = []
        while j < n and lines[j].strip() and not lines[j].startswith(("|", "- ", "![")) and not re.match(r"#{2,4}\s", lines[j]) \
                and lines[j].strip() != r"\[":
            para.append(lines[j])
            j += 1
        out += paragraph(para)
        i = j
    return out


def sections(md):
    """Map a key to the lines of a `##` section (up to the next `##`)."""
    idx = [(i, l) for i, l in enumerate(md) if l.startswith("## ")]
    res = {}
    for k, (i, l) in enumerate(idx):
        end = idx[k + 1][0] if k + 1 < len(idx) else len(md)
        res[l[3:].strip()] = md[i:end]
    return res


def find(secs, prefix):
    hits = [k for k in secs if k.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return secs[hits[0]]


def strip_heading(lines):
    return lines[1:]


def write(name, header, body_lines, landscape=False):
    os.makedirs(OUT, exist_ok=True)
    text = [r"% Generated once by scripts/paper_md2tex.py from docs/paper/draft.md; edit this file from now on.", header]
    if landscape:
        text.append(r"\begin{landscape}")
    text += body_lines
    if landscape:
        text.append(r"\end{landscape}")
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(text) + "\n")


def main():
    md = open(DRAFT).read().split("\n")
    secs = sections(md)
    # appendix letters follow the order of \section after \appendix in paper.tex: A, B, C, D, E, F, G, H
    a = find(secs, "Appendix A.")
    b = find(secs, "Appendix B.")
    c = find(secs, "Appendix C.")
    write("app_A.tex", r"\section{Proof sketches and non-implications}\label{app:A}", convert(strip_heading(a)))
    write("app_B.tex", r"\section{Experimental protocol and evidence map}\label{app:B}", convert(strip_heading(b)))
    write("app_C.tex", r"\section{Claim contract}\label{app:C}", convert(strip_heading(c)), landscape=True)
    d = convert(find(secs, "3. Method")) + [""] + convert(find(secs, "4. Theory"))
    write("app_D.tex", r"\section{Method and theory in full}\label{app:D}" "\n"
          r"\noindent Record sections 3 and 4 of the manuscript; section numbers below follow the record numbering.", d)
    e = []
    for pre in ("5. Mechanism", "6. Same-rank", "7. Stopping"):
        e += convert(find(secs, pre)) + [""]
    write("app_E.tex", r"\section{Symbolic studies in full}\label{app:E}" "\n"
          r"\noindent Record sections 5 to 7: the oracle and learned-writer mechanism experiments, the same-rank and historical budget controls, and the stopping studies.", e)
    f = convert(find(secs, "8. A single language-model module"))
    write("app_F.tex", r"\section{Language-model studies in full}\label{app:F}" "\n"
          r"\noindent Record section 8: the access protocol, every writer screen, the registered oracle-reader confirmation at two backbones, DPFP and DPFP-X, the per-step readouts, and the summary subsection.", f)
    g = convert(find(secs, "9. Limitations"))
    write("app_G.tex", r"\section{Limitations and reproducibility in full}\label{app:G}" "\n"
          r"\noindent Record section 9.", g)
    h = convert(find(secs, "2. Related work"))
    write("app_H.tex", r"\section{Related work in full}\label{app:H}" "\n"
          r"\noindent Record section 2.", h)
    # references list -> refs.bib
    ref = find(secs, "References and source records")
    entries = []
    for ln in ref:
        m = re.match(r"- (.+?), \[\*(.+?)\*\]\((https?://[^)]+)\)(.*)$", ln)
        if m:
            entries.append((m.group(1), m.group(2), m.group(3), m.group(4).strip()))
    bib = []
    for k, (au, ti, url, rest) in enumerate(entries, 1):
        key = "r%03d" % k
        extra = rest.strip(" ().")
        bib.append("@misc{%s,\n  author = {{%s}},\n  title = {{%s}},\n  note = {\\url{%s}%s}\n}\n" % (
            key, au.replace("&", r"\&"), ti.replace("&", r"\&").replace("#", r"\#").replace("%", r"\%"),
            url.replace("%", r"\%").replace("#", r"\#"), (" (" + extra + ")") if extra else ""))
    with open(os.path.join(OUT, "refs.bib"), "w") as fh:
        fh.write("\n".join(bib))
    print("wrote", len(entries), "bib entries")


if __name__ == "__main__":
    sys.exit(main())
