"""Rebuild the versioned name pool using only local candidates and model cache.

Run this on a compute node, never the login node. No network access is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from looplora.infra.seeds import derive_seed

MODEL = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
ROOT = Path(__file__).resolve().parent
LOCALES = ("en_US", "en_GB", "de_DE", "fr_FR", "it_IT", "en_IN")


def extract_candidates(source_dir: Path) -> dict:
    """Extract literal names without importing or executing Faker providers."""
    given: set[str] = set()
    surnames: set[str] = set()
    sources = []
    for locale in LOCALES:
        filename = "en-us" if locale == "en_US" else locale
        raw = (source_dir / f"looplora-faker-person-{filename}.py").read_bytes()
        sources.append({
            "url": f"https://raw.githubusercontent.com/joke2k/faker/v37.12.0/faker/providers/person/{locale}/__init__.py",
            "blake2b_128": hashlib.blake2b(raw, digest_size=16).hexdigest(),
        })
        for node in ast.walk(ast.parse(raw)):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id not in {"first_names_male", "first_names_female", "last_names"}:
                continue
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "OrderedDict":
                values = [pair[0] for pair in ast.literal_eval(value.args[0])]
            else:
                values = ast.literal_eval(value)
            (surnames if target.id == "last_names" else given).update(values)
    return {"version": "faker-person-candidates-v1", "sources": sources,
            "given_names": sorted(given), "surnames": sorted(surnames)}


def build(tokenizer, candidates: dict) -> dict:
    """Prefer given names, then permute labels independently of split/category."""
    names, token_ids, kinds = [], [], []
    seen_names: set[str] = {"Unknown"}
    seen_ids: set[int] = set()
    for key in ("given_names", "surnames"):
        for name in sorted(set(candidates[key])):
            if name in seen_names or re.fullmatch(r"[A-Z][a-z]{1,19}", name) is None:
                continue
            text = " " + name
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids) != text or ids[0] in seen_ids:
                continue
            seen_names.add(name)
            seen_ids.add(ids[0])
            names.append(name)
            token_ids.append(ids[0])
            kinds.append("given" if key == "given_names" else "surname")
    if len(names) < 1024:
        raise ValueError(f"Only {len(names)} qualifying names; need at least 1024")
    missing_ids = tokenizer.encode(" Unknown", add_special_tokens=False)
    if len(missing_ids) != 1 or tokenizer.decode(missing_ids) != " Unknown":
        raise ValueError("Missing-answer label must be exactly one token")
    selected = sorted(zip(names[:1024], token_ids[:1024], kinds[:1024], strict=True),
                      key=lambda item: (derive_seed(0, "name-pool-order", item[0]), item[0]))
    return {"version": "qwen3-names-v1", "model": MODEL, "revision": REVISION,
            "candidate_version": candidates["version"],
            "selection": "ASCII [A-Z][a-z]{1,19}; given names then surnames, lexical within each; unique names and token IDs; exact encode/decode of one leading space; select first 1024; Unknown reserved; order selected names by (derive_seed(0, 'name-pool-order', name), name)",
            "order_seed": 0,
            "names": [item[0] for item in selected], "token_ids": [item[1] for item in selected],
            "kinds": [item[2] for item in selected],
            "insufficient_name": "Unknown", "insufficient_token_id": missing_ids[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-faker", type=Path, help="Directory of the six pinned provider sources")
    args = parser.parse_args()
    candidate_path = ROOT / "name_candidates.v1.json"
    if args.extract_faker is not None:
        candidates = extract_candidates(args.extract_faker)
        candidate_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n")
    else:
        candidates = json.loads(candidate_path.read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pool = build(tokenizer, candidates)
    (ROOT / "qwen3_names.v1.json").write_text(json.dumps(pool, indent=2) + "\n")
    print(json.dumps({"names": len(pool["names"]), "given": pool["kinds"].count("given"),
                      "surname": pool["kinds"].count("surname"), "revision": REVISION}))


if __name__ == "__main__":
    main()
