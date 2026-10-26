"""Capacity check for the LLM oracle's value-to-key map (review C1, numpy only).

The P1-L oracle stores key = pair_codes[head, 0] and value = global_codes[tail],
two independent random unit tables. A read returns the bridge entity's value
code, so every hop after the first needs the reader to map that value code to
the bridge's unrelated pair code. One reader step can only apply
q = d + Q relu(U m + c) with r hidden units. This script fits that map over the
seen entity pool, for random pair codes and for rotation keys
normalize(Pi @ global_codes[e]), and replays T2 episodes (chains 8 and 4,
E = 12, delta rule with lambda = beta = 1) to measure hop-2 addressing.
The hop-1 query is the exact key, so the numbers are an upper bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def fit_linear(x: np.ndarray, y: np.ndarray):
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    return lambda m: m @ w


def fit_mlp(x: np.ndarray, y: np.ndarray, hidden: int, rng: np.random.Generator,
            steps: int, lr: float = 3e-3):
    """Full-batch Adam on q = d + relu(m U + c) Q, the per-step reader family."""
    n, dim = x.shape
    params = {
        "U": rng.normal(0, 1 / np.sqrt(dim), (dim, hidden)),
        "c": np.zeros(hidden),
        "Q": rng.normal(0, 1 / np.sqrt(hidden), (hidden, y.shape[1])),
        "d": np.zeros(y.shape[1]),
    }
    moments = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
    for t in range(1, steps + 1):
        pre = x @ params["U"] + params["c"]
        h = np.maximum(pre, 0)
        err = (h @ params["Q"] + params["d"] - y) * (2 / n)
        dh = (err @ params["Q"].T) * (pre > 0)
        grads = {"U": x.T @ dh, "c": dh.sum(0), "Q": h.T @ err, "d": err.sum(0)}
        for k, g in grads.items():
            m1, m2 = moments[k]
            m1 *= .9; m1 += .1 * g
            m2 *= .999; m2 += .001 * g * g
            params[k] -= lr * (m1 / (1 - .9 ** t)) / (np.sqrt(m2 / (1 - .999 ** t)) + 1e-8)
    return lambda m: np.maximum(m @ params["U"] + params["c"], 0) @ params["Q"] + params["d"]


def mean_cos(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sum(unit(pred) * unit(target), axis=-1)))


def episodes(codes: np.ndarray, keys: np.ndarray, step, rng: np.random.Generator,
             count: int, lengths=(8, 4)) -> dict[str, float]:
    """Replay chain episodes; the question asks hop 2 from the long chain's end."""
    n, r = codes.shape
    hits = {"hop1_decode": 0, "hop2_address": 0, "hop2_decode": 0, "hop2_decode_full": 0}
    for _ in range(count):
        nodes = rng.choice(n, sum(lengths) + len(lengths), replace=False)
        facts, offset = [], 0
        for length in lengths:
            facts += [(nodes[offset + i], nodes[offset + i + 1]) for i in range(length)]
            offset += length + 1
        memory = np.zeros((r, r))
        for i in rng.permutation(len(facts)):
            head, tail = facts[i]
            k, v = keys[head], codes[tail]
            memory += np.outer(v - memory @ k, k)
        start, bridge, answer = nodes[lengths[0] - 2], nodes[lengths[0] - 1], nodes[lengths[0]]
        read1 = memory @ keys[start]
        hits["hop1_decode"] += int(nodes[np.argmax(codes[nodes] @ read1)] == bridge)
        query = step(read1[None])[0]
        heads = np.array([h for h, _ in facts])
        hits["hop2_address"] += int(heads[np.argmax(keys[heads] @ query)] == bridge)
        read2 = memory @ query
        hits["hop2_decode"] += int(nodes[np.argmax(codes[nodes] @ read2)] == answer)
        hits["hop2_decode_full"] += int(np.argmax(codes @ read2) == answer)
    return {k: v / count for k, v in hits.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=768)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--mlp-steps", type=int, default=1500)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for r in args.ranks:
        for seed in range(args.seeds):
            rng = np.random.default_rng([r, seed])
            codes = unit(rng.standard_normal((args.entities, r)))
            pair = unit(rng.standard_normal((args.entities, r)))
            rotation, _ = np.linalg.qr(rng.standard_normal((r, r)))
            rotated = unit(codes @ rotation.T)
            for scheme, keys in (("pair_codes", pair), ("rotation", rotated)):
                for family in ("linear", "mlp"):
                    if family == "linear":
                        step = fit_linear(codes, keys)
                    else:
                        step = fit_mlp(codes, keys, r, rng, args.mlp_steps)
                    row = {"r": r, "seed": seed, "keys": scheme, "map": family,
                           "fit_cos": mean_cos(step(codes), keys)}
                    row.update(episodes(codes, keys, step, rng, args.episodes))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    summary = {}
    for row in rows:
        key = f"r{row['r']}/{row['keys']}/{row['map']}"
        summary.setdefault(key, []).append(row)
    table = {k: {m: float(np.mean([x[m] for x in v])) for m in
                 ("fit_cos", "hop1_decode", "hop2_address", "hop2_decode", "hop2_decode_full")}
             for k, v in summary.items()}
    for k, v in table.items():
        print(k, " ".join(f"{m}={x:.3f}" for m, x in v.items()))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                        "rows": rows, "mean": table}, indent=2) + "\n")


if __name__ == "__main__":
    main()
