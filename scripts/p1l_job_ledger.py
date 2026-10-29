#!/usr/bin/env python3
"""Bounded stdlib scheduler accounting for this worker's P1-L jobs (login-safe)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

from run_p1l_main import save


REPO = Path("/work/xg24i002/x10041/LoopLora")
HELPER = Path("/work/xg24i002/x10041/.codex/skills/miyabi-pbs-cmd/scripts/qstat_json.py")
# Each study accounts only for its own job names: P1L-S's `ps-<stage>-<tag>` never matches a P1-L prefix.
LEDGERS = {
    "p1l": dict(name="P1-L", task="WP-P1L", since="2026-10-27T04:07",
                prefixes=("pf-p1l", "size-p1l", "sm-p1l", "t-p1l", "p1l-", "ev-p1l", "fin-p1l"),
                budget_rule="B2: per-job8nodes x2h; no total cap"),
    "p1l-s": dict(name="P1L-S", task="P1L-S", since="2026-10-27T00:00", prefixes=("ps-",),
                  budget_rule="P1L-S: per-job 1 node x 2h; node-hours recorded per job"),
    "p5l": dict(name="P5L", task="WP-P5L", since="2026-10-29T04:00",
                prefixes=("t-p5l", "sm-p5l", "tr-p5l", "ev-p5l", "fin-p5l"),
                budget_rule="P5L: tests plus smoke <= 0.5 node-h; per-job 1 node x 2h with --scale-up docs/prereg/P5L.md"),
    "b8w": dict(name="B8W", task="WP-L10", since="2026-10-29T13:00",
                prefixes=("t-b8w", "sm-b8w", "tr-b8w", "ev-b8w", "fin-b8w"),
                budget_rule="B8W: 1.0-2.0 node-h in total including tests and smokes; per-job 1 node x 1 h"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--study", choices=tuple(LEDGERS), default="p1l")
    args = parser.parse_args()
    ledger = LEDGERS[args.study]
    prefixes = ledger["prefixes"]
    submissions = REPO / "logs/pbs/submissions.jsonl"
    if submissions.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("submission metadata exceeds the bounded control-plane scan")
    entries = []
    with submissions.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["submitted"] >= ledger["since"] and row["name"].startswith(prefixes):
                entries.append(row)
    history = subprocess.run(["/usr/bin/python3", "-I", str(HELPER), "-H", "--hday", "2", "--hnum", "200",
        "--fields", "job_id,job_name,status,elapsed_seconds,nodes"], capture_output=True, text=True, check=True)
    active = subprocess.run(["/usr/bin/python3", "-I", str(HELPER),
        "--fields", "job_id,job_name,status,elapsed_seconds,nodes"], capture_output=True, text=True, check=True)
    states = {}
    for result in (history, active):
        value = json.loads(result.stdout)
        if not value["ok"]:
            raise ValueError(value)
        states.update({r["job_id"].split(".")[0]: r for r in value.get("jobs", [])})
    rows, total, final = [], 0, True
    for entry in entries:
        job = states.get(entry["job_id"].split(".")[0])
        if job is None:
            exact = subprocess.run(["/usr/bin/python3", "-I", str(HELPER), "-H", entry["job_id"],
                "--fields", "job_id,job_name,status,elapsed_seconds,nodes"],
                capture_output=True, text=True, check=True)
            value = json.loads(exact.stdout)
            if not value["ok"] or len(value.get("jobs", [])) != 1:
                raise ValueError(f"PBS did not return job{entry['job_id']}; no silent accounting omission")
            job = value["jobs"][0]
        log = Path(entry["log"])
        exit_file = Path(str(log) + ".exit")
        exit_code = int(exit_file.read_text()) if exit_file.exists() else None
        host = None
        if log.exists():
            with log.open() as stream:
                header = stream.readline(4096)
            match = re.search(r"host=(\S+)", header)
            host = match.group(1) if match else None
        elapsed, nodes = job["elapsed_seconds"] or 0, job["nodes"] or 0
        total += elapsed * nodes
        done = job["status"] in {"FINISH", "EXPIRED"}
        final = final and done
        rows.append(dict(job_id=entry["job_id"], name=entry["name"], host=host,
            status=job["status"], nodes=nodes, allocated_seconds=elapsed, node_hours=elapsed*nodes/3600,
            requested_walltime=entry["walltime"], exit_code=exit_code,
            source=entry["source"], argv=entry["argv"], log=str(log)))
    if args.require_final and not final:
        raise ValueError(f"some {ledger['name']} jobs are still queued/running")
    result = dict(task=ledger["task"], budget_rule=ledger["budget_rule"], final=final,
        allocated_node_seconds=total, allocated_node_hours=total/3600,
        accounting="qstat observed elapsed_seconds times nodes; failed jobs included; live totals provisional",
        jobs=rows)
    save(args.out, result)
    print(json.dumps(dict(jobs=len(rows), allocated_node_hours=total/3600, final=final)))


if __name__ == "__main__":
    main()
