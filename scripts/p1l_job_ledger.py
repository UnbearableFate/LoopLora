#!/usr/bin/env python3
"""Bounded stdlib scheduler accounting for this worker's P1-L jobs (login-safe)."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re
import subprocess

from run_p1l_main import save


REPO = Path("/work/xg24i002/x10041/LoopLora")
HELPER = Path("/work/xg24i002/x10041/.codex/skills/miyabi-pbs-cmd/scripts/qstat_json.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args()
    with (REPO / "logs/pbs/submissions.jsonl").open() as stream:
        entries = [json.loads(line) for line in deque(stream, maxlen=256)]
    prefixes = ("pf-p1l", "size-p1l", "sm-p1l", "t-p1l", "p1l-", "ev-p1l", "fin-p1l")
    entries = [r for r in entries if r["submitted"] >= "2026-10-27T04:07"
               and r["name"].startswith(prefixes)]
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
            raise ValueError(f"PBS did not return job{entry['job_id']}; no silent accounting omission")
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
        raise ValueError("some P1-L jobs are still queued/running")
    result = dict(task="WP-P1L", budget_rule="B2: per-job8nodes x2h; no total cap", final=final,
        allocated_node_seconds=total, allocated_node_hours=total/3600,
        accounting="qstat observed elapsed_seconds times nodes; failed jobs included; live totals provisional",
        jobs=rows)
    save(args.out, result)
    print(json.dumps(dict(jobs=len(rows), allocated_node_hours=total/3600, final=final)))


if __name__ == "__main__":
    main()
