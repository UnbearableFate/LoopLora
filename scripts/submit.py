#!/usr/bin/python3
"""Submit a LoopLora compute job (login node, stdlib only).

Enforces the current-stage cap from the user (2026-10-26): one experiment may
use at most 8 nodes x 1 h, or 8 nodes x 2 h when results already obtained
support scaling up (pass --scale-up with the decision/prereg that justifies it).
The cap is a check, not the request; size walltime from measured runs.

Examples:
  scripts/submit.py --name pytest --walltime 00:15:00 --worktree . --wait -- -m pytest -q tests
  scripts/submit.py --name p1s-screen --queue regular-g --walltime 00:40:00 \
      --commit HEAD -- -m looplora.sym.run --config configs/p1s.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/work/xg24i002/x10041/LoopLora")
LOG_ROOT = REPO / "logs" / "pbs"
JOB_SCRIPT = REPO / "scripts" / "pbs" / "job.pbs"
GROUP = "xg24i002"
MAX_NODES = 8          # user cap: <= 8 nodes per experiment
MAX_WALLTIME_S = 3600  # user cap: <= 1 h per experiment
SCALE_UP_WALLTIME_S = 7200  # user cap with --scale-up: <= 2 h when earlier results justify it
QUEUES = {"debug-g": 1800, "regular-g": None}


def parse_walltime(text: str) -> int:
    parts = [int(p) for p in text.split(":")]
    if len(parts) != 3:
        raise SystemExit(f"walltime must be HH:MM:SS, got {text!r}")
    h, m, s = parts
    return h * 3600 + m * 60 + s


def resolve_commit(ref: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify", ref + "^{commit}"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def job_alive(job_id: str) -> bool:
    out = subprocess.run(["qstat", "-f", job_id], capture_output=True, text=True)
    return out.returncode == 0 and "Job Id:" in out.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--queue", default="debug-g", choices=sorted(QUEUES))
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--walltime", required=True, help="HH:MM:SS")
    ap.add_argument("--scale-up", metavar="DOC",
                    help="repo path of the decision/prereg whose results justify a job above 1 h (<= 2 h)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--commit", help="git ref to snapshot (experiments)")
    src.add_argument("--worktree", help="run a working tree in place (tests/development only)")
    ap.add_argument("--log", help="log path (default logs/pbs/<name>-<stamp>.log)")
    ap.add_argument("--wait", action="store_true", help="block until the job ends and print the log tail")
    ap.add_argument("--wait-timeout", type=int, help="seconds (default: walltime + 30 min)")
    ap.add_argument("--tail", type=int, default=60)
    ap.add_argument("argv", nargs=argparse.REMAINDER, help="-- then python argv")
    a = ap.parse_args()

    argv = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
    if not argv:
        raise SystemExit("missing python argv after --")
    wall = parse_walltime(a.walltime)
    if not 1 <= a.nodes <= MAX_NODES:
        raise SystemExit(f"nodes={a.nodes} violates the cap (1..{MAX_NODES})")
    if a.scale_up and not (REPO / a.scale_up).is_file():
        raise SystemExit(f"--scale-up document {a.scale_up!r} does not exist in {REPO}")
    if wall > (SCALE_UP_WALLTIME_S if a.scale_up else MAX_WALLTIME_S):
        raise SystemExit(f"walltime {a.walltime} exceeds the per-experiment cap "
                         f"({'2 h with --scale-up' if a.scale_up else '1 h; 2 h needs --scale-up'})")
    qmax = QUEUES[a.queue]
    if qmax is not None and wall > qmax:
        raise SystemExit(f"walltime {a.walltime} exceeds {a.queue} limit")
    if a.nodes > 1:
        raise SystemExit("multi-node launch is not implemented in job.pbs yet")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    log = Path(a.log).absolute() if a.log else LOG_ROOT / f"{a.name}-{stamp}.log"
    if a.commit:
        source = {"kind": "commit", "ref": resolve_commit(a.commit)}
    else:
        source = {"kind": "worktree", "ref": str(Path(a.worktree).absolute())}
    spec = {"name": a.name, "queue": a.queue, "nodes": a.nodes, "walltime": a.walltime,
            "scale_up": a.scale_up, "source": source, "argv": argv, "log": str(log), "submitted": dt.datetime.now().isoformat()}
    spec_path = LOG_ROOT / "specs" / f"{a.name}-{stamp}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=1))

    cmd = ["qsub", "-q", a.queue, "-W", f"group_list={GROUP}", "-l", f"select={a.nodes}:mpiprocs=1",
           "-l", f"walltime={a.walltime}", "-N", a.name[:15], "-j", "oe", "-o", str(log) + ".pbs",
           "-v", f"LL_SPEC={spec_path}", str(JOB_SCRIPT)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(LOG_ROOT))
    if res.returncode != 0:
        print(res.stdout, res.stderr, file=sys.stderr)
        raise SystemExit(f"qsub failed rc={res.returncode}")
    job_id = res.stdout.strip()
    rec = dict(spec, job_id=job_id, spec_path=str(spec_path))
    with open(LOG_ROOT / "submissions.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({"job_id": job_id, "log": str(log), "spec": str(spec_path), "source": source}))
    if not a.wait:
        return 0

    exit_file = Path(str(log) + ".exit")
    deadline = time.time() + (a.wait_timeout or wall + 1800)
    missing_since = None
    while time.time() < deadline:
        if exit_file.exists():
            break
        if job_alive(job_id):
            missing_since = None
        else:
            missing_since = missing_since or time.time()
            if time.time() - missing_since > 90:  # ended without writing .exit
                break
        time.sleep(15)
    if log.exists():
        lines = log.read_text(errors="replace").splitlines()
        print("\n".join(lines[-a.tail:]))
    if not exit_file.exists():
        print(f"[submit] job {job_id} has no exit marker (timeout or killed); see {log}.pbs", file=sys.stderr)
        return 3
    return int(exit_file.read_text().strip() or 1)


if __name__ == "__main__":
    sys.exit(main())
