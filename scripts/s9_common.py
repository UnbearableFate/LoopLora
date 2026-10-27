"""Shared helpers of the WP-S9 scripts (stdlib only: importing this module never imports torch)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from audit_p4s_stopping import (
    digest_checkpoint, markdown_table, no_confirmation_path, read_csv, read_json, require, require_compute,
)

__all__ = ["absolute_path", "digest_checkpoint", "git_locked_file", "interval", "markdown_table",
           "no_confirmation_path", "now", "percent", "read_csv", "read_json", "require", "require_compute",
           "table_lines", "verify_git_lock", "write_csv", "write_json"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def absolute_path(value: str | Path, what: str) -> Path:
    """``value`` as a ``Path``; a relative path is refused so that no result lands in a job's working directory."""
    path = Path(value)
    require(path.is_absolute(), f"{what} must be an absolute path")
    return path


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    """Write rows with the union of their keys (first-seen order) unless ``fieldnames`` is given."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), restval="", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float, digits: int = 2) -> str:
    return f"{100.0 * float(value):+.{digits}f}"


def interval(row: Mapping[str, Any], digits: int = 2) -> str:
    """'+x.xx [+lo, +hi]' in percentage points for a row with difference/ci_low/ci_high."""
    return f"{percent(row['difference'], digits)} [{percent(row['ci_low'], digits)}, {percent(row['ci_high'], digits)}]"


def table_lines(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    return markdown_table(columns, rows)


def git_locked_file(path: str | Path, commit: str) -> bytes:
    """Bytes of ``path`` at ``commit`` in the repository that contains ``path`` (``git show``, never the workspace)."""
    require(isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "a Git lock needs a full 40-character commit hash")
    path = Path(path).resolve()
    root = Path(subprocess.check_output(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                                        text=True).strip()).resolve()
    return subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{path.relative_to(root).as_posix()}"])


def verify_git_lock(path: str | Path, commit: str, what: str) -> None:
    """The workspace file must be byte-identical to its committed lock."""
    require(git_locked_file(path, commit) == Path(path).read_bytes(), f"{what} differs from its Git lock {commit[:12]}")
