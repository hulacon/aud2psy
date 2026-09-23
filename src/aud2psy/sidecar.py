"""`aud2psy sidecar refresh`: bring existing sidecars up to schema 1.1.

Contract B 1.1 adds a `nulls` map to every model entry (what a NaN in each
column means). Feature values do not change between 1.0 and 1.1, so an old
family is brought forward by rewriting its `.meta.json` only; a CSV is never
written.

A refreshed sidecar is checked against the tables it describes: each model's
own columns are read from the family's `output` tables, and a column holding
NaN without a declaration refuses the refresh for that sidecar (nothing is
written). A declaration therefore never goes onto data it does not describe,
and a producer defect surfaces here rather than in a downstream fit.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .exceptions import Aud2PsyError
from .metadata import SCHEMA_VERSION, stamp_nulls


@dataclass
class RefreshResult:
    path: Path
    status: str  # "refreshed" | "unchanged" | "refused" | "skipped"
    undeclared: dict[str, list[str]] = field(default_factory=dict)  # model -> NaN columns
    note: str = ""


def find_sidecars(paths: list[str | Path]) -> list[Path]:
    """Sidecar files named directly, plus every `*.meta.json` under a directory."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(p.rglob("*.meta.json")))
        elif p.name.endswith(".meta.json") and p.is_file():
            found.append(p)
        else:
            raise Aud2PsyError(f"{p} is neither a directory nor an existing .meta.json sidecar")
    return found


def _table_path(sidecar: Path, recorded: str) -> Path:
    """The recorded table path, or the same filename beside the sidecar if the tree moved."""
    p = Path(recorded)
    if p.is_file():
        return p
    beside = sidecar.parent / p.name
    if beside.is_file():
        return beside
    raise Aud2PsyError(
        f"{sidecar}: output table {recorded} is missing (also not beside the sidecar). "
        "The refresh checks nulls against the data, so it cannot proceed without it."
    )


def _nan_columns(sidecar: Path, meta: dict) -> dict[str, set[str]]:
    """Per model: those of its declared columns that hold any NaN in the family's tables."""
    import pandas as pd

    wanted = {name: set(entry.get("columns") or ()) for name, entry in meta["models"].items()}
    all_wanted = set().union(*wanted.values()) if wanted else set()
    nan_cols: set[str] = set()
    for table in (meta.get("output") or {}).values():
        path = _table_path(sidecar, table["path"])
        df = pd.read_csv(path, usecols=lambda c: c in all_wanted)
        # numeric columns only: a string column's empty cell reads back as NaN
        # but is not a null in the contract's sense
        num = df.select_dtypes(include="number")
        nan_cols |= {c for c in num.columns if num[c].isna().any()}
    return {name: cols & nan_cols for name, cols in wanted.items()}


def refresh_sidecar(path: str | Path, *, dry_run: bool = False) -> RefreshResult:
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get("extractor") != "aud2psy":
        return RefreshResult(path, "skipped", note=f"extractor {meta.get('extractor')!r}")
    new = copy.deepcopy(meta)
    stamp_nulls(new["models"])
    undeclared = {
        name: sorted(cols - set(new["models"][name]["nulls"]))
        for name, cols in _nan_columns(path, new).items()
    }
    undeclared = {k: v for k, v in undeclared.items() if v}
    if undeclared:
        return RefreshResult(path, "refused", undeclared,
                             note="NaN in undeclared column(s): a producer defect; nothing written")
    new["schema_version"] = SCHEMA_VERSION
    if new == meta:
        return RefreshResult(path, "unchanged")
    from . import __version__

    new.setdefault("refreshed", []).append({
        "by": f"aud2psy {__version__}",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": ["schema_version", "models.*.nulls"],
        "from_schema_version": meta.get("schema_version"),
    })
    if not dry_run:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=2))
        os.replace(tmp, path)
    return RefreshResult(path, "refreshed")
