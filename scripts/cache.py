#!/usr/bin/env python3
"""Inspect and manage the result cache.

Subcommands:
  list                  — show all cache keys, record counts, total shots.
  inspect <key>         — show every record under a key.
  invalidate <run_id>   — soft-invalidate a single record.
  prune                 — drop soft-invalidated records from disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import cache as _cache  # noqa: E402


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def cmd_list(_args: argparse.Namespace) -> int:
    keys = _cache.list_keys()
    if not keys:
        print("(cache is empty)")
        return 0
    print(f"{'key':16s}  {'records':>7s}  {'shots':>8s}  newest")
    for key in keys:
        records = _cache.get(key)
        if not records:
            continue
        shots = sum(int(r.get("shots", 0) or 0) for r in records)
        newest = max((r.get("created_at", 0) or 0) for r in records)
        print(f"{key[:16]}  {len(records):>7d}  {shots:>8d}  {_fmt_ts(newest)}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    # Allow either full key or prefix match (first hit wins).
    keys = [k for k in _cache.list_keys() if k.startswith(args.key)]
    if not keys:
        print(f"no key matches {args.key!r}", file=sys.stderr)
        return 1
    key = keys[0]
    records = _cache._read_all(key)  # include invalid for inspect
    print(f"key: {key}  records: {len(records)}")
    for r in records:
        valid = "OK " if r.get("valid", True) else "INV"
        print(
            f"  [{valid}] run_id={r.get('run_id', '-')[:12]}  "
            f"shots={r.get('shots', 0)}  "
            f"created={_fmt_ts(r.get('created_at'))}"
        )
        if args.verbose:
            print(json.dumps(r, indent=2, default=str))
    return 0


def cmd_invalidate(args: argparse.Namespace) -> int:
    ok = _cache.invalidate(args.run_id)
    print("invalidated" if ok else f"no record with run_id={args.run_id}")
    return 0 if ok else 1


def cmd_prune(_args: argparse.Namespace) -> int:
    dropped = _cache.prune_invalid()
    print(f"dropped {dropped} invalid record(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list keys").set_defaults(fn=cmd_list)

    p_insp = sub.add_parser("inspect", help="inspect a key")
    p_insp.add_argument("key", help="full key or prefix")
    p_insp.add_argument("-v", "--verbose", action="store_true")
    p_insp.set_defaults(fn=cmd_inspect)

    p_inv = sub.add_parser("invalidate", help="soft-invalidate a record")
    p_inv.add_argument("run_id")
    p_inv.set_defaults(fn=cmd_invalidate)

    sub.add_parser("prune", help="drop invalid records").set_defaults(fn=cmd_prune)

    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
