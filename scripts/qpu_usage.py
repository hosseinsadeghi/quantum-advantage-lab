#!/usr/bin/env python3
"""Inspect the IonQ QPU usage log.

Subcommands:
  summary               — totals: jobs, shots, cost; per-backend breakdown.
  list                  — recent jobs (joined submitted/completed events).
  raw                   — dump every event verbatim.
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

from backend import qpu_usage as _u  # noqa: E402


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_money(x: float | None) -> str:
    return f"${float(x):.4f}" if x else "-"


def _join_by_job(records: list[dict]) -> list[dict]:
    """Merge each job's submitted + completed events into one row."""
    rows: dict[str, dict] = {}
    for r in records:
        jid = r.get("job_id") or ""
        rows.setdefault(jid, {"job_id": jid})
        rows[jid].update(r)  # completed overlays submitted
    return sorted(
        rows.values(), key=lambda r: r.get("timestamp_unix") or 0
    )


def cmd_summary(_args: argparse.Namespace) -> int:
    s = _u.summarize()
    print(f"log: {s['log_path']}")
    print(f"  total jobs           : {s['total_jobs']}")
    print(f"  completed            : {s['completed_jobs']}")
    print(f"  total shots          : {s['total_shots']:,}")
    print(f"  total cost           : {_fmt_money(s['total_cost_usd'])}")
    print(f"  total exec time (s)  : {s['total_execution_time_seconds']:.3f}")
    by_be = s.get("by_backend") or {}
    if by_be:
        print()
        print(f"  {'backend':24s}  {'jobs':>5s}  {'shots':>10s}  {'cost':>10s}")
        for be, slot in sorted(by_be.items()):
            print(
                f"  {be[:24]:24s}  {slot['jobs']:>5d}  "
                f"{slot['shots']:>10,d}  {_fmt_money(slot['cost_usd']):>10s}"
            )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = _join_by_job(_u.read_all())
    if not rows:
        print("(no QPU jobs logged yet)")
        return 0
    rows = rows[-args.limit :] if args.limit else rows
    print(
        f"{'submitted':19s}  {'job_id':12s}  {'backend':16s}  "
        f"{'qb':>3s}  {'shots':>6s}  {'status':10s}  {'cost':>8s}"
    )
    for r in rows:
        circ = r.get("circuit") or {}
        print(
            f"{_fmt_ts(r.get('timestamp_unix')):19s}  "
            f"{(r.get('job_id') or '-')[:12]:12s}  "
            f"{(r.get('backend') or '-')[:16]:16s}  "
            f"{circ.get('n_qubits') or '-':>3}  "
            f"{r.get('shots') or 0:>6d}  "
            f"{(r.get('status') or r.get('event') or '-'):10s}  "
            f"{_fmt_money(r.get('cost_usd')):>8s}"
        )
    return 0


def cmd_raw(args: argparse.Namespace) -> int:
    for r in _u.read_all():
        print(json.dumps(r, indent=2 if args.pretty else None, default=str))
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    """Refetch IonQ state for one or all open jobs."""
    if args.job_id:
        rec = _u.refresh(args.job_id)
        if rec is None:
            print(f"could not refresh {args.job_id} (not in log or IonQ unreachable)", file=sys.stderr)
            return 1
        print(json.dumps(rec, indent=2, default=str))
        return 0
    recs = _u.refresh_open_jobs()
    print(f"refreshed {len(recs)} open job(s)")
    for r in recs:
        print(f"  {r['job_id']}  status={r.get('ionq_status')}  cost_usd={r.get('cost_usd')}")
    return 0


def cmd_update_cost(args: argparse.Namespace) -> int:
    """Refetch cost from IonQ and merge it into existing records in-place."""
    if args.job_id:
        diff = _u.update_cost(args.job_id)
        if diff is None:
            print(f"could not update {args.job_id} (not in log or IonQ unreachable)", file=sys.stderr)
            return 1
        print(json.dumps(diff, indent=2, default=str))
        return 0
    results = _u.update_cost_all()
    print(f"updated {sum(1 for v in results.values() if v is not None)}/{len(results)} job(s)")
    for jid, diff in results.items():
        if diff is None:
            print(f"  {jid}  (skipped: not in log or IonQ unreachable)")
        else:
            print(f"  {jid}  status={diff.get('ionq_status')}  cost_usd={diff.get('cost_usd')}  exec_s={diff.get('execution_time_seconds')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="totals + per-backend breakdown").set_defaults(fn=cmd_summary)

    plist = sub.add_parser("list", help="recent jobs")
    plist.add_argument("--limit", type=int, default=20, help="rows to show (default 20; 0 = all)")
    plist.set_defaults(fn=cmd_list)

    praw = sub.add_parser("raw", help="dump every event")
    praw.add_argument("--pretty", action="store_true")
    praw.set_defaults(fn=cmd_raw)

    pref = sub.add_parser("refresh", help="refetch IonQ state for a job (or all open jobs)")
    pref.add_argument("job_id", nargs="?", help="job UUID (default: all open jobs)")
    pref.set_defaults(fn=cmd_refresh)

    puc = sub.add_parser("update-cost", help="refetch IonQ cost and update records in-place (no append)")
    puc.add_argument("job_id", nargs="?", help="job UUID (default: every job in log)")
    puc.set_defaults(fn=cmd_update_cost)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
