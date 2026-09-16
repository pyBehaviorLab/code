"""
Headless video zone-statistics, one code path for old (``B/M/V``) and new
(``#``-header) ``*_video_data.txt`` files.

Detects the format via the existing readers, canonicalizes the session, runs the
shared :mod:`zone_stats_full` analysis (state grouping, time-bins,
REGIONS_HIERARCHY, IA facing) and writes ``<stem>_zone_stats.xlsx``.

Usage:
    python -m tools.offline_analysis.analyze_cli <txt> [<txt> ...] \\
        [--bin-min N] [--out DIR] [--head-idx I] [--body-idx J] \\
        [--min-dwell-ms MS] [--angle-deg DEG] [--ia-confirm-ms MS]
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List

from .video_data_parser import parse_video_data
from .analysis_input import (
    suggest_head_body_indices, analyze_session_to_excel,
)


def _build_params(args) -> dict:
    return {
        "time_bin_enabled": bool(args.bin_min and args.bin_min > 0),
        "time_bin_min": float(args.bin_min or 0.0),
        "min_dwell_time": args.min_dwell_ms / 1000.0,
        "angle_threshold": args.angle_deg,
        "ia_confirm_ms": args.ia_confirm_ms,
        "head_idx": args.head_idx,
        "body_idx": args.body_idx,
        "source": "offline_analysis.analyze_cli",
    }


def analyze_one(txt_path: str, args) -> int:
    session = parse_video_data(txt_path)
    h, b = args.head_idx, args.body_idx
    if h is None or b is None:
        sh, sb = suggest_head_body_indices(session)
        h = sh if h is None else h
        b = sb if b is None else b

    out_dir = args.out or os.path.dirname(os.path.abspath(txt_path))
    params = _build_params(argparse.Namespace(**{**vars(args), "head_idx": h, "body_idx": b}))

    path = analyze_session_to_excel(
        session, out_dir, head_idx=h, body_idx=b, params=params,
        min_dwell_time=args.min_dwell_ms / 1000.0,
        angle_threshold=args.angle_deg, ia_confirm_ms=args.ia_confirm_ms,
    )
    if path:
        print(f"[ok]   {os.path.basename(txt_path)} -> {path}")
        return 0
    print(f"[warn] {os.path.basename(txt_path)}: no analyzable data "
          f"(no states / no pose / empty zones)")
    return 1


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(
        prog="analyze_cli",
        description="Video zone statistics for old + new _video_data.txt files.")
    p.add_argument("inputs", nargs="+", help="One or more *_video_data.txt files")
    p.add_argument("--bin-min", type=float, default=0.0,
                   help="Time-bin length in minutes (0 = no binning)")
    p.add_argument("--out", default=None,
                   help="Output directory (default: alongside each input)")
    p.add_argument("--head-idx", type=int, default=None,
                   help="Head keypoint index (default: auto-suggest -> 0)")
    p.add_argument("--body-idx", type=int, default=None,
                   help="Body keypoint index (default: auto-suggest -> 1)")
    p.add_argument("--min-dwell-ms", type=float, default=200.0,
                   help="Min dwell time to count an entry (ms)")
    p.add_argument("--angle-deg", type=float, default=45.0,
                   help="IA facing angle threshold (deg)")
    p.add_argument("--ia-confirm-ms", type=float, default=100.0,
                   help="IA in-zone / facing confirm time (ms)")
    args = p.parse_args(argv)

    rc = 0
    for txt in args.inputs:
        if not os.path.isfile(txt):
            print(f"[err]  not found: {txt}")
            rc = 2
            continue
        try:
            rc = analyze_one(txt, args) or rc
        except Exception as e:
            import traceback
            print(f"[err]  {os.path.basename(txt)}: {e}")
            traceback.print_exc()
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
