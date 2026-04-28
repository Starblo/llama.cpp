#!/usr/bin/env python3
"""
LLM Inference Benchmark Suite — Top-Level Launcher
===================================================
Thesis: KV Cache Profiling on Consumer GPU (RTX 5060, 8 GB GDDR7)

Dispatches to individual test scripts.  Each test is a standalone module
that can also be run directly (e.g. ``python test1_baseline.py``).

Available tests
---------------
  1  test1_baseline     — Baseline throughput & context-length scaling
  2  (reserved)         — KV offload single-model deep dive
  3  (reserved)         — KV offload cross-model comparison

Usage
-----
  python run_tests.py                # run all available tests
  python run_tests.py -t 1           # run test 1 only
  python run_tests.py --list         # show test descriptions
  python run_tests.py --dry-run      # print commands only
  python run_tests.py -r 3           # override repeat count
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# ensure parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Test Registry ──────────────────────────────────────────────────────────────
#
# Each entry maps a test ID to:
#   module  - importable module name (relative to this directory)
#   desc    - human-readable description

TESTS = {
    1: {
        "module": "test1_baseline",
        "desc":   "Baseline throughput & context-length scaling (4 models × ngl=99)",
    },
    # 2 and 3 will be added once Test 1 results are analyzed
    # 2: {
    #     "module": "test2_offload_deep",
    #     "desc":   "KV offload single-model deep dive (LLaMA-8B × context × ngl)",
    # },
    # 3: {
    #     "module": "test3_offload_cross",
    #     "desc":   "KV offload cross-model comparison (4 models × ngl sweep)",
    # },
}


def run_test(test_id: int, repeats: int, dry_run: bool, **kwargs) -> bool:
    """Import and execute a single test module. Returns True on success."""
    cfg = TESTS[test_id]
    print(f"\n{'━' * 60}")
    print(f"  Launching Test {test_id}: {cfg['desc']}")
    print(f"{'━' * 60}\n")

    try:
        mod = importlib.import_module(cfg["module"])
        mod.run(repeats, dry_run, **kwargs)
        return True
    except Exception as exc:
        print(f"\n  ✗ Test {test_id} failed: {exc}\n")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Inference Benchmark Suite — Top-Level Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-t", "--tests",
        nargs="*",
        type=int,
        default=None,
        metavar="N",
        help="Test IDs to run (default: all available). E.g. -t 1",
    )
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=5,
        metavar="N",
        help="Repeat count per configuration (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        metavar="S",
        help="Seconds per single bench run (default: 600)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List test descriptions and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable tests:")
        for tid, cfg in sorted(TESTS.items()):
            print(f"  {tid}  {cfg['module']:<25}  {cfg['desc']}")
        print()
        return

    requested = args.tests if args.tests else sorted(TESTS.keys())

    # validate
    for tid in requested:
        if tid not in TESTS:
            print(f"Error: unknown test ID {tid}. Use --list to see available tests.")
            sys.exit(1)

    print(f"\nTests to run: {requested}")
    print(f"Repeats:      {args.repeats}")
    print(f"Timeout:      {args.timeout}s per run")
    print(f"Dry-run:      {args.dry_run}")

    failed = []
    for tid in requested:
        ok = run_test(tid, args.repeats, args.dry_run, timeout=args.timeout)
        if not ok:
            failed.append(tid)

    print("\n" + "=" * 60)
    if failed:
        print(f"  Tests failed: {failed}")
    else:
        print("  All tests completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
