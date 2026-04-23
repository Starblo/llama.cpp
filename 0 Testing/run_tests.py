#!/usr/bin/env python3
"""
LLM Inference Benchmark Suite
Thesis: KV Cache Profiling on Consumer GPU (RTX 5060, 8 GB GDDR7)

Tests
-----
  1  baseline       — all 4 models, prompt 512–8192 tokens, 128 output
  2  decode_scaling — LLaMA 3.1 8B, prompt 512, output 64–1024 tokens
  3  kv_offload     — LLaMA 3.1 8B, prompt 2048, 128 output, ngl 0–99
  4  cross_model    — all 4 models, prompt 512/2048/8192, 128 output

Usage
-----
  python run_tests.py                # run all 4 tests
  python run_tests.py -t 1 3        # run tests 1 and 3 only
  python run_tests.py --dry-run     # print commands, do not execute
  python run_tests.py --list        # show test descriptions and exit
  python run_tests.py -r 3          # override repeat count (default 5)
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(r"D:\NetShare\Coding\llama.cpp")
BENCH_EXE   = REPO_ROOT / "build" / "bin" / "Release" / "llama-bench.exe"
MODELS_DIR  = REPO_ROOT / "models" / "4model"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS: Dict[str, Path] = {
    "llama-3.1-8b": MODELS_DIR / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "gemma-2-9b":   MODELS_DIR / "gemma-2-9b-it-Q4_K_M.gguf",
    "phi-3.1-mini": MODELS_DIR / "Phi-3.1-mini-128k-instruct-Q4_K_M.gguf",
    "qwen2.5-7b":   MODELS_DIR / "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
}

# ── Test Definitions ───────────────────────────────────────────────────────────
#
# Each test is a dict with:
#   name    str         — used in the output filename
#   desc    str         — human-readable description
#   models  List[str]   — keys from MODELS dict
#   ngl     List[int]   — GPU layer counts; multiple values → separate runs merged
#   prompt  List[int]   — prompt token counts (comma-joined in the -p flag)
#   gen     List[int]   — generation token counts (comma-joined in the -n flag)

DEFAULT_REPEATS = 5

TESTS: Dict[int, Dict[str, Any]] = {
    1: {
        "name":   "baseline",
        "desc":   "Basic throughput baseline — all 4 models, prompt 512–8192 tokens",
        "models": ["llama-3.1-8b", "gemma-2-9b", "phi-3.1-mini", "qwen2.5-7b"],
        "ngl":    [99],
        "prompt": [512, 2048, 4096, 8192],
        "gen":    [128],
    },
    2: {
        "name":   "decode_scaling",
        "desc":   "Decode scaling — LLaMA 3.1 8B, output 64–1024 tokens",
        "models": ["llama-3.1-8b"],
        "ngl":    [99],
        "prompt": [512],
        "gen":    [64, 128, 256, 512, 1024],
    },
    3: {
        "name":   "kv_offload",
        "desc":   "KV cache offload — LLaMA 3.1 8B, GPU layers 0/8/16/24/99",
        "models": ["llama-3.1-8b"],
        "ngl":    [99, 24, 16, 8, 0],   # each ngl → separate subprocess call
        "prompt": [2048],
        "gen":    [128],
    },
    4: {
        "name":   "cross_model",
        "desc":   "Cross-model comparison — all 4 models, prompt 512/2048/8192",
        "models": ["llama-3.1-8b", "gemma-2-9b", "phi-3.1-mini", "qwen2.5-7b"],
        "ngl":    [99],
        "prompt": [512, 2048, 8192],
        "gen":    [128],
    },
}

# ── Logging ────────────────────────────────────────────────────────────────────

log = logging.getLogger("bench")


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    log.info("Log: %s", log_path)


# ── Pre-flight ─────────────────────────────────────────────────────────────────

def preflight_check(requested_tests: List[int]) -> bool:
    ok = True
    if not BENCH_EXE.is_file():
        log.error("llama-bench.exe not found: %s", BENCH_EXE)
        ok = False

    needed_models: set[str] = set()
    for tid in requested_tests:
        needed_models.update(TESTS[tid]["models"])

    for name in needed_models:
        path = MODELS[name]
        if not path.is_file():
            log.error("Model not found [%s]: %s", name, path)
            ok = False
        else:
            size_gb = path.stat().st_size / 1024**3
            log.info("  %-16s  %.2f GB  %s", name, size_gb, path.name)

    return ok


# ── Command Building & Execution ───────────────────────────────────────────────

def build_cmd(
    models: List[str],
    ngl: int,
    prompt: List[int],
    gen: List[int],
    repeats: int,
) -> List[str]:
    cmd = [str(BENCH_EXE)]
    for m in models:
        cmd += ["-m", str(MODELS[m])]
    cmd += ["-ngl",  str(ngl)]
    cmd += ["-p",    ",".join(map(str, prompt))]
    cmd += ["-n",    ",".join(map(str, gen))]
    cmd += ["-r",    str(repeats)]
    cmd += ["-o",    "csv"]
    return cmd


def run_cmd(cmd: List[str], dry_run: bool, label: str, timeout: int = 7200) -> str:
    """Run llama-bench; return raw CSV text from stdout.

    Timeout is generous (2 h) to accommodate slow CPU-offload runs.
    """
    log.info("[%s] %s", label, " ".join(cmd))
    if dry_run:
        return ""

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.error("[%s] Timed out after %ds", label, timeout)
        raise

    elapsed = time.monotonic() - t0
    log.info("[%s] Done in %.1fs  (rc=%d)", label, elapsed, proc.returncode)

    if proc.returncode != 0:
        log.error("[%s] stderr:\n%s", label, proc.stderr.strip())
        raise RuntimeError(f"llama-bench exited {proc.returncode}")

    # Log progress lines from stderr at DEBUG level so they appear in the log
    if proc.stderr.strip():
        for line in proc.stderr.strip().splitlines():
            log.debug("[%s] >> %s", label, line)

    return proc.stdout


# ── CSV Helpers ────────────────────────────────────────────────────────────────

def merge_csv_blocks(blocks: List[str]) -> str:
    """Concatenate multiple CSV outputs that each have their own header row."""
    merged: List[str] = []
    header: str | None = None
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
            merged.append(header)
        # Skip the header in subsequent blocks
        merged.extend(lines[1:])
    return "\n".join(merged) + "\n"


def row_count(csv_text: str) -> int:
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    return len(rows)


def save_csv(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info("Saved %d data rows → %s", row_count(text), path)


# ── Test Runner ────────────────────────────────────────────────────────────────

def run_test(test_id: int, run_dir: Path, repeats: int, dry_run: bool) -> None:
    cfg   = TESTS[test_id]
    label_base = f"T{test_id}:{cfg['name']}"

    log.info("")
    log.info("━" * 60)
    log.info("TEST %d  %s", test_id, cfg["desc"])
    log.info("  models : %s", cfg["models"])
    log.info("  ngl    : %s", cfg["ngl"])
    log.info("  prompt : %s", cfg["prompt"])
    log.info("  gen    : %s", cfg["gen"])
    log.info("  repeats: %d", repeats)
    log.info("━" * 60)

    csv_blocks: List[str] = []

    for ngl in cfg["ngl"]:
        label = f"{label_base}|ngl={ngl}"
        cmd   = build_cmd(cfg["models"], ngl, cfg["prompt"], cfg["gen"], repeats)
        try:
            block = run_cmd(cmd, dry_run, label)
            csv_blocks.append(block)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            log.error("[%s] Skipping remaining ngl values for this test. Error: %s", label, exc)
            break

    if not dry_run:
        merged   = merge_csv_blocks(csv_blocks)
        out_path = run_dir / f"test{test_id}_{cfg['name']}.csv"
        save_csv(merged, out_path)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM inference benchmark runner for KV cache thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-t", "--tests",
        nargs="*",
        type=int,
        choices=list(TESTS.keys()),
        default=list(TESTS.keys()),
        metavar="N",
        help="Test IDs to run (default: all).  E.g. -t 1 3",
    )
    parser.add_argument(
        "-r", "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        metavar="N",
        help=f"Repeat count per configuration (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List test descriptions and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable tests:")
        for tid, cfg in TESTS.items():
            print(f"  {tid}  {cfg['name']:<20}  {cfg['desc']}")
        print()
        return

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / ts
    setup_logging(run_dir)

    log.info("llama.cpp bench suite  —  %s", ts)
    log.info("Tests requested : %s", sorted(args.tests))
    log.info("Repeats         : %d", args.repeats)
    log.info("Dry-run         : %s", args.dry_run)

    if not args.dry_run:
        log.info("")
        log.info("Pre-flight checks:")
        if not preflight_check(args.tests):
            log.error("Pre-flight failed — aborting.")
            sys.exit(1)

    failed: List[int] = []
    for tid in sorted(args.tests):
        try:
            run_test(tid, run_dir, args.repeats, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            log.error("Test %d failed: %s", tid, exc)
            failed.append(tid)

    log.info("")
    if failed:
        log.warning("Tests failed: %s", failed)
    else:
        log.info("All tests completed successfully.")
    log.info("Results directory: %s", run_dir)


if __name__ == "__main__":
    main()
