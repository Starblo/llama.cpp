#!/usr/bin/env python3
"""
Test 1 — Baseline Throughput & Context-Length Scaling
=====================================================

Purpose
-------
Establish the full-GPU (ngl=99) baseline for all 4 models across a wide
range of context lengths.  By pushing context from 512 up to 32768+, this
test discovers each model's VRAM overflow boundary — the context length at
which the KV cache + model weights exceed 8 GB and performance collapses.

Configuration
-------------
  Models  : LLaMA-3.1-8B, Gemma-2-9B, Qwen2.5-7B, Phi-3.1-Mini
  ngl     : 99  (full GPU offload)
  prompt  : 512, 1024, 2048, 4096, 8192, 16384, 32768
  gen     : 128
  repeats : 5

Expected findings
-----------------
  - Phi (2.2 GB weights) should survive the longest context.
  - Gemma (5.75 GB) should hit VRAM overflow earliest.
  - The "cliff edge" in prefill speed pinpoints the overflow for each model.

If a context length causes OOM, that run is skipped and logged.

Usage
-----
  python test1_baseline.py              # run everything
  python test1_baseline.py --dry-run    # print commands only
  python test1_baseline.py -r 3         # override repeat count
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ensure parent directory is on sys.path so `lib` can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.common import (
    MODELS,
    build_cmd,
    log,
    make_run_dir,
    merge_csv_blocks,
    preflight_check,
    run_cmd,
    save_csv,
    setup_logging,
)

# ── Test configuration ─────────────────────────────────────────────────────────

TEST_NAME = "test1_baseline"
DEFAULT_REPEATS = 5
DEFAULT_TIMEOUT = 600  # seconds per single llama-bench invocation

ALL_MODELS  = ["llama-3.1-8b", "gemma-2-9b", "phi-3.1-mini", "qwen2.5-7b"]
PROMPT_LENS = [
    512, 1024, 2048, 3072,                                    # early baseline (dense sampling)
    4096, 6144, 8192, 10240, 12288, 14336, 16384,             # cliff zone (step 2K)
    20480, 24576, 28672, 32768,                               # OOM zone (step 4K)
    40960, 49152, 65536, 81920, 98304, 131072,                # extended range (for Qwen)
]
GEN_TOKENS  = [128]
NGL         = 99


# ── Runner ─────────────────────────────────────────────────────────────────────

def run(repeats: int, dry_run: bool, timeout: int = DEFAULT_TIMEOUT) -> None:
    run_dir = make_run_dir(TEST_NAME)
    setup_logging(run_dir)

    log.info("=" * 60)
    log.info("Test 1 — Baseline Throughput & Context-Length Scaling")
    log.info("=" * 60)
    log.info("  models  : %s", ALL_MODELS)
    log.info("  ngl     : %d", NGL)
    log.info("  prompts : %s", PROMPT_LENS)
    log.info("  gen     : %s", GEN_TOKENS)
    log.info("  repeats : %d", repeats)
    log.info("  timeout : %ds per run", timeout)
    log.info("  dry-run : %s", dry_run)

    if not dry_run and not preflight_check(ALL_MODELS):
        log.error("Pre-flight failed — aborting.")
        sys.exit(1)

    # Run each model individually so that an OOM on one model
    # does not abort the entire test.
    csv_blocks: list[str] = []

    for model_name in ALL_MODELS:
        short = MODELS[model_name]["short"]

        # Try progressively longer context lengths; stop on failure.
        model_blocks: list[str] = []
        for plen in PROMPT_LENS:
            label = f"T1|{short}|p={plen}"
            cmd = build_cmd([model_name], NGL, [plen], GEN_TOKENS, repeats)
            try:
                block = run_cmd(cmd, label, dry_run=dry_run, timeout=timeout)
                model_blocks.append(block)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                log.warning(
                    "[%s] Failed (likely OOM/timeout at context %d). Skipping "
                    "longer contexts for this model. Error: %s",
                    label, plen, exc,
                )
                break

        csv_blocks.extend(model_blocks)

    if not dry_run and csv_blocks:
        merged = merge_csv_blocks(csv_blocks)
        save_csv(merged, run_dir / f"{TEST_NAME}.csv")

    log.info("")
    log.info("Results directory: %s", run_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 1 — Baseline throughput & context scaling",
    )
    parser.add_argument("-r", "--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Seconds per single bench run (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.repeats, args.dry_run, args.timeout)


if __name__ == "__main__":
    main()
