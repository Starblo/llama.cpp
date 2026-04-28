"""
Shared infrastructure for the LLM Inference Benchmark Suite.
Thesis: KV Cache Profiling on Consumer GPU (RTX 5060, 8 GB GDDR7)

Provides:
    - Path constants (repo root, bench executable, model files)
    - Model registry with metadata
    - Logging setup
    - llama-bench command builder and runner
    - CSV merge / save helpers
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(r"D:\NetShare\Coding\llama.cpp")
BENCH_EXE   = REPO_ROOT / "build" / "bin" / "Release" / "llama-bench.exe"
MODELS_DIR  = REPO_ROOT / "models" / "4model"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# ── Model Registry ─────────────────────────────────────────────────────────────
#
# Each entry:
#   file      - GGUF filename
#   short     - display name for plots / tables
#   layers    - total transformer layers (for ngl sweep upper bound)
#   weight_gb - Q4_K_M file size in GB (1000-based, matches HuggingFace)
#   layers    - num_hidden_layers (from GGUF metadata block_count)
#   ngl_max   - layers + 1 (embedding layer), the effective max for -ngl

MODELS: Dict[str, Dict[str, Any]] = {
    "llama-3.1-8b": {
        "file":      "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "short":     "LLaMA-3.1-8B",
        "layers":    32,
        "ngl_max":   33,
        "weight_gb": 4.92,
    },
    "gemma-2-9b": {
        "file":      "gemma-2-9b-it-Q4_K_M.gguf",
        "short":     "Gemma-2-9B",
        "layers":    42,
        "ngl_max":   43,
        "weight_gb": 5.76,
    },
    "phi-3.1-mini": {
        "file":      "Phi-3.1-mini-128k-instruct-Q4_K_M.gguf",
        "short":     "Phi-3.1-Mini",
        "layers":    32,
        "ngl_max":   33,
        "weight_gb": 2.39,
    },
    "qwen2.5-7b": {
        "file":      "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "short":     "Qwen2.5-7B",
        "layers":    28,
        "ngl_max":   29,
        "weight_gb": 4.68,
    },
}

MODEL_ORDER = ["LLaMA-3.1-8B", "Gemma-2-9B", "Qwen2.5-7B", "Phi-3.1-Mini"]

def model_path(name: str) -> Path:
    """Return the full path to a model's GGUF file."""
    return MODELS_DIR / MODELS[name]["file"]


def short_name(name: str) -> str:
    """Return the short display name for a model."""
    return MODELS[name]["short"]


# ── Logging ────────────────────────────────────────────────────────────────────

log = logging.getLogger("bench")


def setup_logging(run_dir: Path) -> None:
    """Configure console + file logging into *run_dir*/run.log."""
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

def preflight_check(model_names: List[str]) -> bool:
    """Verify that the bench executable and required model files exist."""
    ok = True
    if not BENCH_EXE.is_file():
        log.error("llama-bench.exe not found: %s", BENCH_EXE)
        ok = False

    for name in model_names:
        p = model_path(name)
        if not p.is_file():
            log.error("Model not found [%s]: %s", name, p)
            ok = False
        else:
            size_gb = p.stat().st_size / 1024**3
            log.info("  %-16s  %.2f GB  %s", name, size_gb, p.name)

    return ok


# ── Command Building & Execution ──────────────────────────────────────────────

def build_cmd(
    model_names: List[str],
    ngl: int,
    prompt: List[int],
    gen: List[int],
    repeats: int,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    """Build a llama-bench command line."""
    cmd = [str(BENCH_EXE)]
    for m in model_names:
        cmd += ["-m", str(model_path(m))]
    cmd += ["-ngl",  str(ngl)]
    cmd += ["-p",    ",".join(map(str, prompt))]
    cmd += ["-n",    ",".join(map(str, gen))]
    cmd += ["-r",    str(repeats)]
    cmd += ["-o",    "csv"]
    if extra_flags:
        cmd += extra_flags
    return cmd


def run_cmd(cmd: List[str], label: str, dry_run: bool = False,
            timeout: int = 7200) -> str:
    """Run a llama-bench command; return raw CSV stdout.

    *timeout* defaults to 2 h to accommodate slow CPU-offload runs.
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
        merged.extend(lines[1:])
    return "\n".join(merged) + "\n"


def row_count(csv_text: str) -> int:
    return len(list(csv.DictReader(io.StringIO(csv_text))))


def save_csv(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info("Saved %d data rows → %s", row_count(text), path)


def make_run_dir(test_name: str) -> Path:
    """Create a timestamped results directory for a specific test."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / test_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
