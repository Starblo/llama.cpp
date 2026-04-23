#!/usr/bin/env python3
"""
LLM Benchmark Results Analyzer
Thesis: KV Cache Profiling on Consumer GPU (RTX 5060, 8 GB GDDR7)

Usage
-----
  python analyze.py                           # latest run
  python analyze.py results/20250423_143022   # specific run directory
  python analyze.py --all                     # every run in results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

RESULTS_DIR = Path(__file__).parent / "results"

# ── Model display names ────────────────────────────────────────────────────────

_MODEL_SHORT: Dict[str, str] = {
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf": "LLaMA-3.1-8B",
    "gemma-2-9b-it-Q4_K_M.gguf":               "Gemma-2-9B",
    "Phi-3.1-mini-128k-instruct-Q4_K_M.gguf":  "Phi-3.1-Mini",
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf":   "Qwen2.5-7B",
}

MODEL_ORDER = ["LLaMA-3.1-8B", "Gemma-2-9B", "Qwen2.5-7B", "Phi-3.1-Mini"]


def short_model(filename: str) -> str:
    return _MODEL_SHORT.get(Path(filename).name, Path(filename).stem)


# ── CSV loading & row classification ──────────────────────────────────────────

def load_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default


def _i(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (ValueError, TypeError):
        return default


def is_pp(row: Dict[str, str]) -> bool:
    """Prefill row: n_prompt > 0, n_gen == 0."""
    return _i(row, "n_prompt") > 0 and _i(row, "n_gen") == 0


def is_tg(row: Dict[str, str]) -> bool:
    """Token-generation row: n_prompt == 0, n_gen > 0."""
    return _i(row, "n_prompt") == 0 and _i(row, "n_gen") > 0


# ── Pivot / transposed table printing ─────────────────────────────────────────

def _col_widths(headers: List[str], rows: List[List[str]]) -> List[int]:
    all_rows = [headers] + rows
    return [max(len(str(r[i])) for r in all_rows) + 2 for i in range(len(headers))]


def print_pivot(row_header: str, col_headers: List[str], rows: List[Tuple]) -> None:
    """Print a pivot table.

    rows: list of (row_label, col_val_0, col_val_1, ...) tuples
    """
    if not rows:
        print("    (no data)")
        return
    headers   = [row_header] + col_headers
    str_rows  = [[str(c) for c in r] for r in rows]
    widths    = _col_widths(headers, str_rows)
    fmt       = "  ".join(f"{{:<{w}}}" for w in widths)
    sep       = "  ".join("─" * w for w in widths)
    print("  " + fmt.format(*headers))
    print("  " + sep)
    for r in str_rows:
        print("  " + fmt.format(*r))


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title: str) -> None:
    print(f"\n  ── {title}")


# ── Test 1: Basic Throughput Baseline ─────────────────────────────────────────

def analyze_test1(rows: List[Dict], out_dir: Path) -> None:
    section("Test 1 — Basic Throughput Baseline")

    pp = [r for r in rows if is_pp(r)]
    tg = [r for r in rows if is_tg(r)]

    # ── Prefill pivot: rows = prompt lengths, cols = models ──────────────────
    subsection("Prefill (pp) — t/s  [rows = prompt tokens, cols = model]")

    prompt_lens = sorted(set(_i(r, "n_prompt") for r in pp))
    models = [m for m in MODEL_ORDER
              if any(short_model(r.get("model_filename","")) == m for r in pp)]

    # build lookup: (model, n_prompt) → "avg ± std"
    pp_lookup: Dict[Tuple[str, int], str] = {}
    for r in pp:
        key = (short_model(r.get("model_filename","")), _i(r, "n_prompt"))
        pp_lookup[key] = f"{_f(r,'avg_ts'):>8.1f} ± {_f(r,'stddev_ts'):.1f}"

    pivot_rows = []
    for pl in prompt_lens:
        row = [f"{pl:>6}"]
        for m in models:
            row.append(pp_lookup.get((m, pl), "—"))
        pivot_rows.append(tuple(row))
    print_pivot("Prompt", models, pivot_rows)

    # ── Decode table: rows = models ───────────────────────────────────────────
    subsection("Decode (tg) — 128 tokens")

    tg_rows_fmt = []
    for r in sorted(tg, key=lambda r: MODEL_ORDER.index(
            short_model(r.get("model_filename",""))
            if short_model(r.get("model_filename","")) in MODEL_ORDER else 99)):
        model  = short_model(r.get("model_filename",""))
        avg_ts = _f(r, "avg_ts")
        std_ts = _f(r, "stddev_ts")
        ms_tok = 1000.0 / avg_ts if avg_ts > 0 else 0
        tg_rows_fmt.append((model, f"{avg_ts:>8.2f}", f"± {std_ts:.2f}", f"{ms_tok:.3f}"))
    print_pivot("Model", ["t/s", "StdDev", "ms/token"], tg_rows_fmt)

    if HAS_MATPLOTLIB:
        _plot_test1(pp, tg, models, prompt_lens, out_dir)


def _plot_test1(pp, tg, models, prompt_lens, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test 1 — Basic Throughput Baseline", fontweight="bold")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax = axes[0]
    for i, model in enumerate(models):
        pts = sorted(
            (_i(r, "n_prompt"), _f(r, "avg_ts"))
            for r in pp if short_model(r.get("model_filename","")) == model
        )
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, marker="o", label=model, color=colors[i % len(colors)])
    ax.set_title("Prefill speed vs prompt length")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("t/s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax = axes[1]
    tg_vals = {short_model(r.get("model_filename","")): _f(r, "avg_ts") for r in tg}
    ordered = [m for m in MODEL_ORDER if m in tg_vals]
    ax.bar(ordered, [tg_vals[m] for m in ordered],
           color=[colors[i % len(colors)] for i in range(len(ordered))])
    ax.set_title("Decode speed (128 gen tokens)")
    ax.set_ylabel("t/s")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "test1_throughput.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {p.name}")


# ── Test 2: Decode Scaling ─────────────────────────────────────────────────────

def analyze_test2(rows: List[Dict], out_dir: Path) -> None:
    section("Test 2 — Decode Scaling (LLaMA 3.1 8B)")
    subsection("rows = output token count")

    tg = [r for r in rows if is_tg(r)]
    if not tg:
        print("    (no tg rows found)")
        return

    tg_sorted = sorted(tg, key=lambda r: _i(r, "n_gen"))
    pivot_rows = []
    for r in tg_sorted:
        n_g    = _i(r, "n_gen")
        avg_ts = _f(r, "avg_ts")
        std_ts = _f(r, "stddev_ts")
        ms_tok = 1000.0 / avg_ts if avg_ts > 0 else 0
        pivot_rows.append((f"{n_g:>6}", f"{avg_ts:>8.2f}", f"± {std_ts:.2f}", f"{ms_tok:.3f}"))
    print_pivot("n_gen", ["t/s", "StdDev", "ms/token"], pivot_rows)

    if HAS_MATPLOTLIB:
        _plot_test2(tg_sorted, out_dir)


def _plot_test2(tg, out_dir):
    pts = [(_i(r, "n_gen"), _f(r, "avg_ts")) for r in tg]
    if not pts:
        return
    x, y = zip(*pts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Test 2 — Decode Scaling (LLaMA 3.1 8B)", fontweight="bold")

    axes[0].plot(x, y, marker="o", color="tab:blue")
    axes[0].set_title("Decode speed vs output length")
    axes[0].set_xlabel("n_gen (tokens)")
    axes[0].set_ylabel("t/s")
    axes[0].grid(True, alpha=0.3)

    ms = [1000.0 / v if v > 0 else 0 for v in y]
    axes[1].plot(x, ms, marker="o", color="tab:orange")
    axes[1].set_title("Latency per token vs output length")
    axes[1].set_xlabel("n_gen (tokens)")
    axes[1].set_ylabel("ms / token")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "test2_decode_scaling.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {p.name}")


# ── Test 3: KV Cache Offload ───────────────────────────────────────────────────

def analyze_test3(rows: List[Dict], out_dir: Path) -> None:
    section("Test 3 — KV Cache Offload (LLaMA 3.1 8B)")
    print("  ngl = GPU layers (0 = full CPU, 99 = full GPU)")
    subsection("rows = n_gpu_layers, cols = prefill / decode")

    pp = [r for r in rows if is_pp(r)]
    tg = [r for r in rows if is_tg(r)]

    pp_by_ngl: Dict[int, Tuple[float, float]] = {
        _i(r, "n_gpu_layers"): (_f(r, "avg_ts"), _f(r, "stddev_ts")) for r in pp
    }
    tg_by_ngl: Dict[int, Tuple[float, float]] = {
        _i(r, "n_gpu_layers"): (_f(r, "avg_ts"), _f(r, "stddev_ts")) for r in tg
    }

    baseline_tg = tg_by_ngl.get(99, (1.0, 0.0))[0] or 1.0
    ngl_vals = sorted(set(list(pp_by_ngl.keys()) + list(tg_by_ngl.keys())), reverse=True)

    pivot_rows = []
    for ngl in ngl_vals:
        pp_ts, pp_std = pp_by_ngl.get(ngl, (0.0, 0.0))
        tg_ts, tg_std = tg_by_ngl.get(ngl, (0.0, 0.0))
        pct = tg_ts / baseline_tg * 100 if baseline_tg > 0 else 0
        note = "← GPU-only baseline" if ngl >= 99 else ""
        pivot_rows.append((
            f"{ngl:>3}",
            f"{pp_ts:>8.1f} ± {pp_std:.1f}" if pp_ts else "—",
            f"{tg_ts:>8.2f} ± {tg_std:.2f}" if tg_ts else "—",
            f"{pct:>6.1f}%",
            note,
        ))
    print_pivot("ngl", ["Prefill (t/s)", "Decode (t/s)", "Decode vs baseline", "Note"],
                pivot_rows)

    if HAS_MATPLOTLIB:
        _plot_test3(pp_by_ngl, tg_by_ngl, ngl_vals, baseline_tg, out_dir)


def _plot_test3(pp_by_ngl, tg_by_ngl, ngl_vals, baseline_tg, out_dir):
    ngl_sorted = sorted(ngl_vals)
    pp_pts = [(n, pp_by_ngl[n][0]) for n in ngl_sorted if n in pp_by_ngl]
    tg_pts = [(n, tg_by_ngl[n][0]) for n in ngl_sorted if n in tg_by_ngl]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Test 3 — KV Cache Offload (LLaMA 3.1 8B)", fontweight="bold")

    if pp_pts:
        x, y = zip(*pp_pts)
        axes[0].plot(x, y, marker="s", color="tab:green", label="Prefill")
    if tg_pts:
        x2, y2 = zip(*tg_pts)
        axes[0].plot(x2, y2, marker="o", color="tab:orange", label="Decode")
    axes[0].set_title("Speed vs GPU layers")
    axes[0].set_xlabel("n_gpu_layers  (0 = full CPU)")
    axes[0].set_ylabel("t/s")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if tg_pts:
        x2, y2 = zip(*tg_pts)
        pct = [v / baseline_tg * 100 for v in y2]
        axes[1].plot(x2, pct, marker="o", color="tab:red")
        axes[1].axhline(100, linestyle="--", color="gray", linewidth=0.8)
        axes[1].set_title("Decode speed — % of GPU-only baseline")
        axes[1].set_xlabel("n_gpu_layers")
        axes[1].set_ylabel("% of baseline")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / "test3_kv_offload.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {p.name}")


# ── Test 4: Cross-Model Comparison ────────────────────────────────────────────

def analyze_test4(rows: List[Dict], out_dir: Path) -> None:
    section("Test 4 — Cross-Model Comparison")

    pp = [r for r in rows if is_pp(r)]
    tg = [r for r in rows if is_tg(r)]

    models = [m for m in MODEL_ORDER
              if any(short_model(r.get("model_filename","")) == m for r in pp)]

    # ── Prefill pivot: rows = prompt lengths, cols = models ──────────────────
    subsection("Prefill (pp) — t/s  [rows = prompt tokens, cols = model]")

    prompt_lens = sorted(set(_i(r, "n_prompt") for r in pp))
    pp_lookup: Dict[Tuple[str, int], str] = {}
    for r in pp:
        key = (short_model(r.get("model_filename","")), _i(r, "n_prompt"))
        pp_lookup[key] = f"{_f(r,'avg_ts'):>8.1f} ± {_f(r,'stddev_ts'):.1f}"

    pp_pivot = []
    for pl in prompt_lens:
        row = [f"{pl:>6}"]
        for m in models:
            row.append(pp_lookup.get((m, pl), "—"))
        pp_pivot.append(tuple(row))
    print_pivot("Prompt", models, pp_pivot)

    # ── Decode pivot: rows = models ───────────────────────────────────────────
    subsection("Decode (tg) — 128 tokens  [rows = model]")

    tg_lookup: Dict[str, Tuple[float, float]] = {}
    for r in tg:
        m = short_model(r.get("model_filename",""))
        tg_lookup[m] = (_f(r, "avg_ts"), _f(r, "stddev_ts"))

    tg_pivot = []
    for m in models:
        avg_ts, std_ts = tg_lookup.get(m, (0.0, 0.0))
        ms_tok = 1000.0 / avg_ts if avg_ts > 0 else 0
        tg_pivot.append((m, f"{avg_ts:>8.2f}", f"± {std_ts:.2f}", f"{ms_tok:.3f}"))
    print_pivot("Model", ["t/s", "StdDev", "ms/token"], tg_pivot)

    if HAS_MATPLOTLIB:
        _plot_test4(pp, tg, models, prompt_lens, out_dir)


def _plot_test4(pp, tg, models, prompt_lens, out_dir):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test 4 — Cross-Model Comparison", fontweight="bold")

    # Grouped bar: prefill per prompt length per model
    ax = axes[0]
    n_groups = len(prompt_lens)
    n_models = len(models)
    bar_w = 0.8 / n_models
    for i, model in enumerate(models):
        vals = []
        for pl in prompt_lens:
            match = [_f(r, "avg_ts") for r in pp
                     if short_model(r.get("model_filename","")) == model
                     and _i(r, "n_prompt") == pl]
            vals.append(match[0] if match else 0)
        offsets = [j + i * bar_w - (n_models - 1) * bar_w / 2
                   for j in range(n_groups)]
        ax.bar(offsets, vals, width=bar_w, label=model,
               color=colors[i % len(colors)])
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"{pl:,}" for pl in prompt_lens])
    ax.set_title("Prefill speed by model & prompt length")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("t/s")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Decode bar
    ax = axes[1]
    tg_vals = {short_model(r.get("model_filename","")): _f(r, "avg_ts") for r in tg}
    ordered = [m for m in MODEL_ORDER if m in tg_vals]
    ax.bar(ordered, [tg_vals[m] for m in ordered],
           color=[colors[i % len(colors)] for i in range(len(ordered))])
    ax.set_title("Decode speed (128 gen tokens)")
    ax.set_ylabel("t/s")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = out_dir / "test4_cross_model.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot → {p.name}")


# ── Run Directory Dispatch ─────────────────────────────────────────────────────

_ANALYZERS = {
    "test1": analyze_test1,
    "test2": analyze_test2,
    "test3": analyze_test3,
    "test4": analyze_test4,
}


def analyze_run(run_dir: Path) -> None:
    print(f"\n{'─'*70}")
    print(f"  Run: {run_dir.name}")
    print(f"{'─'*70}")

    csv_files = sorted(run_dir.glob("test*.csv"))
    if not csv_files:
        print("  No CSV files found.")
        return

    for csv_path in csv_files:
        rows = load_csv(csv_path)
        if not rows:
            print(f"  {csv_path.name}: empty — skipping")
            continue
        matched = False
        for key, fn in _ANALYZERS.items():
            if key in csv_path.stem:
                fn(rows, run_dir)
                matched = True
                break
        if not matched:
            print(f"\n  {csv_path.name}: unknown — columns: {list(rows[0].keys())}")
    print()


# ── Entry Point ────────────────────────────────────────────────────────────────

def find_latest_run() -> Optional[Path]:
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze llama-bench results")
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="Run directory (default: latest)")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all run directories")
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("Note: matplotlib not installed — plots skipped.")
        print("      pip install matplotlib\n")

    if args.all:
        if not RESULTS_DIR.exists():
            print("results/ not found. Run run_tests.py first.")
            sys.exit(1)
        for run_dir in sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir()):
            analyze_run(run_dir)
    elif args.run_dir:
        if not args.run_dir.is_dir():
            print(f"Not found: {args.run_dir}")
            sys.exit(1)
        analyze_run(args.run_dir)
    else:
        run_dir = find_latest_run()
        if run_dir is None:
            print("No runs found. Run run_tests.py first.")
            sys.exit(1)
        analyze_run(run_dir)


if __name__ == "__main__":
    main()
