#!/usr/bin/env python3
"""
Analyze Test 1 — Baseline Throughput & Context-Length Scaling
=============================================================

Reads the CSV produced by test1_baseline.py and generates:
  1. Prefill speed vs context length (line chart, one line per model)
  2. Decode speed bar chart (grouped by model)
  3. VRAM overflow boundary summary table
  4. Console printout of all key numbers

Usage
-----
  python test1_analyze.py                               # latest run
  python test1_analyze.py results/test1_baseline/xxxxx  # specific run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.common import MODELS, MODEL_ORDER, RESULTS_DIR

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0))
    except (ValueError, TypeError):
        return 0.0

def _i(row: dict, key: str) -> int:
    try:
        return int(row.get(key, 0))
    except (ValueError, TypeError):
        return 0

def is_pp(row: dict) -> bool:
    return _i(row, "n_prompt") > 0 and _i(row, "n_gen") == 0

def is_tg(row: dict) -> bool:
    return _i(row, "n_prompt") == 0 and _i(row, "n_gen") > 0

def model_short(row: dict) -> str:
    fname = Path(row.get("model_filename", "")).name
    for m in MODELS.values():
        if m["file"] == fname:
            return m["short"]
    return fname

def load_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def assign_context_to_tg(rows: List[dict]) -> List[dict]:
    """Pair each decode row with the context length of its preceding prefill row."""
    tg_with_ctx = []
    last_pp_ctx: Dict[str, int] = {}
    for r in rows:
        ms = model_short(r)
        if is_pp(r):
            last_pp_ctx[ms] = _i(r, "n_prompt")
        elif is_tg(r) and ms in last_pp_ctx:
            r = dict(r)
            r["_context"] = last_pp_ctx[ms]
            tg_with_ctx.append(r)
    return tg_with_ctx


# Planned context schedule (must mirror test1_baseline.PROMPT_LENS).
# Used to infer the next (failed) context length after the last successful run.
PLANNED_PROMPT_LENS = [
    512, 1024, 2048, 3072,
    4096, 6144, 8192, 10240, 12288, 14336, 16384,
    20480, 24576, 28672, 32768,
    40960, 49152, 65536, 81920, 98304, 131072,
]

CLIFF_DROP_THRESHOLD = 0.50  # >50% drop between consecutive points = cliff


def detect_cliff(points: List[Tuple[int, float, float]]) -> Optional[Tuple[int, float, float]]:
    """Return (cliff_ctx, baseline_speed, cliff_speed) for the first significant drop."""
    if not points:
        return None
    baseline_ctx, baseline_speed, _ = points[0]
    for i in range(1, len(points)):
        prev_ctx, prev_speed, _ = points[i - 1]
        curr_ctx, curr_speed, _ = points[i]
        if prev_speed > 0:
            drop = (prev_speed - curr_speed) / prev_speed
            if drop > CLIFF_DROP_THRESHOLD:
                return curr_ctx, baseline_speed, curr_speed
        if baseline_speed > 0:
            cumulative_drop = (baseline_speed - curr_speed) / baseline_speed
            if cumulative_drop > 0.70:
                return curr_ctx, baseline_speed, curr_speed
    return None


def next_planned_ctx(last_ctx: int) -> Optional[int]:
    """Given the last successful context, return the next planned context (the OOM candidate)."""
    for c in PLANNED_PROMPT_LENS:
        if c > last_ctx:
            return c
    return None


def model_pp_points(pp: List[dict], m: str) -> List[Tuple[int, float, float]]:
    return sorted(
        [(_i(r, "n_prompt"), _f(r, "avg_ts"), _f(r, "stddev_ts")) for r in pp if model_short(r) == m],
        key=lambda x: x[0],
    )


def model_tg_points(tg_ctx: List[dict], m: str) -> List[Tuple[int, float, float]]:
    return sorted(
        [(r["_context"], _f(r, "avg_ts"), _f(r, "stddev_ts")) for r in tg_ctx if model_short(r) == m],
        key=lambda x: x[0],
    )


# ── Console Report ─────────────────────────────────────────────────────────────

def print_report(pp: List[dict], tg: List[dict]) -> None:
    models = [m for m in MODEL_ORDER if any(model_short(r) == m for r in pp)]
    prompt_lens = sorted(set(_i(r, "n_prompt") for r in pp))

    # ── Prefill table ──
    print("\n" + "=" * 70)
    print("  Prefill Speed (t/s) — ngl=99, gen=128")
    print("  Platform: RTX 5060 (8 GB GDDR7), Q4_K_M, FP16 KV cache")
    print("=" * 70)

    header = f"  {'Context':>8}"
    for m in models:
        header += f"  {m:>16}"
    print(header)
    print("  " + "─" * (10 + 18 * len(models)))

    for pl in prompt_lens:
        row_str = f"  {pl:>8}"
        for m in models:
            match = [r for r in pp if model_short(r) == m and _i(r, "n_prompt") == pl]
            if match:
                val = _f(match[0], "avg_ts")
                row_str += f"  {val:>16.1f}"
            else:
                row_str += f"  {'(OOM)':>16}"
        print(row_str)

    # ── Decode table ──
    print("\n  Decode Speed (t/s) — 128 gen tokens")
    print("  " + "─" * 50)
    for m in models:
        match = [r for r in tg if model_short(r) == m]
        if match:
            val = _f(match[0], "avg_ts")
            ms = 1000 / val if val > 0 else 0
            print(f"  {m:<20}  {val:>8.2f} t/s  ({ms:.1f} ms/token)")
        else:
            print(f"  {m:<20}  (no data)")

    # ── VRAM overflow summary ──
    print("\n  VRAM Overflow Detection")
    print("  " + "─" * 50)
    for m in models:
        pts = sorted(
            [(_i(r, "n_prompt"), _f(r, "avg_ts"), _f(r, "stddev_ts")) for r in pp if model_short(r) == m],
            key=lambda x: x[0],
        )
        if len(pts) < 2:
            print(f"  {m:<20}  insufficient data")
            continue

        cliff = None
        for i in range(1, len(pts)):
            prev_speed = pts[i - 1][1]
            curr_speed = pts[i][1]
            if prev_speed > 0:
                drop = (prev_speed - curr_speed) / prev_speed
                if drop > 0.50:
                    cliff = pts[i][0]
                    break

        max_ctx = pts[-1][0]
        if cliff:
            weight_gb = next(
                (v["weight_gb"] for v in MODELS.values() if v["short"] == m), 0
            )
            print(
                f"  {m:<20}  CLIFF at context={cliff:,} "
                f"(weight={weight_gb:.1f} GB, "
                f"estimated KV≈{cliff * 128 / 1024 / 1024:.1f} GB)"
            )
        else:
            print(f"  {m:<20}  no cliff detected (max tested: {max_ctx:,})")

    # ── Performance Cliff Analysis (consolidated) ──
    print("\n  Performance Cliff Analysis")
    print("  " + "─" * 70)
    print(f"  {'Model':<16} {'Baseline':>10} {'Cliff@ctx':>12} "
          f"{'Drop%':>7} {'Speed@cliff':>12} {'OOM@ctx':>10}")
    print("  " + "─" * 70)
    for m in models:
        pts = model_pp_points(pp, m)
        if not pts:
            continue
        baseline = pts[0][1]
        cliff = detect_cliff(pts)
        last_ctx = pts[-1][0]
        oom = next_planned_ctx(last_ctx)
        if cliff:
            cx, prev_s, cy = cliff
            drop_pct = (prev_s - cy) / prev_s * 100 if prev_s > 0 else 0
            cliff_str = f"{cx:>12,}"
            drop_str  = f"{drop_pct:>6.0f}%"
            speed_str = f"{cy:>12.1f}"
        else:
            cliff_str = f"{'—':>12}"
            drop_str  = f"{'—':>7}"
            speed_str = f"{'—':>12}"
        oom_str = f"{oom:>10,}" if oom is not None else f"{'—':>10}"
        print(f"  {m:<16} {baseline:>10.1f} {cliff_str} "
              f"{drop_str} {speed_str} {oom_str}")
    print()

    print()


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot(pp: List[dict], tg_ctx: List[dict], out_dir: Path) -> None:
    if not HAS_MPL:
        print("  matplotlib not installed — plots skipped.")
        return

    models = [m for m in MODEL_ORDER if any(model_short(r) == m for r in pp)]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Test 1 — Baseline Throughput & Context-Length Scaling\n"
        "RTX 5060 (8 GB GDDR7) · Q4_K_M · FP16 KV · ngl=99",
        fontweight="bold", fontsize=11,
    )

    # ── Left: Prefill vs context length ──
    ax = axes[0]
    for i, m in enumerate(models):
        pts = model_pp_points(pp, m)
        if not pts:
            continue
        x, y, yerr = zip(*pts)
        color = colors[i % len(colors)]
        ax.errorbar(x, y, yerr=yerr, marker="o", label=m, color=color,
                    linewidth=2, markersize=6, capsize=3, capthick=1)

        cliff = detect_cliff(pts)
        if cliff:
            cx, _, cy = cliff
            ax.plot(cx, cy, marker="o", markersize=14,
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=2, zorder=5)

        last_ctx, last_y, _ = pts[-1]
        nxt = next_planned_ctx(last_ctx)
        if nxt is not None:
            ax.scatter(nxt, last_y, marker="x", s=120,
                       color=color, linewidths=2.5, zorder=5)

    ax.set_title("Prefill Speed vs Context Length")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Prompt processing speed (t/s)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Right: Decode speed vs context length ──
    ax = axes[1]
    for i, m in enumerate(models):
        pts = model_tg_points(tg_ctx, m)
        if not pts:
            continue
        x, y, yerr = zip(*pts)
        color = colors[i % len(colors)]
        ax.errorbar(x, y, yerr=yerr, marker="o", label=m, color=color,
                    linewidth=2, markersize=6, capsize=3, capthick=1)

        cliff = detect_cliff(pts)
        if cliff:
            cx, _, cy = cliff
            ax.plot(cx, cy, marker="o", markersize=14,
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=2, zorder=5)

        last_ctx, last_y, _ = pts[-1]
        nxt = next_planned_ctx(last_ctx)
        if nxt is not None:
            ax.scatter(nxt, last_y, marker="x", s=120,
                       color=color, linewidths=2.5, zorder=5)

    ax.set_title("Decode Speed vs Context Length")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Token generation speed (t/s)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Shared legend explaining cliff/OOM markers
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="red", markeredgewidth=2, markersize=12,
               label="Cliff (>50% drop)"),
        Line2D([0], [0], marker="x", color="red", markersize=10,
               linewidth=0, markeredgewidth=2.5, label="OOM (next ctx)"),
    ]
    fig.legend(handles=proxies, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    p = out_dir / "test1_baseline.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {p}")


# ── Entry ──────────────────────────────────────────────────────────────────────

def find_latest_run() -> Optional[Path]:
    base = RESULTS_DIR / "test1_baseline"
    if not base.exists():
        return None
    runs = sorted(d for d in base.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Test 1 results")
    parser.add_argument("run_dir", nargs="?", type=Path)
    args = parser.parse_args()

    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_dir = find_latest_run()

    if run_dir is None or not run_dir.is_dir():
        print("No Test 1 results found. Run test1_baseline.py first.")
        sys.exit(1)

    csv_path = run_dir / "test1_baseline.csv"
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    rows = load_csv(csv_path)
    pp = [r for r in rows if is_pp(r)]
    tg = [r for r in rows if is_tg(r)]
    tg_ctx = assign_context_to_tg(rows)

    print(f"\n  Run: {run_dir}")
    print(f"  Rows: {len(pp)} prefill, {len(tg)} decode")
    print_report(pp, tg)
    plot(pp, tg_ctx, run_dir)


if __name__ == "__main__":
    main()
