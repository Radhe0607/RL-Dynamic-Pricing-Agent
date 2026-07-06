"""
eval_summary_export.py
-----------------------
Model Evaluation Summary Export for the DQN Dynamic Pricing Agent.

This module generates a focused, exportable evaluation summary after
a trained model has been evaluated against the pricing environment.
It is the final step in the evaluation pipeline and is designed to
give a clear, business-readable answer to:

    "How well does the trained agent perform as a pricing engine?"

Separation of concerns
-----------------------
  evaluate.py          — orchestrates greedy rollouts (episode data)
  analytics.py         — computes KPIs from per-step episode data
  comparison.py        — runs all strategies and compares KPIs
  eval_summary_export  — accepts any KPI dict and exports it (this file)

The module is intentionally strategy-agnostic: it works equally well
with KPIs produced by the DQN agent, any baseline, or a manual dict.

KPIs supported
--------------
  total_revenue       : Sum of (price × demand) across all episodes.
  avg_reward          : Mean episode reward across all evaluation episodes.
  occupancy_rate      : total_bookings / (episodes × max_steps), in [0, 1].
  booking_rate        : Fraction of steps where demand > 0.
  avg_selling_price   : Mean price charged across all steps and episodes.

Additional metadata included in exports
----------------------------------------
  n_episodes          : Number of evaluation episodes.
  total_steps         : Total environment steps across all episodes.
  std_reward          : Std deviation of episode rewards.

Outputs (saved to outputs/reports/ by default)
-----------------------------------------------
  eval_summary_<label>_<timestamp>.csv   — tabular, one row per metric
  eval_summary_<label>_<timestamp>.json  — full dict with metadata

Public API
----------
    build_eval_summary(kpis, label) → dict
        Pure function.  Normalises and annotates a KPI dict.

    print_eval_summary(kpis, label) → None
        Prints a formatted console panel.  No I/O.

    export_eval_summary(kpis, label, output_dir) → tuple[str, str]
        Saves CSV and JSON.  Returns (csv_path, json_path).

    generate_eval_summary(kpis, label, output_dir) → tuple[str, str]
        One-call convenience: print → export → return paths.

Usage
-----
    # From evaluate.py (integration)
    from src.evaluation.eval_summary_export import generate_eval_summary

    kpis = {
        "total_revenue":    18432.0,
        "avg_reward":       102.4,
        "occupancy_rate":   0.72,
        "booking_rate":     0.65,
        "avg_selling_price": 55.0,
        "n_episodes":        20,
        "total_steps":       4000,
        "std_reward":        14.2,
    }
    generate_eval_summary(kpis, label="DQN Agent")

    # Standalone CLI
    python -m src.evaluation.eval_summary_export
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> str:
    """Create *path* and any missing parent directories.

    Args:
        path (str): Target directory.

    Returns:
        str: The same *path* (for chaining).
    """
    os.makedirs(path, exist_ok=True)
    return path


def _safe_label(label: str) -> str:
    """Convert a display label to a filename-safe slug.

    Args:
        label (str): Human-readable label (e.g. ``"DQN Agent"``).

    Returns:
        str: Filename-safe version (e.g. ``"dqn_agent"``).
    """
    return label.lower().replace(" ", "_").replace("/", "-").replace("\\", "-")


def _pct_bar(value: float, total: float = 1.0, width: int = 20) -> str:
    """Render a small ASCII progress bar proportional to value/total.

    Args:
        value (float): Numerator.
        total (float): Denominator (1.0 for fractions already in [0,1]).
        width (int):   Bar interior width in characters.

    Returns:
        str: e.g. ``"[████████████░░░░░░░░]  60.0%"``
    """
    pct    = max(0.0, min(1.0, value / max(total, 1e-9))) * 100
    filled = int(pct / 100 * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


# ---------------------------------------------------------------------------
# KPI normaliser — pure, no I/O
# ---------------------------------------------------------------------------

def build_eval_summary(
    kpis:  Dict[str, float],
    label: str = "Agent",
) -> Dict[str, object]:
    """Normalise, annotate, and return an evaluation KPI summary dict.

    This function is **pure** — it performs no I/O and has no side effects.
    It accepts a raw KPI dict (as produced by ``_aggregate_kpis`` in
    ``comparison.py`` or by any evaluation harness) and returns a clean,
    fully annotated summary dict ready for printing or exporting.

    Keys guaranteed in the output
    ------------------------------
    All five required KPIs are present and typed as float.
    Missing keys default to 0.0 so callers are not required to supply all.

    Args:
        kpis  (dict): Raw KPI dict.  Accepted keys (all optional):
                      ``total_revenue``, ``avg_reward``, ``occupancy_rate``,
                      ``booking_rate``, ``avg_selling_price``,
                      ``n_episodes``, ``total_steps``, ``std_reward``.
        label (str):  Human-readable name for the agent / strategy.
                      Embedded in the output and used in filenames.

    Returns:
        dict: Normalised summary with guaranteed keys and metadata.
    """
    summary = {
        # ── Required KPIs ─────────────────────────────────────────────
        "total_revenue":       round(float(kpis.get("total_revenue",       0.0)), 4),
        "avg_reward":          round(float(kpis.get("avg_reward",
                                      kpis.get("mean_reward",              0.0))), 4),
        "occupancy_rate":      round(float(kpis.get("occupancy_rate",      0.0)), 4),
        "booking_rate":        round(float(kpis.get("booking_rate",        0.0)), 4),
        "avg_selling_price":   round(float(kpis.get("avg_selling_price",   0.0)), 4),
        # ── Optional supplementary ────────────────────────────────────
        "std_reward":          round(float(kpis.get("std_reward",          0.0)), 4),
        "n_episodes":          int(kpis.get("n_episodes",    0)),
        "total_steps":         int(kpis.get("total_steps",   0)),
        # ── Metadata ──────────────────────────────────────────────────
        "label":               label,
        "generated_at":        datetime.now(tz=timezone.utc).isoformat(),
    }
    return summary


# ---------------------------------------------------------------------------
# Console printer — no file I/O
# ---------------------------------------------------------------------------

def print_eval_summary(
    kpis:  Dict[str, float],
    label: str = "Agent",
) -> None:
    """Print a formatted evaluation summary panel to stdout.

    The panel shows all five required KPIs plus supplementary info in
    clearly grouped, aligned sections with inline percentage bars.

    Args:
        kpis  (dict): Raw or normalised KPI dict.
        label (str):  Display name for the agent / strategy.

    Example output::

        ╔══════════════════════════════════════════════════════════════╗
        ║  Model Evaluation Summary  |  DQN Agent                     ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  BUSINESS KPIs                                               ║
        ║    Total Revenue         :      18432.0000                   ║
        ║    Average Reward        :        102.4000  ± 14.2000        ║
        ║    Occupancy Rate        :         72.0 %  [████████████░░░░░░░░]  72.0%
        ║    Booking Rate          :         65.0 %  [█████████████░░░░░░░░]  65.0%
        ║    Avg Selling Price     :         £55.0000                  ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  EVALUATION RUN                                              ║
        ║    Episodes evaluated    :             20                    ║
        ║    Total steps           :           4000                    ║
        ╚══════════════════════════════════════════════════════════════╝
    """
    s   = build_eval_summary(kpis, label)
    w   = 66
    sep = "═" * w

    def _line(content: str = "") -> None:
        print(f"║  {content:<{w - 4}}║")

    print(f"╔{sep}╗")
    print(f"║  {'Model Evaluation Summary  |  ' + label:<{w - 4}}║")
    print(f"╠{sep}╣")

    # ── Business KPIs ─────────────────────────────────────────────────────
    _line("BUSINESS KPIs")
    _line("─" * (w - 4))

    _line(f"  {'Total Revenue':<26}: £{s['total_revenue']:>14,.4f}")
    _line(f"  {'Average Reward':<26}:  {s['avg_reward']:>14.4f}"
          f"  ± {s['std_reward']:.4f}")

    occ_pct = s["occupancy_rate"] * 100
    _line(f"  {'Occupancy Rate':<26}:  {occ_pct:>13.1f} %"
          f"  {_pct_bar(s['occupancy_rate'], width=16)}")

    bk_pct = s["booking_rate"] * 100
    _line(f"  {'Booking Rate':<26}:  {bk_pct:>13.1f} %"
          f"  {_pct_bar(s['booking_rate'], width=16)}")

    _line(f"  {'Avg Selling Price':<26}: £{s['avg_selling_price']:>14.4f}")

    print(f"╠{sep}╣")

    # ── Run metadata ──────────────────────────────────────────────────────
    _line("EVALUATION RUN")
    _line("─" * (w - 4))

    if s["n_episodes"]:
        _line(f"  {'Episodes evaluated':<26}:  {s['n_episodes']:>14,}")
    if s["total_steps"]:
        _line(f"  {'Total env steps':<26}:  {s['total_steps']:>14,}")
    if s["n_episodes"] and s["total_steps"]:
        avg_steps = s["total_steps"] / s["n_episodes"]
        _line(f"  {'Avg steps / episode':<26}:  {avg_steps:>14.1f}")

    _line(f"  {'Generated at':<26}:  {s['generated_at'][:19].replace('T', ' ')} UTC")

    print(f"╚{sep}╝")
    print()


# ---------------------------------------------------------------------------
# CSV + JSON export
# ---------------------------------------------------------------------------

def export_eval_summary(
    kpis:       Dict[str, float],
    label:      str = "Agent",
    output_dir: str = "outputs/reports",
) -> Tuple[str, str]:
    """Save the evaluation summary to CSV and JSON files.

    Both files are written to *output_dir* with a timestamp in the
    filename so successive evaluations do not overwrite each other.

    CSV layout
    ----------
    One header row + one data row; each column is one KPI.  This format
    is easy to import into Excel, pandas, or BI tools.

    JSON layout
    -----------
    A metadata envelope containing the full summary dict, the agent label,
    and an ISO-8601 timestamp.

    Args:
        kpis       (dict): Raw or normalised KPI dict.
        label      (str):  Display / filename label. Default ``"Agent"``.
        output_dir (str):  Output directory. Default ``"outputs/reports"``.
                           Created automatically if it does not exist.

    Returns:
        tuple[str, str]: ``(csv_path, json_path)`` — absolute paths of
                         the saved files.
    """
    _ensure_dir(output_dir)

    s         = build_eval_summary(kpis, label)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug      = _safe_label(label)
    base      = f"eval_summary_{slug}_{timestamp}"
    csv_path  = os.path.join(output_dir, base + ".csv")
    json_path = os.path.join(output_dir, base + ".json")

    # ── CSV — one row, one column per KPI ─────────────────────────────────
    # The five required KPIs appear first; supplementary fields follow.
    ordered_keys = [
        "total_revenue",
        "avg_reward",
        "occupancy_rate",
        "booking_rate",
        "avg_selling_price",
        "std_reward",
        "n_episodes",
        "total_steps",
        "label",
        "generated_at",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerow({k: s.get(k, "") for k in ordered_keys})

    print(f"  [eval_summary] CSV  saved → {csv_path}")

    # ── JSON — full summary with metadata envelope ─────────────────────────
    payload = {
        "generated_at": s["generated_at"],
        "label":        label,
        "kpis": {
            "total_revenue":     s["total_revenue"],
            "avg_reward":        s["avg_reward"],
            "std_reward":        s["std_reward"],
            "occupancy_rate":    s["occupancy_rate"],
            "booking_rate":      s["booking_rate"],
            "avg_selling_price": s["avg_selling_price"],
        },
        "run_info": {
            "n_episodes":  s["n_episodes"],
            "total_steps": s["total_steps"],
        },
    }

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  [eval_summary] JSON saved → {json_path}")

    return csv_path, json_path


# ---------------------------------------------------------------------------
# One-call entry point
# ---------------------------------------------------------------------------

def generate_eval_summary(
    kpis:       Dict[str, float],
    label:      str = "Agent",
    output_dir: str = "outputs/reports",
) -> Tuple[str, str]:
    """Print and export the evaluation summary in a single call.

    This is the integration point for the evaluation pipeline.
    It chains ``print_eval_summary`` → ``export_eval_summary``.

    Args:
        kpis       (dict): Raw KPI dict containing any subset of:
                           ``total_revenue``, ``avg_reward``,
                           ``occupancy_rate``, ``booking_rate``,
                           ``avg_selling_price``, ``n_episodes``,
                           ``total_steps``, ``std_reward``.
        label      (str):  Human-readable agent / strategy name.
                           Used in the console header and filenames.
                           Default ``"Agent"``.
        output_dir (str):  Directory for CSV and JSON files.
                           Default ``"outputs/reports"``.

    Returns:
        tuple[str, str]: ``(csv_path, json_path)`` of the saved files.

    Example::

        from src.evaluation.eval_summary_export import generate_eval_summary

        kpis = {
            "total_revenue":    18432.0,
            "avg_reward":       102.4,
            "occupancy_rate":   0.72,
            "booking_rate":     0.65,
            "avg_selling_price": 55.0,
            "n_episodes":       20,
            "total_steps":      4000,
            "std_reward":       14.2,
        }
        csv_path, json_path = generate_eval_summary(kpis, label="DQN Agent")
    """
    # Step 1 — Print formatted panel to console
    print_eval_summary(kpis, label=label)

    # Step 2 — Save CSV + JSON to disk
    return export_eval_summary(kpis, label=label, output_dir=output_dir)


# ---------------------------------------------------------------------------
# CLI — standalone demo with synthetic KPIs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    label = sys.argv[1] if len(sys.argv) > 1 else "DQN Agent (demo)"

    print(f"\n  [eval_summary_export] Demo mode — label: {label}\n")

    # Synthetic KPIs — replace with real values from run_evaluation() or
    # compare_strategies() in production use
    demo_kpis: Dict[str, float] = {
        "total_revenue":     18432.50,
        "avg_reward":          102.40,
        "std_reward":           14.20,
        "occupancy_rate":        0.72,
        "booking_rate":          0.65,
        "avg_selling_price":    55.00,
        "n_episodes":           20,
        "total_steps":        4000,
    }

    csv_p, json_p = generate_eval_summary(demo_kpis, label=label)
    print(f"\n  Files written:")
    print(f"    CSV  → {csv_p}")
    print(f"    JSON → {json_p}")
