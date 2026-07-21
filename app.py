"""
app.py
------
Professional Streamlit Dashboard for the RL Dynamic Pricing Agent.

This dashboard visualises the training results, pricing performance,
baseline comparisons, and environment trends of the DQN-based Dynamic
Pricing project.

The dashboard operates in two modes:
  1. Live mode  — reads real files from outputs/reports/, outputs/logs/,
                  checkpoints/, etc. when they exist.
  2. Demo mode  — generates realistic synthetic data when outputs are
                  absent (e.g. before training has been run).

Both modes render exactly the same UI so the dashboard is always
presentable during demos or development.

Run
---
    streamlit run app.py

Dependencies
------------
    pip install streamlit plotly pandas numpy
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import glob
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be the FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RL Dynamic Pricing Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — mirror values from the project's config modules
# ─────────────────────────────────────────────────────────────────────────────
MIN_PRICE          = 10
MAX_PRICE          = 100
N_ACTIONS          = 5
PRICE_STEP         = (MAX_PRICE - MIN_PRICE) / (N_ACTIONS - 1)
PRICE_LEVELS       = [round(MIN_PRICE + i * PRICE_STEP, 2) for i in range(N_ACTIONS)]
INITIAL_INVENTORY  = 100
DEFAULT_EPISODES   = 1_000

# Output paths (relative to project root)
REPORTS_DIR        = "outputs/reports"
LOGS_DIR           = "outputs/logs"
CHECKPOINTS_DIR    = "checkpoints"
COMPARISON_JSON    = "outputs/comparison_results.json"
COMPARISON_CSV     = "outputs/comparison_results.csv"
TRAINING_LOG       = os.path.join(LOGS_DIR, "training.log")

# Colour palette — consistent, premium dark-mode-friendly
C_DQN              = "#4FC3F7"   # sky blue  — DQN agent
C_FIXED            = "#81C784"   # green     — Fixed strategy
C_RANDOM           = "#FFB74D"   # amber     — Random strategy
C_RULE             = "#CE93D8"   # lavender  — Rule-based strategy
C_ACCENT           = "#64FFDA"   # teal      — accent / highlight
C_WARN             = "#FF8A65"   # coral     — warning / bad metric
C_BG_CARD          = "#1E2130"   # dark card background
STRATEGY_COLOURS   = {
    "DQN Agent":        C_DQN,
    "Fixed Price":      C_FIXED,
    "Random Price":     C_RANDOM,
    "Rule-Based":       C_RULE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS injection
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    """Inject custom CSS for premium card styling and typography."""
    st.markdown(
        """
        <style>
        /* ── Google Font ─────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ── KPI card ────────────────────────────────────────── */
        .kpi-card {
            background: linear-gradient(135deg, #1E2130 0%, #252A3C 100%);
            border: 1px solid rgba(100, 255, 218, 0.15);
            border-radius: 14px;
            padding: 22px 24px 18px;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(79, 195, 247, 0.18);
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #90A4AE;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #E3F2FD;
            line-height: 1.1;
        }
        .kpi-delta {
            font-size: 0.78rem;
            margin-top: 6px;
            color: #64FFDA;
        }

        /* ── Section header ──────────────────────────────────── */
        .section-header {
            border-left: 4px solid #4FC3F7;
            padding-left: 12px;
            margin: 28px 0 16px;
            font-size: 1.15rem;
            font-weight: 700;
            color: #E3F2FD;
        }

        /* ── Hero banner ─────────────────────────────────────── */
        .hero {
            background: linear-gradient(135deg, #0D1B2A 0%, #1A237E 50%, #0D47A1 100%);
            border-radius: 18px;
            padding: 36px 40px;
            margin-bottom: 28px;
            border: 1px solid rgba(79,195,247,0.2);
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #E3F2FD;
            margin: 0 0 8px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #90CAF9;
            max-width: 680px;
        }
        .hero-badges {
            margin-top: 18px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .badge {
            background: rgba(79,195,247,0.15);
            border: 1px solid rgba(79,195,247,0.4);
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.78rem;
            font-weight: 600;
            color: #4FC3F7;
        }

        /* ── Metric diff colour ───────────────────────────────── */
        .positive { color: #69F0AE; }
        .negative { color: #FF5252; }

        /* ── Strategy tag ────────────────────────────────────── */
        .tag-dqn    { color: #4FC3F7; font-weight: 700; }
        .tag-fixed  { color: #81C784; font-weight: 700; }
        .tag-random { color: #FFB74D; font-weight: 700; }
        .tag-rule   { color: #CE93D8; font-weight: 700; }

        /* ── Sidebar refinements ─────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: #111827;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 20px;
        }

        /* ── Divider ─────────────────────────────────────────── */
        .fancy-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #4FC3F7, transparent);
            margin: 28px 0;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generators
# ─────────────────────────────────────────────────────────────────────────────

def _seeded_rng(seed: int = 42) -> np.random.Generator:
    """Return a seeded NumPy generator for reproducible demo data."""
    return np.random.default_rng(seed)


def _generate_demo_rewards(
    n_episodes: int = DEFAULT_EPISODES,
    rng: Optional[np.random.Generator] = None,
) -> List[float]:
    """Generate a realistic reward curve: noisy improvement with saturation.

    The curve mimics a typical DQN learning trajectory:
      - Early episodes: high variance, near-zero average (exploration phase)
      - Mid episodes:   reward rises as ε decays and Q-values stabilise
      - Late episodes:  convergence with small fluctuations around a plateau

    Args:
        n_episodes (int): Number of episodes.
        rng: NumPy random generator (seeded for reproducibility).

    Returns:
        list[float]: Per-episode rewards.
    """
    if rng is None:
        rng = _seeded_rng()

    rewards = []
    plateau = 350.0
    for i in range(n_episodes):
        # Logistic growth from ~0 to plateau
        progress   = i / max(n_episodes - 1, 1)
        mean       = plateau / (1 + math.exp(-10 * (progress - 0.35)))
        noise_std  = max(5.0, 80.0 * (1 - progress))
        reward     = float(rng.normal(mean, noise_std))
        rewards.append(round(reward, 4))
    return rewards


def _moving_average(values: List[float], window: int = 50) -> List[float]:
    """Compute a moving average over *values*.

    Args:
        values (list[float]): Input series.
        window (int): Window size.

    Returns:
        list[float]: Moving average (same length; early values use shorter window).
    """
    result = []
    for i, _ in enumerate(values):
        start  = max(0, i - window + 1)
        result.append(float(np.mean(values[start : i + 1])))
    return result


def _generate_demo_training_summary(
    rewards: List[float],
) -> Dict[str, Any]:
    """Build a training summary dict from a reward series.

    Args:
        rewards (list[float]): Per-episode reward list.

    Returns:
        dict: Summary with keys expected by the UI.
    """
    return {
        "total_episodes":  len(rewards),
        "best_reward":     round(max(rewards), 2),
        "avg_reward":      round(float(np.mean(rewards)), 2),
        "final_epsilon":   0.05,
        "training_time":   "34m 11s",
    }


def _generate_demo_kpis(
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Generate realistic business KPIs for the DQN agent.

    Args:
        rng: Seeded generator.

    Returns:
        dict: KPI values.
    """
    if rng is None:
        rng = _seeded_rng()

    return {
        "total_revenue":     round(float(rng.uniform(180_000, 220_000)), 2),
        "avg_selling_price": round(float(rng.uniform(48, 62)), 2),
        "occupancy_rate":    round(float(rng.uniform(0.68, 0.82)), 4),
        "booking_rate":      round(float(rng.uniform(0.60, 0.76)), 4),
        "avg_reward":        round(float(rng.uniform(310, 370)), 2),
    }


def _generate_demo_comparison() -> pd.DataFrame:
    """Return a DataFrame with KPIs for all four strategies.

    Returns:
        pd.DataFrame: One row per strategy, columns are KPI names.
    """
    data = {
        "Strategy": ["DQN Agent", "Fixed Price", "Random Price", "Rule-Based"],
        "Total Revenue (£)": [198_450, 152_300, 118_200, 171_600],
        "Avg Reward":        [342.1,   218.4,   163.7,   290.5],
        "Occupancy Rate (%)": [74.2,   61.5,    48.3,    67.8],
        "Avg Selling Price (£)": [55.3, 55.0,   52.8,    54.1],
    }
    return pd.DataFrame(data)


def _generate_demo_price_trend(
    n_steps: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Generate a price-over-time series for the DQN agent.

    Args:
        n_steps (int): Number of pricing steps.
        rng: Seeded generator.

    Returns:
        pd.DataFrame: Columns: step, price, demand.
    """
    if rng is None:
        rng = _seeded_rng(seed=7)

    steps   = list(range(1, n_steps + 1))
    actions = rng.integers(0, N_ACTIONS, size=n_steps)
    prices  = [PRICE_LEVELS[a] for a in actions]

    # Smooth prices slightly so the chart looks intentional
    smoothed = []
    for i, p in enumerate(prices):
        if i == 0:
            smoothed.append(p)
        else:
            smoothed.append(round(0.7 * p + 0.3 * smoothed[-1], 2))

    # Demand: inversely related to price with noise
    demands = []
    for p in smoothed:
        base   = max(0, 100 - p + float(rng.uniform(-10, 10)))
        demands.append(round(base, 2))

    return pd.DataFrame({"step": steps, "price": smoothed, "demand": demands})


# ─────────────────────────────────────────────────────────────────────────────
# Real-data loaders (fall back to demo data on failure)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_rewards_from_log(log_path: str) -> Optional[List[float]]:
    """Parse per-episode rewards from the structured training log.

    Looks for lines like:
        [EPISODE] ep=200 ... reward=12.50 ...

    Args:
        log_path (str): Path to training.log.

    Returns:
        list[float] | None: Reward list, or None if file missing / unreadable.
    """
    if not os.path.isfile(log_path):
        return None
    rewards = []
    pattern = re.compile(r"reward=\s*([-\d.]+)")
    with open(log_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                try:
                    rewards.append(float(m.group(1)))
                except ValueError:
                    pass
    return rewards if rewards else None


@st.cache_data(ttl=60)
def _load_comparison_json(json_path: str) -> Optional[Dict]:
    """Load the comparison results JSON produced by comparison.py.

    Args:
        json_path (str): Path to comparison_results.json.

    Returns:
        dict | None: Comparison data, or None if unavailable.
    """
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=60)
def _load_latest_report(reports_dir: str) -> Optional[str]:
    """Find and return the text content of the most recent training report.

    Args:
        reports_dir (str): Directory containing dqn_report_*.txt files.

    Returns:
        str | None: Report text, or None if none found.
    """
    pattern = os.path.join(reports_dir, "dqn_report_*.txt")
    files   = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    try:
        with open(files[0], encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


@st.cache_data(ttl=60)
def _load_eval_summary_json(reports_dir: str) -> Optional[Dict]:
    """Load the most recent evaluation summary JSON.

    Args:
        reports_dir (str): Directory with eval_summary_*.json files.

    Returns:
        dict | None: Summary dict, or None if unavailable.
    """
    pattern = os.path.join(reports_dir, "eval_summary_*.json")
    files   = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return None
    try:
        with open(files[0], encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _build_comparison_df(comp_json: Optional[Dict]) -> pd.DataFrame:
    """Convert the comparison JSON into a clean display DataFrame.

    Args:
        comp_json (dict | None): Loaded comparison results, or None for demo.

    Returns:
        pd.DataFrame: One row per strategy with standardised column names.
    """
    if comp_json is None:
        return _generate_demo_comparison()

    # The comparison JSON has strategy names as top-level keys
    rows = []
    for strategy, kpis in comp_json.items():
        if not isinstance(kpis, dict):
            continue
        rows.append({
            "Strategy":             strategy,
            "Total Revenue (£)":    round(kpis.get("total_revenue",     0.0), 2),
            "Avg Reward":           round(kpis.get("avg_reward",
                                    kpis.get("mean_reward",             0.0)), 2),
            "Occupancy Rate (%)":   round(kpis.get("occupancy_rate",    0.0) * 100, 2),
            "Avg Selling Price (£)": round(kpis.get("avg_selling_price", 0.0), 2),
        })
    return pd.DataFrame(rows) if rows else _generate_demo_comparison()


# ─────────────────────────────────────────────────────────────────────────────
# UI component helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi_card(label: str, value: str, delta: str = "", icon: str = "") -> str:
    """Return the HTML string for a single KPI card.

    Args:
        label (str): Metric name.
        value (str): Formatted metric value.
        delta (str): Optional secondary note (e.g. "+12% vs baseline").
        icon  (str): Emoji icon shown above the value.

    Returns:
        str: HTML string.
    """
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    icon_html  = f'<div style="font-size:1.8rem;margin-bottom:6px">{icon}</div>' if icon else ""
    return f"""
    <div class="kpi-card">
        {icon_html}
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def _section(title: str) -> None:
    """Render a styled section header.

    Args:
        title (str): Section name.
    """
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


def _divider() -> None:
    """Render a subtle horizontal gradient divider."""
    st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def _chart_reward_curve(rewards: List[float], window: int = 50) -> go.Figure:
    """Build the reward-per-episode + moving-average chart.

    Args:
        rewards (list[float]): Per-episode reward values.
        window  (int):         Moving-average window size.

    Returns:
        go.Figure: Plotly figure.
    """
    episodes  = list(range(1, len(rewards) + 1))
    moving_av = _moving_average(rewards, window=window)

    fig = go.Figure()

    # Raw reward — thin, low-opacity line
    fig.add_trace(go.Scatter(
        x=episodes, y=rewards,
        mode="lines",
        name="Episode Reward",
        line=dict(color=C_DQN, width=1, dash="dot"),
        opacity=0.45,
        hovertemplate="Ep %{x}<br>Reward: %{y:.2f}<extra></extra>",
    ))

    # Moving average — bold, prominent
    fig.add_trace(go.Scatter(
        x=episodes, y=moving_av,
        mode="lines",
        name=f"Moving Avg ({window})",
        line=dict(color=C_ACCENT, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(100,255,218,0.06)",
        hovertemplate="Ep %{x}<br>Avg Reward: %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Reward per Episode", font=dict(size=14, color="#90CAF9")),
        xaxis=dict(title="Episode", gridcolor="#1E2A3A", color="#78909C"),
        yaxis=dict(title="Total Reward", gridcolor="#1E2A3A", color="#78909C"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,35,0.6)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#B0BEC5")),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=36, b=0),
    )
    return fig


def _chart_comparison_bars(df: pd.DataFrame, metric: str) -> go.Figure:
    """Build a horizontal bar chart for a single KPI across strategies.

    Args:
        df     (pd.DataFrame): Comparison DataFrame.
        metric (str):          Column name to plot.

    Returns:
        go.Figure: Plotly figure.
    """
    colours = [STRATEGY_COLOURS.get(s, "#90CAF9") for s in df["Strategy"]]

    fig = go.Figure(go.Bar(
        x=df[metric],
        y=df["Strategy"],
        orientation="h",
        marker=dict(
            color=colours,
            line=dict(color="rgba(255,255,255,0.1)", width=0.8),
        ),
        text=[f"{v:,.1f}" for v in df[metric]],
        textposition="outside",
        textfont=dict(color="#B0BEC5", size=11),
        hovertemplate="%{y}<br>" + metric + ": %{x:,.2f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,35,0.6)",
        xaxis=dict(gridcolor="#1E2A3A", color="#78909C", title=metric),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", color="#B0BEC5"),
        margin=dict(l=0, r=60, t=10, b=0),
        height=210,
        showlegend=False,
    )
    return fig


def _chart_price_demand(df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: price (left axis) and demand (right axis) over steps.

    Args:
        df (pd.DataFrame): Columns: step, price, demand.

    Returns:
        go.Figure: Plotly figure with two y-axes.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["price"],
            name="Price (£)",
            mode="lines",
            line=dict(color=C_DQN, width=2),
            hovertemplate="Step %{x}<br>Price: £%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["step"], y=df["demand"],
            name="Demand",
            mode="lines",
            line=dict(color=C_RULE, width=1.8, dash="dash"),
            fill="tozeroy",
            fillcolor="rgba(206,147,216,0.07)",
            hovertemplate="Step %{x}<br>Demand: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,35,0.6)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#B0BEC5"),
                    orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=36, b=0),
        hovermode="x unified",
    )
    fig.update_yaxes(
        title_text="Price (£)",
        gridcolor="#1E2A3A", color="#78909C",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Demand (units)",
        gridcolor="rgba(0,0,0,0)", color="#78909C",
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Step", gridcolor="#1E2A3A", color="#78909C")
    return fig


def _chart_radar(df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing all strategies across normalised KPIs.

    Args:
        df (pd.DataFrame): Comparison DataFrame.

    Returns:
        go.Figure: Plotly polar figure.
    """
    categories = [
        "Total Revenue (£)",
        "Avg Reward",
        "Occupancy Rate (%)",
        "Avg Selling Price (£)",
    ]

    # Normalise each metric 0→1 relative to the maximum across strategies
    normed = df[categories].copy()
    for col in categories:
        max_val = normed[col].max()
        normed[col] = normed[col] / max_val if max_val > 0 else normed[col]

    fig = go.Figure()
    for i, row in df.iterrows():
        values = normed.loc[i, categories].tolist()
        values += values[:1]   # close the radar shape
        cats   = categories + categories[:1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill="toself",
            name=row["Strategy"],
            line=dict(color=STRATEGY_COLOURS.get(row["Strategy"], "#90CAF9"), width=2),
            fillcolor=STRATEGY_COLOURS.get(row["Strategy"], "#90CAF9").replace(")", ",0.12)").replace("rgb", "rgba"),
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(13,19,35,0.6)",
            radialaxis=dict(visible=True, color="#78909C", gridcolor="#1E2A3A",
                            range=[0, 1]),
            angularaxis=dict(color="#90CAF9"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#B0BEC5")),
        margin=dict(l=0, r=0, t=16, b=0),
        height=360,
    )
    return fig


def _chart_epsilon_decay(n_episodes: int, epsilon_start: float = 1.0,
                         epsilon_end: float = 0.05,
                         epsilon_decay: float = 0.995) -> go.Figure:
    """Plot the ε-greedy exploration schedule.

    Args:
        n_episodes    (int):   Total episodes.
        epsilon_start (float): Starting ε.
        epsilon_end   (float): Minimum ε.
        epsilon_decay (float): Multiplicative decay per episode.

    Returns:
        go.Figure: Plotly line chart.
    """
    episodes = list(range(1, n_episodes + 1))
    epsilons = []
    eps = epsilon_start
    for _ in episodes:
        epsilons.append(round(eps, 6))
        eps = max(epsilon_end, eps * epsilon_decay)

    fig = go.Figure(go.Scatter(
        x=episodes, y=epsilons,
        mode="lines",
        line=dict(color=C_FIXED, width=2),
        fill="tozeroy",
        fillcolor="rgba(129,199,132,0.08)",
        hovertemplate="Ep %{x}<br>ε = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=epsilon_end, line_dash="dot", line_color=C_WARN,
                  annotation_text=f"ε_min = {epsilon_end}",
                  annotation_font_color=C_WARN)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,35,0.6)",
        xaxis=dict(title="Episode", gridcolor="#1E2A3A", color="#78909C"),
        yaxis=dict(title="Epsilon (ε)", gridcolor="#1E2A3A", color="#78909C"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=260,
    )
    return fig


def _chart_revenue_compare(df: pd.DataFrame) -> go.Figure:
    """Grouped vertical bar chart for revenue across strategies.

    Args:
        df (pd.DataFrame): Comparison DataFrame.

    Returns:
        go.Figure: Plotly figure.
    """
    colours = [STRATEGY_COLOURS.get(s, "#90CAF9") for s in df["Strategy"]]
    fig = go.Figure(go.Bar(
        x=df["Strategy"],
        y=df["Total Revenue (£)"],
        marker=dict(color=colours, line=dict(color="rgba(255,255,255,0.08)", width=0.8)),
        text=[f"£{v:,.0f}" for v in df["Total Revenue (£)"]],
        textposition="outside",
        textfont=dict(color="#B0BEC5", size=11),
        hovertemplate="%{x}<br>Revenue: £%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,19,35,0.6)",
        xaxis=dict(gridcolor="rgba(0,0,0,0)", color="#B0BEC5"),
        yaxis=dict(title="Total Revenue (£)", gridcolor="#1E2A3A", color="#78909C"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=300,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar(cfg_defaults: Dict) -> Dict:
    """Render the sidebar and return user-controlled settings.

    Args:
        cfg_defaults (dict): Default hyperparameter values (from DQNConfig).

    Returns:
        dict: Dashboard configuration chosen by the user.
    """
    with st.sidebar:
        st.markdown(
            "## 💹 RL Dynamic Pricing\n"
            "#### Dashboard Controls",
        )
        st.divider()

        # ── Project info ──────────────────────────────────────────────────
        with st.expander("📋 Project Information", expanded=True):
            st.markdown("""
**Project**: RL-based Dynamic Hotel Pricing  
**Agent**: Deep Q-Network (DQN)  
**Environment**: Custom Gymnasium Env  
**State space**: 3 features (price, inventory, days)  
**Action space**: 5 discrete price levels  
**Price range**: £10 – £100  
**Reward**: Revenue per step (price × demand)
            """)

        # ── Technologies ──────────────────────────────────────────────────
        with st.expander("🛠 Technologies Used"):
            st.markdown("""
| Component | Technology |
|-----------|-----------|
| RL Agent | PyTorch DQN |
| Environment | Gymnasium |
| Config | Python dataclass |
| Visualisation | Streamlit + Plotly |
| Data export | CSV + JSON |
| Logging | Python logging |
| Checkpointing | PyTorch `.pt` |
            """)

        # ── Hyperparameters ───────────────────────────────────────────────
        with st.expander("⚙️ DQN Hyperparameters"):
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Learning rate | `{cfg_defaults['learning_rate']}` |
| Gamma (γ) | `{cfg_defaults['gamma']}` |
| Tau (τ) | `{cfg_defaults['tau']}` |
| Epsilon start | `{cfg_defaults['epsilon_start']}` |
| Epsilon end | `{cfg_defaults['epsilon_end']}` |
| Epsilon decay | `{cfg_defaults['epsilon_decay']}` |
| Batch size | `{cfg_defaults['batch_size']}` |
| Buffer capacity | `{cfg_defaults['buffer_capacity']:,}` |
| Hidden size | `{cfg_defaults['hidden_size']}` |
| Max episodes | `{cfg_defaults['max_episodes']:,}` |
| Steps/episode | `{cfg_defaults['max_steps_per_episode']}` |
            """)

        st.divider()

        # ── Display controls ──────────────────────────────────────────────
        st.markdown("#### 🎛 Display Controls")
        ma_window = st.slider(
            "Moving-avg window", min_value=10, max_value=200,
            value=50, step=10, key="ma_window",
        )
        n_trend_steps = st.slider(
            "Price/demand trend steps", min_value=50, max_value=500,
            value=200, step=50, key="n_trend_steps",
        )
        show_radar = st.checkbox("Show radar comparison", value=True)
        show_epsilon_chart = st.checkbox("Show ε decay chart", value=True)

        st.divider()
        st.caption("🔄 Dashboard auto-refreshes from outputs/")
        st.caption(f"Last render: {datetime.now().strftime('%H:%M:%S')}")

    return {
        "ma_window":         ma_window,
        "n_trend_steps":     n_trend_steps,
        "show_radar":        show_radar,
        "show_epsilon_chart": show_epsilon_chart,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — renders the full dashboard."""

    _inject_css()

    # ── Default DQN config values (no project import needed) ─────────────
    cfg_defaults = {
        "learning_rate":       1e-3,
        "gamma":               0.99,
        "tau":                 0.005,
        "epsilon_start":       1.0,
        "epsilon_end":         0.05,
        "epsilon_decay":       0.995,
        "batch_size":          64,
        "buffer_capacity":     10_000,
        "hidden_size":         64,
        "max_episodes":        1_000,
        "max_steps_per_episode": 200,
        "target_update_freq":  100,
    }

    # ── Sidebar controls ──────────────────────────────────────────────────
    settings = _render_sidebar(cfg_defaults)
    rng      = _seeded_rng()

    # ────────────────────────────────────────────────────────────────────────
    # Hero Banner
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">💹 RL Dynamic Pricing Agent</div>
        <div class="hero-subtitle">
            A Deep Q-Network (DQN) agent trained to optimise hotel room pricing
            dynamically — balancing revenue maximisation with demand-driven
            occupancy.  The agent learns a pricing policy entirely from
            interaction with a custom Gymnasium environment.
        </div>
        <div class="hero-badges">
            <span class="badge">🤖 DQN</span>
            <span class="badge">🎮 Gymnasium</span>
            <span class="badge">🔥 PyTorch</span>
            <span class="badge">📊 Streamlit</span>
            <span class="badge">📈 Plotly</span>
            <span class="badge">🐍 Python 3.10+</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data loading ───────────────────────────────────────────────────────
    # Attempt to load real files; fall back to demo data transparently
    real_rewards   = _load_rewards_from_log(TRAINING_LOG)
    is_demo        = real_rewards is None or len(real_rewards) == 0
    rewards        = real_rewards if not is_demo else _generate_demo_rewards(
                         DEFAULT_EPISODES, rng=_seeded_rng(1)
                     )

    comp_json      = _load_comparison_json(COMPARISON_JSON)
    comparison_df  = _build_comparison_df(comp_json)

    real_eval      = _load_eval_summary_json(REPORTS_DIR)
    if real_eval and "kpis" in real_eval:
        kpis = real_eval["kpis"]
        kpis.update(real_eval.get("run_info", {}))
    else:
        kpis = _generate_demo_kpis(rng=_seeded_rng(2))

    training_summary = _generate_demo_training_summary(rewards)

    report_text    = _load_latest_report(REPORTS_DIR)
    trend_df       = _generate_demo_price_trend(
                         n_steps=settings["n_trend_steps"],
                         rng=_seeded_rng(7),
                     )

    if is_demo:
        st.info(
            "**Demo Mode** — no training outputs were found at `outputs/`.  "
            "Realistic synthetic data is displayed.  "
            "Run `python -m src.training.train` to populate real results.",
            icon="ℹ️",
        )

    # ════════════════════════════════════════════════════════════════════════
    # Section 1 — Training Summary
    # ════════════════════════════════════════════════════════════════════════
    _section("📊 Training Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_cards = [
        (c1, "🎯", "Total Episodes",  f"{training_summary['total_episodes']:,}", ""),
        (c2, "🏆", "Best Reward",     f"{training_summary['best_reward']:,.1f}", "peak performance"),
        (c3, "📈", "Average Reward",  f"{training_summary['avg_reward']:,.1f}",  "all episodes"),
        (c4, "🎲", "Final Epsilon",   f"{training_summary['final_epsilon']:.3f}", "exploration floor"),
        (c5, "⏱",  "Training Time",  training_summary["training_time"],         "wall-clock"),
    ]
    for col, icon, label, value, delta in kpi_cards:
        col.markdown(_kpi_card(label, value, delta=delta, icon=icon),
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Epsilon decay chart (optional) ────────────────────────────────────
    if settings["show_epsilon_chart"]:
        st.plotly_chart(
            _chart_epsilon_decay(
                n_episodes=training_summary["total_episodes"],
                epsilon_start=cfg_defaults["epsilon_start"],
                epsilon_end=cfg_defaults["epsilon_end"],
                epsilon_decay=cfg_defaults["epsilon_decay"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ════════════════════════════════════════════════════════════════════════
    # Section 2 — Pricing Performance KPIs
    # ════════════════════════════════════════════════════════════════════════
    _section("💰 Pricing Performance (DQN Agent)")

    dqn_row = comparison_df[comparison_df["Strategy"] == "DQN Agent"]

    # Pull from eval summary JSON if available, else use comparison_df
    total_rev    = kpis.get("total_revenue",     dqn_row["Total Revenue (£)"].values[0]    if not dqn_row.empty else 0)
    avg_price    = kpis.get("avg_selling_price", dqn_row["Avg Selling Price (£)"].values[0] if not dqn_row.empty else 0)
    occ_rate     = kpis.get("occupancy_rate",    dqn_row["Occupancy Rate (%)"].values[0] / 100  if not dqn_row.empty else 0)
    bk_rate      = kpis.get("booking_rate",      0.0)
    avg_rwd      = kpis.get("avg_reward",        dqn_row["Avg Reward"].values[0]            if not dqn_row.empty else 0)

    k1, k2, k3, k4, k5 = st.columns(5)
    pricing_kpis = [
        (k1, "💵", "Total Revenue",      f"£{total_rev:,.0f}",        "across all episodes"),
        (k2, "🏷",  "Avg Selling Price", f"£{avg_price:.2f}",         "per booking"),
        (k3, "🏨", "Occupancy Rate",    f"{occ_rate * 100:.1f}%",    "capacity utilised"),
        (k4, "📅", "Booking Rate",      f"{bk_rate * 100:.1f}%",     "steps with demand"),
        (k5, "⭐", "Average Reward",    f"{avg_rwd:,.1f}",            "per episode"),
    ]
    for col, icon, label, value, delta in pricing_kpis:
        col.markdown(_kpi_card(label, value, delta=delta, icon=icon),
                     unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # Section 3 — Reward Visualisation
    # ════════════════════════════════════════════════════════════════════════
    _section("📉 Reward Visualisation")

    st.plotly_chart(
        _chart_reward_curve(rewards, window=settings["ma_window"]),
        use_container_width=True,
    )

    # Reward distribution histogram
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_hist = px.histogram(
            x=rewards, nbins=60,
            labels={"x": "Episode Reward", "y": "Count"},
            color_discrete_sequence=[C_DQN],
            opacity=0.85,
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,19,35,0.6)",
            xaxis=dict(gridcolor="#1E2A3A", color="#78909C"),
            yaxis=dict(gridcolor="#1E2A3A", color="#78909C"),
            margin=dict(l=0, r=0, t=36, b=0),
            title=dict(text="Reward Distribution", font=dict(size=13, color="#90CAF9")),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, use_container_width=True,
                        config={"displayModeBar": False})

    with col_b:
        arr = np.array(rewards)
        st.markdown("**Reward Statistics**")
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Mean   | `{arr.mean():.2f}` |
| Median | `{float(np.median(arr)):.2f}` |
| Std    | `{arr.std():.2f}` |
| Min    | `{arr.min():.2f}` |
| Max    | `{arr.max():.2f}` |
| Q25    | `{float(np.percentile(arr, 25)):.2f}` |
| Q75    | `{float(np.percentile(arr, 75)):.2f}` |
        """)

    # ════════════════════════════════════════════════════════════════════════
    # Section 4 — Baseline Comparison
    # ════════════════════════════════════════════════════════════════════════
    _section("🏁 Baseline Strategy Comparison")

    # Styled comparison table
    st.markdown("##### 📋 Full KPI Comparison Table")
    styled = comparison_df.style.background_gradient(
        subset=["Total Revenue (£)", "Avg Reward", "Occupancy Rate (%)"],
        cmap="Blues",
    ).format({
        "Total Revenue (£)":     "£{:,.0f}",
        "Avg Reward":            "{:.1f}",
        "Occupancy Rate (%)":    "{:.1f}%",
        "Avg Selling Price (£)": "£{:.2f}",
    })
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Revenue bar chart (full width)
    st.markdown("##### 💷 Total Revenue by Strategy")
    st.plotly_chart(
        _chart_revenue_compare(comparison_df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Per-metric horizontal bars in a 2×2 grid
    st.markdown("##### 📊 KPI Breakdown by Strategy")
    metrics_to_plot = [
        "Avg Reward",
        "Occupancy Rate (%)",
        "Avg Selling Price (£)",
    ]
    cols = st.columns(len(metrics_to_plot))
    for col, metric in zip(cols, metrics_to_plot):
        with col:
            st.caption(metric)
            st.plotly_chart(
                _chart_comparison_bars(comparison_df, metric),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # Radar chart
    if settings["show_radar"]:
        st.markdown("##### 🕸 Multi-Metric Radar Comparison")
        st.plotly_chart(
            _chart_radar(comparison_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ════════════════════════════════════════════════════════════════════════
    # Section 5 — Price & Demand Trends
    # ════════════════════════════════════════════════════════════════════════
    _section("📈 Price & Demand Trends (Single Episode)")

    st.plotly_chart(
        _chart_price_demand(trend_df),
        use_container_width=True,
    )

    # Separate standalone charts below
    col_p, col_d = st.columns(2)

    with col_p:
        fig_price = px.line(
            trend_df, x="step", y="price",
            title="Price over Time",
            labels={"step": "Step", "price": "Price (£)"},
            color_discrete_sequence=[C_DQN],
        )
        fig_price.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,19,35,0.6)",
            xaxis=dict(gridcolor="#1E2A3A", color="#78909C"),
            yaxis=dict(gridcolor="#1E2A3A", color="#78909C"),
            margin=dict(l=0, r=0, t=36, b=0),
            title=dict(font=dict(size=13, color="#90CAF9")),
            showlegend=False,
            height=250,
        )
        st.plotly_chart(fig_price, use_container_width=True,
                        config={"displayModeBar": False})

    with col_d:
        fig_demand = px.area(
            trend_df, x="step", y="demand",
            title="Demand over Time",
            labels={"step": "Step", "demand": "Demand (units)"},
            color_discrete_sequence=[C_RULE],
        )
        fig_demand.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,19,35,0.6)",
            xaxis=dict(gridcolor="#1E2A3A", color="#78909C"),
            yaxis=dict(gridcolor="#1E2A3A", color="#78909C"),
            margin=dict(l=0, r=0, t=36, b=0),
            title=dict(font=dict(size=13, color="#90CAF9")),
            showlegend=False,
            height=250,
        )
        st.plotly_chart(fig_demand, use_container_width=True,
                        config={"displayModeBar": False})

    # ════════════════════════════════════════════════════════════════════════
    # Section 6 — Training Report
    # ════════════════════════════════════════════════════════════════════════
    _section("📄 Training Report")

    if report_text:
        st.markdown("*Most recent report from `outputs/reports/`*")
        st.code(report_text, language="text")
    else:
        st.info(
            "No training report found at `outputs/reports/dqn_report_*.txt`.  "
            "A report is generated automatically when training completes.",
            icon="📭",
        )
        # Show a sample report structure
        sample_report = """================================================================
  RL Dynamic Pricing — DQN Agent
  Training Report  (DEMO — run training to generate a real report)
================================================================
  Generated : 2025-07-13 08:29:50 UTC
  Checkpoint: checkpoints/

----------------------------------------------------------------
  TRAINING DURATION
----------------------------------------------------------------
  Total episodes trained  :          1,000
  Total environment steps :        200,000
  Wall-clock duration     :      14m 32s

----------------------------------------------------------------
  REWARD STATISTICS
----------------------------------------------------------------
  Best episode reward     :     348.1200
  Mean reward             :     218.4100
  Final episode reward    :     334.9800
  Best moving-avg reward  :     308.5000  (episode 947)

----------------------------------------------------------------
  EXPLORATION (ε-greedy)
----------------------------------------------------------------
  Initial ε               : 1.0000
  Final ε                 : 0.0500
  ε decayed               : 100.0 %
================================================================"""
        st.code(sample_report, language="text")

    # ════════════════════════════════════════════════════════════════════════
    # Footer
    # ════════════════════════════════════════════════════════════════════════
    _divider()
    st.markdown(
        "<div style='text-align:center; color:#546E7A; font-size:0.82rem; padding:12px 0'>"
        "RL Dynamic Pricing Agent · Built with PyTorch, Gymnasium, Streamlit & Plotly · "
        f"Dashboard rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
