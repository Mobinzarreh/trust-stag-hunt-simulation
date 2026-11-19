#!/usr/bin/env python3
"""
Figure 1 of the paper (default, --kind combined): belief, confidence and outcomes over the course of an
interaction. Panels (a)-(d): Gradual Deterioration, Break and Repair, Strategic Cheater and Expectation Violation,
trust-sensitive (TS) and discounted (Disc) agents, aggregated over 300 runs; panel (e): one typical run of each agent
against the Strategic Cheater (closest to the agent's median total payoff and median number of failed
collaborations, standardized distance, ties by run id). Drawn at 6.5 in with 9 pt type.

--kind scheduled adaptive draws two larger four-agent figures for all seven non-stationary partners instead. Each
block is one partner (blocks on a 2 x 2 grid), content aggregated over all 300 runs, on one linear concentration
scale:
    belief      per-round median of the collaboration estimate p-hat of each agent, interquartile band for the
                trust-sensitive (TS) and discounted (Disc) agents; threshold u = 2/3 dotted
    confidence  concentration of each agent's belief distribution on one linear scale: 2 eps + t for TS (median and
                interquartile band), alpha + beta for the Beta agents (the same in every run, so a single line); the
                cumulative agent's concentration leaves the axis after round 10 (it reaches 102)
    outcomes    one strip per agent; each round's column is split by the share of the 300 runs that ended in each of
                the four outcomes (successful collaboration, failed collaboration, positive surprise, mutual
                independence)
Shading marks the rounds in which the partner's collaboration probability is below u (scheduled partners only).

'scheduled': the four partners whose reliability changes on a schedule (Single-Cycle, Gradual Deterioration, Break
and Repair, Reliability Ramp); 'adaptive': the three partners that act on a model of the focal agent (Strategic
Cheater, Expectation Violation, Distrust Lock).

Nothing is simulated: the runs are those of run_benchmark_comparison.py (results/benchmarks/) and
run_metric_panel.py (results/metric_panel/runs/current/); the per-agent payoff means are checked against Table 4
before plotting.

Usage:
    python scripts/make_dynamics_figures.py            # results/figures/paper_dynamics_combined.{png,pdf} (Figure 1)
    python scripts/make_dynamics_figures.py --kind scheduled adaptive   # the two four-agent figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from trust_bayesian_agent_comparison.config import (
    DECISION_THRESHOLD, EPS, FIGURES_DIR, NUM_ROUNDS, RECENCY_BENCHMARK_KEY, RESULTS_DIR, SLIDING_WINDOW_W)
from trust_bayesian_agent_comparison.partners.registry import partner_factory

BENCH = RESULTS_DIR / "benchmarks"
PANEL_RUNS = RESULTS_DIR / "metric_panel" / "runs" / "current"
# agent -> (label, colour, line style, line width, IQR band)
AGENTS = {
    "trust": ("TS", "#2E86AB", "-", 2.4, True),
    "disc": ("Disc", "#A23B72", "--", 2.0, True),
    "win": ("SW", "#E08A3C", "-.", 1.8, False),
    "cum": ("Cum", "#6C757D", (0, (1, 1.4)), 2.0, False),
}
B1_FILE = {"trust": ("cumulative", 1), "cum": ("cumulative", 2), "disc": (RECENCY_BENCHMARK_KEY, 2),
           "win": (f"window_w{SLIDING_WINDOW_W}", 2)}
PANEL_FILE = {"trust": "trust", "cum": "cumulative", "disc": "discpp0.9", "win": f"win{SLIDING_WINDOW_W}"}
PANEL_PARTNERS = {"BreakRepair", "ReliabilityRamp", "DistrustLock"}

FIGURES = {
    "scheduled": [("SingleCycle", "Single-Cycle"), ("GradualDeterioration", "Gradual Deterioration"),
                  ("BreakRepair", "Break and Repair"), ("ReliabilityRamp", "Reliability Ramp")],
    "adaptive": [("StrategicCheater", "Strategic Cheater"), ("ExpectationViolation", "Expectation Violation"),
                 ("DistrustLock", "Distrust Lock")],
}
# Table 4 of the manuscript (mean total payoff; TS, Disc, Win, Cum): the plotted runs must be the reported ones
TABLE4 = {
    "SingleCycle": (240.3, 239.3, 242.2, 209.0), "GradualDeterioration": (266.1, 264.5, 265.5, 255.2),
    "BreakRepair": (292.9, 295.0, 300.8, 273.7), "ReliabilityRamp": (286.3, 287.5, 291.6, 273.1),
    "StrategicCheater": (330.8, 321.9, 316.4, 303.7), "ExpectationViolation": (289.0, 269.5, 261.3, 260.8),
    "DistrustLock": (272.1, 283.4, 285.4, 265.7),
}
# outcome code -> (label, colour); stacked from the bottom of each strip in this order
OUTCOMES = [("Successful collaboration", "#28A745"), ("Failed collaboration", "#DC3545"),
            ("Positive surprise", "#ADB5BD"), ("Mutual independence", "#4A90D9")]
THRESH_C, SHADE_C = "#C73E1D", "#EFEDE6"
TITLE_KW = dict(fontsize=14, fontweight="bold", fontstyle="italic", pad=7, loc="left", color="#1C3A5E",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#EDF2F8", edgecolor="#B0C4D8", alpha=0.9))
FS_ROW, FS_Y, FS_TICK, FS_LEG = 13, 13, 12, 12
CONC_TOP = 14.5


def path(partner, agent):
    if partner in PANEL_PARTNERS:
        return PANEL_RUNS / f"{partner}_{PANEL_FILE[agent]}.csv"
    d, slot = B1_FILE[agent]
    return BENCH / d / f"{partner}_agent{slot}.csv"


def load(partner, agent):
    df = pd.read_csv(path(partner, agent))
    df["conc"] = 2 * EPS + df["trust_level"] if "trust_level" in df else df["alpha"] + df["beta"]
    a, o = df.agent_action.astype(int), df.partner_action.astype(int)
    df["outcome"] = np.select([(a == 1) & (o == 1), (a == 1) & (o == 0), (a == 0) & (o == 1)], [0, 1, 2], 3)
    return df


def check_table4(partner, data):
    for k, agent in enumerate(("trust", "disc", "win", "cum")):
        got = data[agent].groupby("run_id").agent_payoff.sum().mean()
        assert abs(got - TABLE4[partner][k]) < 0.051, (partner, agent, got, TABLE4[partner][k])


def unreliable_spans(partner):
    """Rounds in which the partner's scheduled collaboration probability is below u, as (start, end) spans."""
    p = partner_factory(partner)()
    if not hasattr(p, "coop_prob"):
        return []
    low = np.array([p.coop_prob(k) for k in range(NUM_ROUNDS)]) < DECISION_THRESHOLD
    spans, k = [], 0
    while k < NUM_ROUNDS:
        if low[k]:
            j = k
            while j + 1 < NUM_ROUNDS and low[j + 1]:
                j += 1
            spans.append((k, j + 1))
            k = j + 1
        else:
            k += 1
    return spans


def outcome_image(df, height=60):
    """(height x rounds x 4) image: each column split by the share of runs in each outcome (bottom to top)."""
    share = (df.groupby(["round", "outcome"]).size().unstack(fill_value=0)
             .reindex(columns=range(4), fill_value=0))
    share = share.div(share.sum(axis=1), axis=0).to_numpy()
    rgb = np.array([matplotlib.colors.to_rgba(c) for _, c in OUTCOMES])
    img = np.ones((height, share.shape[0], 4))
    edges = np.cumsum(share, axis=1)
    centers = (np.arange(height) + 0.5) / height
    for r in range(share.shape[0]):
        idx = np.searchsorted(edges[r], centers, side="right").clip(0, 3)
        img[:, r] = rgb[idx]
    return img


def draw_block(fig, spec, partner, title, letter, data, left, bottom):
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, height_ratios=[3.0, 2.4, 2.2], hspace=0.06)
    ax_b = fig.add_subplot(inner[0])
    ax_c = fig.add_subplot(inner[1], sharex=ax_b)
    ax_o = fig.add_subplot(inner[2], sharex=ax_b)
    spans = unreliable_spans(partner)
    for ax in (ax_b, ax_c):
        for s, e in spans:
            ax.axvspan(s - 0.5, e - 0.5, color=SHADE_C, zorder=0, lw=0)
    for agent, (label, color, ls, lw, band) in AGENTS.items():
        df = data[agent]
        for ax, col in ((ax_b, "agent_belief"), (ax_c, "conc")):
            g = df.groupby("round")[col]
            med = g.median()
            if band and not (col == "conc" and agent != "trust"):
                ax.fill_between(med.index, g.quantile(0.25), g.quantile(0.75), color=color, alpha=0.17, lw=0, zorder=1)
            ax.plot(med.index, med.values, color=color, ls=ls, lw=lw, zorder=3 if agent == "trust" else 2)
    ax_b.axhline(DECISION_THRESHOLD, color=THRESH_C, ls=":", lw=1.6, alpha=0.8, zorder=2)
    # the cumulative agent's concentration leaves the axis: say where it goes
    cum = data["cum"].groupby("round").conc.median()
    exit_r = int(cum[cum > CONC_TOP].index.min())
    ax_c.annotate("Cum continues to 102", xy=(exit_r, CONC_TOP), xytext=(exit_r + 2, 13.3), fontsize=11,
                  color=AGENTS["cum"][1], ha="left", va="center")

    strips = [outcome_image(data[a]) for a in AGENTS]
    h = strips[0].shape[0]
    gap = np.ones((max(2, h // 12), strips[0].shape[1], 4))
    stack = []
    for k, s in enumerate(reversed(strips)):  # Cum at the bottom, TS on top
        if k:
            stack.append(gap)
        stack.append(s)
    img = np.vstack(stack)
    ax_o.imshow(img, aspect="auto", interpolation="nearest", origin="lower",
                extent=[-0.5, NUM_ROUNDS - 0.5, 0, img.shape[0]])
    centers = [(k * (h + gap.shape[0])) + h / 2 for k in range(len(strips))]
    ax_o.set_yticks(centers)
    ax_o.set_yticklabels([AGENTS[a][0] for a in reversed(list(AGENTS))], fontsize=FS_TICK, fontweight="bold")
    ax_o.tick_params(axis="y", length=0, pad=6)
    ax_o.grid(False)
    for s in ("top", "left", "right"):
        ax_o.spines[s].set_visible(False)

    ax_b.set_title(f"({letter})  {title}", **TITLE_KW)
    ax_b.set_ylim(-0.03, 1.03)
    ax_b.set_yticks([0, 1 / 3, 2 / 3, 1])
    ax_b.set_yticklabels(["0", "1/3", "2/3", "1"])
    ax_c.set_ylim(0, CONC_TOP)
    ax_c.set_yticks([2, 6, 12])
    ax_b.set_xlim(-0.5, NUM_ROUNDS - 0.5)
    for ax in (ax_b, ax_c):
        ax.grid(alpha=0.25)
        ax.tick_params(labelbottom=False, labelsize=FS_TICK)
        if not left:
            ax.tick_params(labelleft=False)
    if bottom:
        ax_o.set_xlabel("Round", fontsize=FS_Y, fontweight="bold")
        ax_o.tick_params(axis="x", labelsize=FS_TICK)
    else:
        ax_o.tick_params(labelbottom=False)
    if left:
        ax_b.set_ylabel(r"Estimate $\hat{p}_k$", fontsize=FS_Y, fontweight="bold")
        ax_c.set_ylabel("Concentration", fontsize=FS_Y, fontweight="bold")
        for ax, rl in ((ax_b, "Estimate"), (ax_c, "Concentration"), (ax_o, "Outcomes")):
            ax.annotate(rl, xy=(-0.15, 0.5), xycoords="axes fraction", fontsize=FS_ROW, fontweight="bold",
                        color="#333333", ha="center", va="center", rotation=90, annotation_clip=False)
    else:
        ax_o.tick_params(labelleft=False)


def legend_handles(with_shading):
    h = [Line2D([0], [0], color=c, ls=ls, lw=lw, label=lab) for lab, c, ls, lw, _ in AGENTS.values()]
    h += [Patch(facecolor=AGENTS["trust"][1], alpha=0.25, label="Interquartile range (TS, Disc)"),
          Line2D([0], [0], color=THRESH_C, ls=":", lw=1.6, label="Threshold u = 2/3")]
    if with_shading:
        h.append(Patch(facecolor=SHADE_C, edgecolor="#cccccc", label="Partner below u"))
    h += [Patch(facecolor=c, label=lab) for lab, c in OUTCOMES]
    return h


def make(kind, out_dir):
    panels = FIGURES[kind]
    data = {}
    for partner, _ in panels:
        data[partner] = {a: load(partner, a) for a in AGENTS}
        check_table4(partner, data[partner])
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15.5, 14))
    outer = gridspec.GridSpec(2, 2, figure=fig, hspace=0.16, wspace=0.08, left=0.1, right=0.985, top=0.97,
                              bottom=0.1 if kind == "scheduled" else 0.04)
    for k, (partner, title) in enumerate(panels):
        r, c = divmod(k, 2)
        draw_block(fig, outer[r, c], partner, title, "ABCD"[k], data[partner], left=c == 0,
                   bottom=(r == 1) or (kind == "adaptive" and k == 1))
    if kind == "scheduled":
        fig.legend(handles=legend_handles(True), loc="lower center", ncol=6, fontsize=FS_LEG, framealpha=0.95,
                   edgecolor="#cccccc", bbox_to_anchor=(0.54, 0.012), handlelength=2.2, columnspacing=1.5)
    else:  # the fourth cell holds the legend
        ax = fig.add_subplot(outer[1, 1])
        ax.axis("off")
        ax.legend(handles=legend_handles(False), loc="center", ncol=1, fontsize=FS_LEG + 1, framealpha=0.95,
                  edgecolor="#cccccc", handlelength=2.6)
    out = out_dir / f"paper_dynamics_{kind}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} and {out.with_suffix('.pdf')}")
    return out


C_FS = 9
C_AGENTS = {"trust": ("TS", "#2E86AB", "-", 1.8), "disc": ("Disc", "#A23B72", "--", 1.5)}
C_NAMES = {"trust": "TS: trust-sensitive agent", "disc": "Disc: discounted Bayesian agent"}
# outcome code -> (label, colour); colourblind-checked (all-pairs CVD dE 8.3, normal-vision floor 15.5), replacing the
# red/green pair of OUTCOMES (dE 3.5 under deuteranopia). Stacked from the bottom in this order, so failed
# collaboration sits on the baseline of every strip.
C_OUTC = {1: ("Failed collaboration", "#E1702A"), 0: ("Successful collaboration", "#1A7F5E"),
          2: ("Positive surprise", "#B0B4BA"), 3: ("Mutual independence", "#4C72B0")}
C_STACK = [1, 0, 2, 3]
C_FRAME = "#1C3A5E"
C_CONC_TOP = 13.2
C_PANELS = [("GradualDeterioration", "Gradual Deterioration"), ("BreakRepair", "Break and Repair"),
            ("StrategicCheater", "Strategic Cheater"), ("ExpectationViolation", "Expectation Violation")]
C_ZOOM = "StrategicCheater"
C_CONC_LABEL = r"$\alpha_k+\beta_k$"
# figure height (in); 7.7 leaves room for the caption on a 9 in text page. Margins and frame offsets are set in inches.
C_HEIGHT = 7.7
C_RC = {"font.size": C_FS, "axes.labelsize": C_FS, "xtick.labelsize": C_FS, "ytick.labelsize": C_FS,
        "legend.fontsize": C_FS, "font.family": "DejaVu Sans", "axes.edgecolor": "#B8B8B8", "axes.linewidth": 0.7,
        "xtick.color": "#444", "ytick.color": "#444", "axes.labelcolor": "#222", "xtick.major.size": 2,
        "ytick.major.size": 2, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5}


def _codes(df):
    a, o = df.agent_action.to_numpy(int), df.partner_action.to_numpy(int)
    return np.select([(a == 1) & (o == 1), (a == 1) & (o == 0), (a == 0) & (o == 1)], [0, 1, 2], 3)


def _share_image(df, height=60):
    d = df.assign(outcome=_codes(df))
    share = d.groupby(["round", "outcome"]).size().unstack(fill_value=0).reindex(columns=C_STACK, fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).to_numpy()
    rgb = np.array([matplotlib.colors.to_rgba(C_OUTC[c][1]) for c in C_STACK])
    img = np.ones((height, share.shape[0], 4))
    edges = np.cumsum(share, axis=1)
    centers = (np.arange(height) + 0.5) / height
    for r in range(share.shape[0]):
        img[:, r] = rgb[np.searchsorted(edges[r], centers, side="right").clip(0, 3)]
    return img


def typical_run(df):
    """Run closest to the agent's median total payoff and median number of failed collaborations (standardized
    squared distance; ties broken by the lower run id). A stated rule, so the shown run is not chosen by eye."""
    pi = df.groupby("run_id").agent_payoff.sum()
    fc = ((df.agent_action == 1) & (df.partner_action == 0)).groupby(df.run_id).sum()
    z = ((pi - pi.median()) / pi.std()) ** 2 + ((fc - fc.median()) / max(fc.std(), 1e-9)) ** 2
    return int(z.sort_values(kind="stable").index[0])


def _style(ax, left, ylabel, yt, ytl, ylim):
    ax.set_ylim(*ylim)
    ax.set_yticks(yt)
    ax.set_yticklabels(ytl if left else [])
    if left:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.5, NUM_ROUNDS - 0.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])


def _strips(ax, imgs, left, labels):
    h = imgs[0].shape[0]
    gap = np.ones((8, imgs[0].shape[1], 4))
    stack = []
    for k, s in enumerate(reversed(imgs)):
        if k:
            stack.append(gap)
        stack.append(s)
    img = np.vstack(stack)
    ax.imshow(img, aspect="auto", interpolation="nearest", origin="lower",
              extent=[-0.5, imgs[0].shape[1] - 0.5, 0, img.shape[0]])
    ax.set_yticks([(k * (h + gap.shape[0])) + h / 2 for k in range(len(imgs))])
    ax.set_yticklabels(list(reversed(labels)) if left else [])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, NUM_ROUNDS - 0.5)
    ax.set_xticks([0, 20, 40, 60, 80, 100])


def _block(fig, spec, partner, title, letter, data, left):
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, height_ratios=[2.4, 1.3, 1.25], hspace=0.09)
    ax_b, ax_c, ax_o = (fig.add_subplot(inner[i]) for i in range(3))
    for ax in (ax_b, ax_c):
        for s, e in unreliable_spans(partner):
            ax.axvspan(s - 0.5, e - 0.5, color=SHADE_C, zorder=0, lw=0)
    for agent, (_, col, ls, lw) in C_AGENTS.items():
        df = data[agent]
        for ax, c in ((ax_b, "agent_belief"), (ax_c, "conc")):
            g = df.groupby("round")[c]
            med = g.median()
            if not (c == "conc" and agent != "trust"):  # a Beta agent's concentration is the same in every run
                ax.fill_between(med.index, g.quantile(0.25), g.quantile(0.75), color=col, alpha=0.17, lw=0, zorder=1)
            ax.plot(med.index, med.values, color=col, ls=ls, lw=lw, zorder=3 if agent == "trust" else 2)
    ax_b.axhline(DECISION_THRESHOLD, color=THRESH_C, ls=":", lw=1.1, zorder=2)
    _strips(ax_o, [_share_image(data[a]) for a in C_AGENTS], left, [C_AGENTS[a][0] for a in C_AGENTS])
    ax_b.set_title(f"({letter})", loc="left", fontsize=C_FS + 0.5, fontweight="bold", color=C_FRAME, pad=2)
    _style(ax_b, left, r"Estimate $\hat p_k$", [0, 1 / 3, 2 / 3, 1], ["0", "1/3", "2/3", "1"], (-0.03, 1.03))
    _style(ax_c, left, C_CONC_LABEL, [0, 6, 12], ["0", "6", "12"], (0, C_CONC_TOP))
    for ax in (ax_b, ax_c):
        ax.tick_params(labelbottom=False)
    return ax_b, ax_c, ax_o


def _zoom(fig, spec, partner, title):
    inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, height_ratios=[2.6, 1.55, 0.95], hspace=0.09)
    ax_b, ax_c, ax_o = (fig.add_subplot(inner[i]) for i in range(3))
    ax_b.axhline(DECISION_THRESHOLD, color=THRESH_C, ls=":", lw=1.1, zorder=2)
    runs, info = {}, []
    for agent, (lab, col, ls, lw) in C_AGENTS.items():
        df = load(partner, agent)
        rid = typical_run(df)
        run = df[df.run_id == rid].sort_values("round")
        runs[agent] = run
        k = run["round"].to_numpy()
        ax_b.plot(k, run.agent_belief, color=col, ls=ls, lw=lw, zorder=4 if agent == "trust" else 3)
        ax_c.plot(k, run.conc, color=col, ls=ls, lw=lw, zorder=4)
        c10 = run.conc.to_numpy()[10:]
        info.append(f"{lab}: run {rid}, payoff {run.agent_payoff.sum():.0f}, failed collaborations "
                    f"{int((_codes(run) == 1).sum())}, largest concentration drop "
                    f"{(np.maximum.accumulate(c10) - c10).max():.2f}")
    imgs = []
    for agent in C_AGENTS:
        rgb = np.array([matplotlib.colors.to_rgba(C_OUTC[c][1]) for c in _codes(runs[agent])])
        imgs.append(np.repeat(rgb[None, :, :], 30, axis=0))
    _strips(ax_o, imgs, True, [C_AGENTS[a][0] for a in C_AGENTS])
    for x in np.arange(-0.5, NUM_ROUNDS, 1):  # thin white lines between rounds
        ax_o.axvline(x, color="white", lw=0.35)
    ax_b.set_title("(e)", loc="left", fontsize=C_FS + 0.5, fontweight="bold", color=C_FRAME, pad=2)
    _style(ax_b, True, r"Estimate $\hat p_k$", [0, 1 / 3, 2 / 3, 1], ["0", "1/3", "2/3", "1"], (-0.03, 1.03))
    _style(ax_c, True, C_CONC_LABEL, [0, 6, 12], ["0", "6", "12"], (0, C_CONC_TOP))
    for ax in (ax_b, ax_c):
        ax.tick_params(labelbottom=False)
    ax_o.set_xlabel("Round", labelpad=1)
    return (ax_b, ax_c, ax_o), info


def _frame_and_connect(fig, src, dst):
    """Frame the source block outside its axes (so no data is covered), frame the zoom, join the two."""
    k = 8.8 / fig.get_figheight()  # offsets were tuned on an 8.8 in canvas; keep them the same in inches
    sb, so = src[0].get_position(), src[2].get_position()
    sx0, sx1, sy1, sy0 = 0.012, sb.x1 + 0.024, sb.y1 + 0.022 * k, so.y0 - 0.026 * k
    db, do = dst[0].get_position(), dst[2].get_position()
    dx0, dx1, dy1, dy0 = 0.012, 0.992, db.y1 + 0.024 * k, do.y0 - 0.036 * k
    for x0, y0, x1, y1 in ((sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1)):
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fill=False, ec=C_FRAME,
                                 lw=1.1, zorder=10))
    for xs, xd in ((sx0, dx0), (sx1, dx1)):
        fig.add_artist(Line2D([xs, xd], [sy0, dy1], transform=fig.transFigure, color=C_FRAME, lw=0.8,
                              ls=(0, (2, 2)), zorder=10))


def _legend(fig, y):
    h = [Line2D([0], [0], color=c, ls=ls, lw=lw, label=C_NAMES[a]) for a, (_, c, ls, lw) in C_AGENTS.items()]
    h += [Patch(facecolor="#9bb7c9", label="interquartile range over 300 runs"),
          Line2D([0], [0], color=THRESH_C, ls=":", lw=1.1, label="threshold u = 2/3"),
          Patch(facecolor=SHADE_C, edgecolor="#cccccc", label="partner below u")]
    h += [Patch(facecolor=C_OUTC[c][1], label=C_OUTC[c][0]) for c in C_STACK]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y), handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, labelspacing=0.35)


def make_combined(out_dir):
    data = {}
    for partner, _ in C_PANELS:
        data[partner] = {a: load(partner, a) for a in AGENTS}
        check_table4(partner, data[partner])
    with plt.rc_context(C_RC):
        fig = plt.figure(figsize=(6.5, C_HEIGHT))
        outer = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2.0, 2.0, 2.15], hspace=0.36 * 8.8 / C_HEIGHT * 0.93,
                                  wspace=0.13, left=0.1, right=0.972, top=1 - 0.22 / C_HEIGHT,
                                  bottom=1.23 / C_HEIGHT)
        blocks = {}
        for k, (partner, title) in enumerate(C_PANELS):
            r, c = divmod(k, 2)
            blocks[partner] = _block(fig, outer[r, c], partner, title, "abcd"[k], data[partner], left=(c == 0))
        zaxes, info = _zoom(fig, outer[2, :], C_ZOOM, dict(C_PANELS)[C_ZOOM])
        _legend(fig, 0.0)
        out = out_dir / "paper_dynamics_combined.png"
        fig.savefig(out, dpi=300, facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
    print("\n".join(info))
    print(f"Saved {out} and {out.with_suffix('.pdf')}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--kind", nargs="+", default=["combined"], choices=["combined", *FIGURES])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind in args.kind:
        if kind == "combined":
            make_combined(args.out_dir)
        else:
            make(kind, args.out_dir)


if __name__ == "__main__":
    main()
