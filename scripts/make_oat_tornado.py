#!/usr/bin/env python3
"""
Tornado small multiples for the one-factor-at-a-time sensitivity study.

One panel per partner. Each row is a parameter; the two bars are the paired effect of its low and high
value on the chosen metric, relative to the submitted baseline, with 95% CI whiskers (300 runs).
Rows keep the same order in every panel (by mean absolute swing over partners), so panels can be read
side by side. Each panel has its own x scale. theta, which is shared with the benchmarks, moves payoff
far more than the trust-specific parameters; where its bar would hide the others it is clipped at the
panel edge and its value printed.

Reads results/oat_sensitivity/effects.csv from run_oat_sensitivity.py.

Usage:
    python scripts/make_oat_tornado.py                 # total payoff
    python scripts/make_oat_tornado.py --metric FC     # failed collaboration (pp)
    python scripts/make_oat_tornado.py --gap --paper   # Figure 2 of the paper

--gap plots the change in the payoff DIFFERENCE between the
trust-sensitive and discounted Bayesian agents (TS - Disc) instead of the trust agent's own payoff change, on one common
x axis for all nine panels (+-10, which holds every bar and 95% CI), low and high value on one row, the tested values
in the row labels of the left column, and the reversal line at minus the baseline gap (an arrow and value where it lies
off the axis). For the trust-only parameters the discounted agent does not change, so the gap change equals the trust
agent's own change; for the shared theta it subtracts the discounted agent's change at the same theta, which absorbs
theta's -20 to -26 bars (both agents lose them). Paired over run ids within an agent; the two agents' streams are
independent, so SE = sqrt(SE_TS^2 + SE_Disc^2). Reads results/oat_sensitivity/per_run.csv. Drawn at 6.5 in, 9 pt.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.config import RESULTS_DIR, FIGURES_DIR
from trust_bayesian_agent_comparison.partners import MANUSCRIPT_PARTNERS

PARTNER_LABELS = {
    "AlwaysCooperate": "Always Collaborate", "AlwaysDefect": "Always Independent", "Random": "Random (p = 0.5)",
    "TitForTat": "TFT", "SuspiciousTFT": "STFT", "SingleCycle": "Single-Cycle",
    "GradualDeterioration": "Gradual Deterioration", "StrategicCheater": "Strategic Cheater",
    "ExpectationViolation": "Expectation Violation",
}
SYMBOLS = {"theta": "θ", "eta_x": "η_x", "gamma_x": "γ_x", "gamma_t": "γ_t", "omega": "ω", "lambda": "λ", "nu": "ν",
           "mu": "μ", "psi": "ψ", "sigma": "σ", "t_max": "t_max/ε",
           "c": "c", "x_0": "x_0", "t_0": "t_0"}
METRIC_TITLES = {"Pi": "total payoff", "FC": "failed collaboration (pp)", "SC": "successful collaboration (pp)"}
# Ordinal pair from one hue: light = low value, dark = high value
LOW, HIGH = "#86b6ef", "#1c5cab"
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BENCH_MARK, FLIP = "#b5651d", "#8a1c1c"
REF_BENCH = "disc_pp_rho0.90"
REF_BENCH_LABEL = "DiscPP (ρ = 0.9)"
SHARED_PARAMS = ("theta",)
PAPER_PARAMS = ("theta", "t_max", "nu", "gamma_x", "c")
PAPER_SYMBOLS = {"theta": r"$\theta$", "t_max": r"$t_{\max}/\varepsilon$", "nu": r"$\nu$", "gamma_x": r"$\gamma_x$",
                 "c": r"$c$", "eta_x": r"$\eta_x$", "gamma_t": r"$\gamma_t$", "omega": r"$\omega$",
                 "sigma": r"$\sigma$", "x_0": r"$x_0$", "t_0": r"$t_0$"}


def fmt(v):
    return "-" if pd.isna(v) else f"{v:.3g}"


def benchmark_theta_effects(path: Path, metric: str) -> pd.DataFrame | None:
    """
    The reference benchmark's OWN low/high effect on the shared parameter theta, so the theta row can
    show both agents. Returns partner -> (low_effect, high_effect), or None if the file is absent.
    Only payoff is stored per theta, so the overlay is drawn for the Pi figure only.
    """
    if metric != "Pi" or not path.exists():
        return None
    t = pd.read_csv(path)
    t = t[t.benchmark == REF_BENCH]
    piv = t.pivot_table(index="partner", columns="theta", values="bench_Pi")
    lo_t, base_t, hi_t = 1.0, 2.0, 3.0
    if not {lo_t, base_t, hi_t}.issubset(piv.columns):
        return None
    return pd.DataFrame({"low_effect": piv[lo_t] - piv[base_t], "high_effect": piv[hi_t] - piv[base_t]})


def ordering_gap(path: Path) -> pd.Series | None:
    """
    Baseline trust-minus-benchmark payoff gap per partner. A parameter swing larger than -gap would
    reverse the ordering of the two agents.
    """
    if not path.exists():
        return None
    o = pd.read_csv(path)
    b = o[(o.variant == "baseline") & (o.benchmark == REF_BENCH)]
    return b.set_index("partner")["dPi"] if len(b) else None


GAP_LIM = 10.5
GAP_FS = 9


def gap_effects(per_run: Path, params) -> tuple[pd.DataFrame, pd.Series]:
    """Change in the TS - Disc payoff gap for each parameter, level and partner, with its SE; and the baseline gap.
    TS settings are paired with the TS baseline by run id; for theta, the discounted agent's run at the same theta is
    paired with its own run at the default theta = 2."""
    pr = pd.read_csv(per_run)
    ts = pr[pr.agent == "trust"]
    ds = pr[pr.agent == REF_BENCH]
    base = ts[ts.variant == "baseline"].set_index(["partner", "run_id"]).Pi
    dbase = ds[ds.variant == "theta_2"].set_index(["partner", "run_id"]).Pi
    gap0 = base.groupby("partner").mean() - dbase.groupby("partner").mean()
    meta = pd.read_csv(per_run.parent / "effects.csv")
    rows = []
    for par in params:
        for lev in ("low", "high"):
            d = ts[ts.variant == f"{par}_{lev}"].set_index(["partner", "run_id"]).Pi - base
            assert d.notna().all() and len(d) == len(base), (par, lev)
            dd = None
            if par in SHARED_PARAMS:
                th = float(meta[(meta.parameter == par)].iloc[0][f"{lev}_value"])
                dd = ds[ds.variant == f"theta_{th:g}"].set_index(["partner", "run_id"]).Pi - dbase
                assert dd.notna().all() and len(dd) == len(dbase), (par, lev)
            for partner, g in d.groupby("partner"):
                m, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
                if dd is not None:
                    h = dd.loc[partner]
                    m -= h.mean()
                    se = np.sqrt(se ** 2 + (h.std(ddof=1) / np.sqrt(len(h))) ** 2)
                rows.append((par, lev, partner, m, se))
    out = pd.DataFrame(rows, columns=["parameter", "level", "partner", "effect", "se"])
    # the trust-only rows must equal the trust agent's own effect in effects.csv
    own = meta[meta.metric == "Pi"].set_index(["parameter", "partner"])
    for _, r in out[~out.parameter.isin(SHARED_PARAMS)].iterrows():
        assert abs(own.loc[(r.parameter, r.partner), f"{r.level}_effect"] - r.effect) < 1e-4  # effects.csv is rounded to 5 decimals
    return out, gap0


def make_gap_figure(args):
    params = list(PAPER_PARAMS) if args.params is None else args.params
    eff, gap0 = gap_effects(args.input.parent / "per_run.csv", params)
    meta = pd.read_csv(args.input)
    vals = {p: meta[meta.parameter == p].iloc[0] for p in params}
    ext = (eff.effect.abs() + 1.96 * eff.se).max()
    assert ext < GAP_LIM, f"a bar or interval reaches {ext:.2f}, beyond the common axis"
    rc = {"font.size": GAP_FS, "axes.labelsize": GAP_FS, "xtick.labelsize": GAP_FS, "ytick.labelsize": GAP_FS,
          "font.family": "DejaVu Sans"}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 3, figsize=(6.5, 6.0), sharex=True, sharey=True)
        y = np.arange(len(params))
        for k, (ax, partner) in enumerate(zip(axes.flat, MANUSCRIPT_PARTNERS)):
            sub = eff[eff.partner == partner].set_index(["parameter", "level"])
            ax.axvline(0, color=AXIS, linewidth=0.9, zorder=1)
            for level, color, height, dy in (("low", LOW, 0.66, -0.2), ("high", HIGH, 0.34, 0.0)):
                v = np.array([sub.loc[(p, level), "effect"] for p in params])
                se = np.array([sub.loc[(p, level), "se"] for p in params])
                ax.barh(y, v, height=height, color=color, edgecolor="none", zorder=2 if level == "low" else 3)
                ax.errorbar(v, y + dy, xerr=1.96 * se, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.4,
                            zorder=4)
            flip = -float(gap0[partner])
            if abs(flip) <= GAP_LIM:
                ax.axvline(flip, color=FLIP, linewidth=1.0, linestyle=(0, (3, 2)), zorder=5)
            else:  # off the common axis: an arrow at the edge, labelled, so the panel explains itself
                xa = np.sign(flip) * GAP_LIM
                ax.annotate("reversal at " + f"{flip:+.1f}".replace("-", "−"), xy=(xa, len(params) - 0.45),
                            xytext=(xa - np.sign(flip) * 1.1, len(params) - 0.45), color=FLIP, fontsize=GAP_FS,
                            ha="left" if flip < 0 else "right", va="center",
                            arrowprops=dict(arrowstyle="-|>", color=FLIP, lw=0.9, shrinkA=0, shrinkB=0),
                            annotation_clip=False)
            ax.set_xlim(-GAP_LIM, GAP_LIM)
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.set_xticklabels(["−10", "−5", "0", "5", "10"])
            ax.set_ylim(len(params) - 0.45 + (0.55 if abs(flip) > GAP_LIM else 0), -0.6)
            ax.set_yticks(y)
            # symbols only: the tested values are given once, in the legend entries for low and high
            ax.set_yticklabels([PAPER_SYMBOLS.get(p, p) for p in params])
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", length=2, pad=1.5, colors=SECONDARY)
            ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            # the TS - Disc difference at the default parameters, in parentheses (defined once, below the legend)
            ax.set_title(f"{PARTNER_LABELS[partner].replace('Random (p = 0.5)', 'Random-(p)')} "
                         "(" + f"{float(gap0[partner]):+.1f}".replace("-", "−") + ")", fontsize=GAP_FS, color=INK,
                         loc="left", pad=2)
        for ax in axes[-1]:
            ax.set_xlabel("change in TS − Disc payoff", color=SECONDARY, labelpad=1)
        def tested(level):
            return ", ".join(f"{PAPER_SYMBOLS.get(p, p)} = {fmt(vals[p][level + '_value'])}" for p in params)
        handles = [Patch(color=LOW, label=f"low value ({tested('low')})"),
                   Patch(color=HIGH, label=f"high value ({tested('high')})"),
                   Line2D([], [], color=INK, lw=0.7, marker="|", markersize=5, label="95% confidence interval"),
                   Line2D([], [], color=FLIP, linewidth=1.0, linestyle=(0, (3, 2)),
                          label="change that would reverse the ordering of TS and Disc (arrow if off the scale)")]
        fig.legend(handles=handles, loc="lower left", ncol=1, frameon=False, bbox_to_anchor=(0.01, 0.03),
                   handlelength=1.6, handletextpad=0.5, labelspacing=0.3)
        fig.text(0.01, 0.006, "In parentheses after each partner: the TS − Disc payoff difference at the default "
                 "parameters.", ha="left", va="bottom", color=SECONDARY)
        fig.tight_layout(rect=(0, 0.175, 1, 1), h_pad=0.9, w_pad=1.2)
        out = args.output or FIGURES_DIR / ("paper_oat_tornado_gap.png" if args.paper else "oat_tornado_gap.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, facecolor="white")
        fig.savefig(out.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
    big = eff.loc[eff.effect.abs().idxmax()]
    th = eff[eff.parameter == "theta"].effect.abs().max() if "theta" in params else float("nan")
    print(f"largest gap change {big.effect:+.2f} ({big.parameter} {big.level}, {big.partner}); theta at most {th:.2f}; "
          f"largest bar + 1.96 SE {ext:.2f}")
    print(f"Saved {out} and {out.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", default="Pi", choices=list(METRIC_TITLES))
    parser.add_argument("--input", type=Path, default=RESULTS_DIR / "oat_sensitivity" / "effects.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--params", nargs="+", default=None,
                        help="parameters to show (default: all; with --paper: " + " ".join(PAPER_PARAMS) + ")")
    parser.add_argument("--gap", action="store_true")
    args = parser.parse_args()
    if args.gap:
        return make_gap_figure(args)
    global REF_BENCH_LABEL
    if args.paper:
        REF_BENCH_LABEL = "discounted Bayesian agent (ρ = 0.9)"

    eff = pd.read_csv(args.input)
    e = eff[eff.metric == args.metric].copy()
    bench_theta = benchmark_theta_effects(args.input.parent / "theta_benchmarks.csv", args.metric)
    gaps = ordering_gap(args.input.parent / "ordering_robustness.csv") if args.metric == "Pi" else None
    order = (e.assign(a=e.swing.abs()).groupby("parameter").a.mean().sort_values(ascending=False).index.tolist())
    keep = args.params if args.params is not None else (list(PAPER_PARAMS) if args.paper else ["all"])
    if keep != ["all"]:
        missing = set(keep) - set(order)
        if missing:
            raise SystemExit(f"unknown parameters: {sorted(missing)}")
        order = [p for p in order if p in keep]
    labels = {}
    for p in order:
        r = e[e.parameter == p].iloc[0]
        sym = PAPER_SYMBOLS.get(p, SYMBOLS.get(p, p)) if args.paper else SYMBOLS.get(p, p)
        labels[p] = f"{sym}  ({fmt(r.low_value)} | {fmt(r.high_value)})"

    if args.paper:
        fig, axes = plt.subplots(3, 3, figsize=(10.0, 2.6 + 0.78 * len(order)))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12.5 if len(order) >= 11 else 3.0 + 0.95 * len(order)))
    y = np.arange(len(order))
    for ax, partner in zip(axes.flat, MANUSCRIPT_PARTNERS):
        sub = e[e.partner == partner].set_index("parameter").loc[order]
        # x range from every parameter except theta, so theta cannot flatten the rest
        others = sub.drop(index="theta", errors="ignore")[["low_lo", "low_hi", "high_lo", "high_hi"]].abs().to_numpy()
        lim = max(np.nanmax(others) * 1.25, 0.5)
        for offset, level, color in ((-0.19, "low", LOW), (0.19, "high", HIGH)):
            vals = sub[f"{level}_effect"].to_numpy()
            lo, hi = sub[f"{level}_lo"].to_numpy(), sub[f"{level}_hi"].to_numpy()
            shown = np.clip(vals, -lim, lim)
            ax.barh(y + offset, shown, height=0.36, color=color, edgecolor="none", zorder=2)
            ok = np.isfinite(vals) & (np.abs(vals) <= lim)
            ax.errorbar(vals[ok], (y + offset)[ok], xerr=[vals[ok] - lo[ok], hi[ok] - vals[ok]], fmt="none",
                        ecolor=SECONDARY, elinewidth=0.9, capsize=1.8, zorder=3)
            for yi, v in zip(y + offset, vals):
                if np.isfinite(v) and abs(v) > lim:
                    ax.text(np.sign(v) * lim * 0.98, yi, f"{v:+.1f}", ha="right" if v > 0 else "left",
                            va="center", fontsize=7, color="white", fontweight="bold", zorder=4)
        if bench_theta is not None and partner in bench_theta.index and "theta" in order:
            yt = order.index("theta")
            for offset, level in ((-0.19, "low"), (0.19, "high")):
                bv = float(bench_theta.loc[partner, f"{level}_effect"])
                ax.plot(np.clip(bv, -lim, lim), yt + offset, marker="o", markersize=5.5,
                        markerfacecolor="none", markeredgecolor=BENCH_MARK, markeredgewidth=1.5,
                        linestyle="none", zorder=5)

        if gaps is not None and partner in gaps.index:
            flip = -float(gaps[partner])
            if abs(flip) <= lim:
                ax.axvline(flip, color=FLIP, linewidth=1.1, linestyle=(0, (4, 2)), zorder=4)
            elif args.paper:
                pass
            else:
                ax.text(0.985 if flip > 0 else 0.015, 0.03,
                        f"ordering flips at {flip:+.0f} (off scale)", transform=ax.transAxes,
                        ha="right" if flip > 0 else "left", va="bottom", fontsize=6.8, color=FLIP)

        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[p] for p in order], fontsize=7.5, color=SECONDARY)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
        ax.tick_params(axis="x", labelsize=7.5, colors=MUTED, length=0)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        base = sub["baseline_mean"].iloc[0]
        if args.paper:
            ax.set_title(f"{PARTNER_LABELS[partner]} (baseline {base:.1f})", fontsize=8.5, color=INK, loc="left")
        else:
            ax.set_title(f"{PARTNER_LABELS[partner]}   (baseline {base:.1f})", fontsize=9.5, color=INK, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel(f"change in {METRIC_TITLES[args.metric]} vs baseline", fontsize=8.5, color=SECONDARY)

    handles = [Patch(color=LOW, label="low value"), Patch(color=HIGH, label="high value")]
    if bench_theta is not None:
        handles.append(Line2D([], [], marker="o", markersize=5.5, markerfacecolor="none",
                              markeredgecolor=BENCH_MARK, markeredgewidth=1.5, linestyle="none",
                              label=f"{REF_BENCH_LABEL}, same θ"))
    if gaps is not None:
        handles.append(Line2D([], [], color=FLIP, linewidth=1.1, linestyle=(0, (4, 2)),
                              label=f"swing that reverses the ordering vs {REF_BENCH_LABEL}"))
    fig.legend(handles=handles, loc="upper right", ncol=2 if args.paper else len(handles), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.99, 0.995))
    if not args.paper:
        fig.suptitle(f"One-factor-at-a-time sensitivity of the trust agent: {METRIC_TITLES[args.metric]}",
                     fontsize=12, x=0.01, ha="left", y=0.995, color=INK)
    note = ("Paired effect of each parameter's low and high value (in parentheses) vs the revised-model "
            "baseline, 300 runs, whiskers = 95% CI. Rows ordered by mean absolute swing over partners. Each "
            "panel has its own scale;\nbars that run off the panel are printed with their value (θ, shared with "
            "the benchmarks). Effective memory halves/doubles for γ; ±50% for scale parameters; t_0 one-sided.")
    if bench_theta is not None or gaps is not None:
        # The blanket claim "no parameter reaches the line" is false (t_max and gamma_x cross it on the
        # near-tie partners), so the caption states what ordering_robustness.csv actually shows.
        note += ("\nθ is the only parameter the benchmarks share, so its row also carries the benchmark's own "
                 "effect at the same θ; the other parameters do not exist in a Beta model. The dashed line marks "
                 "the swing that would reverse\nthe two agents' ordering on that partner. It is crossed only where "
                 "the baseline gap is 1 to 3 points; on Strategic Cheater and Expectation Violation, which carry "
                 "the result, nothing comes near it. Of the 189\ntrust-vs-DiscPP cells, 26 change the significance "
                 "verdict, 8 of them the ceiling t_max and 6 the shared θ.")
    if not args.paper:
        fig.text(0.01, 0.003, note, fontsize=7.5, color=SECONDARY)
    top = 1 - (0.62 if args.paper else 0.35) / fig.get_figheight()
    fig.tight_layout(rect=(0, 0.0 if args.paper else 0.062, 1, top), h_pad=2.2, w_pad=2.0)
    out = args.output or FIGURES_DIR / (f"paper_oat_tornado_{args.metric}.png" if args.paper
                                        else f"oat_tornado_{args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="#fcfcfb")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
