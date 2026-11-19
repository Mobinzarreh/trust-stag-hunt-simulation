#!/usr/bin/env python3
"""
Metric panel: every behavioral and outcome metric for the four agents.

Scores the trust-sensitive agent and the three Beta benchmarks (cumulative, prior-preserving
discounted rho = 0.9, window w = 10) on all metric families in trust_bayesian_agent_comparison/analysis/
behavioral_metrics.py (outcome layer, M1-M3, M5-M7, the Axelrod profile), with 95% CIs and
trust-minus-benchmark differences. Two configurations:

    current       the revised partners (expectation-gated Strategic Cheater, 100-round deterioration),
                  read from the main runs in results/benchmarks/ (run_benchmark_comparison.py), plus the
                  diagnostic partners, simulated here: BreakRepair, ReliabilityRamp and DistrustLock (the
                  submitted Strategic Cheater rule). This is the configuration the paper reports.
    submitted_v1  the submitted partner configuration. Only Strategic Cheater and Gradual
                  Deterioration differ; it is scored only where the submitted runs are present

The seven partners that did not change since submission share their runs (and so their metrics)
between the two configurations. The controlled-shock harness (M1) runs once per agent.

Writes results/metric_panel/: cells_long.csv, diffs_long.csv, runs_long.csv, shock.csv,
schedule_pooled.csv, scoreboard.csv, redundancy.csv, catalog.csv and panel_report.txt (every
metric as a partner x agent table). New runs are cached under results/metric_panel/runs/.

The tables, scoreboard and redundancy check cover the economic metrics plus the panel metrics
(PANEL_METRICS plus A_p); --all restores every metric.

Usage:
    python scripts/run_metric_panel.py               # reuse cached runs
    python scripts/run_metric_panel.py --overwrite   # regenerate the runs simulated here
    python scripts/run_metric_panel.py --boot 50     # fewer bootstrap replicates (faster)
    python scripts/run_metric_panel.py --all         # report every metric, dropped ones included
    python scripts/run_metric_panel.py --recency plain   # the same panel with the baseline-discounting variant
                                                         # (results/metric_panel_plainDisc/)
"""

import argparse
import zlib
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

from trust_bayesian_agent_comparison.agents import (
    FocalAgent,
    BayesianFocalAgent,
    PriorPreservingDiscountedBetaAgent,
    DiscountedBetaAgent,
    SlidingWindowBetaAgent,
)
from trust_bayesian_agent_comparison.analysis import run_cell, run_cells
from trust_bayesian_agent_comparison.analysis.behavioral_metrics import (
    CATALOG,
    SHOCK_CATALOG,
    N_BOOT,
    ECONOMIC_METRICS,
    PANEL_METRICS,
    PANEL_SHOCK_METRICS,
    catalog_frame,
    compare_cells,
    concat_traces,
    make_trace,
    pooled,
    score_cell,
    shock_profile,
)
from trust_bayesian_agent_comparison.config import (
    DISCOUNTED_BETA_RHO,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    MC_BASE_SEED,
    N_JOBS,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    RESULTS_DIR,
    SLIDING_WINDOW_W,
)
from trust_bayesian_agent_comparison.partners import (
    COLLABORATIVE_PARTNERS,
    DIAGNOSTIC_PARTNERS,
    MANUSCRIPT_PARTNERS,
    PARTNER_CATEGORY,
    partner_factory,
)

OUT = RESULTS_DIR / "metric_panel"
BENCH = RESULTS_DIR / "benchmarks"

COLLAB = False
COLLAB_RUNS = RESULTS_DIR / "collaborative_partners" / "runs"
COLLAB_FILE = {"trust": "trust", "cumulative": "cumulative", "discpp0.9": "discpp", "win10": "window",
               "disc0.9": "disc_plain"}
COLLAB_REFERENCE = ("TitForTat", "SuspiciousTFT")
TITLE = "METRIC PANEL (four agents, main runs + diagnostic partners)"

# key -> (label, factory, RNG slot, B1 results subdirectory). Slots follow B1: the trust agent is
# agent1, every benchmark agent2 (common random numbers across benchmarks).
AGENTS = {
    "trust": ("Trust", FocalAgent, 1, "cumulative"),
    "cumulative": ("Cumulative", BayesianFocalAgent, 2, "cumulative"),
    "discpp0.9": (RECENCY_BENCHMARK_LABEL, partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO), 2,
                  RECENCY_BENCHMARK_KEY),
    "win10": (f"Window w={SLIDING_WINDOW_W}", partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W), 2,
              f"window_w{SLIDING_WINDOW_W}"),
}
BENCHMARKS = [k for k in AGENTS if k != "trust"]
ADAPTIVE = ("discpp0.9", "win10")   # recency-matched benchmarks (cumulative is the non-adaptive reference)

CHANGED_SINCE_SUBMISSION = ("GradualDeterioration", "StrategicCheater")
CONFIG_PARTNERS = {
    "current": MANUSCRIPT_PARTNERS + DIAGNOSTIC_PARTNERS,
    "submitted_v1": MANUSCRIPT_PARTNERS,
}
SCOPES = {
    "nonclock": lambda p: p not in ("AlwaysCooperate", "AlwaysDefect"),
    "exploiters": lambda p: p in MANUSCRIPT_PARTNERS and PARTNER_CATEGORY[p] == "non-stationary",
    "breakrepair": lambda p: p == "BreakRepair",
    "ramp": lambda p: p == "ReliabilityRamp",
}
TRAJ_CHECK = ["agent_belief", "agent_action", "partner_action", "agent_payoff"]
REPORTED = set(ECONOMIC_METRICS) | set(PANEL_METRICS)


def reported_catalog() -> dict:
    """CATALOG restricted to the reported metrics, in catalog order."""
    return {m: v for m, v in CATALOG.items() if m in REPORTED}


# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------
def run_source(config: str, partner: str, agent: str) -> Path:
    """CSV holding the runs of one cell (existing B1/submitted runs, or runs simulated here)."""
    if COLLAB:
        key = "trust_v1" if (agent == "trust" and config.startswith("submitted_v1")) else COLLAB_FILE[agent]
        return COLLAB_RUNS / f"{partner}__{key}.csv"
    _, _, slot, bench_dir = AGENTS[agent]
    if partner in DIAGNOSTIC_PARTNERS:
        return OUT / "runs" / "current" / f"{partner}_{agent}.csv"
    if config == "submitted_v1" and partner in CHANGED_SINCE_SUBMISSION:
        submitted = RESULTS_DIR / f"{partner}_agent{slot}.csv"
        if agent in ("trust", "cumulative") and submitted.exists():
            return submitted
        return OUT / "runs" / "submitted_v1" / f"{partner}_{agent}.csv"
    return BENCH / bench_dir / f"{partner}_agent{slot}.csv"


def cell_agent_factory(config: str, agent: str):
    factory = AGENTS[agent][1]
    return partial(FocalAgent, submitted_v1=True) if (config.startswith("submitted_v1") and agent == "trust") else factory


def cell_partner_factory(config: str, partner: str):
    return partner_factory(partner, NUM_ROUNDS, submitted_v1=(config == "submitted_v1" and not COLLAB))


def simulate_missing(overwrite: bool, n_jobs: int):
    """Simulate the cells this script owns (diagnostic partners, submitted adaptive benchmarks)."""
    todo = {}
    for config, partners in CONFIG_PARTNERS.items():
        for p in partners:
            for agent, (_, factory, slot, _) in AGENTS.items():
                path = run_source(config, p, agent)
                if OUT not in path.parents or (path.exists() and not overwrite) or path in todo:
                    continue
                todo[path] = dict(agent_factory=cell_agent_factory(config, agent), partner_factory=cell_partner_factory(config, p),
                                  slot=slot, num_runs=NUM_MONTE_CARLO_RUNS, base_seed=MC_BASE_SEED,
                                  num_rounds=NUM_ROUNDS, return_trajectories=True)
    if not todo:
        return
    print(f"Simulating {len(todo)} cells ({NUM_MONTE_CARLO_RUNS} runs each)...")
    results = run_cells({str(k): v for k, v in todo.items()}, n_jobs=n_jobs)
    for path, (_, traj) in results.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        traj.to_csv(path, index=False)


def check_sources(n_check: int = 5):
    """The files this script reads must be what it says they are (same seeds, slots and partners)."""
    if COLLAB:
        for config, partners in CONFIG_PARTNERS.items():
            for p in partners:
                for agent, (_, _, slot, _) in AGENTS.items():
                    _, fresh = run_cell(cell_agent_factory(config, agent), cell_partner_factory(config, p), slot,
                                        num_runs=n_check, return_trajectories=True)
                    cached = pd.read_csv(run_source(config, p, agent))
                    cached = cached[cached["run_id"] < n_check]
                    if not np.allclose(fresh[TRAJ_CHECK].to_numpy(float), cached[TRAJ_CHECK].to_numpy(float)):
                        raise AssertionError(f"{run_source(config, p, agent)} does not reproduce ({config}, {p}, {agent})")
        print("Run sources check: the stored collaborative runs reproduce from their seeds.")
        return
    for config, p, agent in [("current", "StrategicCheater", "trust"), ("current", "GradualDeterioration", "cumulative"),
                             ("submitted_v1", "StrategicCheater", "trust"),
                             ("submitted_v1", "GradualDeterioration", "trust"),
                             ("submitted_v1", "StrategicCheater", "cumulative")]:
        _, _, slot, _ = AGENTS[agent]
        factory = cell_agent_factory(config, agent)
        _, fresh = run_cell(factory, cell_partner_factory(config, p), slot, num_runs=n_check,
                            return_trajectories=True)
        cached = pd.read_csv(run_source(config, p, agent))
        cached = cached[cached["run_id"] < n_check]
        if not np.allclose(fresh[TRAJ_CHECK].to_numpy(float), cached[TRAJ_CHECK].to_numpy(float)):
            raise AssertionError(f"{run_source(config, p, agent)} does not reproduce ({config}, {p}, {agent})")
    print("Run sources check: cached current and submitted runs reproduce from their seeds.")


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------
def score_one(config: str, partner: str, agent: str, n_boot: int):
    df = pd.read_csv(run_source(config, partner, agent))
    tr = make_trace(df, cell_partner_factory(config, partner)())
    seed = zlib.crc32(f"{config}|{partner}|{agent}".encode())
    return score_cell(tr, n_boot=n_boot, seed=seed)


def score_all(n_boot: int, n_jobs: int) -> dict:
    """{(config, partner, agent): CellResult}. Unchanged partners are scored once and shared."""
    keys = [("current", p, a) for p in CONFIG_PARTNERS["current"] for a in AGENTS]
    if COLLAB:
        keys = [(c, p, a) for c, ps in CONFIG_PARTNERS.items() for p in ps for a in AGENTS]
    else:
        keys += [("submitted_v1", p, a) for p in CHANGED_SINCE_SUBMISSION for a in AGENTS]
    print(f"Scoring {len(keys)} cells ({n_boot} bootstrap replicates for pooled metrics)...")
    res = Parallel(n_jobs=n_jobs)(delayed(score_one)(c, p, a, n_boot) for c, p, a in keys)
    cells = dict(zip(keys, res))
    if not COLLAB:
        for p in CONFIG_PARTNERS["submitted_v1"]:
            if p not in CHANGED_SINCE_SUBMISSION:
                for a in AGENTS:
                    cells[("submitted_v1", p, a)] = cells[("current", p, a)]
    return cells


def long_tables(cells: dict):
    rows, diffs, runs = [], [], []
    for (config, p, a), cr in cells.items():
        s = cr.summary.assign(config=config, partner=p, agent=a)
        rows.append(s)
        runs.append(cr.runs.assign(config=config, partner=p, agent=a, run_id=np.arange(len(cr.runs))))
        if a == "trust":
            for b in BENCHMARKS:
                d = compare_cells(cr, cells[(config, p, b)]).assign(config=config, partner=p, benchmark=b)
                diffs.append(d)
    return pd.concat(rows, ignore_index=True), pd.concat(diffs, ignore_index=True), pd.concat(runs, ignore_index=True)


def scoreboard(cells_long: pd.DataFrame, diffs_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per config and metric: on each partner in scope where all four agents have a value, W if the
    trust agent is significantly better than all three benchmarks (95% CI), L if significantly worse
    than at least one, ~ otherwise; and the trust agent's rank (1 = best of four). For signature
    metrics 'better' means showing the predicted human pattern more strongly.
    """
    out = []
    for config, partners in CONFIG_PARTNERS.items():
        for metric, (fam, label, direction, kind, scope, status) in reported_catalog().items():
            if direction == 0:
                continue
            marks, marks2, ranks, wins, losses = [], [], [], [], []
            for p in partners:
                if scope in SCOPES and not SCOPES[scope](p):
                    continue
                v = cells_long[(cells_long.config == config) & (cells_long.partner == p)
                               & (cells_long.metric == metric)].set_index("agent")["value"]
                if len(v.reindex(list(AGENTS)).dropna()) < len(AGENTS):
                    continue
                d = diffs_long[(diffs_long.config == config) & (diffs_long.partner == p)
                               & (diffs_long.metric == metric)].set_index("benchmark")
                if len(d) < len(BENCHMARKS):
                    continue
                better = [(direction * d.loc[b, "lo"] > 0) if direction > 0 else (d.loc[b, "hi"] < 0)
                          for b in BENCHMARKS]
                worse = [(d.loc[b, "hi"] < 0) if direction > 0 else (d.loc[b, "lo"] > 0) for b in BENCHMARKS]
                mark = "W" if all(better) else ("L" if any(worse) else "~")
                marks.append(mark)
                # same mark against the two adaptive (recency-matched) benchmarks only
                ad = [BENCHMARKS.index(b) for b in ADAPTIVE]
                marks2.append("W" if all(better[i] for i in ad) else ("L" if any(worse[i] for i in ad) else "~"))
                ranks.append(int((direction * v > direction * v["trust"]).sum()) + 1)
                (wins if mark == "W" else losses if mark == "L" else []).append(p)
            if not marks:
                continue
            out.append(dict(config=config, metric=metric, family=fam, kind=kind, status=status,
                            n_partners=len(marks), W=marks.count("W"), L=marks.count("L"),
                            neutral=marks.count("~"), W_adaptive=marks2.count("W"), L_adaptive=marks2.count("L"),
                            trust_first=ranks.count(1),
                            mean_rank=float(np.mean(ranks)), wins=", ".join(wins), losses=", ".join(losses)))
    return pd.DataFrame(out)


def redundancy(cells_long: pd.DataFrame, config: str = "current") -> pd.DataFrame:
    """
    Pairwise redundancy of the per-cell metrics: Spearman correlation across all agent x partner
    cells, and the mean Kendall tau between the two metrics' orderings of the four agents within a
    partner (|tau| near 1 = the two metrics always rank the agents the same way, or exactly reversed).
    """
    w = (cells_long[cells_long.config == config]
         .pivot_table(index=["partner", "agent"], columns="metric", values="value"))
    metrics = [m for m in reported_catalog() if m in w.columns and m not in ("HB", "t_mean")]
    rows = []
    for m1, m2 in combinations(metrics, 2):
        both = w[[m1, m2]].dropna()
        if len(both) < 12:
            continue
        taus = []
        for _, g in both.groupby(level="partner"):
            if len(g) == len(AGENTS) and g[m1].nunique() > 1 and g[m2].nunique() > 1:
                taus.append(kendalltau(g[m1], g[m2]).statistic)
        rows.append(dict(m1=m1, m2=m2, cells=len(both), spearman_cells=both[m1].corr(both[m2], method="spearman"),
                         partners=len(taus), mean_tau=float(np.mean(taus)) if taus else np.nan))
    r = pd.DataFrame(rows)
    r["abs_tau"] = r["mean_tau"].abs()
    return r.sort_values("abs_tau", ascending=False)


def pooled_schedule(cells_by_agent_traces: dict) -> pd.DataFrame:
    """M7 schedule pooled over the nine manuscript partners (current configuration), per agent."""
    rows = []
    for agent, tr in cells_by_agent_traces.items():
        res = pooled(tr)
        rows.append(dict(agent=agent, **{k: res[k] for k in ("sched_slope", "sched_offset", "sched_q0", "sched_q1")}))
    return pd.DataFrame(rows)


def schedule_traces() -> dict:
    traces = {}
    for agent in AGENTS:
        trs = [make_trace(pd.read_csv(run_source("current", p, agent)), cell_partner_factory("current", p)())
               for p in MANUSCRIPT_PARTNERS]
        traces[agent] = concat_traces(trs)
    return traces


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def fmt(v, metric):
    if not np.isfinite(v):
        return "      ."
    if metric in ("Pi", "HB", "partner_pay", "welfare", "U_ba", "CVaR10", "exp_loss"):
        return f"{v:7.1f}"
    if abs(v) >= 10:
        return f"{v:7.1f}"
    return f"{v:7.3f}"


def report(cells_long, diffs_long, board, shock, sched, red, path: Path):
    lines = []
    add = lines.append
    add(TITLE)
    add("Marks after the trust value: W = significantly better than all three benchmarks, "
        "L = significantly worse than at least one, ~ = neither (95% CIs). Descriptive metrics carry no mark.\n")
    labels = {a: AGENTS[a][0] for a in AGENTS}
    for config, partners in CONFIG_PARTNERS.items():
        add("=" * 100)
        add(f"CONFIGURATION: {config}")
        add("=" * 100)
        c = cells_long[cells_long.config == config]
        d = diffs_long[diffs_long.config == config]
        for metric, (fam, label, direction, kind, scope, status) in reported_catalog().items():
            sub = c[c.metric == metric]
            if sub.empty:
                continue
            if config == "submitted_v1" and not COLLAB:
                sub = sub[sub.partner.isin(CHANGED_SINCE_SUBMISSION)]
                if sub.empty:
                    continue
            add(f"\n[{fam}] {metric}: {label}   (direction {direction:+d}, {kind}, scope {scope}, {status})")
            add(f"{'partner':<22}" + "".join(f"{labels[a]:>15}" for a in AGENTS))
            for p in partners:
                v = sub[sub.partner == p].set_index("agent")["value"].reindex(list(AGENTS))
                if v.isna().all():
                    continue
                mark = ""
                if direction != 0 and v.notna().all():
                    dd = d[(d.partner == p) & (d.metric == metric)].set_index("benchmark")
                    if len(dd) == len(BENCHMARKS):
                        better = [(dd.loc[b, "lo"] > 0) if direction > 0 else (dd.loc[b, "hi"] < 0) for b in BENCHMARKS]
                        worse = [(dd.loc[b, "hi"] < 0) if direction > 0 else (dd.loc[b, "lo"] > 0) for b in BENCHMARKS]
                        mark = "W" if all(better) else ("L" if any(worse) else "~")
                cells = [fmt(v[a], metric) + (f" {mark}" if a == "trust" else "  ") for a in AGENTS]
                add(f"{p:<22}" + "".join(f"{x:>15}" for x in cells))
    add("\n" + "=" * 100)
    add("SCOREBOARD (trust agent, per metric)")
    add("=" * 100)
    add(board.to_string(index=False))
    add("\n" + "=" * 100)
    add("M1 CONTROLLED-SHOCK HARNESS (identical evidence, 100 seeds; trust noise only)")
    add("=" * 100)
    add(shock.to_string())
    if sched is not None:
        add("\nM7 schedule pooled over the nine manuscript partners (current):")
        add(sched.to_string(index=False))
    add("\n" + "=" * 100)
    add("REDUNDANCY: metric pairs that order the four agents the same way within partners (|mean tau| >= 0.8)")
    add("=" * 100)
    add(red[(red.abs_tau >= 0.8) & (red.partners >= 5)].to_string(index=False))
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="Regenerate the runs simulated by this script")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="Bootstrap replicates for pooled metrics")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--all", action="store_true", help="Report every metric, including the dropped ones")
    ap.add_argument("--recency", choices=("prior", "plain"), default="prior",
                    help="Recency benchmark: prior-preserving DiscPP (default) or the plain discounted model")
    ap.add_argument("--partners", choices=("manuscript", "collaborative"), default="manuscript",
                    help="manuscript (default): the nine plus diagnostic partners; collaborative: the five co-adaptive "
                         "partners and two reference rows, read from results/collaborative_partners/")
    args = ap.parse_args()
    if args.recency == "plain":
        global OUT, ADAPTIVE
        swapped = {("disc0.9" if k == "discpp0.9" else k):
                   ((f"Disc rho={DISCOUNTED_BETA_RHO:g}", partial(DiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO), 2,
                     f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}") if k == "discpp0.9" else v)
                   for k, v in AGENTS.items()}
        AGENTS.clear()
        AGENTS.update(swapped)
        BENCHMARKS[:] = [k for k in AGENTS if k != "trust"]
        ADAPTIVE = ("disc0.9", "win10")
        OUT = RESULTS_DIR / "metric_panel_plainDisc"
    if args.partners == "collaborative":
        global COLLAB, CONFIG_PARTNERS, TITLE
        COLLAB = True
        # submitted_v1 = the submitted-paper trust agent against the same partners (stage 1 file ..._trust_v1.csv)
        CONFIG_PARTNERS = {"current": COLLABORATIVE_PARTNERS, "reference": COLLAB_REFERENCE,
                           "submitted_v1": COLLABORATIVE_PARTNERS, "submitted_v1_reference": COLLAB_REFERENCE}
        OUT = RESULTS_DIR / ("metric_panel_collaborative_plainDisc" if args.recency == "plain"
                             else "metric_panel_collaborative")
        TITLE = ("METRIC PANEL, COLLABORATIVE PARTNERS (four agents; stage 1 runs; 'current' = the five co-adaptive "
                 "partners, 'reference' = TitForTat and SuspiciousTFT; 'submitted_v1' and 'submitted_v1_reference' = the "
                 "same partners with the submitted-paper trust agent)")
    if args.all:
        REPORTED.update(CATALOG)
    OUT.mkdir(parents=True, exist_ok=True)

    simulate_missing(args.overwrite, args.n_jobs)
    check_sources()
    cells = score_all(args.boot, args.n_jobs)
    cells_long, diffs_long, runs_long = long_tables(cells)
    cells_long["category"] = cells_long["partner"].map(PARTNER_CATEGORY)

    print("Shock harness...")
    shock = pd.DataFrame({a: shock_profile(AGENTS[a][1])[1] for a in AGENTS})
    shock = shock.loc[[k for k in SHOCK_CATALOG if k in shock.index] +
                      [k for k in shock.index if k not in SHOCK_CATALOG]]
    if COLLAB:
        shock["trust_submitted"] = shock_profile(partial(FocalAgent, submitted_v1=True))[1]
    if not args.all:
        shock = shock.loc[list(PANEL_SHOCK_METRICS) + ["dp_neg", "dp_pos"]]
    sched = pooled_schedule(schedule_traces()) if args.all and not COLLAB else None
    board = scoreboard(cells_long, diffs_long)
    red = redundancy(cells_long)

    cells_long.to_csv(OUT / "cells_long.csv", index=False)
    diffs_long.to_csv(OUT / "diffs_long.csv", index=False)
    runs_long.to_csv(OUT / "runs_long.csv", index=False)
    shock.to_csv(OUT / "shock.csv")
    if sched is not None:
        sched.to_csv(OUT / "schedule_pooled.csv", index=False)
    board.to_csv(OUT / "scoreboard.csv", index=False)
    red.to_csv(OUT / "redundancy.csv", index=False)
    catalog_frame().to_csv(OUT / "catalog.csv", index=False)
    report(cells_long, diffs_long, board, shock, sched, red, OUT / "panel_report.txt")

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)
    print("\nSCOREBOARD (current):")
    print(board[board.config == "current"].drop(columns=["config"]).to_string(index=False))
    print("\nSHOCK HARNESS:")
    print(shock.round(3).to_string())
    print(f"\nFull tables: {OUT / 'panel_report.txt'}")


if __name__ == "__main__":
    main()
