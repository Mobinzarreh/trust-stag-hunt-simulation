#!/usr/bin/env python3
"""
Game-structure sweep: does the trust agent's advantage depend on the normalized deviation loss u?
(Table B2 of the paper: the manuscript regime, coupled/coupled initialization.)

Games. Every game is a symmetric Stag Hunt (R > M >= P > S) built by config.payoff_matrix_for_u,
equal to the study matrix (R = 4, M = 3, P = 2, S = 0) at u = 2/3. Under the shared logit rule the
collaboration margin is EV(C) - EV(I) = D (p - u), where D = (P - S) + (R - M), so the agent
collaborates with probability sigma(theta * D * (p - u)): u sets the threshold and D sets the
decision sharpness. The sweep therefore runs TWO REGIMES, which differ in whether D is held fixed:

  manuscript (primary, reported)  The games published in the paper. M = 3 and P = 2 throughout;
      below u = 2/3 only the upside moves (S = 0, R = 1 + 2/u, so u = 0.2 gives R = 11, Table 2),
      above it only the downside moves (R = 4, S = 2 - u/(1-u), so u = 0.8 gives S = -2, Table 3).
      D = max(2/u, 1/(1-u)) is NOT constant: it is 10 at u = 0.2, 3 at u = 2/3 and 10 at u = 0.9,
      so theta * D is U-shaped and u is confounded with decision sharpness. These are the matrices
      a reader sees in the paper, which is why they are the ones reported.
  fixed_D (control)  Holds D = 3 at every u, so the threshold moves alone and any difference from
      the manuscript regime is attributable to sharpness, not to u. Two matrices with the same u
      and D behave identically (checked below), so this regime carries a primary family
      (fixed_stag: R = 4, S = 0, P = 3u, M = 1 + 3u) and rescores every run from its outcome
      counts under a second family (fixed_hare: P = 2, M = 3, R = 6 - 3u, S = 2 - 3u) with no new
      simulation. No companion family is defined for the manuscript regime: the invariance it would
      show is a property of the decision rule (behavior depends on the payoffs only through u and
      D), which the fixed_D pair already establishes, so the manuscript *_alt columns are empty.

Payoff scale is not comparable across u in either regime (R ranges 4 to 11), so read the
trust-minus-benchmark contrasts and the normalized metrics (EF, SC, FC), not raw Pi, along u.

Initialization. The paper sets x_0 = 1 - u, and the belief-driven partners mirror it
(x_hat_0 = 1 - u), so changing u also moves where both start. x_0 acts mainly through which side
of 0.5 it is on (run_bounds_init_sensitivity.py), and 1 - u reaches 0.5 at u = 0.5. So every u runs a 2 x 2 initialization
design:
    agent_init    coupled (x_0 = 1 - u)      or fixed (x_0 = 1/3, the study value)
    partner_init  coupled (x_hat_0 = 1 - u)  or fixed (x_hat_0 = 1/3)
coupled/coupled is the paper's design. fixed/fixed holds all initialization at the study values,
so only the payoffs change. The benchmarks start from Beta(1, 1) and have no x_0, so only
partner_init applies to them. At u = 2/3 all four conditions coincide.

Agents and seeding. The trust agent runs in the agent-1 RNG slot and the benchmarks (cumulative,
prior-preserving discounted rho = 0.9, window w = 10) in the agent-2 slot, as in run_benchmark_comparison.py, so trust-minus-benchmark
differences are unpaired (Welch CIs). At u = 2/3 the sweep must reproduce the main runs bit for bit. This is
checked against results/benchmarks/, both trajectories and per-run metrics.

Outputs (in --out, default results/game_structure/; every file has a leading 'regime' column):
    per_run.csv     one row per run: Pi, HB, SC, FC, outcome counts, mean confidence, and
                    Pi_alt / HB_alt under the regime's second game family (NaN for manuscript)
    summary.csv     mean and SD per cell
    contrasts.csv   trust minus benchmark for every regime, u, partner, benchmark and
                    initialization condition: D, theta * D, Pi (and Pi_alt in fixed_D), EF, SC
                    and FC with 95% CIs
    crossovers.csv  u values where the payoff difference changes sign (linear interpolation)

Usage:
    python scripts/run_game_structure_sweep.py                  # reuse per_run.csv if it matches
    python scripts/run_game_structure_sweep.py --overwrite      # regenerate
    python scripts/run_game_structure_sweep.py --runs 20 --out results/game_structure_smoke --overwrite
"""

import argparse
import io
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.agents import (
    FocalAgent,
    BayesianFocalAgent,
    PriorPreservingDiscountedBetaAgent,
    DiscountedBetaAgent,
    SlidingWindowBetaAgent,
)
from trust_bayesian_agent_comparison.analysis import (
    run_cells,
    per_run_metrics,
    payoff_from_counts,
    welch_ci,
    sig_mark,
    OUTCOME_COUNT_COLS,
)
from trust_bayesian_agent_comparison.partners import (
    MANUSCRIPT_PARTNERS,
    PARTNER_CATEGORY,
    BeliefDrivenPartnerBase,
    build_partner,
    partner_factory,
)
from trust_bayesian_agent_comparison.simulation import run_agent_simulation, slot_rng
from trust_bayesian_agent_comparison.config import (
    RESULTS_DIR,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    MC_BASE_SEED,
    N_JOBS,
    PAYOFF_MATRIX,
    DECISION_THRESHOLD,
    DISCOUNTED_BETA_RHO,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    SLIDING_WINDOW_W,
    U_SWEEP_GRID,
    GAME_REGIMES,
    payoff_letters,
    INVERSE_TEMPERATURE,
    payoff_matrix_for_u,
    stag_indifference_threshold,
    total_deviation_loss,
    best_response_payoff,
)

REGIMES = tuple(GAME_REGIMES)
MAIN_REGIME, CONTROL_REGIME = "manuscript", "fixed_D"
U_STUDY = 2 / 3
X0_STUDY = 1 - DECISION_THRESHOLD
B1_DIR = RESULTS_DIR / "benchmarks"

TRUST = "trust"
# Benchmark key (same as the B1 result directory) -> (label, agent constructor)
BENCHMARKS = {
    "cumulative": ("Cumulative", BayesianFocalAgent),
    RECENCY_BENCHMARK_KEY: (RECENCY_BENCHMARK_LABEL,
                            partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO)),
    f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}": (f"Disc rho={DISCOUNTED_BETA_RHO:.2f}",
                                                 partial(DiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO)),
    f"window_w{SLIDING_WINDOW_W}": (f"Window w={SLIDING_WINDOW_W}",
                                    partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W)),
}
LABELS = {TRUST: "Trust", **{k: v[0] for k, v in BENCHMARKS.items()}}

INITS = ("coupled", "fixed")
NA = "na"
CELL_KEYS = ["regime", "u", "partner", "agent", "agent_init", "partner_init"]
METRIC_COLS = ["Pi", "HB", "SC", "FC", *OUTCOME_COUNT_COLS]
TRAJECTORY_COLS = ["round", "agent_belief", "agent_action", "partner_action", "agent_payoff",
                   "partner_payoff", "trust_level", "signal", "alpha", "beta", "run_id"]


def u_label(u):
    return "2/3" if np.isclose(u, U_STUDY) else f"{u:g}"


def init_x0(u, mode):
    """x_0 for the agent or x_hat_0 for the belief-driven partners under an initialization mode."""
    return 1 - u if mode == "coupled" else X0_STUDY


def belief_driven(partners):
    return tuple(p for p in partners if isinstance(build_partner(p), BeliefDrivenPartnerBase))


# ----------------------------------------------------------------------------------------------
# Checks that do not need the sweep
# ----------------------------------------------------------------------------------------------

PUBLISHED_TABLES = {0.2: (11.0, 3.0, 2.0, 0.0),   # Table 2
                    0.8: (4.0, 3.0, 2.0, -2.0)}   # Table 3


def game_str(m):
    L = payoff_letters(m)
    return "(" + ", ".join(f"{L[k]:.4g}" for k in "RMPS") + ")"


def check_games():
    """Every family is a Stag Hunt with threshold u and equals the study matrix at 2/3; fixed_D holds D = 3."""
    for regime, fams in GAME_REGIMES.items():
        for fam in (f for f in fams if f is not None):
            if not np.array_equal(payoff_matrix_for_u(U_STUDY, fam), PAYOFF_MATRIX):
                raise AssertionError(f"{fam} at u = 2/3 is not the study matrix")
            for u in U_SWEEP_GRID:
                m = payoff_matrix_for_u(u, fam)
                L = payoff_letters(m)
                if not (L["R"] > L["M"] >= L["P"] > L["S"]):
                    raise AssertionError(f"{fam} at u = {u} is not a Stag Hunt: {L}")
                if abs(stag_indifference_threshold(m) - u) > 1e-12:
                    raise AssertionError(f"{fam} at u = {u}: threshold wrong")
                if regime == CONTROL_REGIME and abs(total_deviation_loss(m) - 3) > 1e-12:
                    raise AssertionError(f"{fam} at u = {u}: D is not 3")
    main_family = GAME_REGIMES[MAIN_REGIME][0]
    for u, want in PUBLISHED_TABLES.items():
        got = tuple(payoff_letters(payoff_matrix_for_u(u, main_family))[k] for k in "RMPS")
        if got != want:
            raise AssertionError(f"{main_family} at u = {u} is {got}, published table says {want}")
    print("Games: every family is a Stag Hunt with threshold u on the whole grid and equals the study "
          "matrix at u = 2/3;\nfixed_D holds D = 3; the manuscript family reproduces published Tables 2 "
          "(u = 0.2) and 3 (u = 0.8) exactly.")
    fams = [f for fs in GAME_REGIMES.values() for f in fs if f is not None]
    print(f"{'u':>6s}  " + "  ".join(f"{f + ' (R, M, P, S)':>26s}" for f in fams)
          + f"  {'D manuscript':>12s}  {'theta*D':>7s}")
    for u in U_SWEEP_GRID:
        m_main = payoff_matrix_for_u(u, main_family)
        d = total_deviation_loss(m_main)
        print(f"{u_label(u):>6s}  " + "  ".join(f"{game_str(payoff_matrix_for_u(u, f)):>26s}" for f in fams)
              + f"  {d:12.2f}  {INVERSE_TEMPERATURE * d:7.1f}")


def check_family_invariance(partners, us=(0.2, 0.5, 0.9), seeds=(42, 43)):
    """Same u and D, different payoffs: behavior must be identical, payoffs must rescore from counts."""
    agents = {TRUST: FocalAgent, **{k: v[1] for k, v in BENCHMARKS.items()}}
    n = 0
    for u in us:
        mats = {fam: payoff_matrix_for_u(u, fam) for fam in GAME_REGIMES[CONTROL_REGIME]}
        for p in partners:
            for key, ctor in agents.items():
                kw = {"u_i": 1 - u} if key == TRUST else {}
                for seed in seeds:
                    runs = {fam: run_agent_simulation(ctor(payoff_matrix=m, **kw),
                                                      partner_factory(p, NUM_ROUNDS, False, m, 1 - u)(),
                                                      NUM_ROUNDS, seed, rng=slot_rng(seed, 1))
                            for fam, m in mats.items()}
                    a, b = runs.values()
                    for col in ("agent_belief", "agent_action", "partner_action"):
                        if not np.array_equal(a[col].to_numpy(), b[col].to_numpy()):
                            raise AssertionError(f"Families differ in {col}: u={u} {p} {key} seed {seed}")
                    for fam, df in runs.items():
                        counts = {c: v for c, v in zip(OUTCOME_COUNT_COLS, _counts(df))}
                        if abs(payoff_from_counts(counts, mats[fam]) - df.agent_payoff.sum()) > 1e-9:
                            raise AssertionError(f"Rescoring from counts fails: u={u} {p} {key} {fam}")
                    n += 1
    print(f"Family invariance: {n} run pairs behave identically under {' and '.join(mats)} (same u, D = 3); "
          f"payoffs rescore exactly from outcome counts.")


def _counts(df):
    a, o = df.agent_action.to_numpy(), df.partner_action.to_numpy()
    return (((a == 1) & (o == 1)).sum(), ((a == 1) & (o == 0)).sum(),
            ((a == 0) & (o == 1)).sum(), ((a == 0) & (o == 0)).sum())


# ----------------------------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------------------------

def build_cells(partners, num_runs, us=U_SWEEP_GRID, return_trajectories=False, regimes=REGIMES):
    cells = {}
    for regime in regimes:
        primary = GAME_REGIMES[regime][0]
        for u in us:
            m = payoff_matrix_for_u(u, primary)
            for p in partners:
                for pinit in INITS:
                    pf = partner_factory(p, NUM_ROUNDS, False, m, init_x0(u, pinit))
                    for ainit in INITS:
                        cells[(regime, u, p, TRUST, ainit, pinit)] = dict(
                            agent_factory=partial(FocalAgent, payoff_matrix=m, u_i=init_x0(u, ainit)),
                            partner_factory=pf, slot=1, num_runs=num_runs, return_trajectories=return_trajectories)
                    for b, (_, ctor) in BENCHMARKS.items():
                        cells[(regime, u, p, b, NA, pinit)] = dict(
                            agent_factory=partial(ctor, payoff_matrix=m),
                            partner_factory=pf, slot=2, num_runs=num_runs, return_trajectories=return_trajectories)
    return cells


def add_alt_family(per_run):
    """Rescore every run under its regime's second game family (if any) from its outcome counts."""
    per_run = per_run.copy()
    per_run["Pi_alt"] = np.nan
    per_run["HB_alt"] = np.nan
    for regime, (_, alt) in GAME_REGIMES.items():
        if alt is None:
            continue
        for u in U_SWEEP_GRID:
            mask = (per_run.regime == regime) & (per_run.u == u)
            m_alt = payoff_matrix_for_u(u, alt)
            sub = per_run.loc[mask]
            best = best_response_payoff(m_alt)
            per_run.loc[mask, "Pi_alt"] = payoff_from_counts(sub, m_alt)
            per_run.loc[mask, "HB_alt"] = best[0] * (sub.n_CI + sub.n_II) + best[1] * (sub.n_CC + sub.n_IC)
    return per_run


def run_sweep(partners, num_runs, out_dir, overwrite, n_jobs):
    path, meta_path = out_dir / "per_run.csv", out_dir / "per_run_meta.json"
    meta = {"num_runs": num_runs, "base_seed": MC_BASE_SEED, "num_rounds": NUM_ROUNDS,
            "u_grid": list(U_SWEEP_GRID), "partners": list(partners),
            "regimes": {r: list(f) for r, f in GAME_REGIMES.items()},
            "benchmarks": list(BENCHMARKS)}
    if not overwrite and path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{path} was made with different settings; rerun with --overwrite")
        print(f"Loading cached sweep from {path}")
        # round_trip parsing keeps u (and every float) exactly as written
        return pd.read_csv(path, float_precision="round_trip")

    cells = build_cells(partners, num_runs)
    print(f"Running {len(cells)} cells x {num_runs} runs ...")
    res = run_cells(cells, n_jobs=n_jobs, verbose=2)
    frames = [df.assign(**dict(zip(CELL_KEYS, key))) for key, df in res.items()]
    per_run = add_alt_family(pd.concat(frames, ignore_index=True))
    per_run = per_run[CELL_KEYS + ["run_id"] + [c for c in per_run.columns if c not in CELL_KEYS + ["run_id"]]]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"Saved {len(per_run)} runs to {path}")
    return per_run


# ----------------------------------------------------------------------------------------------
# Checks on the sweep
# ----------------------------------------------------------------------------------------------

def check_b1_regression(per_run, partners, num_runs, n_jobs):
    """u = 2/3 must reproduce the B1 runs: trajectories (via CSV, as stored) and per-run metrics."""
    if num_runs != NUM_MONTE_CARLO_RUNS:
        print("B1 regression check skipped (needs the full run count).")
        return
    cells = build_cells(partners, num_runs, us=(U_STUDY,), return_trajectories=True, regimes=(MAIN_REGIME,))
    cells = {k: v for k, v in cells.items() if k[4] in ("coupled", NA) and k[5] == "coupled"}
    res = run_cells(cells, n_jobs=n_jobs)
    by = per_run.set_index(CELL_KEYS + ["run_id"]).sort_index()
    for (_, u, p, a, ainit, pinit), (summary, traj) in res.items():
        b1 = B1_DIR / ("cumulative" if a == TRUST else a) / f"{p}_{'agent1' if a == TRUST else 'agent2'}.csv"
        cached = pd.read_csv(b1)
        fresh = pd.read_csv(io.StringIO(traj.to_csv(index=False)))
        cols = [c for c in TRAJECTORY_COLS if c in cached]
        if not np.array_equal(fresh[cols].to_numpy(float), cached[cols].to_numpy(float)):
            raise AssertionError(f"u = 2/3 trajectories differ from {b1}")
        ref = per_run_metrics(cached)
        if not all(np.array_equal(summary[c].to_numpy(), ref[c].to_numpy()) for c in ("Pi", "HB", "SC", "FC")):
            raise AssertionError(f"u = 2/3 per-run metrics differ from {b1}")
        # every initialization condition coincides at u = 2/3, and the sweep must hold the same runs
        for regime in REGIMES:
            for ai in (INITS if a == TRUST else (NA,)):
                for pi in INITS:
                    sweep = by.loc[(regime, u, p, a, ai, pi)]
                    if not np.array_equal(sweep[METRIC_COLS].to_numpy(float),
                                          summary.set_index("run_id")[METRIC_COLS].to_numpy(float)):
                        raise AssertionError(f"Sweep cell {regime} u=2/3 {p} {a} {ai}/{pi} differs from the B1 rerun")
    print(f"B1 regression: u = 2/3 reproduces all {len(res)} B1 agent x partner cells bit for bit "
          f"(trajectories and per-run metrics), in all four initialization conditions and in both regimes "
          f"({', '.join(REGIMES)}).")


def check_init_scope(per_run, partners):
    """partner_init may only matter for belief-driven partners."""
    bd = belief_driven(partners)
    by = per_run.set_index(CELL_KEYS + ["run_id"]).sort_index()
    for regime in REGIMES:
        for u in U_SWEEP_GRID:
            for p in partners:
                if p in bd:
                    continue
                for a in [TRUST, *BENCHMARKS]:
                    for ai in (INITS if a == TRUST else (NA,)):
                        x = by.loc[(regime, u, p, a, ai, "coupled")][METRIC_COLS].to_numpy(float)
                        y = by.loc[(regime, u, p, a, ai, "fixed")][METRIC_COLS].to_numpy(float)
                        if not np.array_equal(x, y):
                            raise AssertionError(f"partner_init changed a non-belief-driven partner: "
                                                 f"{regime} u={u} {p} {a}")
    print(f"Initialization scope: partner_init changes nothing outside the belief-driven partners {bd}.")


# ----------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------

def ratio_ci(pi, hb):
    """Ratio of means 100 * mean(Pi) / mean(HB) and its delta-method standard error."""
    pi, hb = np.asarray(pi, float), np.asarray(hb, float)
    r = pi.mean() / hb.mean()
    n = len(pi)
    var = (pi.var(ddof=1) - 2 * r * np.cov(pi, hb)[0, 1] + r ** 2 * hb.var(ddof=1)) / (n * hb.mean() ** 2)
    return 100 * r, 100 * np.sqrt(max(var, 0.0))


def summarize(per_run):
    g = per_run.groupby(CELL_KEYS, sort=False)
    out = g.agg(Pi=("Pi", "mean"), Pi_sd=("Pi", "std"), HB=("HB", "mean"), Pi_alt=("Pi_alt", "mean"),
                HB_alt=("HB_alt", "mean"), SC=("SC", "mean"), FC=("FC", "mean"), t_mean=("t_mean", "mean"),
                n=("run_id", "count")).reset_index()
    out["EF"] = 100 * out.Pi / out.HB
    out["EF_alt"] = 100 * out.Pi_alt / out.HB_alt
    out[["SC", "FC"]] *= 100
    return out


def condition_name(ainit, pinit):
    return f"{ainit}/{pinit}"


def contrasts(per_run, partners):
    groups = {k: g.sort_values("run_id") for k, g in per_run.groupby(CELL_KEYS, sort=False)}
    rows = []
    for regime in REGIMES:
        primary, alt = GAME_REGIMES[regime]
        for u in U_SWEEP_GRID:
            D = total_deviation_loss(payoff_matrix_for_u(u, primary))
            for p in partners:
                for b in BENCHMARKS:
                    for ainit in INITS:
                        for pinit in INITS:
                            t = groups[(regime, u, p, TRUST, ainit, pinit)]
                            m = groups[(regime, u, p, b, NA, pinit)]
                            row = {"regime": regime, "u": u, "u_label": u_label(u), "D": D,
                                   "theta_D": INVERSE_TEMPERATURE * D, "partner": p,
                                   "category": PARTNER_CATEGORY[p], "benchmark": b,
                                   "condition": condition_name(ainit, pinit),
                                   "trust_Pi": t.Pi.mean(), "bench_Pi": m.Pi.mean()}
                            for metric, scale in (("Pi", 1), ("Pi_alt", 1), ("SC", 100), ("FC", 100)):
                                if metric.endswith("_alt") and alt is None:
                                    row.update({f"d{metric}": np.nan, f"d{metric}_lo": np.nan,
                                                f"d{metric}_hi": np.nan})
                                    continue
                                d, lo, hi = welch_ci(t[metric], m[metric], scale)
                                row.update({f"d{metric}": d, f"d{metric}_lo": lo, f"d{metric}_hi": hi})
                            for suffix in ("", "_alt"):
                                if suffix and alt is None:
                                    row.update({"dEF_alt": np.nan, "dEF_alt_lo": np.nan, "dEF_alt_hi": np.nan})
                                    continue
                                ef_t, se_t = ratio_ci(t["Pi" + suffix], t["HB" + suffix])
                                ef_m, se_m = ratio_ci(m["Pi" + suffix], m["HB" + suffix])
                                d, se = ef_t - ef_m, np.hypot(se_t, se_m)
                                row.update({f"dEF{suffix}": d, f"dEF{suffix}_lo": d - 1.96 * se,
                                            f"dEF{suffix}_hi": d + 1.96 * se})
                            rows.append(row)
    return pd.DataFrame(rows)


def crossovers(con):
    """Sign changes of the payoff difference along u, by linear interpolation between grid points."""
    rows = []
    for (r, p, b, c), g in con.groupby(["regime", "partner", "benchmark", "condition"], sort=False):
        g = g.sort_values("u")
        u, d = g.u.to_numpy(), g.dPi.to_numpy()
        lo, hi = g.dPi_lo.to_numpy(), g.dPi_hi.to_numpy()
        for i in range(len(u) - 1):
            if np.sign(d[i]) != np.sign(d[i + 1]) and d[i] != d[i + 1]:
                rows.append({"regime": r, "partner": p, "benchmark": b, "condition": c,
                             "u_cross": u[i] + (u[i + 1] - u[i]) * d[i] / (d[i] - d[i + 1]),
                             "u_left": u[i], "d_left": d[i], "sig_left": sig_mark(lo[i], hi[i]),
                             "u_right": u[i + 1], "d_right": d[i + 1], "sig_right": sig_mark(lo[i + 1], hi[i + 1])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------------------------

def bar(title):
    print(f"\n{'=' * 118}\n{title}\n{'=' * 118}")


def print_grid(con, partners, regime, benchmark, condition, metric="dPi", fmt="{:+6.1f}"):
    sub = con[(con.regime == regime) & (con.benchmark == benchmark) & (con.condition == condition)]
    sub = sub.set_index(["partner", "u"])
    print(f"{'Partner':22s}" + "".join(f"{u_label(u):>9s}" for u in U_SWEEP_GRID))
    for p in partners:
        cells = []
        for u in U_SWEEP_GRID:
            r = sub.loc[(p, u)]
            cells.append(f"{fmt.format(r[metric]) + (sig_mark(r[metric + '_lo'], r[metric + '_hi']) or ' '):>9s}")
        print(f"{p:22s}" + "".join(cells))


REGIME_TITLES = {MAIN_REGIME: "MANUSCRIPT GAMES (published tables; D and theta*D vary with u)",
                 CONTROL_REGIME: "D = 3 CONTROL (only the threshold moves)"}


def regime_flips(con, metric, condition=None):
    key = ["u", "partner", "benchmark", "condition"]
    j = con[con.regime == MAIN_REGIME].set_index(key).join(
        con[con.regime == CONTROL_REGIME].set_index(key), lsuffix="_man", rsuffix="_ctl").reset_index()
    if condition is not None:
        j = j[j.condition == condition]
    flip = [sig_mark(a, b) != sig_mark(c, d) for a, b, c, d in
            zip(j[f"{metric}_lo_man"], j[f"{metric}_hi_man"], j[f"{metric}_lo_ctl"], j[f"{metric}_hi_ctl"])]
    return j, j[flip]


def print_tables(per_run, summ, con, cross, partners):
    bd = belief_driven(partners)
    main = condition_name("coupled", "coupled")
    ctrl = condition_name("fixed", "fixed")
    disc = RECENCY_BENCHMARK_KEY
    discs = (RECENCY_BENCHMARK_KEY, f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}")
    mt, ct = REGIME_TITLES[MAIN_REGIME], REGIME_TITLES[CONTROL_REGIME]

    for regime in REGIMES:
        bar(f"[{REGIME_TITLES[regime]}]\nTrust agent total payoff by u (paper design: x_0 = x_hat_0 = 1 - u)")
        s = summ[(summ.regime == regime) & (summ.agent == TRUST)
                 & (summ.agent_init == "coupled") & (summ.partner_init == "coupled")]
        s = s.set_index(["partner", "u"])
        print(f"{'Partner':22s}" + "".join(f"{u_label(u):>9s}" for u in U_SWEEP_GRID))
        for p in partners:
            print(f"{p:22s}" + "".join(f"{s.loc[(p, u), 'Pi']:9.1f}" for u in U_SWEEP_GRID))

    for condition, title in ((main, "PAPER DESIGN (x_0 = x_hat_0 = 1 - u)"),
                             (ctrl, "FIXED-INITIALIZATION CONTROL (x_0 = x_hat_0 = 1/3 at every u)")):
        for b in BENCHMARKS:
            bar(f"[{mt}] {title}\nTrust minus {LABELS[b]}, total payoff (+/- = 95% Welch CI above/below 0)")
            print_grid(con, partners, MAIN_REGIME, b, condition)

    for condition, title in ((main, "paper design"), (ctrl, "fixed-initialization control")):
      for disc in discs:
        bar(f"[{mt}]\nTrust minus {LABELS[disc]}, efficiency EF (pp), {title}")
        print_grid(con, partners, MAIN_REGIME, disc, condition, metric="dEF")
        bar(f"[{mt}]\nTrust minus {LABELS[disc]}, failed collaboration FC (pp; negative = trust fails less), {title}")
        print_grid(con, partners, MAIN_REGIME, disc, condition, metric="dFC")

    for b in BENCHMARKS:
        bar(f"[{ct}] PAPER DESIGN\nTrust minus {LABELS[b]}, total payoff (+/- = 95% Welch CI above/below 0)")
        print_grid(con, partners, CONTROL_REGIME, b, main)
    bar(f"[{ct}] PAPER DESIGN\nTrust minus {LABELS[disc]}, efficiency EF (pp)")
    print_grid(con, partners, CONTROL_REGIME, disc, main, metric="dEF")

    bar("Threshold vs sharpness: trust-minus-benchmark contrasts whose significance sign differs between the\n"
        "manuscript games and the D = 3 control (same u, partner, benchmark, initialization). There, theta*D,\n"
        "not u, decides the verdict. Identical at u = 2/3, where both regimes are the study matrix.")
    for metric, label in (("dPi", "payoff"), ("dEF", "efficiency EF"), ("dSC", "successful collab. SC"),
                          ("dFC", "failed collab. FC")):
        j, flip = regime_flips(con, metric)
        print(f"  {label:24s}: {len(flip):4d} of {len(j)} contrasts change significance sign between regimes")
    j, flip = regime_flips(con, "dEF", condition=main)
    print(f"\nEF flips in the paper design ({len(flip)} of {len(j)}):")
    if len(flip):
        fmt = lambda d, lo, hi: f"{d:+.1f}{sig_mark(lo, hi) or ' '}"
        out = pd.DataFrame({
            "u": flip.u_label_man, "theta*D": flip.theta_D_man.round(1), "partner": flip.partner,
            "benchmark": flip.benchmark.map(LABELS),
            "dEF manuscript": [fmt(*x) for x in zip(flip.dEF_man, flip.dEF_lo_man, flip.dEF_hi_man)],
            "dEF D=3": [fmt(*x) for x in zip(flip.dEF_ctl, flip.dEF_lo_ctl, flip.dEF_hi_ctl)]})
        print(out.to_string(index=False))

    fd = con[con.regime == CONTROL_REGIME]
    alt = GAME_REGIMES[CONTROL_REGIME][1]
    bar(f"[{ct}] Payoff-family robustness: trust-minus-benchmark payoff under "
        f"{GAME_REGIMES[CONTROL_REGIME][0]} vs {alt}\n"
        "(same runs, rescored; cells whose significance sign differs between families)")
    differ = fd[fd.apply(lambda r: sig_mark(r.dPi_lo, r.dPi_hi) != sig_mark(r.dPi_alt_lo, r.dPi_alt_hi), axis=1)]
    print(f"{len(differ)} of {len(fd)} contrasts change significance sign.")
    if len(differ):
        print(differ[["u_label", "partner", "benchmark", "condition", "dPi", "dPi_lo", "dPi_hi",
                      "dPi_alt", "dPi_alt_lo", "dPi_alt_hi"]].round(1).to_string(index=False))
    differ_ef = fd[fd.apply(lambda r: sig_mark(r.dEF_lo, r.dEF_hi) != sig_mark(r.dEF_alt_lo, r.dEF_alt_hi), axis=1)]
    print(f"{len(differ_ef)} of {len(fd)} EF contrasts change significance sign between families.")

    for b in BENCHMARKS:
        bar(f"[{mt}]\nInitialization decomposition, belief-driven partners: trust minus {LABELS[b]}, total payoff\n"
            "agent_init/partner_init; coupled = 1 - u, fixed = 1/3")
        conds = [condition_name(a, p) for a in INITS for p in INITS]
        sub = con[(con.regime == MAIN_REGIME) & (con.benchmark == b)].set_index(["partner", "condition", "u"])
        for p in bd:
            print(f"\n  {p}")
            print(f"  {'condition':18s}" + "".join(f"{u_label(u):>9s}" for u in U_SWEEP_GRID))
            for c in conds:
                cells = [f"{sub.loc[(p, c, u), 'dPi']:+6.1f}{sig_mark(sub.loc[(p, c, u), 'dPi_lo'], sub.loc[(p, c, u), 'dPi_hi']) or ' '}"
                         for u in U_SWEEP_GRID]
                print(f"  {c:18s}" + "".join(f"{x:>9s}" for x in cells))

    bar("Significant reversals: payoff difference changes sign between neighboring u values whose CIs\n"
        "both exclude 0 (linear interpolation; both regimes; paper design and fixed-init control;\n"
        "all sign changes in crossovers.csv)")
    c = cross[cross.condition.isin([main, ctrl]) & (cross.sig_left != "") & (cross.sig_right != "")].copy() \
        if len(cross) else cross
    if len(c):
        c["benchmark"] = c.benchmark.map(LABELS)
        print(c.round(2).to_string(index=False))
    else:
        print("None.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate runs even if cached results exist")
    parser.add_argument("--runs", type=int, default=NUM_MONTE_CARLO_RUNS, help="Monte Carlo runs per cell")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "game_structure", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Parallel jobs (-1 = all cores)")
    args = parser.parse_args()

    partners = list(MANUSCRIPT_PARTNERS)
    print(f"u grid: {', '.join(u_label(u) for u in U_SWEEP_GRID)}; regimes: "
          + "; ".join(f"{r} = {' + '.join(f for f in fs if f)}" for r, fs in GAME_REGIMES.items())
          + f" (reported: {MAIN_REGIME})")
    print(f"Agents: Trust (slot 1), {', '.join(LABELS[b] for b in BENCHMARKS)} (slot 2)")
    print(f"Partners: {len(partners)}, runs: {args.runs}, rounds: {NUM_ROUNDS}, base seed: {MC_BASE_SEED}")

    check_games()
    check_family_invariance(partners)
    per_run = run_sweep(partners, args.runs, args.out, args.overwrite, args.n_jobs)
    check_b1_regression(per_run, partners, args.runs, args.n_jobs)
    check_init_scope(per_run, partners)

    summ = summarize(per_run)
    con = contrasts(per_run, partners)
    cross = crossovers(con)
    print_tables(per_run, summ, con, cross, partners)

    # round results to 4 decimals but keep u exact, so files can be matched on the grid values
    for df, name in ((summ, "summary.csv"), (con, "contrasts.csv"), (cross, "crossovers.csv")):
        df.round({c: 4 for c in df.columns if c not in ("u", "u_left", "u_right")}).to_csv(args.out / name, index=False)
    print(f"\nSaved summary.csv, contrasts.csv, crossovers.csv to {args.out}")


if __name__ == "__main__":
    main()
