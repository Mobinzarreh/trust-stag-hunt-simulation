#!/usr/bin/env python3
"""
Recency sweep (Section 5.4 of the paper). Is there ANY memory setting at which a Beta-Bernoulli learner
reproduces the trust agent's behavior?

A "best against best" comparison, run in the direction that makes the claim hardest to attack: the trust
agent gets its ONE declared setting (the default FocalAgent) on every partner and every metric, while each
benchmark family is
allowed to search its whole memory grid. Two opponents are built from that grid:

    best-overall   one rho (or w) per family, chosen once and used on all 13 partners, by a stated
                   criterion. A fair opponent: like the trust agent, it has a single setting.
    best-per-partner ("oracle") the setting chosen separately on each partner AND each metric, to beat
                   the trust agent there. This opponent knows the partner before meeting it, which no
                   agent in the study does, so it is an upper bound on the family and not a model.

The question is deliberately not "which rho earns the most payoff". Near-myopic settings win payoff,
and the paper does not claim payoff dominance. The question is whether any setting reproduces the SIGNATURES: the asymmetric
response A_p, earlier-and-steadier detection, slow repair, and the failed-collaboration reduction.

Families (all four benchmarks of the study, swept):
    disc_pp   PriorPreservingDiscountedBetaAgent(rho)  the primary recency benchmark
    disc      DiscountedBetaAgent(rho)                 the published Parhizkar form (prior decays too)
    window    SlidingWindowBetaAgent(w)                the moving average
    rho = 1 and w >= K all reproduce BayesianFocalAgent bit for bit, which is asserted below.

Writes results/recency_sweep/:
    settings.csv     the grid: family, parameter, effective memory, concentration ceiling
    cells_long.csv   setting x partner x metric, value with a 95% CI
    diffs_long.csv   trust minus setting, paired on run_id (per-run metrics only)
    envelope.csv     per partner x metric: the best setting in the family and whether trust still wins
    best_overall.csv the single best setting per family under each criterion, and the trust comparison
    shock.csv        the scripted-evidence harness (A_p, A_t) for every setting
    report.txt       the readable tables

Usage:
    python scripts/run_recency_sweep.py                 # reuse stored tables if present
    python scripts/run_recency_sweep.py --overwrite      # re-simulate and re-score
    python scripts/run_recency_sweep.py --boot 50        # fewer bootstrap replicates (faster)
    python scripts/run_recency_sweep.py --quick          # a coarse grid, for a smoke test
"""

import argparse
import zlib
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from trust_bayesian_agent_comparison.agents import (
    BayesianFocalAgent,
    DiscountedBetaAgent,
    FocalAgent,
    PriorPreservingDiscountedBetaAgent,
    SlidingWindowBetaAgent,
)
from trust_bayesian_agent_comparison.analysis import paired_ci, run_cell
from trust_bayesian_agent_comparison.analysis.behavioral_metrics import (
    CATALOG,
    ECONOMIC_METRICS,
    N_BOOT,
    PANEL_METRICS,
    POOLED_METRICS,
    make_trace,
    score_cell,
    shock_profile,
)
from trust_bayesian_agent_comparison.config import (
    DISCOUNTED_BETA_RHO,
    MC_BASE_SEED,
    N_JOBS,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    RECENCY_BENCHMARK_KEY,
    RESULTS_DIR,
    SLIDING_WINDOW_W,
)
from trust_bayesian_agent_comparison.partners import (
    DIAGNOSTIC_PARTNERS,
    MANUSCRIPT_PARTNERS,
    PARTNER_CATEGORY,
    partner_factory,
)

OUT = RESULTS_DIR / "recency_sweep"
BENCH_DIR = RESULTS_DIR / "benchmarks"
PANEL_RUNS = RESULTS_DIR / "metric_panel" / "runs" / "current"
PANEL_RUNS_PLAIN = RESULTS_DIR / "metric_panel_plainDisc" / "runs" / "current"
PARTNERS = list(MANUSCRIPT_PARTNERS) + list(DIAGNOSTIC_PARTNERS)
TRAJ_CHECK = ["agent_belief", "agent_action", "partner_action", "agent_payoff"]

# The filtered metric set of variant-comparison.md Section 11.1, plus the two structural footprints.
REPORTED = list(ECONOMIC_METRICS) + list(PANEL_METRICS)

TRUST = "trust_B"

# ---------------------------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------------------------
RHO_GRID = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00)
W_GRID = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100)
RHO_GRID_QUICK = (0.50, 0.70, 0.90, 1.00)
W_GRID_QUICK = (1, 5, 10, 100)

FAMILY_LABEL = {"disc_pp": "DiscPP (prior-preserving)", "disc": "Disc plain", "window": "Window"}


def build_settings(quick: bool = False) -> dict:
    """key -> dict(family, param, label, factory). Keys are stable and used as the agent id everywhere."""
    rho_grid = RHO_GRID_QUICK if quick else RHO_GRID
    w_grid = W_GRID_QUICK if quick else W_GRID
    s = {}
    for rho in rho_grid:
        s[f"disc_pp_{rho:.2f}"] = dict(
            family="disc_pp", param=rho, label=f"DiscPP rho={rho:.2f}",
            factory=partial(PriorPreservingDiscountedBetaAgent, rho=rho),
            # effective sample size of a discounted average, and the Beta concentration ceiling
            n_eff=(1 / (1 - rho) if rho < 1 else np.inf),
            ceiling=(2 + 1 / (1 - rho) if rho < 1 else np.inf))
    for rho in rho_grid:
        s[f"disc_{rho:.2f}"] = dict(
            family="disc", param=rho, label=f"Disc plain rho={rho:.2f}",
            factory=partial(DiscountedBetaAgent, rho=rho),
            n_eff=(1 / (1 - rho) if rho < 1 else np.inf),
            ceiling=(2 / (1 - rho) if rho < 1 else np.inf))
    for w in w_grid:
        s[f"window_{w}"] = dict(
            family="window", param=w, label=f"Window w={w}",
            factory=partial(SlidingWindowBetaAgent, window=w),
            n_eff=float(w), ceiling=2 + min(w, NUM_ROUNDS))
    return s


def agent_factory(key: str, settings: dict):
    return FocalAgent if key == TRUST else settings[key]["factory"]


def slot_of(key: str) -> int:
    """Trust runs in RNG slot 1, benchmarks in slot 2, exactly as in B1, the panel and the variant study."""
    return 1 if key == TRUST else 2


# ---------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------
def verify(settings: dict, n_check: int = 10) -> None:
    """Six identities. Any failure means this sweep is not the same simulation as the stored tables."""
    checks = 0
    cols = TRAJ_CHECK + ["trust_level", "signal"]

    def runs(fac, partner, slot, n=n_check):
        return run_cell(fac, partner_factory(partner, NUM_ROUNDS), slot,
                        num_runs=n, return_trajectories=True)[1]

    probe = ("StrategicCheater", "TitForTat", "GradualDeterioration")

    # 1-3. rho = 1 (both discounted families) and w >= K reproduce the cumulative model bit for bit
    for key, name in (("disc_pp_1.00", "DiscPP at rho = 1"), ("disc_1.00", "plain Disc at rho = 1"),
                      (f"window_{max(W_GRID)}", f"the window at w = {max(W_GRID)} >= K")):
        if key not in settings:
            continue
        for p in probe:
            a = runs(settings[key]["factory"], p, 2)
            b = runs(BayesianFocalAgent, p, 2)
            worst = np.abs(a[TRAJ_CHECK].to_numpy(float) - b[TRAJ_CHECK].to_numpy(float)).max()
            assert worst == 0.0, (key, p, worst)
        checks += 1
        print(f"{checks}. {name} reproduces the cumulative benchmark bit for bit")

    # 4. the trust agent is the package default and reproduces the cached B1 / panel runs
    worst_t = 0.0
    for p in PARTNERS:
        src = (PANEL_RUNS / f"{p}_trust.csv" if p in DIAGNOSTIC_PARTNERS
               else BENCH_DIR / "cumulative" / f"{p}_agent1.csv")
        if not src.exists():
            continue
        cached = pd.read_csv(src)
        cached = cached[cached["run_id"] < n_check]
        fresh = runs(FocalAgent, p, 1)
        worst_t = max(worst_t, float(np.abs(fresh[TRAJ_CHECK].to_numpy(float)
                                            - cached[TRAJ_CHECK].to_numpy(float)).max()))
    assert worst_t < 1e-12, worst_t
    checks += 1
    print(f"{checks}. the trust agent (version B) reproduces the cached B1 / panel runs "
          f"on every partner (max abs diff {worst_t:.1e})")

    # 5. the two grid points that are the study's own benchmarks reproduce their cached files
    worst_b = 0.0
    for key, bench_dir, panel_key in ((f"disc_pp_{DISCOUNTED_BETA_RHO:.2f}", RECENCY_BENCHMARK_KEY, "discpp0.9"),
                                      (f"window_{SLIDING_WINDOW_W}", f"window_w{SLIDING_WINDOW_W}", "win10"),
                                      (f"disc_{DISCOUNTED_BETA_RHO:.2f}",
                                       f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}", "disc0.9")):
        if key not in settings:
            continue
        for p in PARTNERS:
            if p in DIAGNOSTIC_PARTNERS:
                base = PANEL_RUNS_PLAIN if panel_key == "disc0.9" else PANEL_RUNS
                src = base / f"{p}_{panel_key}.csv"
            else:
                src = BENCH_DIR / bench_dir / f"{p}_agent2.csv"
            if not src.exists():
                continue
            cached = pd.read_csv(src)
            cached = cached[cached["run_id"] < n_check]
            fresh = runs(settings[key]["factory"], p, 2)
            worst_b = max(worst_b, float(np.abs(fresh[TRAJ_CHECK].to_numpy(float)
                                                - cached[TRAJ_CHECK].to_numpy(float)).max()))
    assert worst_b < 1e-12, worst_b
    checks += 1
    print(f"{checks}. the grid points rho = {DISCOUNTED_BETA_RHO} and w = {SLIDING_WINDOW_W} reproduce the "
          f"cached benchmark runs (max abs diff {worst_b:.1e})")

    # 6. every swept agent shares the decision stage with the trust agent (same inv_temp, stochastic logit)
    for key, s in settings.items():
        ag = s["factory"]()
        assert ag.inv_temp == FocalAgent().inv_temp and ag.stochastic, key
    checks += 1
    print(f"{checks}. all {len(settings)} swept settings share the decision rule with the trust agent "
          f"(theta = {FocalAgent().inv_temp}, stochastic logit)")
    print(f"Verification: {checks} checks passed.\n")


# ---------------------------------------------------------------------------------------------
# Runs and scoring
# ---------------------------------------------------------------------------------------------
def run_and_score(partner: str, key: str, settings: dict, n_boot: int):
    traj = run_cell(agent_factory(key, settings), partner_factory(partner, NUM_ROUNDS), slot_of(key),
                    num_runs=NUM_MONTE_CARLO_RUNS, base_seed=MC_BASE_SEED, num_rounds=NUM_ROUNDS,
                    return_trajectories=True)[1]
    tr = make_trace(traj, partner_factory(partner, NUM_ROUNDS)())
    return score_cell(tr, n_boot=n_boot, seed=zlib.crc32(f"{partner}|{key}".encode()))


def score_all(settings: dict, partners: list, n_boot: int, n_jobs: int):
    keys = [TRUST] + list(settings)
    cells = [(p, k) for p in partners for k in keys]
    print(f"Simulating and scoring {len(cells)} cells "
          f"({NUM_MONTE_CARLO_RUNS} runs, {n_boot} bootstrap replicates)...")
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_and_score)(p, k, settings, n_boot) for p, k in cells)
    return {(p, k): r for (p, k), r in zip(cells, res)}


def paired_table(a, b) -> pd.DataFrame:
    """Paired difference a - b over matched run ids, per-run metrics only."""
    rows = []
    for col in a.runs.columns.intersection(b.runs.columns):
        if col in POOLED_METRICS:
            continue
        x, y = a.runs[col].to_numpy(float), b.runs[col].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        d, lo, hi = paired_ci(x[ok], y[ok])
        rows.append(dict(metric=col, diff=d, lo=lo, hi=hi, n=int(ok.sum())))
    sa, sb = a.summary.set_index("metric")["value"], b.summary.set_index("metric")["value"]
    for col in POOLED_METRICS:
        if col in sa.index and col in sb.index:
            rows.append(dict(metric=col, diff=sa[col] - sb[col], lo=np.nan, hi=np.nan, n=np.nan))
    return pd.DataFrame(rows)


def long_tables(scored: dict, settings: dict, partners: list):
    rows, diffs = [], []
    for (p, k), cr in scored.items():
        rows.append(cr.summary.assign(partner=p, agent=k))
    for p in partners:
        t = scored[(p, TRUST)]
        for k in settings:
            diffs.append(paired_table(t, scored[(p, k)]).assign(partner=p, agent=TRUST, benchmark=k))
    return pd.concat(rows, ignore_index=True), pd.concat(diffs, ignore_index=True)


SCOPES = {
    "nonclock": lambda p: p not in ("AlwaysCooperate", "AlwaysDefect"),
    "exploiters": lambda p: p in MANUSCRIPT_PARTNERS and PARTNER_CATEGORY[p] == "non-stationary",
    "breakrepair": lambda p: p == "BreakRepair",
    "ramp": lambda p: p == "ReliabilityRamp",
}


def in_scope(metric: str, partner: str) -> bool:
    scope = CATALOG[metric][4]
    return SCOPES[scope](partner) if scope in SCOPES else True


# ---------------------------------------------------------------------------------------------
# The two opponents
# ---------------------------------------------------------------------------------------------
def envelope(cells_long: pd.DataFrame, diffs_long: pd.DataFrame, settings: dict,
             partners: list) -> pd.DataFrame:
    """
    Best-per-partner (oracle). For every partner x metric x family, the setting whose value is best in
    the metric's own direction, and whether the trust agent is still significantly better than THAT
    setting. direction = 0 metrics (signatures) have no "better", so they report the trust value, the
    full range of the family and whether the trust value falls outside it.
    """
    out = []
    for metric in REPORTED:
        if metric not in CATALOG:
            continue
        direction = CATALOG[metric][2]
        for p in partners:
            if not in_scope(metric, p):
                continue
            v = cells_long[(cells_long.partner == p) & (cells_long.metric == metric)].set_index("agent")["value"]
            if TRUST not in v.index or not np.isfinite(v.get(TRUST, np.nan)):
                continue
            t_val = float(v[TRUST])
            for fam in FAMILY_LABEL:
                keys = [k for k in settings if settings[k]["family"] == fam and k in v.index
                        and np.isfinite(v[k])]
                if not keys:
                    continue
                vals = v[keys].astype(float)
                row = dict(partner=p, metric=metric, family=fam, direction=direction, trust=t_val,
                           fam_min=float(vals.min()), fam_max=float(vals.max()),
                           fam_argmin=str(vals.idxmin()), fam_argmax=str(vals.idxmax()))
                if direction == 0:
                    # a signature: no better/worse, only whether the trust agent is outside the family's range
                    row.update(best=np.nan, best_key="", trust_beats_best=np.nan,
                               outside=bool(t_val > vals.max() or t_val < vals.min()))
                else:
                    best_key = str(vals.idxmax() if direction > 0 else vals.idxmin())
                    best = float(vals.max() if direction > 0 else vals.min())
                    d = diffs_long[(diffs_long.partner == p) & (diffs_long.metric == metric)
                                   & (diffs_long.benchmark == best_key)]
                    if len(d) == 1 and np.isfinite(d.iloc[0]["lo"]):
                        lo, hi = float(d.iloc[0]["lo"]), float(d.iloc[0]["hi"])
                        beats = (lo > 0) if direction > 0 else (hi < 0)
                        loses = (hi < 0) if direction > 0 else (lo > 0)
                    else:
                        beats = direction * (t_val - best) > 0
                        loses = direction * (t_val - best) < 0
                    row.update(best=best, best_key=best_key, trust_beats_best=bool(beats),
                               trust_loses_to_best=bool(loses), outside=np.nan)
                out.append(row)
    return pd.DataFrame(out)


def best_overall(cells_long: pd.DataFrame, settings: dict, partners: list) -> pd.DataFrame:
    """
    One setting per family, chosen once for all partners, under three stated criteria. Each is a
    different idea of what the benchmark is "for", and no single one is privileged.

        payoff        mean total payoff over the nine manuscript partners
        fc            mean failed-collaboration rate over the partners where any occur (lower is better)
        detect        mean detection delay on the change partners (lower is better)

    The trust agent's own value under each criterion is reported beside it.
    """
    nine = [p for p in partners if p in MANUSCRIPT_PARTNERS]
    rows = []
    crits = {
        "payoff": ("Pi", +1, nine),
        "fc": ("FC", -1, nine),
        "detect": ("delay_down_sust", -1, [p for p in partners if in_scope("delay_down_sust", p)]),
    }
    for crit, (metric, direction, scope) in crits.items():
        v = cells_long[(cells_long.metric == metric) & (cells_long.partner.isin(scope))]
        m = v.pivot_table(index="agent", columns="partner", values="value").mean(axis=1)
        t_val = float(m.get(TRUST, np.nan))
        for fam in FAMILY_LABEL:
            keys = [k for k in settings if settings[k]["family"] == fam and k in m.index
                    and np.isfinite(m[k])]
            if not keys:
                continue
            s = m[keys].astype(float)
            best_key = str(s.idxmax() if direction > 0 else s.idxmin())
            rows.append(dict(criterion=crit, metric=metric, family=fam, best_key=best_key,
                             best_param=settings[best_key]["param"], best_value=float(s[best_key]),
                             trust_value=t_val, trust_minus_best=t_val - float(s[best_key]),
                             matched_key=_matched(fam, settings),
                             matched_value=float(m.get(_matched(fam, settings), np.nan)),
                             n_partners=len(scope)))
    return pd.DataFrame(rows)


def _matched(fam: str, settings: dict) -> str:
    """The study's own setting in each family (rho = 0.9, w = 10)."""
    key = {"disc_pp": f"disc_pp_{DISCOUNTED_BETA_RHO:.2f}", "disc": f"disc_{DISCOUNTED_BETA_RHO:.2f}",
           "window": f"window_{SLIDING_WINDOW_W}"}[fam]
    return key if key in settings else ""


def shock_table(settings: dict, n_seeds: int = 100) -> pd.DataFrame:
    """A_p and A_t on the scripted single-shock harness: identical histories for every agent."""
    rows = []
    for key in [TRUST] + list(settings):
        # shock_profile returns (per-seed frame, summary Series); only the summary is needed here
        _, prof = shock_profile(agent_factory(key, settings), seeds=range(n_seeds))
        rows.append(dict(agent=key, **{k: float(prof.get(k, np.nan)) for k in
                                       ("A_p", "A_t", "dp_neg", "dp_pos", "regain_over_lose")}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def write_report(path: Path, settings: dict, cells: pd.DataFrame, env: pd.DataFrame,
                 bo: pd.DataFrame, shock: pd.DataFrame, partners: list) -> None:
    L = []
    A = L.append
    A("Recency sweep. The trust agent's one setting against every memory setting of the benchmarks.")
    A("=" * 110)
    A(f"{len(settings)} benchmark settings x {len(partners)} partners, {NUM_MONTE_CARLO_RUNS} runs, u = 2/3.")
    A("The trust agent is the default FocalAgent at its single declared setting everywhere.")
    A("")

    A("1. STRUCTURAL FOOTPRINTS (scripted single-shock harness, identical histories)")
    A("-" * 110)
    s = shock.set_index("agent")
    A(f"   trust agent      A_p = {s.loc[TRUST, 'A_p']:.3f}   A_t = {s.loc[TRUST, 'A_t']:.3f}")
    bench = s.drop(index=TRUST)
    A(f"   every benchmark  A_p in [{bench.A_p.min():.6f}, {bench.A_p.max():.6f}] "
      f"over all {len(bench)} settings")
    A("   A_p is the ratio of the estimate's movement after an unexpected defection to its movement")
    A("   after an unexpected collaboration. A symmetric conjugate update gives exactly 1 at every rho")
    A("   and w, which is what the range above shows.")
    A("")

    A("2. CV_within: confidence as state rather than a clock")
    A("-" * 110)
    cv = cells[cells.metric == "CV_within"].pivot_table(index="partner", columns="agent", values="value")
    if TRUST in cv.columns:
        bc = [c for c in cv.columns if c != TRUST]
        for p in cv.index:
            A(f"   {p:24s} trust {cv.loc[p, TRUST]:+.3f}   benchmarks "
              f"[{cv.loc[p, bc].min():+.3f}, {cv.loc[p, bc].max():+.3f}]")
    A("")

    A("3. THE ORACLE ENVELOPE: can ANY setting beat the trust agent, partner by partner?")
    A("-" * 110)
    A("   For each partner x metric, the best setting in the family is picked AFTER seeing the result,")
    A("   so this opponent knows the partner in advance. 'trust wins' means the trust agent is still")
    A("   significantly better than that best setting.")
    A("")
    scored = env[env.direction != 0]
    for fam in FAMILY_LABEL:
        f = scored[scored.family == fam]
        if not len(f):
            continue
        A(f"   {FAMILY_LABEL[fam]}")
        for metric in REPORTED:
            m = f[f.metric == metric]
            if not len(m):
                continue
            w = int(m.trust_beats_best.sum())
            l = int(m.trust_loses_to_best.sum())
            A(f"      {metric:20s} trust beats the best setting on {w:2d} of {len(m):2d} partners, "
              f"loses on {l:2d}")
        A("")

    A("4. SIGNATURES (direction 0): is the trust value outside the family's whole range?")
    A("-" * 110)
    sig = env[env.direction == 0]
    for metric in sorted(sig.metric.unique()):
        m = sig[sig.metric == metric]
        A(f"   {metric}")
        for _, r in m.iterrows():
            mark = "OUTSIDE" if r.outside else "inside "
            A(f"      {r.partner:22s} {FAMILY_LABEL[r.family]:28s} trust {r.trust:8.2f}   "
              f"family [{r.fam_min:8.2f}, {r.fam_max:8.2f}]  {mark}")
        A("")

    A("5. BEST-OVERALL: one setting per family, used on every partner")
    A("-" * 110)
    for crit in bo.criterion.unique():
        b = bo[bo.criterion == crit]
        A(f"   criterion = {crit} ({b.iloc[0].metric}, mean over {b.iloc[0].n_partners} partners)")
        A(f"      trust agent {b.iloc[0].trust_value:8.2f}")
        for _, r in b.iterrows():
            A(f"      {FAMILY_LABEL[r.family]:28s} best {r.best_key:16s} {r.best_value:8.2f}   "
              f"trust - best {r.trust_minus_best:+7.2f}   (matched {r.matched_key}: {r.matched_value:.2f})")
        A("")

    path.write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="re-simulate and re-score everything")
    ap.add_argument("--boot", type=int, default=N_BOOT, help="bootstrap replicates per cell")
    ap.add_argument("--quick", action="store_true", help="coarse grid, for a smoke test")
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    settings = build_settings(quick=args.quick)
    partners = PARTNERS

    print(f"T9 recency sweep: {len(settings)} settings x {len(partners)} partners")
    print(f"  trust agent: version B (package default), one setting everywhere")
    print(f"  output: {OUT}\n")

    if not args.skip_verify:
        verify(settings)

    cells_path = OUT / "cells_long.csv"
    if cells_path.exists() and not args.overwrite:
        print(f"Reusing {cells_path} (pass --overwrite to re-run).")
        cells = pd.read_csv(cells_path)
        diffs = pd.read_csv(OUT / "diffs_long.csv")
        shock = pd.read_csv(OUT / "shock.csv")
    else:
        scored = score_all(settings, partners, args.boot, args.jobs)
        cells, diffs = long_tables(scored, settings, partners)
        shock = shock_table(settings)
        cells.to_csv(cells_path, index=False)
        diffs.to_csv(OUT / "diffs_long.csv", index=False)
        shock.to_csv(OUT / "shock.csv", index=False)

    pd.DataFrame([dict(key=k, **{a: b for a, b in v.items() if a != "factory"})
                  for k, v in settings.items()]).to_csv(OUT / "settings.csv", index=False)
    env = envelope(cells, diffs, settings, partners)
    bo = best_overall(cells, settings, partners)
    env.to_csv(OUT / "envelope.csv", index=False)
    bo.to_csv(OUT / "best_overall.csv", index=False)
    write_report(OUT / "report.txt", settings, cells, env, bo, shock, partners)
    print(f"\nWrote {OUT}/ (settings, cells_long, diffs_long, envelope, best_overall, shock, report.txt)")


if __name__ == "__main__":
    main()
