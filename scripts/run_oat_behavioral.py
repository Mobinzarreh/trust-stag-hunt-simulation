#!/usr/bin/env python3
"""
One-factor-at-a-time sensitivity of the BEHAVIORAL results, on the 12 reported partners (Section 5.3 and
Appendix B of the paper).

run_oat_sensitivity.py records payoff, SC and FC on the nine manuscript partners only, so it cannot say whether
detection, repair, forgiveness or the Distrust Lock cost survive a parameter change. This script runs the same
design (config.OAT_DESIGN, every row at its low and high value, the rest at the package default) on the nine
manuscript partners plus Break and Repair, Reliability Ramp and Distrust Lock, and scores every run with the metric
panel's code (behavioral_metrics.per_run), the code behind results/variants.

Seeds and slots are those of B1, the variant comparison and the OAT: run i uses seed MC_BASE_SEED + i, trust variants
run in RNG slot 1 and benchmarks in slot 2. So:
    * every trust variant shares its random numbers with the baseline run for run: effects against the baseline are
      paired (common random numbers);
    * trust and benchmark streams are independent: trust-minus-benchmark differences use the two-sample SE.
theta is shared with the benchmarks, so on the theta rows the benchmarks run at the same theta (1 and 3); every other
row is compared with the benchmarks at the default theta = 2, read from results/variants/runs_long.csv.

Checks, asserted before anything is written:
    baseline trust, 12 partners, 9 metrics   = results/variants/runs_long.csv agent B, run for run
    every trust variant, 9 manuscript partners, Pi / SC / FC = results/oat_sensitivity/per_run.csv, run for run
    benchmarks at theta = 1 and 3, 9 manuscript partners, Pi / SC / FC = the same file

Reporting follows the session-4 convention (no stars): mean (SE); trust minus DiscPP (SE). Whether a baseline result
survives a setting is judged with a Holm correction within each metric's table (all settings x partners of that
metric): "holds" (same sign as at the baseline and significant), "level" (not significant), "reverses" (opposite sign
and significant). A result counts only where it exists at the baseline (significant there under the same rule).

Outputs (results/oat_behavioral/):
    per_run.csv     agent, variant, partner, run_id and the nine metrics
    levels.csv      mean, sd, se, n per agent x variant x partner x metric
    effects.csv     trust variant minus trust baseline, paired (effect, se)
    contrasts.csv   trust variant minus each benchmark at the matching theta (diff, se, p, p_holm, verdict vs DiscPP)
    shock.csv       A_p and the regain/lose counts of every trust variant (scripted-evidence harness)
    report.md       the survival tables

Usage:
    python scripts/run_oat_behavioral.py              # reuse per_run.csv if present
    python scripts/run_oat_behavioral.py --overwrite  # regenerate
"""

import argparse
import json
import math
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from trust_bayesian_agent_comparison.agents import (BayesianFocalAgent, FocalAgent,
                                                    PriorPreservingDiscountedBetaAgent, SlidingWindowBetaAgent)
from trust_bayesian_agent_comparison.analysis import run_cell
from trust_bayesian_agent_comparison.analysis.behavioral_metrics import CATALOG, make_trace, per_run, shock_profile
from trust_bayesian_agent_comparison.config import (DISCOUNTED_BETA_RHO, EPS, INVERSE_TEMPERATURE, MC_BASE_SEED,
                                                    N_JOBS, NUM_MONTE_CARLO_RUNS, NUM_ROUNDS, OAT_DESIGN, RESULTS_DIR,
                                                    SLIDING_WINDOW_W)
from trust_bayesian_agent_comparison.partners import DIAGNOSTIC_PARTNERS, MANUSCRIPT_PARTNERS, partner_factory

OUT = RESULTS_DIR / "oat_behavioral"
PARTNERS = list(MANUSCRIPT_PARTNERS) + list(DIAGNOSTIC_PARTNERS)
METRICS = ["Pi", "SC", "FC", "partner_pay", "delay_down_sust", "delay_up_sust", "crossings", "brier_phat",
           "forgiveness"]
# metric: (display scale, decimals, unit)
SHOW = {"Pi": (1, 1, "payoff points"), "SC": (100, 1, "points of rounds"), "FC": (100, 2, "points of rounds"),
        "partner_pay": (1, 1, "the exploiter's payoff points"), "delay_down_sust": (1, 2, "rounds"),
        "delay_up_sust": (1, 2, "rounds"), "crossings": (1, 2, "crossings per run"),
        "brier_phat": (1, 4, "Brier score"), "forgiveness": (100, 1, "points")}
BENCH = {"discpp0.9": partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO),
         "win10": partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W),
         "cumulative": BayesianFocalAgent}
OAT_BENCH_KEY = {"discpp0.9": f"disc_pp_rho{DISCOUNTED_BETA_RHO:.2f}", "win10": f"window_w{SLIDING_WINDOW_W}",
                 "cumulative": "cumulative"}
BASE = "baseline"
THETA_KW = OAT_DESIGN["theta"][0]
EXPLOITERS = ["SingleCycle", "GradualDeterioration", "StrategicCheater", "ExpectationViolation"]
PRETTY = {"AlwaysCooperate": "Always Collab.", "AlwaysDefect": "Always Indep.", "Random": "Random",
          "TitForTat": "TFT", "SuspiciousTFT": "STFT", "SingleCycle": "Single Cycle",
          "GradualDeterioration": "Grad. Det.", "StrategicCheater": "Str. Cheater",
          "ExpectationViolation": "Exp. Viol.", "BreakRepair": "Break & Repair", "ReliabilityRamp": "Rel. Ramp",
          "DistrustLock": "Distrust Lock"}
Z95 = 1.959964


def variants() -> dict:
    """key -> (parameter, level, value, FocalAgent overrides). Same keys as run_oat_sensitivity.py, without its theta extras."""
    out = {BASE: (BASE, BASE, np.nan, {})}
    for param, (kw, _, low, high) in OAT_DESIGN.items():
        for level, value in (("low", low), ("high", high)):
            if value is not None:
                out[f"{param}_{level}"] = (param, level, value, {kw: value})
    return out


def theta_of(key: str) -> float:
    ov = variants()[key][3]
    return float(ov.get(THETA_KW, INVERSE_TEMPERATURE))


def score(agent_fac, partner: str, slot: int, inv_temp: float) -> pd.DataFrame:
    """300 runs of one cell, scored per run with the panel's metric code."""
    traj = run_cell(agent_fac, partner_factory(partner, NUM_ROUNDS), slot, num_runs=NUM_MONTE_CARLO_RUNS,
                    base_seed=MC_BASE_SEED, num_rounds=NUM_ROUNDS, return_trajectories=True)[1]
    tr = make_trace(traj, partner_factory(partner, NUM_ROUNDS)(), inv_temp=inv_temp, eps=EPS)
    runs = per_run(tr)[METRICS]
    runs.insert(0, "run_id", np.arange(len(runs)))
    return runs


def cells() -> list:
    out = []
    for key, (_, _, _, ov) in variants().items():
        for p in PARTNERS:
            out.append(("trust", key, p, partial(FocalAgent, **ov), 1, theta_of(key)))
    for b, ctor in BENCH.items():
        for level in ("low", "high"):
            th = OAT_DESIGN["theta"][2 if level == "low" else 3]
            for p in PARTNERS:
                out.append((b, f"theta_{level}", p, partial(ctor, inv_temp=th), 2, th))
    return out


def simulate(overwrite: bool, n_jobs: int) -> pd.DataFrame:
    path, meta_path = OUT / "per_run.csv", OUT / "per_run_meta.json"
    meta = {"num_runs": NUM_MONTE_CARLO_RUNS, "base_seed": MC_BASE_SEED, "num_rounds": NUM_ROUNDS,
            "partners": PARTNERS, "design": {k: list(v) for k, v in OAT_DESIGN.items()}, "metrics": METRICS}
    if not overwrite and path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{path} was made with different settings; rerun with --overwrite")
        print(f"Loading cached runs from {path}")
        return pd.read_csv(path, float_precision="round_trip")
    todo = cells()
    print(f"Simulating {len(todo)} cells x {NUM_MONTE_CARLO_RUNS} runs ...")
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(score)(fac, p, slot, th) for _, _, p, fac, slot, th in todo)
    df = pd.concat([r.assign(agent=a, variant=v, partner=p) for (a, v, p, *_), r in zip(todo, res)], ignore_index=True)
    df = df[["agent", "variant", "partner", "run_id"] + METRICS]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=1))
    return df


def with_default_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Add the benchmarks at the default theta from the variant comparison (same seeds, slot 2, same metric code)."""
    v = pd.read_csv(RESULTS_DIR / "variants" / "runs_long.csv")
    v = v[v.agent.isin(list(BENCH)) & v.partner.isin(PARTNERS)][["agent", "partner", "run_id"] + METRICS]
    return pd.concat([df, v.assign(variant=BASE)[df.columns]], ignore_index=True)


def verify(df: pd.DataFrame) -> list:
    msgs = []
    v = pd.read_csv(RESULTS_DIR / "variants" / "runs_long.csv")
    worst = 0.0
    for p in PARTNERS:
        ours = df[(df.agent == "trust") & (df.variant == BASE) & (df.partner == p)].set_index("run_id")[METRICS]
        ref = v[(v.agent == "B") & (v.partner == p)].set_index("run_id")[METRICS]
        a, b = ours.to_numpy(float), ref.loc[ours.index].to_numpy(float)
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        gap = np.nanmax(np.abs(a - b)) if np.isfinite(a).any() else 0.0
        assert same_nan and gap < 1e-9, f"baseline on {p} differs from results/variants (gap {gap})"
        worst = max(worst, gap)
    msgs.append(f"baseline trust = results/variants agent B on all {len(PARTNERS)} partners x {len(METRICS)} metrics "
                f"(largest gap {worst:.1e})")
    oat = pd.read_csv(RESULTS_DIR / "oat_sensitivity" / "per_run.csv")
    n = 0
    for (agent, variant, p), g in df.groupby(["agent", "variant", "partner"]):
        if p not in MANUSCRIPT_PARTNERS:
            continue
        oa, ov = ("trust", variant) if agent == "trust" else (OAT_BENCH_KEY[agent], None)
        if agent != "trust":
            if variant == BASE:
                ov = "theta_2"
            else:
                ov = f"theta_{OAT_DESIGN['theta'][2 if variant == 'theta_low' else 3]:g}"
        ref = oat[(oat.agent == oa) & (oat.variant == ov) & (oat.partner == p)].set_index("run_id")
        assert len(ref) == NUM_MONTE_CARLO_RUNS, (oa, ov, p, len(ref))
        g = g.set_index("run_id")
        for c in ("Pi", "SC", "FC"):
            assert np.allclose(g[c].to_numpy(float), ref.loc[g.index, c].to_numpy(float), atol=1e-9, rtol=0), \
                f"{agent}/{variant}/{p}/{c} differs from results/oat_sensitivity"
        n += 1
    msgs.append(f"Pi, SC and FC equal results/oat_sensitivity/per_run.csv run for run on {n} cells (every trust variant "
                "and the benchmarks at theta = 1, 2, 3 on the nine manuscript partners)")
    return msgs


def stats(x) -> tuple:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return x.mean(), sd, sd / math.sqrt(len(x)), len(x)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    running, m = 0.0, len(order)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def analyse(df: pd.DataFrame):
    keys = list(variants())
    lev = []
    for (a, v, p), g in df.groupby(["agent", "variant", "partner"]):
        for m in METRICS:
            mean, sd, se, n = stats(g[m])
            lev.append(dict(agent=a, variant=v, partner=p, metric=m, mean=mean, sd=sd, se=se, n=n))
    levels = pd.DataFrame(lev)

    trust = df[df.agent == "trust"]
    base = trust[trust.variant == BASE].set_index(["partner", "run_id"])
    eff = []
    for v in keys[1:]:
        g = trust[trust.variant == v].set_index(["partner", "run_id"])
        for p in PARTNERS:
            for m in METRICS:
                d = (g.loc[p, m] - base.loc[p, m]).dropna()
                if len(d) == 0:
                    continue
                eff.append(dict(variant=v, parameter=variants()[v][0], level=variants()[v][1], value=variants()[v][2],
                                partner=p, metric=m, effect=d.mean(),
                                se=d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else np.nan, n=len(d)))
    effects = pd.DataFrame(eff)

    L = levels.set_index(["agent", "variant", "partner", "metric"])
    con = []
    for v in keys:
        bvar = {"theta_low": "theta_low", "theta_high": "theta_high"}.get(v, BASE)
        for p in PARTNERS:
            for m in METRICS:
                t = L.loc[("trust", v, p, m)]
                if t.n == 0:
                    continue
                for b in BENCH:
                    r = L.loc[(b, bvar, p, m)]
                    if r.n == 0:
                        continue
                    d = t["mean"] - r["mean"]
                    se = math.sqrt(t.se ** 2 + r.se ** 2)
                    pval = math.erfc(abs(d / se) / math.sqrt(2)) if se > 0 else (1.0 if d == 0 else 0.0)
                    con.append(dict(variant=v, parameter=variants()[v][0], level=variants()[v][1],
                                    value=variants()[v][2], partner=p, metric=m, benchmark=b, trust=t["mean"],
                                    bench=r["mean"], diff=d, se=se, p=pval))
    con = pd.DataFrame(con)
    con["p_holm"] = np.nan
    for (m, b), idx in con.groupby(["metric", "benchmark"]).groups.items():
        con.loc[idx, "p_holm"] = holm(con.loc[idx, "p"].to_numpy())
    con["sig"] = con.p_holm < 0.05
    # verdict against the baseline result of the same partner, metric and benchmark
    bl = con[con.variant == BASE].set_index(["partner", "metric", "benchmark"])
    verdict = []
    for r in con.itertuples():
        b0 = bl.loc[(r.partner, r.metric, r.benchmark)]
        if not b0.sig:
            verdict.append("no baseline result")
        elif not r.sig:
            verdict.append("level")
        else:
            verdict.append("holds" if np.sign(r.diff) == np.sign(b0["diff"]) else "reverses")
    con["verdict"] = verdict
    return levels, effects, con


def matched_memory(levels: pd.DataFrame):
    """
    The gamma_x rows change the agent's memory while DiscPP stays at rho = 0.9, so they mix the parameter with a memory
    mismatch. This compares them with DiscPP at the same memory instead (rho = gamma_x), from the T9 sweep
    (results/recency_sweep/cells_long.csv, same seeds and slot; SE recovered from its 95% interval). Its rho = 0.9 cells
    are checked against our DiscPP baseline first.
    """
    sweep = pd.read_csv(RESULTS_DIR / "recency_sweep" / "cells_long.csv")
    sweep = sweep.assign(se=(sweep.hi - sweep.lo) / (2 * Z95)).set_index(["agent", "partner", "metric"])
    L = levels.set_index(["agent", "variant", "partner", "metric"])
    worst = 0.0
    for p in PARTNERS:
        for m in METRICS:
            ours = L.loc[("discpp0.9", BASE, p, m)]
            if ours.n == 0 or (f"disc_pp_{DISCOUNTED_BETA_RHO:.2f}", p, m) not in sweep.index:
                continue
            worst = max(worst, abs(ours["mean"] - sweep.loc[(f"disc_pp_{DISCOUNTED_BETA_RHO:.2f}", p, m), "value"]))
    assert worst < 1e-9, f"T9 DiscPP rho = {DISCOUNTED_BETA_RHO} differs from the DiscPP baseline by {worst}"
    rows = []
    for v in ("gamma_x_low", "gamma_x_high"):
        rho = variants()[v][2]
        for p in PARTNERS:
            for m in METRICS:
                t = L.loc[("trust", v, p, m)]
                key = (f"disc_pp_{rho:.2f}", p, m)
                if t.n == 0 or key not in sweep.index:
                    continue
                b = sweep.loc[key]
                rows.append(dict(variant=v, rho=rho, partner=p, metric=m, diff=t["mean"] - b.value,
                                 se=math.sqrt(t.se ** 2 + b.se ** 2)))
    return pd.DataFrame(rows), f"T9 DiscPP at rho = {DISCOUNTED_BETA_RHO} equals the DiscPP baseline (largest gap {worst:.1e})"


def shock(n_jobs: int) -> pd.DataFrame:
    keys = list(variants())
    res = Parallel(n_jobs=n_jobs)(delayed(shock_profile)(partial(FocalAgent, **variants()[k][3]),
                                                          inv_temp=theta_of(k)) for k in keys)
    return pd.DataFrame({k: r[1] for k, r in zip(keys, res)}).T


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(x) for x in r]) + " |")
    return "\n".join(lines)


def label(v: str) -> str:
    if v == BASE:
        return "baseline"
    param, level, value, _ = variants()[v]
    return f"{param} = {value:g} ({level})"


def matched_table(mm: pd.DataFrame) -> str:
    picks = [("delay_down_sust", ["SingleCycle", "GradualDeterioration", "BreakRepair", "ReliabilityRamp"]),
             ("delay_up_sust", ["BreakRepair", "ReliabilityRamp"]),
             ("FC", ["Random", "SingleCycle", "GradualDeterioration", "StrategicCheater", "ExpectationViolation",
                     "BreakRepair", "ReliabilityRamp", "DistrustLock"]),
             ("Pi", ["StrategicCheater", "ExpectationViolation", "BreakRepair", "DistrustLock"])]
    out = []
    for m, ps in picks:
        scale, nd, _ = SHOW[m]
        tab = {}
        for v in ("gamma_x_low", "gamma_x_high"):
            r = {}
            for p in ps:
                x = mm[(mm.variant == v) & (mm.partner == p) & (mm.metric == m)]
                r[PRETTY[p]] = f"{scale * x['diff'].iloc[0]:+.{nd}f} ({scale * x.se.iloc[0]:.{nd}f})" if len(x) else "."
            tab[f"gamma_x = rho = {variants()[v][2]:g}"] = r
        t = pd.DataFrame(tab).T
        t.index.name = m
        out += [md_table(t), ""]
    return "\n".join(out)


def report(con, sh, checks, mm=None) -> str:
    d = con[con.benchmark == "discpp0.9"]
    keys = list(variants())
    parts = ["# Behavioral OAT sensitivity (trust agent version B, u = 2/3, 12 partners)\n",
             "Generated by `scripts/run_oat_behavioral.py`. Entries are trust minus DiscPP, difference (SE); on the theta rows DiscPP "
             "runs at the same theta. Verdicts use a Holm correction within each metric's table and are counted only where the "
             "result exists at the baseline.\n", "## Checks\n", *[f"- {c}" for c in checks], ""]
    # survival summary
    rows = {}
    for m in METRICS:
        x = d[(d.metric == m) & (d.variant != BASE) & (d.verdict != "no baseline result")]
        vc = x.verdict.value_counts()
        rows[m] = {"baseline results": int(d[(d.metric == m) & (d.variant == BASE) & (d.verdict != "no baseline result")].shape[0]),
                   "cells (settings x partners)": len(x), "holds": int(vc.get("holds", 0)),
                   "level": int(vc.get("level", 0)), "reverses": int(vc.get("reverses", 0))}
    summ = pd.DataFrame(rows).T
    summ.index.name = "metric"
    parts += ["## Survival of the baseline results against DiscPP\n",
              "A baseline result is a partner where trust minus DiscPP is significant at the default setting. Each of the 21 settings "
              "is then scored on those partners: holds, level, or reverses.\n", md_table(summ), ""]
    rev = d[d.verdict == "reverses"]
    if len(rev):
        parts += ["### Reversals\n", md_table(rev.assign(setting=[label(v) for v in rev.variant])
                                               .set_index("setting")[["partner", "metric", "diff", "se"]].round(3)), ""]
    # per-claim tables
    claims = [("delay_down_sust", ["SingleCycle", "GradualDeterioration", "BreakRepair", "ReliabilityRamp"],
               "Detection delay, trust minus DiscPP (rounds; negative = sooner)"),
              ("delay_up_sust", ["BreakRepair", "ReliabilityRamp"], "Repair delay, trust minus DiscPP (rounds; positive = slower)"),
              ("FC", PARTNERS, "Failed collaboration, trust minus DiscPP (points of rounds)"),
              ("Pi", ["StrategicCheater", "ExpectationViolation", "BreakRepair", "DistrustLock"],
               "Payoff, trust minus DiscPP, on the exploiters and the two cost partners"),
              ("partner_pay", EXPLOITERS, "Exploiter's payoff, trust minus DiscPP"),
              ("forgiveness", ["Random", "GradualDeterioration", "BreakRepair", "ReliabilityRamp", "DistrustLock"],
               "Forgiveness, trust minus DiscPP (points)")]
    for m, ps, title in claims:
        scale, nd, _ = SHOW[m]
        tab = {}
        for v in keys:
            r = {}
            for p in ps:
                x = d[(d.variant == v) & (d.partner == p) & (d.metric == m)]
                if len(x):
                    x = x.iloc[0]
                    mark = {"level": " ~", "reverses": " !"}.get(x.verdict, "")
                    r[PRETTY[p]] = f"{scale * x['diff']:+.{nd}f} ({scale * x.se:.{nd}f}){mark}"
                else:
                    r[PRETTY[p]] = "."
            tab[label(v)] = r
        t = pd.DataFrame(tab).T
        t.index.name = "setting"
        parts += [f"## {title}\n", "`~` = no longer significant after the Holm correction, `!` = reversed.\n", md_table(t), ""]
    if mm is not None:
        parts += ["## The gamma_x rows against DiscPP with the same memory\n",
                  "Above, DiscPP stays at rho = 0.9 while the agent's memory moves. Here DiscPP has the same memory as the agent "
                  "(rho = gamma_x, from the T9 sweep): trust minus DiscPP, difference (SE), no correction.\n", matched_table(mm)]
    sh2 = sh[["A_p", "n_lose", "n_regain_min", "regain_over_lose"]].copy()
    sh2.index = [label(k) for k in sh2.index]
    sh2.index.name = "setting"
    parts += ["## Scripted-evidence harness per setting\n",
              "A_p is exactly 1 for every symmetric Beta learner at any setting.\n", md_table(sh2.round(2)), ""]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = ap.parse_args()
    df = with_default_benchmarks(simulate(args.overwrite, args.n_jobs))
    checks = verify(df)
    print("\n".join(checks))
    levels, effects, con = analyse(df)
    mm, mm_check = matched_memory(levels)
    checks.append(mm_check)
    print(mm_check)
    sh = shock(args.n_jobs)
    levels.to_csv(OUT / "levels.csv", index=False)
    effects.to_csv(OUT / "effects.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)
    mm.to_csv(OUT / "matched_memory.csv", index=False)
    sh.to_csv(OUT / "shock.csv")
    (OUT / "report.md").write_text(report(con, sh, checks, mm))
    print(f"wrote {OUT}/per_run.csv, levels.csv, effects.csv, contrasts.csv, shock.csv, report.md")


if __name__ == "__main__":
    main()
