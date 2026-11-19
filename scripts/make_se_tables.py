#!/usr/bin/env python3
"""
Tables 4 to 6 of the paper: main results as mean (SE) and difference (SE).

No new simulations. Reads the per-run metrics of the variant comparison (results/variants/runs_long.csv, from
run_variant_comparison.py) and reports, for the trust agent (version B, the default FocalAgent)
and the three reported benchmarks on the 12 reported partners at u = 2/3:

    level      mean (SE) of each agent, SE = sd / sqrt(n) over the runs where the metric is defined
    contrast   trust minus benchmark, with the SE of the difference

Why the two-sample SE: the trust agent runs in RNG slot 1 and every benchmark in slot 2 (seed + 1_000_000), so
run i of the trust agent and run i of a benchmark share nothing. Their per-run correlation is about 0 (checked
below). SE_diff = sqrt(SE_trust^2 + SE_bench^2). The paired estimate is printed next to it as a check.

The paper judges differences by their size (Section 3.5). For reference, this script also counts differences
under several rules:

    uncorrected   95% interval excludes 0 (the old rule, kept for comparison)
    holm          Holm-adjusted two-sided p < 0.05, one family = every contest in the main tables
                  (9 metrics x 3 benchmarks x the partners where each metric is defined); the strictest family,
                  so a count that survives it survives any narrower one
    holm_table    Holm within each metric (one family per table: partners x benchmarks). Added after the first
                  run showed the all-contest family needs |z| of about 3.7; it is the usual "one table, one
                  family" choice and sits between the other two
    margin        practical margin m, fixed in this docstring before the recount was run:
                    ahead / behind   the whole 95% interval is beyond +m / -m
                    level            the whole 95% interval is inside (-m, +m)
                    unresolved       anything else
                  m is one round's worth of the quantity where the game gives one:
                    Pi, partner_pay      1 payoff point per 100 rounds (R - M = M - P = 1, the smallest
                                         payoff change one round can make)
                    SC, FC               1 point of rounds (one round in 100)
                    delays               1 round
                    crossings            1 crossing per run
                  and a convention where it does not (stated as such, not derived):
                    forgiveness          5 points (reliance in the round after the partner returns)
                    brier_phat           0.01

Outputs (results/tables_se/):
    levels.csv       partner x agent x metric: mean, sd, se, n
    contrasts.csv    partner x benchmark x metric: diff, se, z, p, p_holm, margin, the three rule labels, and
                     the paired estimate for comparison
    claims.csv       each headline count under the three rules
    report.md        the tables formatted as mean (SE) and diff (SE), and the recount

Usage:
    python scripts/make_se_tables.py [--agent B]
"""

import argparse
import math

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.analysis.behavioral_metrics import CATALOG
from trust_bayesian_agent_comparison.config import RESULTS_DIR
from trust_bayesian_agent_comparison.partners import (DIAGNOSTIC_PARTNERS, MANUSCRIPT_PARTNERS,
                                                      PARTNER_CATEGORY)

VAR = RESULTS_DIR / "variants"
OUT = RESULTS_DIR / "tables_se"

BENCH = ["discpp0.9", "win10", "cumulative"]
BENCH_LABEL = {"discpp0.9": "DiscPP 0.9", "win10": "Window 10", "cumulative": "Cumulative"}
ADAPTIVE = ("discpp0.9", "win10")   # crossings is scored against these only (cumulative is inert, not steady)
PARTNERS = list(MANUSCRIPT_PARTNERS) + list(DIAGNOSTIC_PARTNERS)
EXPLOITERS = [p for p in MANUSCRIPT_PARTNERS if PARTNER_CATEGORY[p] == "non-stationary"]
LEAD = ["SingleCycle", "GradualDeterioration", "BreakRepair", "ReliabilityRamp", "StrategicCheater",
        "ExpectationViolation", "DistrustLock"]
PRETTY = {"AlwaysCooperate": "Always Collaborate", "AlwaysDefect": "Always Defect", "Random": "Random",
          "TitForTat": "Tit-for-Tat", "SuspiciousTFT": "Suspicious TFT", "SingleCycle": "Single Cycle",
          "GradualDeterioration": "Gradual Deterioration", "StrategicCheater": "Strategic Cheater",
          "ExpectationViolation": "Expectation Violation", "BreakRepair": "Break and Repair",
          "ReliabilityRamp": "Reliability Ramp", "DistrustLock": "Distrust Lock"}

# metric: (display scale, decimals, margin in stored units, unit)
METRICS = {
    "Pi": (1, 1, 1.0, "payoff points over 100 rounds"),
    "SC": (100, 1, 0.01, "points of rounds, both collaborated"),
    "FC": (100, 2, 0.01, "points of rounds, agent collaborated and the partner did not"),
    "partner_pay": (1, 1, 1.0, "the exploiting partner's payoff points"),
    "delay_down_sust": (1, 2, 1.0, "rounds until reliance is given up for 5 rounds in a row"),
    "crossings": (1, 2, 1.0, "decision-threshold crossings per run"),
    "brier_phat": (1, 4, 0.01, "Brier score of the decision estimate"),
    "delay_up_sust": (1, 2, 1.0, "rounds until reliance is regained for 5 rounds in a row"),
    "forgiveness": (100, 1, 0.05, "points, reliance in the round after the partner returns"),
}
Z95 = 1.959964


def p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (NaN entries are left out of the family and stay NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def stats(x: pd.Series) -> tuple[float, float, float, int]:
    x = x.dropna().to_numpy(float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    sd = x.std(ddof=1) if n > 1 else 0.0
    return x.mean(), sd, sd / math.sqrt(n), n


def defined_partners(levels: pd.DataFrame, metric: str, agent: str) -> list:
    """Partners where the metric exists for the trust agent."""
    lv = levels[(levels.metric == metric) & (levels.agent == agent)].set_index("partner")["mean"]
    ps = [p for p in PARTNERS if p in lv.index and np.isfinite(lv[p])]
    return [p for p in ps if p in EXPLOITERS] if CATALOG[metric][4] == "exploiters" else ps


def label_sig(d, lo, hi, direction):
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo * hi <= 0:
        return "level"
    up = d > 0
    if direction == 0:
        return "higher" if up else "lower"
    return "ahead" if (up == (direction > 0)) else "behind"


def label_margin(d, se, m, direction):
    if not np.isfinite(se):
        return "undefined"
    lo, hi = d - Z95 * se, d + Z95 * se
    if lo > m:
        side = "higher"
    elif hi < -m:
        side = "lower"
    elif lo > -m and hi < m:
        return "level"
    else:
        return "unresolved"
    if direction == 0:
        return side
    return "ahead" if ((side == "higher") == (direction > 0)) else "behind"


def build(runs: pd.DataFrame, agent: str):
    agents = [agent, *BENCH]
    lv = []
    for (p, a), g in runs[runs.agent.isin(agents) & runs.partner.isin(PARTNERS)].groupby(["partner", "agent"]):
        for m in METRICS:
            mean, sd, se, n = stats(g[m])
            lv.append(dict(partner=p, agent=a, metric=m, mean=mean, sd=sd, se=se, n=n))
    levels = pd.DataFrame(lv)
    L = levels.set_index(["partner", "agent", "metric"])

    rows = []
    for m, (_, _, margin, _) in METRICS.items():
        direction = CATALOG[m][2]
        for p in defined_partners(levels, m, agent):
            ta = runs[(runs.agent == agent) & (runs.partner == p)].set_index("run_id")[m]
            for b in BENCH:
                tb = runs[(runs.agent == b) & (runs.partner == p)].set_index("run_id")[m]
                lt, lb = L.loc[(p, agent, m)], L.loc[(p, b, m)]
                if lb.n == 0:
                    continue
                d = lt["mean"] - lb["mean"]
                se = math.sqrt(lt.se ** 2 + lb.se ** 2)
                z = d / se if se > 0 else (0.0 if d == 0 else np.inf * np.sign(d))
                both = pd.concat([ta, tb], axis=1, keys=["a", "b"]).dropna()
                pd_ = (both.a - both.b)
                rows.append(dict(partner=p, benchmark=b, metric=m, direction=direction,
                                 scored=not (m == "crossings" and b not in ADAPTIVE) and direction != 0,
                                 trust=lt["mean"], bench=lb["mean"], diff=d, se=se, z=z,
                                 p=p_two_sided(z) if se > 0 else (1.0 if d == 0 else 0.0), margin=margin,
                                 paired_diff=pd_.mean(), paired_se=pd_.std(ddof=1) / math.sqrt(len(pd_)),
                                 corr=both.a.corr(both.b) if both.a.std() > 0 and both.b.std() > 0 else np.nan))
    con = pd.DataFrame(rows)
    con["p_holm"] = holm(con.p.to_numpy())
    con["p_holm_table"] = np.nan
    for m, idx in con.groupby("metric").groups.items():
        con.loc[idx, "p_holm_table"] = holm(con.loc[idx, "p"].to_numpy())
    con["uncorrected"] = [label_sig(r.diff, r.diff - Z95 * r.se, r.diff + Z95 * r.se, r.direction)
                          for r in con.itertuples()]
    # an interval that excludes 0 exactly when the Holm-adjusted p is below 0.05
    con["holm"] = [label_sig(r.diff, *((r.diff, r.diff) if r.p_holm < 0.05 else (-1.0, 1.0)), r.direction)
                   for r in con.itertuples()]
    con["holm_table"] = [label_sig(r.diff, *((r.diff, r.diff) if r.p_holm_table < 0.05 else (-1.0, 1.0)), r.direction)
                         for r in con.itertuples()]
    con["margin_rule"] = [label_margin(r.diff, r.se, r.margin, r.direction) for r in con.itertuples()]
    return levels, con


def tally(con, metric, bench, partners=None, rule="uncorrected"):
    c = con[(con.metric == metric) & (con.benchmark == bench)]
    if partners is not None:
        c = c[c.partner.isin(partners)]
    return c[rule].value_counts().to_dict(), len(c)


def fmt_counts(d, n, keys):
    return " / ".join(str(d.get(k, 0)) for k in keys) + f" (of {n})"


def claims(con):
    out = []
    rules = ("uncorrected", "holm_table", "holm", "margin_rule")

    def add(claim, metric, bench, partners, keys):
        row = {"claim": claim}
        for r in rules:
            d, n = tally(con, metric, bench, partners, r)
            row[r] = fmt_counts(d, n, keys)
        out.append(row)

    p9 = list(MANUSCRIPT_PARTNERS)
    add("Payoff vs DiscPP, nine manuscript partners: ahead / level / behind / unresolved", "Pi", "discpp0.9", p9,
        ("ahead", "level", "behind", "unresolved"))
    add("Payoff vs DiscPP, twelve partners: ahead / level / behind / unresolved", "Pi", "discpp0.9", None,
        ("ahead", "level", "behind", "unresolved"))
    add("Failed collaboration vs DiscPP: lower (ahead) / level / higher (behind) / unresolved", "FC", "discpp0.9",
        None, ("ahead", "level", "behind", "unresolved"))
    add("Successful collaboration vs DiscPP: higher / level / lower / unresolved", "SC", "discpp0.9", None,
        ("ahead", "level", "behind", "unresolved"))
    add("Detection vs DiscPP: sooner / level / later / unresolved", "delay_down_sust", "discpp0.9", None,
        ("ahead", "level", "behind", "unresolved"))
    add("Repair vs DiscPP (signature): slower / level / faster / unresolved", "delay_up_sust", "discpp0.9", None,
        ("higher", "level", "lower", "unresolved"))
    add("Forgiveness vs DiscPP (signature): lower / level / higher / unresolved", "forgiveness", "discpp0.9", None,
        ("lower", "level", "higher", "unresolved"))
    add("Crossings vs DiscPP: fewer / level / more / unresolved", "crossings", "discpp0.9", None,
        ("ahead", "level", "behind", "unresolved"))
    add("Brier vs DiscPP: lower / level / higher / unresolved", "brier_phat", "discpp0.9", None,
        ("ahead", "level", "behind", "unresolved"))
    # claims that need every benchmark at once
    for claim, metric, want in [("Detection sooner than all three benchmarks: partners", "delay_down_sust", "ahead"),
                                ("Exploiter earns less than against all three benchmarks: partners", "partner_pay", "ahead")]:
        row = {"claim": claim}
        for r in rules:
            c = con[con.metric == metric]
            ps = c.partner.unique()
            k = sum(all(c[(c.partner == p)][r] == want) for p in ps)
            row[r] = f"{k} (of {len(ps)})"
        out.append(row)
    return pd.DataFrame(out).set_index("claim")


def cell(mean, se, scale, nd):
    if not np.isfinite(mean):
        return "."
    return f"{scale * mean:.{nd}f} ({scale * se:.{nd}f})"


def dcell(d, se, scale, nd):
    if not np.isfinite(d):
        return "."
    return f"{scale * d:+.{nd}f} ({scale * se:.{nd}f})"


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [str(v) for v in r]) + " |")
    return "\n".join(lines)


def report(levels, con, cl, agent, checks) -> str:
    L = levels.set_index(["partner", "agent", "metric"])
    parts = [f"# Main results with standard errors (trust version {agent}, u = 2/3)\n",
             "Generated by `scripts/make_se_tables.py` from `results/variants/runs_long.csv`. Each agent has 300 "
             "independent runs per partner; trust and benchmarks run in independent RNG streams. Levels are mean (SE); "
             "contrasts are trust minus benchmark, difference (SE of the difference). About +-2 SE is the 95% interval. "
             "No stars: see `claims` at the end for what survives a multiplicity correction and a practical margin.\n",
             "## Checks\n", *[f"- {c}" for c in checks], ""]
    for m, (scale, nd, margin, unit) in METRICS.items():
        role = "signature, no direction" if CATALOG[m][2] == 0 else ("lower is the scored direction" if CATALOG[m][2] < 0
                                                                     else "higher is the scored direction")
        ps = defined_partners(levels, m, agent)
        rows = {}
        for p in ps:
            r = {f"Trust ({agent})": cell(L.loc[(p, agent, m)]["mean"], L.loc[(p, agent, m)].se, scale, nd)}
            for b in BENCH:
                lb = L.loc[(p, b, m)] if (p, b, m) in L.index else None
                r[BENCH_LABEL[b]] = cell(lb["mean"], lb.se, scale, nd) if lb is not None and lb.n else "."
            for b in BENCH:
                c = con[(con.partner == p) & (con.benchmark == b) & (con.metric == m)]
                r[f"minus {BENCH_LABEL[b]}"] = dcell(c["diff"].iloc[0], c.se.iloc[0], scale, nd) if len(c) else "."
            rows[PRETTY[p]] = r
        df = pd.DataFrame(rows).T
        df.index.name = "partner"
        parts += [f"## {m}: {CATALOG[m][1]}\n", f"Unit: {unit}. Role: {role}. Practical margin: "
                  f"{scale * margin:g}.\n", md_table(df), ""]
    parts += ["## Headline counts under three rules\n",
              "uncorrected = 95% interval excludes 0 (the old stars); holm_table = Holm-adjusted p < 0.05 within "
              f"each metric's table; holm = Holm-adjusted p < 0.05 over all {len(con)} contests in the tables above; "
              "margin = the whole 95% interval beyond the practical margin (ahead or behind), inside it (level), or "
              "neither (unresolved).\n", md_table(cl), ""]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="B", help="trust version key in results/variants (default B, the package default)")
    args = ap.parse_args()

    runs = pd.read_csv(VAR / "runs_long.csv")
    cells = pd.read_csv(VAR / "cells_long.csv")
    levels, con = build(runs, args.agent)

    # identity: our means are the notebook's levels (cells_long), so nothing here is a different result set
    cv = cells.set_index(["partner", "agent", "metric"])["value"]
    lv = levels.set_index(["partner", "agent", "metric"])["mean"]
    common = [k for k in lv.index if k in cv.index and np.isfinite(lv[k]) and np.isfinite(cv[k])]
    worst = max(abs(lv[k] - cv[k]) for k in common)
    assert worst < 1e-9, f"levels differ from results/variants/cells_long.csv by {worst}"
    corr = con["corr"].abs()
    gap = (con["diff"] - con.paired_diff).abs()
    checks = [f"means equal `results/variants/cells_long.csv` on all {len(common)} cells (largest gap {worst:.1e})",
              f"trust vs benchmark per-run correlation: median |r| {corr.median():.3f}, largest {corr.max():.3f} "
              "(independent streams, so the two-sample SE is the right one)",
              f"two-sample vs paired difference: identical on {int((gap < 1e-9).sum())} of {len(con)} contests; "
              f"the rest differ by at most {gap.max():.3f} because pairing drops runs where a delay is undefined "
              "for either agent",
              f"SE ratio paired / two-sample: median {np.nanmedian(con.paired_se / con.se):.3f}"]
    cl = claims(con)

    OUT.mkdir(parents=True, exist_ok=True)
    levels.to_csv(OUT / "levels.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)
    cl.to_csv(OUT / "claims.csv")
    (OUT / "report.md").write_text(report(levels, con, cl, args.agent, checks))
    print("\n".join(checks))
    print()
    print(cl.to_string())
    print(f"\nwrote {OUT}/levels.csv, contrasts.csv, claims.csv, report.md")


if __name__ == "__main__":
    main()
