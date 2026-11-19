#!/usr/bin/env python3
"""
Mechanism ablation of the trust-sensitive agent (Table B1 of the paper).

Each variant switches off one mechanism and keeps every other parameter at its default:
    full                 the default FocalAgent: nu = 3 on unprovoked failed collaborations and psi(u) on surprises
                         in the belief memory, graded confidence, t_max = 10
    no_recency           memory_discount = trust_discount = 1 (cumulative belief and consistency)
    frozen_confidence    t starts at t_max and never moves (no confidence dynamics)
    symmetric            exposed_betrayal_weight = surprise_weight = 1 (violations and surprises weigh like matches in
                         the belief memory: the outcome weights are removed from the only state that carries them)
    no_smoothing         trust_smoothing = 1 (confidence jumps straight to its target)
    no_noise             noise_sigma = 0 (no belief-update noise)
    no_exposure          exposed_betrayal_weight = 1 (nu off)
    no_surprise          surprise_weight = 1 (psi off: a surprise weighs like a match)
    no_attribution       attribution = False (every exposed betrayal weighs nu, provoked or not)
    hit_rate             confidence_scoring = 'hit' (the submitted hard 0/1 expectation)
    submitted            submitted_v1=True: the model as originally submitted, a reference row
symmetric_high is a control for the submitted specification only, and its decomposition below is reported for
the submitted row.

Controls for confidence level versus confidence dynamics:
    fixed_t<k>           confidence held at level k from round 0, k in FIXED_CONFIDENCE_GRID
                         (frozen_confidence is k = 10, the ceiling). frozen_confidence removes the dynamics AND raises
                         the level (the full model averages t of about 4.6 against Expectation Violation),
                         so the level-matched controls separate confidence level from confidence dynamics.
    symmetric_high       (submitted spec only; see submitted_symmetric_high) loss_aversion = lambda_surprise = 3. symmetric removes the asymmetry AND cuts the
                         violation penalty from 3 to 1, so full minus symmetric splits exactly into
                         asymmetry (full minus symmetric_high: only mu changes) plus penalty magnitude
                         (symmetric_high minus symmetric).
The frontier analysis asks whether the dynamic model beats every fixed confidence level on the payoff
trade-off between a stable partner and an exploiting one. Fixed levels are joined in order of k by
straight lines (intermediate levels), the full model's vertical and horizontal distance to that curve
is measured, and bootstrap CIs resample runs (the same run indices for every variant, keeping the
pairing). It also reports whether any single fixed level does at least as well on both partners.

Every variant runs as agent1, the same RNG stream the trust agent had in the main study and in
run_benchmark_comparison.py. The variants consume random numbers in the same order (the noise draw
happens even when sigma = 0), so run i of every variant shares its random numbers with run i of
the full model and the differences are paired. The full model reproduces the main-study trust runs.
Agent2 is the cumulative Bayesian benchmark and is only there because MonteCarloManager is pairwise.

Usage:
    python scripts/run_ablation_study.py                  # reuse cached runs where present
    python scripts/run_ablation_study.py --overwrite      # regenerate every run
    python scripts/run_ablation_study.py --runs 20 --out results/ablation_smoke --overwrite
"""

import argparse
from functools import partial
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.analysis import MonteCarloManager, per_run_metrics
from trust_bayesian_agent_comparison.analysis import paired_ci, sig_mark
from trust_bayesian_agent_comparison.partners import MANUSCRIPT_PARTNERS, PARTNER_CATEGORY, partner_factory
from trust_bayesian_agent_comparison.config import (
    RESULTS_DIR,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    MC_BASE_SEED,
    N_JOBS,
    TRUST_MAX,
    AGENT_T_MAX,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    SLIDING_WINDOW_W,
    FIXED_CONFIDENCE_GRID,
)

# Variant key -> (display label, FocalAgent keyword overrides)
VARIANTS = {
    "full": ("Full", {}),
    "no_recency": ("No recency", dict(memory_discount=1.0, trust_discount=1.0)),
    "frozen_confidence": ("Frozen conf.", dict(t_init=AGENT_T_MAX, trust_smoothing=0.0)),
    "symmetric": ("Symmetric", dict(exposed_betrayal_weight=1.0, surprise_weight=1.0)),
    "no_smoothing": ("No smoothing", dict(trust_smoothing=1.0)),
    "no_noise": ("No noise", dict(noise_sigma=0.0)),
    "no_exposure": ("No nu", dict(exposed_betrayal_weight=1.0)),
    "no_surprise": ("No psi", dict(surprise_weight=1.0)),
    "no_attribution": ("No attribution", dict(attribution=False)),
    "hit_rate": ("Hit-rate conf.", dict(confidence_scoring="hit")),
    "submitted": ("Submitted", dict(submitted_v1=True)),
}
ABLATIONS = ["no_recency", "frozen_confidence", "symmetric", "no_smoothing", "no_noise",
             "no_exposure", "no_surprise", "no_attribution", "hit_rate"]

FIXED_LEVEL_KEYS = {lvl: ("frozen_confidence" if lvl == AGENT_T_MAX else f"fixed_t{lvl:g}") for lvl in FIXED_CONFIDENCE_GRID}
VARIANTS.update({key: (f"Fixed t={lvl:g}", dict(t_init=lvl, trust_smoothing=0.0))
                 for lvl, key in FIXED_LEVEL_KEYS.items() if key != "frozen_confidence"})
VARIANTS["submitted_symmetric"] = ("Submitted, sym. (1,1)", dict(submitted_v1=True, loss_aversion=1.0, lambda_surprise=1.0))
VARIANTS["submitted_symmetric_high"] = ("Submitted, sym. (3,3)", dict(submitted_v1=True, loss_aversion=3.0, lambda_surprise=3.0))
MAIN_VARIANTS = ["full", *ABLATIONS, "submitted"]

EXPLOITERS = ("ExpectationViolation", "StrategicCheater")
N_BOOT = 2000

# Benchmarks from run_benchmark_comparison.py shown for reference (agent2 runs there)
BENCH_DIR = RESULTS_DIR / "benchmarks"
REFERENCES = {
    "cumulative": "Cumulative",
    RECENCY_BENCHMARK_KEY: RECENCY_BENCHMARK_LABEL,
    f"window_w{SLIDING_WINDOW_W}": f"Window w={SLIDING_WINDOW_W}",
}

TRAJECTORY_COLS = ["round", "agent_belief", "agent_action", "partner_action", "agent_payoff", "run_id"]


def run_all(partners, num_runs, out_dir, overwrite, n_jobs):
    """Run every variant. Returns ({variant: {partner: agent1 df}}, {variant: {partner: agent2 df}})."""
    variant_runs = {key: {} for key in VARIANTS}
    other_runs = {key: {} for key in VARIANTS}
    for key, (label, overrides) in VARIANTS.items():
        mc = MonteCarloManager(results_dir=out_dir / key)
        for p in partners:
            print(f"\n--- {label} vs {p}")
            mc.run_monte_carlo(
                agent1_factory=partial(FocalAgent, **overrides),
                agent2_factory=BayesianFocalAgent,
                partner_factory=partner_factory(p, NUM_ROUNDS),
                partner_name=p,
                num_runs=num_runs,
                base_seed=MC_BASE_SEED,
                num_rounds=NUM_ROUNDS,
                n_jobs=n_jobs,
                overwrite=overwrite,
            )
            # Analyse what is on disk so fresh and cached runs are parsed the same way
            df1, df2 = (pd.read_csv(mc.results_dir / f"{p}_{a}.csv") for a in ("agent1", "agent2"))
            for df in (df1, df2):
                if df["run_id"].nunique() != num_runs or df["meta_base_seed"].iloc[0] != MC_BASE_SEED:
                    raise RuntimeError(f"Cached results in {out_dir / key} do not match --runs {num_runs}; "
                                       f"rerun with --overwrite")
            variant_runs[key][p], other_runs[key][p] = df1, df2
    return variant_runs, other_runs


def run_checks(variant_runs, other_runs, partners, num_runs):
    """Sanity checks that the ablation does what it claims."""
    # Agent2 does not depend on agent1, so it must be identical in every variant's run
    for p in partners:
        ref = other_runs["full"][p][TRAJECTORY_COLS].to_numpy(float)
        for key in ABLATIONS:
            if not np.array_equal(ref, other_runs[key][p][TRAJECTORY_COLS].to_numpy(float)):
                raise AssertionError(f"Agent2 runs differ between 'full' and '{key}' on {p}")
    print("Agent2 runs identical across all variants (RNG isolation holds).")

    # Frozen confidence must really be frozen
    for lvl, key in FIXED_LEVEL_KEYS.items():
        for p in partners:
            t = variant_runs[key][p]["trust_level"].to_numpy()
            if not np.all(t == lvl):
                raise AssertionError(f"Fixed-confidence variant {key} moved t on {p}")
    print(f"Every fixed-confidence variant (t = {', '.join(f'{v:g}' for v in FIXED_CONFIDENCE_GRID)}) "
          f"holds its level in every round.")

    # The full model must reproduce the trust-agent runs of the benchmark comparison and main study
    matched = 0
    for p in partners:
        f = BENCH_DIR / "cumulative" / f"{p}_agent1.csv"
        if num_runs == NUM_MONTE_CARLO_RUNS and f.exists():
            cached = pd.read_csv(f)[TRAJECTORY_COLS].to_numpy(float)
            if not np.array_equal(variant_runs["full"][p][TRAJECTORY_COLS].to_numpy(float), cached):
                raise AssertionError(f"Full model differs from the trust-agent runs in {f}")
            matched += 1
    if matched:
        print(f"Full model identical to the benchmark-comparison trust runs on {matched} partners.")


def summarize(variant_runs, partners):
    """Long table: one row per partner x variant, with paired differences full - variant."""
    rows = []
    for p in partners:
        full = per_run_metrics(variant_runs["full"][p])
        for key in VARIANTS:
            m = per_run_metrics(variant_runs[key][p])
            row = {
                "partner": p, "category": PARTNER_CATEGORY[p], "variant": key, "label": VARIANTS[key][0],
                "Pi": m.Pi.mean(), "Pi_sd": m.Pi.std(), "HB": m.HB.mean(),
                "EF": 100 * m.Pi.mean() / m.HB.mean(),
                "SC": 100 * m.SC.mean(), "FC": 100 * m.FC.mean(),
            }
            # Paired by run_id: positive = the switched-off mechanism helps the full model
            for metric, scale in (("Pi", 1), ("SC", 100), ("FC", 100)):
                d = scale * (full[metric] - m[metric])
                se = d.std() / np.sqrt(len(d)) if key != "full" else 0.0
                row[f"worth_{metric}"] = d.mean()
                row[f"worth_{metric}_lo"], row[f"worth_{metric}_hi"] = d.mean() - 1.96 * se, d.mean() + 1.96 * se
            rows.append(row)
    return pd.DataFrame(rows)


def reference_metrics(partners):
    """Per-run metrics for the reference benchmarks, if run_benchmark_comparison.py has been run."""
    refs = {}
    for key in REFERENCES:
        files = {p: BENCH_DIR / key / f"{p}_agent2.csv" for p in partners}
        if all(f.exists() for f in files.values()):
            refs[key] = {p: per_run_metrics(pd.read_csv(f)) for p, f in files.items()}
    return refs


def welch(a, b):
    """Difference of means a - b with an unpaired 95% CI."""
    d = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return d, d - 1.96 * se, d + 1.96 * se


def mark(lo, hi):
    return "+" if lo > 0 else ("-" if hi < 0 else "")


def print_tables(summary, variant_runs, refs, partners):
    by = summary.set_index(["partner", "variant"])
    labels = {k: v[0] for k, v in VARIANTS.items()}

    print(f"\n{'=' * 120}\nTotal payoff by variant (reference benchmarks on the right)\n{'=' * 120}")
    print(f"{'Partner':22s} {'category':15s}" + "".join(f"{labels[k]:>14s}" for k in MAIN_VARIANTS)
          + "".join(f"{REFERENCES[k]:>15s}" for k in refs))
    for p in partners:
        cells = "".join(f"{by.loc[(p, k), 'Pi']:14.1f}" for k in MAIN_VARIANTS)
        ref_cells = "".join(f"{refs[k][p].Pi.mean():15.1f}" for k in refs)
        print(f"{p:22s} {PARTNER_CATEGORY[p]:15s}{cells}{ref_cells}")

    for metric, title in (("Pi", "total payoff"), ("FC", "failed collaboration, pp (negative = mechanism lowers FC)"),
                          ("SC", "successful collaboration, pp")):
        print(f"\n{'=' * 120}\nWhat each mechanism is worth: full minus variant, {title}\n"
              f"paired over runs, mean [95% CI]\n{'=' * 120}")
        print(f"{'Partner':22s}" + "".join(f"{labels[k]:>20s}" for k in ABLATIONS))
        for p in partners:
            cells = []
            for k in ABLATIONS:
                r = by.loc[(p, k)]
                lo, hi = r[f"worth_{metric}_lo"], r[f"worth_{metric}_hi"]
                cells.append(f"{r[f'worth_{metric}']:+6.1f} [{lo:+5.1f},{hi:+5.1f}]{mark(lo, hi):1s}")
            print(f"{p:22s}" + "".join(f"{c:>20s}" for c in cells))

    if refs:
        print(f"\n{'=' * 120}\nVariant minus reference benchmark, total payoff, mean [95% CI] (unpaired)\n{'=' * 120}")
        for ref_key, ref_label in REFERENCES.items():
            if ref_key not in refs:
                continue
            print(f"\n  vs {ref_label}")
            print(f"  {'Partner':22s}" + "".join(f"{labels[k]:>20s}" for k in MAIN_VARIANTS))
            for p in partners:
                cells = []
                for k in MAIN_VARIANTS:
                    d, lo, hi = welch(per_run_metrics(variant_runs[k][p]).Pi, refs[ref_key][p].Pi)
                    cells.append(f"{d:+6.1f} [{lo:+5.1f},{hi:+5.1f}]{mark(lo, hi):1s}")
                print(f"  {p:22s}" + "".join(f"{c:>20s}" for c in cells))


def print_fixed_levels(variant_runs, partners):
    """Payoff at every fixed confidence level, next to the dynamic model, with paired differences."""
    rows = []
    print(f"\n{'=' * 120}\nCONFIDENCE LEVEL VS DYNAMICS: total payoff with t fixed at each level from round 0, "
          f"and the full (dynamic) model\n(* = full minus fixed level, paired 95% CI excludes 0; "
          f"mean t = the full model's average confidence)\n{'=' * 120}")
    print(f"{'Partner':22s}" + "".join(f"{'t=' + format(l, 'g'):>9s}" for l in FIXED_CONFIDENCE_GRID)
          + f"{'Full':>9s}{'mean t':>8s}")
    for p in partners:
        full = per_run_metrics(variant_runs["full"][p])
        cells = []
        for lvl, key in FIXED_LEVEL_KEYS.items():
            m = per_run_metrics(variant_runs[key][p])
            d, lo, hi = paired_ci(full.Pi, m.Pi)
            dfc, lofc, hifc = paired_ci(full.FC, m.FC, 100)
            rows.append({"partner": p, "level": lvl, "variant": key, "Pi": m.Pi.mean(), "FC": 100 * m.FC.mean(),
                         "full_minus_Pi": d, "full_minus_Pi_lo": lo, "full_minus_Pi_hi": hi,
                         "full_minus_FC": dfc, "full_minus_FC_lo": lofc, "full_minus_FC_hi": hifc})
            cells.append(f"{m.Pi.mean():8.1f}{'*' if lo > 0 or hi < 0 else ' '}")
        t_full = variant_runs["full"][p]["trust_level"].mean()
        print(f"{p:22s}" + "".join(cells) + f"{full.Pi.mean():9.1f}{t_full:8.2f}")
    return pd.DataFrame(rows)


def frontier_point(a_lvl, e_lvl, a_full, e_full):
    """
    Distance of the full model from the fixed-level curve in the (stable, exploiter) payoff plane.

    The fixed levels, in order of t, are joined by straight lines. gap_e is the full model's exploiter
    payoff minus the best curve point with the same stable payoff; gap_a the other way round. NaN when
    the full model's payoff lies outside the curve's range. dominated: some single fixed level is at
    least as good on both partners.
    """
    def gap(x_lvl, y_lvl, x_full, y_full):
        best = -np.inf
        for i in range(len(x_lvl) - 1):
            x0, x1, y0, y1 = x_lvl[i], x_lvl[i + 1], y_lvl[i], y_lvl[i + 1]
            if min(x0, x1) <= x_full <= max(x0, x1):
                y = y0 if x1 == x0 else y0 + (y1 - y0) * (x_full - x0) / (x1 - x0)
                best = max(best, y)
        return y_full - best if np.isfinite(best) else np.nan

    return (gap(a_lvl, e_lvl, a_full, e_full), gap(e_lvl, a_lvl, e_full, a_full),
            bool(np.any((a_lvl >= a_full) & (e_lvl >= e_full))))


def frontier_analysis(variant_runs, partners, n_boot=N_BOOT, seed=0):
    """Frontier gaps for every (stable partner, exploiter) pair, with bootstrap 95% CIs."""
    keys = list(FIXED_LEVEL_KEYS.values())
    pi = {p: np.vstack([per_run_metrics(variant_runs[k][p]).Pi.to_numpy() for k in keys + ["full"]])
          for p in partners}
    rng = np.random.default_rng(seed)
    n = pi[partners[0]].shape[1]
    rows = []
    for stable, exploiter in product([p for p in partners if p not in EXPLOITERS], EXPLOITERS):
        A, E = pi[stable], pi[exploiter]
        point = frontier_point(A[:-1].mean(1), E[:-1].mean(1), A[-1].mean(), E[-1].mean())
        boot = []
        for _ in range(n_boot):
            ia, ie = rng.integers(0, n, n), rng.integers(0, n, n)
            a, e = A[:, ia].mean(1), E[:, ie].mean(1)
            boot.append(frontier_point(a[:-1], e[:-1], a[-1], e[-1]))
        boot = np.array(boot, dtype=float)
        row = {"stable": stable, "exploiter": exploiter,
               "full_stable": A[-1].mean(), "full_exploiter": E[-1].mean(),
               "gap_exploiter": point[0], "gap_stable": point[1], "dominated": point[2],
               "p_dominated_boot": boot[:, 2].mean()}
        for j, name in ((0, "gap_exploiter"), (1, "gap_stable")):
            valid = boot[:, j][np.isfinite(boot[:, j])]
            row[f"{name}_defined_boot"] = len(valid) / n_boot
            row[f"{name}_lo"], row[f"{name}_hi"] = (np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan))
        rows.append(row)
    out = pd.DataFrame(rows)

    print(f"\n{'=' * 120}\nFRONTIER: does the dynamic model beat every fixed confidence level on the trade-off "
          f"between a stable partner and an exploiter?\ngap_X = full model's payoff on X minus the fixed-level "
          f"curve at the full model's payoff on the other partner; [bootstrap 95% CI]; n/a = outside the curve.\n"
          f"dominated = some fixed level is at least as good on both (P over {n_boot} bootstrap samples)\n{'=' * 120}")
    print(f"{'stable partner':22s}{'exploiter':22s}{'gap on exploiter':>26s}{'gap on stable':>26s}{'dominated':>14s}")

    def fmt(r, name):
        if not np.isfinite(r[name]):
            return "n/a"
        return f"{r[name]:+.1f} [{r[name + '_lo']:+.1f},{r[name + '_hi']:+.1f}]"

    for _, r in out.iterrows():
        print(f"{r.stable:22s}{r.exploiter:22s}{fmt(r, 'gap_exploiter'):>26s}{fmt(r, 'gap_stable'):>26s}"
              f"{str(r.dominated) + f' ({r.p_dominated_boot:.2f})':>14s}")
    return out


def fixed_level_overall(variant_runs, partners):
    """
    Per fixed level: on how many partners it is significantly better / worse than the dynamic model
    (paired), whether it is at least as good on every partner (point estimates), and the payoff
    averaged over partners with equal weight (paired by run index, since run i of every variant and
    partner uses seed base + i in the agent-1 slot).
    """
    full = {p: per_run_metrics(variant_runs["full"][p]).Pi.to_numpy() for p in partners}
    full_avg = np.mean([full[p] for p in partners], axis=0)
    rows = []
    for lvl, key in FIXED_LEVEL_KEYS.items():
        m = {p: per_run_metrics(variant_runs[key][p]).Pi.to_numpy() for p in partners}
        better = worse = 0
        for p in partners:
            _, lo, hi = paired_ci(m[p], full[p])
            better += lo > 0
            worse += hi < 0
        d, lo, hi = paired_ci(full_avg, np.mean([m[p] for p in partners], axis=0))
        rows.append({"level": lvl, "n_better": better, "n_worse": worse,
                     "weakly_dominates": all(m[p].mean() >= full[p].mean() for p in partners),
                     "full_minus_avg": d, "full_minus_avg_lo": lo, "full_minus_avg_hi": hi})
    out = pd.DataFrame(rows)
    print(f"\n{'=' * 120}\nALL PARTNERS: each fixed confidence level against the dynamic model\n"
          f"better/worse = partners where the fixed level is significantly better/worse (paired 95% CI);\n"
          f"dominates = at least as good on all {len(partners)} partners (point estimates);\n"
          f"avg gap = full minus fixed level, payoff averaged over partners, paired by run [95% CI]\n{'=' * 120}")
    print(f"{'level':>8s}{'better':>8s}{'worse':>8s}{'dominates':>11s}{'avg gap':>24s}")
    for _, r in out.iterrows():
        print(f"{'t=' + format(r.level, 'g'):>8s}{r.n_better:8d}{r.n_worse:8d}{str(r.weakly_dominates):>11s}"
              f"{f'{r.full_minus_avg:+.2f} [{r.full_minus_avg_lo:+.2f},{r.full_minus_avg_hi:+.2f}]':>24s}")
    return out


def asymmetry_decomposition(variant_runs, partners):
    """Submitted specification: submitted - sym(1,1) (B2) = (submitted - sym(3,3)) [asymmetry] + (sym(3,3) - sym(1,1)) [magnitude]."""
    rows = []
    for p in partners:
        full, sym1, sym3 = (per_run_metrics(variant_runs[k][p]) for k in ("submitted", "submitted_symmetric", "submitted_symmetric_high"))
        for metric, scale in (("Pi", 1), ("FC", 100), ("SC", 100)):
            row = {"partner": p, "metric": metric}
            for name, (a, b) in (("b2_asymmetry", (full, sym1)), ("asymmetry", (full, sym3)),
                                 ("magnitude", (sym3, sym1))):
                d, lo, hi = paired_ci(a[metric], b[metric], scale)
                row.update({name: d, f"{name}_lo": lo, f"{name}_hi": hi})
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric, title in (("Pi", "total payoff"), ("FC", "failed collaboration, pp")):
        print(f"\n{'=' * 120}\nASYMMETRY VS PENALTY MAGNITUDE, {title}, paired mean [95% CI]\n"
              f"SUBMITTED specification. B2 'asymmetry' = submitted (3, 0.5) minus symmetric (1, 1); it splits exactly into\n"
              f"asymmetry = submitted minus (3, 3) [only mu changes]  +  magnitude = (3, 3) minus (1, 1)\n{'=' * 120}")
        print(f"{'Partner':22s}{'B2 asymmetry':>24s}{'= asymmetry':>24s}{'+ magnitude':>24s}")
        for _, r in out[out.metric == metric].iterrows():
            cells = [f"{r[n]:+6.1f} [{r[n + '_lo']:+5.1f},{r[n + '_hi']:+5.1f}]{sig_mark(r[n + '_lo'], r[n + '_hi']) or ' '}"
                     for n in ("b2_asymmetry", "asymmetry", "magnitude")]
            print(f"{r.partner:22s}" + "".join(f"{c:>24s}" for c in cells))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate runs even if cached CSVs exist")
    parser.add_argument("--runs", type=int, default=NUM_MONTE_CARLO_RUNS, help="Monte Carlo runs per partner")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ablation", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Parallel jobs (-1 = all cores)")
    args = parser.parse_args()

    partners = list(MANUSCRIPT_PARTNERS)
    print(f"Variants: {', '.join(v[0] for v in VARIANTS.values())}")
    print(f"Partners: {len(partners)}, runs: {args.runs}, rounds: {NUM_ROUNDS}, base seed: {MC_BASE_SEED}")

    variant_runs, other_runs = run_all(partners, args.runs, args.out, args.overwrite, args.n_jobs)
    run_checks(variant_runs, other_runs, partners, args.runs)

    summary = summarize(variant_runs, partners)
    print_tables(summary, variant_runs, reference_metrics(partners) if args.runs == NUM_MONTE_CARLO_RUNS else {},
                 partners)

    levels = print_fixed_levels(variant_runs, partners)
    frontier = frontier_analysis(variant_runs, partners)
    overall = fixed_level_overall(variant_runs, partners)
    asym = asymmetry_decomposition(variant_runs, partners)

    out = args.out / "ablation_summary.csv"
    summary.round(4).to_csv(out, index=False)
    print(f"\nSummary saved to: {out}")
    for df, name in ((levels, "ablation_fixed_levels.csv"), (frontier, "ablation_frontier.csv"),
                     (overall, "ablation_fixed_levels_overall.csv"), (asym, "ablation_asymmetry.csv")):
        df.round(4).to_csv(args.out / name, index=False)
    print(f"Control analyses saved to: {args.out}/ablation_{{fixed_levels,frontier,fixed_levels_overall,asymmetry}}.csv")


if __name__ == "__main__":
    main()
