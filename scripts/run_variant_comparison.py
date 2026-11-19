#!/usr/bin/env python3
"""
Variant comparison: the submitted agent, the current default and every model in between, side by side.
Its runs_long.csv (version B and the benchmarks) is the input of make_se_tables.py, i.e. Tables 4 to 6.

The default FocalAgent is four independent switches away from the submitted agent:

    NU      nu = 3 on an unprovoked exposed failed collaboration in the belief memory, lambda = 1 in t
            (OFF: nu = 1, lambda = 3 in the consistency measure, i.e. the submitted location of the asymmetry)
    PSI     psi(u) = c (1 - u) / u = 0.75 in the belief memory, mu = 1 in t
            (OFF: psi = 1, mu = 0.5 in t)
    GRADED  confidence scores the probability the belief gave to what happened
            (OFF: the submitted hard 0/1 forecast at x >= 1/2)
    ETA     belief step 0.85 (OFF: the submitted 0.70)

This script runs all 16 combinations plus three named extras that the switches cannot express (F, G, H below),
scores them with the metric panel's own code on the 13 panel partners at u = 2/3, and compares every one of them
with the four Beta benchmarks. Nothing else in the agent moves, and the decision stage is identical everywhere,
so every difference comes from how (x, t) is learned.

Named versions:

    A  x0000  the submitted agent (equals FocalAgent(submitted_v1=True), asserted)
    B  x1111  the current default (equals FocalAgent(), asserted, and reproduces the cached B1 trust runs)
    C  x1101  revised belief, submitted-style confidence: hit rate with no category weights
    D  x1000  relocation only: nu moves to the belief, everything else submitted (same parameter count)
    E  x1100  C with the submitted belief step eta = 0.70
    F  ----   revised belief with the submitted confidence equation verbatim (hit, lambda = 3, mu = 0.5)
    G  ----   an intermediate specification (graded, fixed psi = 1.5, eta = 0.70)
    H  ----   B without the attribution clause (every exposed failed collaboration weighs nu)
    I  x1011  relocation plus graded confidence and the larger step, with NO surprise weight. Promoted from the
              factorial after the first run, because it removes almost the whole Distrust Lock cost

Writes results/variants/ (or results/variants_collaborative/ with --partners collaborative):

    variants.csv     the definition table: switches and the resolved agent parameters of every variant
    cells_long.csv   variant x partner x metric: value with a 95% CI
    diffs_long.csv   variant minus benchmark, Welch CI (the metric panel's convention)
    pdiffs_long.csv  variant minus benchmark, paired on run_id (the B1 convention; per-run metrics only)
    vdiffs_long.csv  variant minus version A and variant minus version B, paired (per-run metrics only)
    runs_long.csv    every per-run metric of every cell
    shock.csv        the scripted-evidence harness (A_p, A_t, engage/lose/regain counts) per variant
    scoreboard.csv   per variant and metric: W/L/~ against the four benchmarks, partner by partner
    traj_mean.csv    round-by-round mean (and IQR) of belief, confidence and actions, named versions only
    traj_runs.csv    the first three runs of every named version, for trajectory plots
    report.txt       the readable tables

Usage:
    python scripts/run_variant_comparison.py                      # 13 panel partners, reuse the stored tables
    python scripts/run_variant_comparison.py --overwrite          # re-simulate and re-score everything
    python scripts/run_variant_comparison.py --boot 50            # fewer bootstrap replicates (faster)
    python scripts/run_variant_comparison.py --partners collaborative --overwrite   # the co-adaptive appendix
    python scripts/run_variant_comparison.py --named-only --overwrite               # skip the 11 unnamed cells
    python scripts/run_variant_comparison.py --partners heldout --overwrite         # held-out partners, A and B only

--partners heldout (exploratory, not reported in the paper) runs the six held-out partners for versions A and B
and the four benchmarks only, into results/variants_heldout/. It runs once: no version may be chosen or changed on these results.
It also writes extras_long.csv, per-run window metrics the frozen expectations need (P3 breach windows, the Reformer's
before/after windows, and RegimeSwitch's detection and repair delays, computed per run from the partner's own p* log,
since the panel tiles one instance's p* across runs and would get that partner wrong).
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
    PERSIST_ESTIMATE,
    _change_points,
    _first_sustained,
    reliance_prob,
    CATALOG,
    ECONOMIC_METRICS,
    N_BOOT,
    PANEL_METRICS,
    POOLED_METRICS,
    SHOCK_CATALOG,
    compare_cells,
    make_trace,
    score_cell,
    shock_profile,
)
from trust_bayesian_agent_comparison.config import (
    AGENT_ETA,
    AGENT_LAMBDA_SURPRISE,
    AGENT_LOSS_AVERSION,
    AGENT_T_MAX,
    ATTRIBUTION,
    DISCOUNTED_BETA_RHO,
    ETA,
    EXPOSED_BETRAYAL_WEIGHT,
    LAMBDA_SURPRISE,
    LOSS_AVERSION,
    MC_BASE_SEED,
    N_JOBS,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    RESULTS_DIR,
    SLIDING_WINDOW_W,
    SURPRISE_SCALE,
    stag_indifference_threshold,
)
from trust_bayesian_agent_comparison.partners import (
    COLLABORATIVE_PARTNERS,
    DIAGNOSTIC_PARTNERS,
    HELDOUT_PARTNERS,
    MANUSCRIPT_PARTNERS,
    PARTNER_CATEGORY,
    partner_factory,
)
from trust_bayesian_agent_comparison.simulation import run_agent_simulation, slot_rng

OUT = RESULTS_DIR / "variants"
BENCH_DIR = RESULTS_DIR / "benchmarks"
PANEL_RUNS = RESULTS_DIR / "metric_panel" / "runs" / "current"
PANEL_RUNS_PLAIN = RESULTS_DIR / "metric_panel_plainDisc" / "runs" / "current"
PARTNERS = list(MANUSCRIPT_PARTNERS) + list(DIAGNOSTIC_PARTNERS)
TRAJ_CHECK = ["agent_belief", "agent_action", "partner_action", "agent_payoff"]
REPORTED = list(ECONOMIC_METRICS) + list(PANEL_METRICS) + ["t_mean"]


# ---------------------------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------------------------
SWITCHES = ("NU", "PSI", "GRADED", "ETA")


def switch_kwargs(nu: int, psi: int, graded: int, eta: int) -> dict:
    """FocalAgent keywords of one factorial cell. All four off is the submitted agent, all four on the default."""
    return dict(
        # NU: the violation weight sits in the belief memory (on) or in the consistency measure (off)
        exposed_betrayal_weight=EXPOSED_BETRAYAL_WEIGHT if nu else 1.0,
        attribution=bool(nu) and ATTRIBUTION,
        loss_aversion=AGENT_LOSS_AVERSION if nu else LOSS_AVERSION,
        # PSI: the surprise weight sits in the belief memory, game-scaled (on), or stays as mu in t (off).
        # surprise_weight=None asks FocalAgent for the rule psi(u) = SURPRISE_SCALE (1 - u) / u.
        surprise_weight=None if psi else 1.0,
        lambda_surprise=AGENT_LAMBDA_SURPRISE if psi else LAMBDA_SURPRISE,
        # GRADED: confidence scores a probability (on) or a hard 0/1 forecast (off)
        confidence_scoring="graded" if graded else "hit",
        # ETA: belief step
        eta=AGENT_ETA if eta else ETA,
    )


NAMED_BITS = {(0, 0, 0, 0): "A", (1, 1, 1, 1): "B", (1, 1, 0, 1): "C", (1, 0, 0, 0): "D", (1, 1, 0, 0): "E",
              (1, 0, 1, 1): "I"}
NAMED_ORDER = "ABCDEI"   # order of the switch-reachable named versions; F, G and H are added after them
LABELS = {
    "A": "A submitted",
    "B": "B current default",
    "C": "C revised x, hit t",
    "D": "D relocation only",
    "E": "E C with eta 0.70",
    "F": "F revised x, submitted t",
    "G": "G intermediate (psi 1.5)",
    "H": "H B without attribution",
    "I": "I relocation + graded t",
}
NOTES = {
    "A": "the submitted proof, git tag submitted-v1",
    "B": "default: nu and psi(u) in x, graded confidence, eta 0.85",
    "C": "the user's version C: current belief, confidence as the old hard forecast with no asymmetry",
    "D": "the smallest change: lambda relocated from the consistency measure to the belief memory",
    "E": "C without the step change, so eta stays at the submitted 0.70",
    "F": "Eq. (6) rewritten, Eqs. (8) and (9) reprinted as submitted (asymmetry then sits in both states)",
    "G": "an intermediate specification: a fixed surprise weight instead of the game-scaled rule",
    "H": "tests whether the provocation clause of nu does any work",
    "I": "B without the surprise weight: keeps the asymmetry and the graded confidence, drops psi (and the Distrust Lock cost)",
}


def build_variants(named_only: bool = False) -> dict:
    """key -> dict(name, label, bits, kwargs, note). Named versions first, then the unnamed factorial cells."""
    variants = {}
    for bits, name in sorted(NAMED_BITS.items(), key=lambda kv: NAMED_ORDER.index(kv[1])):
        variants[name] = dict(name=name, label=LABELS[name], bits=bits, kwargs=switch_kwargs(*bits), note=NOTES[name])
    # F, G, H are not switch-reachable: they pair a revised belief with a confidence rule the switches do not produce
    variants["F"] = dict(name="F", label=LABELS["F"], bits=(1, 1, 0, 1), note=NOTES["F"],
                         kwargs={**switch_kwargs(1, 1, 0, 1), "loss_aversion": LOSS_AVERSION,
                                 "lambda_surprise": LAMBDA_SURPRISE})
    variants["G"] = dict(name="G", label=LABELS["G"], bits=(1, 1, 1, 0), note=NOTES["G"],
                         kwargs={**switch_kwargs(1, 1, 1, 0), "surprise_weight": 1.5})
    variants["H"] = dict(name="H", label=LABELS["H"], bits=(1, 1, 1, 1), note=NOTES["H"],
                         kwargs={**switch_kwargs(1, 1, 1, 1), "attribution": False})
    if named_only:
        return variants
    for bits in [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]:
        if bits in NAMED_BITS:
            continue
        key = "x" + "".join(str(b) for b in bits)
        label = key + " " + "+".join([s for s, b in zip(SWITCHES, bits) if b] or ["none"])
        variants[key] = dict(name=key, label=label, bits=bits, kwargs=switch_kwargs(*bits),
                             note="unnamed factorial cell")
    return variants


BENCHMARKS = {
    "cumulative": ("Cumulative", BayesianFocalAgent, "cumulative", "cumulative"),
    "discpp0.9": (RECENCY_BENCHMARK_LABEL, partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO),
                  RECENCY_BENCHMARK_KEY, "discpp0.9"),
    "disc0.9": (f"Disc plain rho={DISCOUNTED_BETA_RHO:.2f}", partial(DiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO),
                f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}", "disc0.9"),
    "win10": (f"Window w={SLIDING_WINDOW_W}", partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W),
              f"window_w{SLIDING_WINDOW_W}", "win10"),
}
# recency-matched benchmarks (the cumulative model is the non-adaptive reference)
ADAPTIVE = ("discpp0.9", "disc0.9", "win10")
VARIANTS = build_variants()
HELDOUT_VERSIONS = ("A", "B")


def agent_factory(key: str):
    if key in BENCHMARKS:
        return BENCHMARKS[key][1]
    return partial(FocalAgent, **VARIANTS[key]["kwargs"])


def slot_of(key: str) -> int:
    """Trust variants run in RNG slot 1, benchmarks in slot 2, exactly as in B1 and the metric panel."""
    return 2 if key in BENCHMARKS else 1


def variant_table() -> pd.DataFrame:
    rows = []
    for key, v in VARIANTS.items():
        ag = agent_factory(key)()
        rows.append(dict(key=key, label=v["label"], note=v["note"],
                         **{s: b for s, b in zip(SWITCHES, v["bits"])},
                         named=key in LABELS, eta=ag.eta, nu=ag.exposed_betrayal_weight,
                         attribution=ag.attribution, psi=ag.surprise_weight, lam=ag.loss_aversion,
                         mu=ag.lambda_surprise, scoring=ag.confidence_scoring, t_max=ag.t_max,
                         gamma_x=ag.memory_discount, gamma_t=ag.trust_discount, omega=ag.trust_smoothing,
                         sigma=ag.noise_sigma, eps=ag.eps, x0=ag.x, t0=ag.t))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Verification: this comparison must be the same simulation as the tables already in results/
# ---------------------------------------------------------------------------------------------
def cached_source(partner: str, key: str) -> Path | None:
    """The stored run file of one cell, if the repo already has one."""
    if key == "B":
        return (PANEL_RUNS / f"{partner}_trust.csv" if partner in DIAGNOSTIC_PARTNERS
                else BENCH_DIR / "cumulative" / f"{partner}_agent1.csv")
    if key in BENCHMARKS:
        _, _, bench_dir, panel_key = BENCHMARKS[key]
        if partner in DIAGNOSTIC_PARTNERS:
            base = PANEL_RUNS_PLAIN if key == "disc0.9" else PANEL_RUNS
            return base / f"{partner}_{panel_key}.csv"
        return BENCH_DIR / bench_dir / f"{partner}_agent2.csv"
    return None


def verify(n_check: int = 10, partners: list | None = None) -> None:
    """Six identities. Any failure means the variants are not comparable with the existing tables."""
    partners = partners or PARTNERS
    checks = 0
    cols = TRAJ_CHECK + ["trust_level", "signal"]

    def runs(key, partner, n=n_check):
        _, traj = run_cell(agent_factory(key), partner_factory(partner, NUM_ROUNDS), slot_of(key),
                           num_runs=n, return_trajectories=True)
        return traj

    # 1. version A is the submitted agent, bit for bit
    for p in ("StrategicCheater", "TitForTat", "Random"):
        a, b = runs("A", p), None
        _, b = run_cell(partial(FocalAgent, submitted_v1=True), partner_factory(p, NUM_ROUNDS), 1,
                        num_runs=n_check, return_trajectories=True)
        worst = np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float)).max()
        assert worst == 0.0, (p, worst)
    checks += 1
    print(f"1. version A reproduces FocalAgent(submitted_v1=True) bit for bit ({n_check} runs, 3 partners)")

    # 2. version B is the package default, bit for bit
    for p in ("StrategicCheater", "TitForTat", "Random"):
        a = runs("B", p)
        _, b = run_cell(FocalAgent, partner_factory(p, NUM_ROUNDS), 1, num_runs=n_check, return_trajectories=True)
        worst = np.abs(a[cols].to_numpy(float) - b[cols].to_numpy(float)).max()
        assert worst == 0.0, (p, worst)
    checks += 1
    print(f"2. version B reproduces FocalAgent() (the default) bit for bit ({n_check} runs, 3 partners)")

    # 3. version B reproduces the cached main-study trust runs; 4. every benchmark reproduces its cached file.
    # The stored files are CSV text, so the tolerance is float round-trip (about 1e-16), not a model difference.
    missing, worst_b, worst_bench = [], 0.0, 0.0
    for key in ["B"] + list(BENCHMARKS):
        for p in partners:
            src = cached_source(p, key)
            if src is None or not src.exists():
                missing.append(f"{key}/{p}")
                continue
            cached = pd.read_csv(src)
            cached = cached[cached["run_id"] < n_check]
            fresh = runs(key, p)
            worst = float(np.abs(fresh[TRAJ_CHECK].to_numpy(float) - cached[TRAJ_CHECK].to_numpy(float)).max())
            assert worst < 1e-12, (key, p, worst, src)
            if key == "B":
                worst_b = max(worst_b, worst)
            else:
                worst_bench = max(worst_bench, worst)
    checks += 2
    print(f"3. version B reproduces the cached B1 / panel trust runs on every partner (max abs diff {worst_b:.1e})")
    print(f"4. all four benchmarks reproduce their cached B1 / panel runs (max abs diff {worst_bench:.1e})"
          + (f"; no stored file for: {', '.join(missing)}" if missing else ""))

    # 5. every variant is a FocalAgent with the same two states and the same decision stage
    u = stag_indifference_threshold(None)
    for key, v in VARIANTS.items():
        ag = agent_factory(key)()
        assert set(vars(ag)) == set(vars(FocalAgent())), key
        assert ag.t_max == AGENT_T_MAX and ag.eps == 1 and ag.x == 1 - u and ag.t == 0.0, key
        assert ag.memory_discount == 0.9 and ag.trust_discount == 0.9 and ag.trust_smoothing == 0.2, key
        assert ag.inv_temp == 2.0 and ag.stochastic, key
    checks += 1
    print(f"5. all {len(VARIANTS)} variants share the state, the bounds, the memories, x_0, t_0 and the decision stage")

    # 6. the two parameterizations of the surprise weight agree at the lead game
    assert FocalAgent().surprise_weight == round(SURPRISE_SCALE * (1 - u) / u, 10) == 0.75, u
    checks += 1
    print(f"6. psi(u) = c (1 - u)/u is 0.75 at the lead game u = {u:.4f}, c = {SURPRISE_SCALE}")
    print(f"Verification: {checks} checks passed.\n")


def heldout_verify(n_check: int = 5) -> None:
    for p in HELDOUT_PARTNERS:
        for key in ("B", "discpp0.9"):
            a, _ = run_cell_logged(agent_factory(key), partner_factory(p, NUM_ROUNDS), slot_of(key), num_runs=n_check)
            _, b = run_cell(agent_factory(key), partner_factory(p, NUM_ROUNDS), slot_of(key), num_runs=n_check,
                            return_trajectories=True)
            assert np.array_equal(a[TRAJ_CHECK].to_numpy(float), b[TRAJ_CHECK].to_numpy(float)), (p, key)
    print(f"H. the logged run loop reproduces run_cell bit for bit on all {len(HELDOUT_PARTNERS)} held-out partners "
          f"(B and DiscPP, {n_check} runs each)\n")


# ---------------------------------------------------------------------------------------------
# Runs and scoring
# ---------------------------------------------------------------------------------------------
def traj_summary(traj: pd.DataFrame, key: str, partner: str):
    """Round-by-round mean and IQR of the states (all runs), and the first three runs in full."""
    cols = ["agent_belief", "agent_action", "partner_action", "agent_payoff", "trust_level", "signal"]
    cols = [c for c in cols if c in traj.columns]
    g = traj.groupby("round")[cols]
    mean = g.mean().add_suffix("_mean")
    q1 = g.quantile(0.25)[[c for c in ("agent_belief", "trust_level") if c in cols]].add_suffix("_q25")
    q3 = g.quantile(0.75)[[c for c in ("agent_belief", "trust_level") if c in cols]].add_suffix("_q75")
    out = pd.concat([mean, q1, q3], axis=1).reset_index().assign(agent=key, partner=partner)
    ex = traj[traj.run_id < 3].copy()
    ex = ex[["run_id", "round"] + cols].assign(agent=key, partner=partner)
    return out, ex


def run_cell_logged(agent_fac, partner_fac, slot: int, num_runs: int = NUM_MONTE_CARLO_RUNS,
                    base_seed: int = MC_BASE_SEED, num_rounds: int = NUM_ROUNDS):
    """run_cell's trajectories, run for run (asserted in heldout_verify), plus each partner's p* log if it keeps one."""
    frames, logs = [], []
    for i in range(num_runs):
        seed = base_seed + i
        agent = agent_fac()
        partner = partner_fac()
        df = run_agent_simulation(agent, partner, num_rounds, seed, rng=slot_rng(seed, slot))
        df["run_id"] = i
        frames.append(df)
        logs.append(getattr(partner, "pstar_log", None))
    pstar = None if logs[0] is None else np.asarray(logs, float)
    return pd.concat(frames, ignore_index=True), pstar


HELDOUT_WINDOWS = {
    "SC_8_37": ("SC", 8, 38),       # P3, after the early block (rounds 5 to 7)
    "SC_53_82": ("SC", 53, 83),     # P3, after the late block (rounds 50 to 52)
    "q_90_99": ("q", 90, 100),      # P3, residue at the end
    "Pi_8_37": ("Pi", 8, 38),
    "Pi_53_82": ("Pi", 53, 83),
    "SC_40_99": ("SC", 40, 100),    # Reformer, after the reform
    "FC_0_39": ("FC", 0, 40),       # Reformer, before it
}


def heldout_extras(tr, pstar_runs) -> pd.DataFrame:
    """Per-run window metrics; for a partner with a per-run p* log, per-run detection and repair delays as well."""
    sc = (tr.a == 1) & (tr.o == 1)
    fc = (tr.a == 1) & (tr.o == 0)
    series = {"SC": sc.astype(float), "FC": fc.astype(float), "q": tr.q, "Pi": tr.pay}
    out = {}
    for name, (what, lo, hi) in HELDOUT_WINDOWS.items():
        v = series[what][:, lo:hi]
        out[name] = v.sum(1) if what == "Pi" else v.mean(1)
    if pstar_runs is not None:
        above = tr.phat >= tr.u
        det, rep_, n_sw = np.full(tr.n, np.nan), np.full(tr.n, np.nan), np.zeros(tr.n)
        for i in range(tr.n):
            changes, _ = _change_points(pstar_runs[i], tr.u)
            n_sw[i] = len(changes)
            bounds = [c for c, _ in changes] + [tr.K]
            seen = set()
            for j, (c, new_above) in enumerate(changes):
                label = "up" if new_above else "down"
                if label in seen:
                    continue
                seen.add(label)
                flags = above[i:i + 1] if new_above else ~above[i:i + 1]
                d = _first_sustained(flags, c, bounds[j + 1], PERSIST_ESTIMATE)[0]
                if label == "down":
                    det[i] = d
                else:
                    rep_[i] = d
        out["rs_detect"] = det
        out["rs_repair"] = rep_
        out["rs_n_switch"] = n_sw
        out["rs_OPD"] = np.abs(tr.q - reliance_prob(pstar_runs, tr.payoff_matrix)).mean(1)
    return pd.DataFrame(out).assign(run_id=np.arange(tr.n))


def run_and_score(partner: str, key: str, n_boot: int, keep_traj: bool):
    """Simulate one cell (300 runs) and score every metric. Trajectories are summarized, never returned in full."""
    pstar_runs = None
    if partner in HELDOUT_PARTNERS:
        traj, pstar_runs = run_cell_logged(agent_factory(key), partner_factory(partner, NUM_ROUNDS), slot_of(key))
    else:
        traj = run_cell(agent_factory(key), partner_factory(partner, NUM_ROUNDS), slot_of(key),
                        num_runs=NUM_MONTE_CARLO_RUNS, base_seed=MC_BASE_SEED, num_rounds=NUM_ROUNDS,
                        return_trajectories=True)[1]
    tr = make_trace(traj, partner_factory(partner, NUM_ROUNDS)())
    cr = score_cell(tr, n_boot=n_boot, seed=zlib.crc32(f"{partner}|{key}".encode()))
    tsum = traj_summary(traj, key, partner) if keep_traj else None
    extras = (heldout_extras(tr, pstar_runs).assign(partner=partner, agent=key)
              if partner in HELDOUT_PARTNERS else None)
    return cr, tsum, extras


def score_all(partners: list, n_boot: int, n_jobs: int):
    keys = list(VARIANTS) + list(BENCHMARKS)
    cells = [(p, k) for p in partners for k in keys]
    print(f"Simulating and scoring {len(cells)} cells ({NUM_MONTE_CARLO_RUNS} runs, {n_boot} bootstrap replicates)...")
    res = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_and_score)(p, k, n_boot, k in LABELS) for p, k in cells)
    scored = {(p, k): r[0] for (p, k), r in zip(cells, res)}
    traj_mean = pd.concat([r[1][0] for r in res if r[1] is not None], ignore_index=True)
    traj_runs = pd.concat([r[1][1] for r in res if r[1] is not None], ignore_index=True)
    extras = [r[2] for r in res if r[2] is not None]
    extras = pd.concat(extras, ignore_index=True) if extras else None
    return scored, traj_mean, traj_runs, extras


def long_tables(scored: dict, partners: list):
    """cells_long, diffs_long (Welch, panel convention), pdiffs_long and vdiffs_long (paired), runs_long."""
    rows, diffs, pdiffs, vdiffs, runs = [], [], [], [], []
    for (p, k), cr in scored.items():
        rows.append(cr.summary.assign(partner=p, agent=k))
        runs.append(cr.runs.assign(partner=p, agent=k, run_id=np.arange(len(cr.runs))))
    for p in partners:
        for k in VARIANTS:
            cr = scored[(p, k)]
            for b in BENCHMARKS:
                diffs.append(compare_cells(cr, scored[(p, b)]).assign(partner=p, agent=k, benchmark=b))
                pdiffs.append(paired_table(cr, scored[(p, b)]).assign(partner=p, agent=k, benchmark=b))
            for base in ("A", "B"):
                if k == base:
                    continue
                vdiffs.append(paired_table(cr, scored[(p, base)]).assign(partner=p, agent=k, baseline=base))
    return (pd.concat(rows, ignore_index=True), pd.concat(diffs, ignore_index=True),
            pd.concat(pdiffs, ignore_index=True), pd.concat(vdiffs, ignore_index=True),
            pd.concat(runs, ignore_index=True))


def paired_table(a, b) -> pd.DataFrame:
    """Paired difference a - b over matched run ids, for the per-run metrics only (pooled ones get no CI)."""
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


SCOPES = {
    "nonclock": lambda p: p not in ("AlwaysCooperate", "AlwaysDefect"),
    "exploiters": lambda p: p in MANUSCRIPT_PARTNERS and PARTNER_CATEGORY[p] == "non-stationary",
    "breakrepair": lambda p: p == "BreakRepair",
    "ramp": lambda p: p == "ReliabilityRamp",
}


def scoreboard(cells_long: pd.DataFrame, diffs_long: pd.DataFrame, partners: list) -> pd.DataFrame:
    """
    Per variant and metric: on each partner where the variant and all four benchmarks have a value, W if the
    variant is significantly better than every benchmark, L if significantly worse than at least one, ~ otherwise.
    The metric panel's rule, with the plain discounted model added to the benchmark set.
    """
    out = []
    for key in VARIANTS:
        for metric, (fam, label, direction, kind, scope, status) in CATALOG.items():
            if direction == 0 or metric not in REPORTED:
                continue
            marks, marks_ad, wins, losses = [], [], [], []
            for p in partners:
                if scope in SCOPES and not SCOPES[scope](p):
                    continue
                v = cells_long[(cells_long.partner == p) & (cells_long.metric == metric)].set_index("agent")["value"]
                if not {key, *BENCHMARKS}.issubset(set(v.dropna().index)):
                    continue
                d = diffs_long[(diffs_long.partner == p) & (diffs_long.agent == key)
                               & (diffs_long.metric == metric)].set_index("benchmark")
                if len(d) < len(BENCHMARKS):
                    continue
                better = {b: (direction * d.loc[b, "lo"] > 0) if direction > 0 else (d.loc[b, "hi"] < 0)
                          for b in BENCHMARKS}
                worse = {b: (d.loc[b, "hi"] < 0) if direction > 0 else (d.loc[b, "lo"] > 0) for b in BENCHMARKS}
                mark = "W" if all(better.values()) else ("L" if any(worse.values()) else "~")
                marks.append(mark)
                marks_ad.append("W" if all(better[b] for b in ADAPTIVE)
                                else ("L" if any(worse[b] for b in ADAPTIVE) else "~"))
                (wins if mark == "W" else losses if mark == "L" else []).append(p)
            if not marks:
                continue
            out.append(dict(agent=key, label=VARIANTS[key]["label"], metric=metric, family=fam, kind=kind,
                            n_partners=len(marks), W=marks.count("W"), L=marks.count("L"),
                            neutral=marks.count("~"), W_adaptive=marks_ad.count("W"),
                            L_adaptive=marks_ad.count("L"), wins=", ".join(wins), losses=", ".join(losses)))
    return pd.DataFrame(out)


def shock_table(n_jobs: int) -> pd.DataFrame:
    """The scripted-evidence harness per variant and benchmark (same seeds as the metric panel)."""
    keys = list(VARIANTS) + list(BENCHMARKS)
    res = Parallel(n_jobs=n_jobs)(delayed(shock_profile)(agent_factory(k)) for k in keys)
    return pd.DataFrame({k: r[1] for k, r in zip(keys, res)})


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def fmt(v, metric):
    if not np.isfinite(v):
        return "      ."
    if metric in ("Pi", "HB", "partner_pay"):
        return f"{v:7.1f}"
    return f"{v:7.1f}" if abs(v) >= 10 else f"{v:7.3f}"


def report(variants: pd.DataFrame, cells_long, diffs_long, vdiffs_long, board, shock, partners, path: Path):
    named = [k for k in VARIANTS if k in LABELS]
    cols = named + list(BENCHMARKS)
    lines = [f"VARIANT COMPARISON, u = 2/3, {NUM_MONTE_CARLO_RUNS} runs, {len(partners)} partners",
             "Named versions A to H, then the four Beta benchmarks. Definitions: see the module docstring.",
             "Marks on a variant: W = significantly better than all four benchmarks, L = significantly worse than at",
             "least one, ~ = neither (95% CI, no multiplicity correction).", ""]
    add = lines.append
    add("=" * 120)
    add("VARIANT DEFINITIONS")
    add("=" * 120)
    add(variants[["key", "label", *SWITCHES, "eta", "nu", "attribution", "psi", "lam", "mu", "scoring"]]
        .to_string(index=False))
    for metric in REPORTED:
        fam, label, direction, kind, scope, status = CATALOG[metric]
        sub = cells_long[cells_long.metric == metric]
        if sub.empty:
            continue
        add("")
        add("=" * 120)
        add(f"[{fam}] {metric}: {label}   (direction {direction:+d}, {kind}, scope {scope})")
        add("=" * 120)
        add(f"{'partner':<22}" + "".join(f"{c:>12}" for c in cols))
        for p in partners:
            if scope in SCOPES and not SCOPES[scope](p):
                continue
            v = sub[sub.partner == p].set_index("agent")["value"]
            if v.reindex(cols).isna().all():
                continue
            cells = []
            for c in cols:
                val = v.get(c, np.nan)
                mark = " "
                if c in VARIANTS and direction != 0 and np.isfinite(val):
                    d = diffs_long[(diffs_long.partner == p) & (diffs_long.agent == c)
                                   & (diffs_long.metric == metric)].set_index("benchmark")
                    if len(d) == len(BENCHMARKS):
                        better = [(direction * d.loc[b, "lo"] > 0) if direction > 0 else (d.loc[b, "hi"] < 0)
                                  for b in BENCHMARKS]
                        worse = [(d.loc[b, "hi"] < 0) if direction > 0 else (d.loc[b, "lo"] > 0) for b in BENCHMARKS]
                        mark = "W" if all(better) else ("L" if any(worse) else "~")
                cells.append(fmt(val, metric).strip() + mark)
            add(f"{p:<22}" + "".join(f"{x:>12}" for x in cells))
    add("")
    add("=" * 120)
    add("PAIRED DIFFERENCES AGAINST VERSION A (submitted), economic metrics, '*' = 95% CI excludes 0")
    add("=" * 120)
    for metric in ("Pi", "SC", "FC"):
        add(f"\n{metric}:")
        add(f"{'partner':<22}" + "".join(f"{c:>12}" for c in named if c != "A"))
        for p in partners:
            row = []
            for c in named:
                if c == "A":
                    continue
                d = vdiffs_long[(vdiffs_long.partner == p) & (vdiffs_long.agent == c)
                                & (vdiffs_long.baseline == "A") & (vdiffs_long.metric == metric)]
                if d.empty:
                    row.append(".")
                    continue
                r = d.iloc[0]
                sc = 100.0 if metric in ("SC", "FC") else 1.0
                row.append(f"{sc * r['diff']:+.2f}" + ("*" if r.lo * r.hi > 0 else ""))
            add(f"{p:<22}" + "".join(f"{x:>12}" for x in row))
    add("")
    add("=" * 120)
    add("SHOCK HARNESS (scripted evidence, identical for every agent)")
    add("=" * 120)
    add(shock.loc[[i for i in shock.index if i in SHOCK_CATALOG], cols].round(3).to_string())
    add("")
    add("=" * 120)
    add("SCOREBOARD, named versions (W/L against all four benchmarks; W_adaptive against the three recency models)")
    add("=" * 120)
    b = board[board.agent.isin(named)]
    add(b[["agent", "metric", "n_partners", "W", "L", "neutral", "W_adaptive", "L_adaptive"]]
        .pivot_table(index="metric", columns="agent", values=["W", "L"], aggfunc="first").to_string())
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------------------------
def main():
    global OUT, PARTNERS, VARIANTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="re-simulate and re-score every cell")
    ap.add_argument("--boot", type=int, default=N_BOOT, help=f"bootstrap replicates (default {N_BOOT})")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    ap.add_argument("--partners", choices=("panel", "collaborative", "heldout"), default="panel")
    ap.add_argument("--named-only", action="store_true", help="skip the 11 unnamed factorial cells")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    if args.named_only:
        VARIANTS = build_variants(named_only=True)
    partners = PARTNERS
    if args.partners == "collaborative":
        partners = list(COLLABORATIVE_PARTNERS)
        OUT = RESULTS_DIR / "variants_collaborative"
    if args.partners == "heldout":
        partners = list(HELDOUT_PARTNERS)
        OUT = RESULTS_DIR / "variants_heldout"
        VARIANTS = {k: v for k, v in build_variants(named_only=True).items() if k in HELDOUT_VERSIONS}
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Variants: {len(VARIANTS)} ({', '.join(k for k in VARIANTS if k in LABELS)} named), "
          f"benchmarks: {len(BENCHMARKS)}, partners: {len(partners)}\n")
    variants = variant_table()
    variants.to_csv(OUT / "variants.csv", index=False)

    if not args.skip_verify:
        verify(partners=[p for p in partners if p in PARTNERS] or None)
        if args.partners == "heldout":
            heldout_verify()

    done = (OUT / "cells_long.csv").exists() and not args.overwrite
    if done:
        print(f"Reusing the stored tables in {OUT} (pass --overwrite to regenerate).")
        cells_long = pd.read_csv(OUT / "cells_long.csv")
        diffs_long = pd.read_csv(OUT / "diffs_long.csv")
        vdiffs_long = pd.read_csv(OUT / "vdiffs_long.csv")
        board = pd.read_csv(OUT / "scoreboard.csv")
        shock = pd.read_csv(OUT / "shock.csv", index_col=0)
    else:
        scored, traj_mean, traj_runs, extras = score_all(partners, args.boot, args.n_jobs)
        cells_long, diffs_long, pdiffs_long, vdiffs_long, runs_long = long_tables(scored, partners)
        board = scoreboard(cells_long, diffs_long, partners)
        shock = shock_table(args.n_jobs)
        for name, df in [("cells_long", cells_long), ("diffs_long", diffs_long), ("pdiffs_long", pdiffs_long),
                         ("vdiffs_long", vdiffs_long), ("runs_long", runs_long), ("scoreboard", board),
                         ("traj_mean", traj_mean), ("traj_runs", traj_runs)]:
            df.to_csv(OUT / f"{name}.csv", index=False)
        shock.to_csv(OUT / "shock.csv")
        if extras is not None:
            extras.to_csv(OUT / "extras_long.csv", index=False)
        print(f"Wrote {OUT}/: " + ", ".join(sorted(f.name for f in OUT.glob("*.csv"))))

    report(variants, cells_long, diffs_long, vdiffs_long, board, shock, partners, OUT / "report.txt")


if __name__ == "__main__":
    main()
