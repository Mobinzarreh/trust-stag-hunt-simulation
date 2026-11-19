"""
Metric panel: behavioral, cognitive and outcome measures computed from simulation traces.

Outcome measures (payoff, successful and failed collaboration) plus process measures, so that every
agent is scored on one fixed set of definitions. The paper reports a subset (the payoff, SC, FC, the
detection and repair delays, the partner's payoff and the asymmetric response A_p); the rest are
computed for exploration. Families:

    O   outcome layer: submitted metrics, partner payoff (deterrence, M4b), welfare, risk functionals
    M1  breakdown and repair: phase delays on the diagnostic partners, the ramp hysteresis area,
        and the controlled-shock harness (shock_profile)
    M2  responsiveness with stability: threshold crossings, false switches, total variation,
        detection delays, integrated adaptation error
    M3  confidence informativeness, one step ahead, on the common concentration scale
    M5  oracle-policy deviation: reliance q_k against the informed policy q*_k
    M6  belief accuracy: Brier of x and p-hat, tracking error, Murphy decomposition
    M7  conditional-collaboration schedule: reliance against the partner's recent collaboration
    AX  Axelrod-style provocability and forgiveness (deferred set, kept because it is cheap)

M4a, the security level against an adversary family, is not computed here.

A trace is one agent against one partner type over n runs of K rounds, reshaped into (n, K) arrays.
All four agents share estimate -> logit reliance q_k -> action, so q_k is recomputed from the
recorded decision estimate p-hat_k (agent_belief) with the agent's game and inverse temperature.

Confidence on the common concentration scale: alpha + beta for the Beta models and 2 eps + t for
the trust agent (both are the concentration of the Beta distribution behind p-hat).

p*_k, the partner's true probability of collaborating in round k, comes from the partner's
coop_prob() when it has one (exogenous schedule: "scheduled" partners). Every other partner in the
registry is deterministic given the history (the TFT variants, Grim Trigger, Pavlov and the
belief-driven partners, whose noise enters their state but not their decision rule), so there
p*_k = o_k exactly. That p*_k depends on the agent's own past actions, so metrics built on it are
on-trajectory quantities there, never regret.
"""

import warnings
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import (
    EPS,
    INVERSE_TEMPERATURE,
    as_payoff_matrix,
    best_response_payoff,
    stag_indifference_threshold,
)

BURN_IN = 10                 # rounds excluded from crossing and false-switch counts
PERSIST_ESTIMATE = 5         # consecutive rounds p-hat must stay on the new side (sustained delay)
PERSIST_ACTION = 3           # consecutive realized actions on the new side (action-based delay)
IAE_WINDOW = 20              # rounds after a change summed in the integrated adaptation error
SCHEDULE_WINDOW = 5          # partner rounds behind the recent collaboration rate (M7)
BRIER_BINS = 10              # equal-width forecast bins for the Murphy decomposition
CVAR_ALPHA = 0.10            # worst share of runs in CVaR
BETRAYAL_AVERSION_LAMBDA = 1.955   # meta-analytic loss aversion (Brown et al. 2024), kink at P
SHOCK_WARMUP = 20            # identical pre-event history in the shock harness
SHOCK_DEPTH = 10             # length of the fixed betrayal episode in the shock harness
SHOCK_MAX_STEPS = 60         # censoring point of the shock harness counts
N_BOOT = 200                 # cluster (run-level) bootstrap replicates for pooled metrics
Z95 = 1.96


# ---------------------------------------------------------------------------------------------
# Metric catalog. direction: +1 higher is better, -1 lower is better, 0 descriptive (no ranking).
# kind: 'performance' (better/worse) or 'signature' (direction is the human pattern the theory
# predicts, e.g. slower repair than breakdown; ranking says which agent shows it most strongly).
# scope: partners where the metric is scored ('all', 'scheduled', 'change', 'nonclock',
# 'exploiters', 'breakrepair', 'ramp'). status: the metric's origin in the planning of the metric panel.
# ---------------------------------------------------------------------------------------------
CATALOG = {
    # Outcome layer
    "Pi":            ("O",  "Total payoff (Eq. 17)", +1, "performance", "all", "submitted"),
    "HB":            ("O",  "Hindsight benchmark (Eq. 20)", 0, "performance", "all", "submitted"),
    "SC":            ("O",  "Successful collaboration rate (Eq. 18)", +1, "performance", "all", "submitted"),
    "FC":            ("O",  "Failed collaboration rate (Eq. 19)", -1, "performance", "all", "submitted"),
    "coop_rate":     ("O",  "Agent collaboration rate", 0, "performance", "all", "descriptive"),
    "partner_pay":   ("O",  "Partner's payoff (deterrence, M4b)", -1, "performance", "exploiters", "M4"),
    "welfare":       ("O",  "Joint welfare (dyad payoff)", +1, "performance", "all", "tested dead end"),
    "pay_sd":        ("O",  "Within-run SD of round payoff", -1, "performance", "all", "tested dead end"),
    "U_ba":          ("O",  "Betrayal-averse utility (lambda 1.955 at P)", +1, "performance", "all", "tested dead end"),
    "CVaR10":        ("O",  "CVaR 10% of Pi (worst decile of runs)", +1, "performance", "all", "tested dead end"),
    # M1 breakdown and repair (trace-based parts; the shock harness is separate)
    "repair_break_ratio": ("M1", "Repair delay / breakdown delay (sustained)", +1, "signature", "change", "M1"),
    "hyst_area":     ("M1", "Hysteresis area: q(down) - q(up) at matched p*", +1, "signature", "ramp", "M1"),
    "lag_down":      ("M1", "Ramp: over-reliance while reliability falls", 0, "signature", "ramp", "M1"),
    "lag_up":        ("M1", "Ramp: under-reliance while reliability recovers", 0, "signature", "ramp", "M1"),
    "act_delay_down": ("M1", "Rounds to 3 realized independence choices after a break", -1, "performance", "change", "M1"),
    "act_delay_up":  ("M1", "Rounds to 3 realized collaboration choices after a repair", 0, "signature", "change", "M1"),
    # M2 responsiveness with stability
    "crossings":     ("M2", "Threshold crossings per run (rounds >= 10)", -1, "performance", "all", "M2"),
    "false_switches": ("M2", "Crossings to the side the informed policy rejects", -1, "performance", "scheduled", "M2"),
    "TV_q":          ("M2", "Total variation of reliance per run", -1, "performance", "all", "M2"),
    "max_jump_q":    ("M2", "Largest one-round change in reliance", -1, "performance", "all", "dropped (wrong way)"),
    "delay_down_first": ("M2", "Detection delay, first crossing below u", -1, "performance", "change", "M2"),
    "delay_down_sust": ("M2", "Detection delay, sustained 5 rounds below u", -1, "performance", "change", "M2"),
    "delay_up_first": ("M2", "Recovery delay, first crossing above u", 0, "signature", "change", "M2"),
    "delay_up_sust": ("M2", "Recovery delay, sustained 5 rounds above u", 0, "signature", "change", "M2"),
    "IAE_down":      ("M2", "Integrated |q - q*| over 20 rounds after a break", -1, "performance", "change", "M2"),
    "IAE_up":        ("M2", "Integrated |q - q*| over 20 rounds after a repair", -1, "performance", "change", "M2"),
    "engage_delay":  ("M2", "Rounds to sustained reliance on a reliable partner from the start", -1, "performance", "scheduled", "M2"),
    # M3 confidence informativeness
    "CV":            ("M3", "Spearman(confidence, -(p-hat - o)^2)", +1, "performance", "nonclock", "M3"),
    "CV_within":     ("M3", "Same, across runs within a round (clock removed; 0 if confidence is a clock)", +1, "performance", "all", "M3 (new variant)"),
    "AUROC":         ("M3", "Confidence discriminates correct next predictions", +1, "performance", "nonclock", "M3"),
    "AUROC_within":  ("M3", "Same, within a round (0.5 if confidence is a clock)", +1, "performance", "all", "M3 (new variant)"),
    "conf_brier_gap": ("M3", "Brier(low-confidence third) - Brier(high third)", +1, "performance", "nonclock", "M3"),
    # M5 oracle-policy deviation
    "OPD":           ("M5", "Mean |q - q*| (oracle-policy deviation)", -1, "performance", "all", "M5"),
    "OR":            ("M5", "Over-reliance: mean q when p* < u", -1, "performance", "all", "M5"),
    "UR":            ("M5", "Under-reliance: mean 1 - q when p* > u", -1, "performance", "all", "M5"),
    "exp_loss":      ("M5", "Expected payoff loss vs best response at p* (dynamic regret if scheduled)", -1, "performance", "all", "M5"),
    # M6 belief accuracy
    "brier_x":       ("M6", "Brier score of the belief x", -1, "performance", "all", "M6"),
    "brier_phat":    ("M6", "Brier score of the decision estimate p-hat", -1, "performance", "all", "M6"),
    "track_phat":    ("M6", "Tracking error (p-hat - p*)^2", -1, "performance", "scheduled", "M6"),
    "track_x":       ("M6", "Tracking error (x - p*)^2", -1, "performance", "scheduled", "M6"),
    "murphy_rel":    ("M6", "Murphy reliability (calibration error) of p-hat", -1, "performance", "all", "M6"),
    "murphy_res":    ("M6", "Murphy resolution of p-hat", +1, "performance", "all", "M6"),
    # M7 conditional-collaboration schedule
    "sched_slope":   ("M7", "Slope of q on partner's 5-round collaboration rate", 0, "signature", "all", "M7"),
    "sched_offset":  ("M7", "Mean schedule offset from the diagonal, q - r", 0, "signature", "all", "M7"),
    "sched_q0":      ("M7", "Reliance after 5 partner independence choices", 0, "signature", "all", "M7"),
    "sched_q1":      ("M7", "Reliance after 5 partner collaborations", 0, "signature", "all", "M7"),
    # Axelrod-style profile (deferred set)
    "provocability": ("AX", "P(independence | partner independent last round)", 0, "signature", "all", "deferred"),
    "forgiveness":   ("AX", "Reliance after the partner returns (I then C)", 0, "signature", "all", "deferred"),
    # Mechanism diagnostic (trust model only)
    "t_mean":        ("D",  "Mean confidence t (trust model only)", 0, "performance", "all", "diagnostic"),
}

# Shock-harness metrics (one value per agent, not per partner)
SHOCK_CATALOG = {
    "n_engage":      ("Collaborations needed to rely on a new partner (from the initial state)", "M1"),
    "n_lose":        ("Consecutive defections to lose reliance after 20 collaborations", "M1"),
    "n_regain_min":  ("Then consecutive collaborations to regain it", "M1"),
    "n_regain_10":   ("Collaborations to regain after 20 C then 10 D (same history for all)", "M1"),
    "n_build":       ("Collaborations to rely after 20 defections (building from distrust)", "M1"),
    "regain_over_lose": ("n_regain_min / n_lose (repair slower than breakdown > 1)", "M1"),
    "dp_neg":        ("Drop in p-hat after one defection, from 20 collaborations", "M1"),
    "dp_pos":        ("Rise in p-hat after one collaboration, from 20 defections", "M1"),
    "A_p":           ("dp_neg / dp_pos (negativity asymmetry in the estimate)", "M1"),
    "dq_neg":        ("Drop in reliance q after one defection", "M1"),
    "dq_pos":        ("Rise in reliance q after one collaboration", "M1"),
    "A_q":           ("dq_neg / dq_pos (asymmetry in reliance)", "M1"),
    "A_t":           ("Confidence drop after a violation / after a surprise (trust only)", "diagnostic"),
}


ECONOMIC_METRICS = ("Pi", "HB", "SC", "FC")
PANEL_METRICS = (
    "partner_pay",                                                    # deterrence (M4b)
    "delay_down_sust", "delay_up_sust", "hyst_area",                  # breakdown, repair, path dependence (M1)
    "crossings",                                                      # stability (M2)
    "CV_within",                                                      # confidence informativeness (M3)
    "UR",                                                             # under-reliance (M5); FC covers over-reliance
    "brier_phat",                                                     # forecast accuracy of the decision estimate (M6)
    "forgiveness",                                                    # Axelrod profile
)
PANEL_SHOCK_METRICS = ("A_p",)                                        # single-shock asymmetry (M1, harness)
DROPPED_METRICS = {
    "coop_rate": "redundant: SC + FC",
    "welfare": "redundant with SC (tau 0.90); payoff rescoring, tested dead end",
    "pay_sd": "redundant with FC (tau 0.90); payoff rescoring, tested dead end",
    "U_ba": "redundant with FC (tau -0.90); payoff rescoring, tested dead end",
    "CVaR10": "payoff rescoring, tested dead end",
    "repair_break_ratio": "derived from the two delays; ratio of small counts, w = 3 scores 4.98 vs trust 3.59",
    "lag_down": "component of hyst_area",
    "lag_up": "component of hyst_area",
    "act_delay_down": "noisier duplicate of delay_down_sust (q is monotone in p-hat)",
    "act_delay_up": "noisier duplicate of delay_up_sust",
    "false_switches": "redundant with crossings (tau 0.92)",
    "max_jump_q": "unfair: bounded by the logit slope, goes the wrong way",
    "delay_down_first": "same agent ordering as delay_down_sust (tau 1.00)",
    "delay_up_first": "same agent ordering as delay_up_sust",
    "IAE_down": "time-integrated duplicate of delay_down_sust and OR",
    "IAE_up": "time-integrated duplicate of delay_up_sust and UR",
    "CV": "unfair: benchmark confidence is a clock (use CV_within)",
    "AUROC": "unfair: clock-driven for the benchmarks",
    "AUROC_within": "mixed and transition-dominated; dropped by the authors",
    "conf_brier_gap": "unfair: clock-driven, undefined for the window",
    "OR": "redundant with FC, already in the economic table (tau 0.90)",
    "OPD": "redundant with OR (tau 0.87)",
    "exp_loss": "redundant with OR (tau 0.93)",
    "track_phat": "same agent ordering as brier_phat (tau 1.00)",
    "track_x": "same agent ordering as brier_x (tau 0.95)",
    "murphy_res": "determined by Brier = REL - RES + UNC given the other two",
    "sched_slope": "M7 dropped: shared logit rule, cannot separate the agents",
    "sched_offset": "M7 dropped",
    "sched_q0": "M7 dropped; same as provocability",
    "sched_q1": "M7 dropped",
    "provocability": "redundant with OR (tau -0.80) and sched_q0",
    "t_mean": "trust-only mechanism diagnostic, not a comparison",
    "n_engage": "duplicate of engage_delay",
    "n_lose": "controlled duplicate of delay_down_sust",
    "n_regain_min": "window mechanics dominate (w = 10 needs 7)",
    "n_regain_10": "controlled duplicate of delay_up_sust",
    "n_build": "covered by delay_up_sust from a deep loss; moves with memory only",
    "regain_over_lose": "built on n_regain_min",
    "A_q": "unfair: about 5 for every agent, set by the logit and u",
    "A_t": "trust-only, restates lambda > mu",
    "TV_q": "redundant with crossings (same pattern: steadier than disc/window, jumpier than cumulative)",
    "engage_delay": "mostly the x0, t0 starting values (trust 2.1 vs 1.0); only 5 partners; UR covers the cost of caution",
    "murphy_rel": "tiny values (0.001 to 0.03), mixed, follows brier_phat",
    "brier_x": "identical to brier_phat for the benchmarks (x = p-hat); for trust it compares the internal signal, a moving average like disc0.9",
}


def catalog_frame() -> pd.DataFrame:
    """The metric catalog as a table."""
    rows = [dict(metric=k, family=v[0], label=v[1], direction=v[2], kind=v[3], scope=v[4], status=v[5])
            for k, v in CATALOG.items()]
    rows += [dict(metric=k, family="M1-shock", label=v[0], direction=0, kind="signature",
                  scope="harness", status=v[1]) for k, v in SHOCK_CATALOG.items()]
    df = pd.DataFrame(rows)
    kept = set(PANEL_METRICS) | set(PANEL_SHOCK_METRICS)
    df["tier"] = ["economic" if m in ECONOMIC_METRICS else "panel" if m in kept else "dropped"
                  for m in df["metric"]]
    df["dropped_because"] = df["metric"].map(DROPPED_METRICS).fillna("")
    return df


# ---------------------------------------------------------------------------------------------
# Decision primitives shared by all four agents
# ---------------------------------------------------------------------------------------------
def action_values(p, payoff_matrix=None):
    """(V_C, V_I): expected payoff of collaborating and of independence if the partner collaborates w.p. p."""
    m = as_payoff_matrix(payoff_matrix)
    p = np.asarray(p, float)
    v_c = (1 - p) * m[1, 0, 0] + p * m[1, 1, 0]
    v_i = (1 - p) * m[0, 0, 0] + p * m[0, 1, 0]
    return v_c, v_i


def reliance_prob(p, payoff_matrix=None, inv_temp: float = INVERSE_TEMPERATURE):
    """q = P(collaborate) under the shared logit rule, given the estimate p (BaseAgent.decide)."""
    v_c, v_i = action_values(p, payoff_matrix)
    return 1.0 / (1.0 + np.exp(-inv_temp * (v_c - v_i)))


# ---------------------------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------------------------
@dataclass
class Trace:
    """One agent against one partner type: (n_runs, K) arrays plus the partner's schedule."""
    phat: np.ndarray          # decision estimate p-hat_k (before the round)
    x: np.ndarray             # behavioral belief: signal x_k for the trust agent, p-hat_k otherwise
    conc: np.ndarray          # confidence on the common concentration scale
    t: Optional[np.ndarray]   # trust agent's confidence t_k (None for the Beta models)
    q: np.ndarray             # reliance probability P(a_k = C)
    a: np.ndarray             # agent action (1 = collaborate)
    o: np.ndarray             # partner action
    pay: np.ndarray           # agent payoff per round
    pay_partner: np.ndarray   # partner payoff per round
    pstar: np.ndarray         # partner's true collaboration probability p*_k
    qstar: np.ndarray         # informed reliance q*_k at p*_k
    scheduled: bool           # p*_k exogenous (coop_prob) rather than o_k
    u: float                  # indifference threshold of the game
    payoff_matrix: np.ndarray
    ramp: Optional[tuple]     # (trough round, ramp length) for ReliabilityRampPartner

    @property
    def n(self) -> int:
        return self.phat.shape[0]

    @property
    def K(self) -> int:
        return self.phat.shape[1]

    def subset(self, idx) -> "Trace":
        """Trace restricted to the runs idx (rows), for the run-level bootstrap."""
        arrays = {f: getattr(self, f)[idx] for f in
                  ("phat", "x", "conc", "q", "a", "o", "pay", "pay_partner", "pstar", "qstar")}
        arrays["t"] = None if self.t is None else self.t[idx]
        return replace(self, **arrays)


def make_trace(df: pd.DataFrame, partner=None, payoff_matrix=None,
               inv_temp: float = INVERSE_TEMPERATURE, eps: float = EPS) -> Trace:
    """
    Build a Trace from a run_agent_simulation / MonteCarloManager frame with a run_id column.

    Args:
        df: round-level results of one agent against one partner type, all runs of equal length
        partner: a fresh instance of that partner type; its coop_prob() (if any) gives p*_k
        payoff_matrix: game played (default: the study matrix)
        inv_temp: the agent's inverse temperature
        eps: the trust agent's Beta baseline (confidence scale 2 eps + t)
    """
    df = df.sort_values(["run_id", "round"], kind="stable")
    n = df["run_id"].nunique()
    K = len(df) // n
    if n * K != len(df) or not (df.groupby("run_id").size() == K).all():
        raise ValueError("all runs must have the same number of rounds")

    def arr(col):
        return df[col].to_numpy(float).reshape(n, K)

    m = as_payoff_matrix(payoff_matrix)
    phat = arr("agent_belief")
    if "trust_level" in df:
        t = arr("trust_level")
        x = arr("signal")
        conc = 2 * eps + t
    else:
        t = None
        x = phat
        conc = arr("alpha") + arr("beta") if {"alpha", "beta"} <= set(df.columns) else np.full_like(phat, np.nan)
    # alpha + beta of the discounted model is the same for every run in a round only up to rounding
    # (alpha and beta differ across runs); round it so within-round comparisons see a constant
    conc = np.round(conc, 10)
    a, o = arr("agent_action"), arr("partner_action")
    scheduled = partner is not None and hasattr(partner, "coop_prob")
    if scheduled:
        pstar = np.tile([partner.coop_prob(k) for k in range(K)], (n, 1)).astype(float)
    else:
        pstar = o.copy()
    ramp = None
    if partner is not None and hasattr(partner, "trough") and hasattr(partner, "down_start"):
        ramp = (partner.trough, partner.trough - partner.down_start)
    return Trace(
        phat=phat, x=x, conc=conc, t=t,
        q=reliance_prob(phat, m, inv_temp), a=a, o=o,
        pay=arr("agent_payoff"), pay_partner=arr("partner_payoff"),
        pstar=pstar, qstar=reliance_prob(pstar, m, inv_temp),
        scheduled=scheduled, u=stag_indifference_threshold(m), payoff_matrix=m, ramp=ramp,
    )


def concat_traces(traces) -> Trace:
    """Stack traces of one agent across partners (rows = runs), e.g. for a pooled M7 schedule."""
    traces = list(traces)
    first = traces[0]
    cat = {f: np.concatenate([getattr(tr, f) for tr in traces]) for f in
           ("phat", "x", "conc", "q", "a", "o", "pay", "pay_partner", "pstar", "qstar")}
    cat["t"] = None if first.t is None else np.concatenate([tr.t for tr in traces])
    return replace(first, scheduled=all(tr.scheduled for tr in traces), ramp=None, **cat)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def _masked_row_mean(values, mask):
    """Row-wise mean of values over mask; NaN for rows with no masked entries."""
    cnt = mask.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (values * mask).sum(1) / cnt
    return np.where(cnt > 0, out, np.nan)


def _first_sustained(flags, start: int, end: int, persist: int):
    """
    Per run, rounds from `start` until the first k in [start, end) that begins `persist` consecutive
    True flags inside [start, end). NaN where that never happens (censored).
    """
    n = flags.shape[0]
    f = flags[:, start:end].astype(int)
    if f.shape[1] < persist:
        return np.full(n, np.nan)
    cs = np.concatenate([np.zeros((n, 1), int), np.cumsum(f, axis=1)], axis=1)
    ok = (cs[:, persist:] - cs[:, :-persist]) == persist
    first = ok.argmax(axis=1).astype(float)
    first[~ok.any(axis=1)] = np.nan
    return first


def _change_points(pstar_row, u):
    """Rounds k >= 1 where the informed policy's side (p* >= u) flips, with the new side."""
    side = pstar_row >= u
    ks = np.flatnonzero(side[1:] != side[:-1]) + 1
    return [(int(k), bool(side[k])) for k in ks], bool(side[0])


def _spearman(a, b):
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def _within_round(score, target, stat, null: float, min_runs: int = 10):
    """
    Average over rounds of stat(score[:, k], target[:, k]) computed across runs, so the round
    number (the clock) cannot drive the result. Rounds where score does not vary across runs carry
    no information beyond the clock and contribute the null value (0 for a correlation, 0.5 for an
    AUROC); rounds where target does not vary are skipped. Weighted by the number of runs.
    """
    vals = []
    for k in range(score.shape[1]):
        s, y = score[:, k], target[:, k].astype(float)
        if len(s) < min_runs or np.ptp(y) == 0:
            continue
        v = null if np.ptp(s) == 0 else stat(s, y)
        vals.append(null if not np.isfinite(v) else v)
    return float(np.mean(vals)) if vals else np.nan


def _auroc(score, positive):
    """Area under the ROC of score for the binary label positive (ties count one half)."""
    positive = np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _murphy(forecast, outcome, bins: int = BRIER_BINS):
    """Murphy (1973) decomposition of the Brier score: reliability, resolution, uncertainty."""
    idx = np.minimum((forecast * bins).astype(int), bins - 1)
    N = len(forecast)
    obar = outcome.mean()
    cnt = np.bincount(idx, minlength=bins)
    fsum = np.bincount(idx, weights=forecast, minlength=bins)
    osum = np.bincount(idx, weights=outcome, minlength=bins)
    nz = cnt > 0
    fbar, obin = fsum[nz] / cnt[nz], osum[nz] / cnt[nz]
    rel = float((cnt[nz] * (fbar - obin) ** 2).sum() / N)
    res = float((cnt[nz] * (obin - obar) ** 2).sum() / N)
    return rel, res, float(obar * (1 - obar))


# ---------------------------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------------------------
def per_run(tr: Trace) -> pd.DataFrame:
    """One row per run with every per-run metric in CATALOG that applies to this trace."""
    m, u = tr.payoff_matrix, tr.u
    a, o, q, qs, ph, ps = tr.a, tr.o, tr.q, tr.qstar, tr.phat, tr.pstar
    n, K = tr.n, tr.K
    nan = np.full(n, np.nan)
    out = {}

    # Outcome layer
    out["Pi"] = tr.pay.sum(1)
    out["HB"] = best_response_payoff(m)[o.astype(int)].sum(1)
    out["SC"] = ((a == 1) & (o == 1)).mean(1)
    out["FC"] = ((a == 1) & (o == 0)).mean(1)
    out["coop_rate"] = a.mean(1)
    out["partner_pay"] = tr.pay_partner.sum(1)
    out["welfare"] = out["Pi"] + out["partner_pay"]
    out["pay_sd"] = tr.pay.std(1, ddof=1)
    gain = tr.pay - m[0, 0, 0]
    out["U_ba"] = np.where(gain >= 0, gain, BETRAYAL_AVERSION_LAMBDA * gain).sum(1)

    # M2: stability
    above = ph >= u
    ab = above[:, BURN_IN:]
    out["crossings"] = (ab[:, 1:] != ab[:, :-1]).sum(1).astype(float)
    if tr.scheduled:
        flipped = ab[:, 1:] != ab[:, :-1]
        wrong_side = ab[:, 1:] != (ps[:, BURN_IN + 1:] >= u)
        out["false_switches"] = (flipped & wrong_side).sum(1).astype(float)
    else:
        out["false_switches"] = nan
    dq = np.abs(np.diff(q, axis=1))
    out["TV_q"] = dq.sum(1)
    out["max_jump_q"] = dq.max(1)

    # M2 / M1: delays after the first change of the informed policy's side in each direction
    for label in ("down", "up"):
        for col in (f"delay_{label}_first", f"delay_{label}_sust", f"act_delay_{label}", f"IAE_{label}"):
            out[col] = nan
    out["engage_delay"] = nan
    if tr.scheduled:
        changes, start_above = _change_points(ps[0], u)
        bounds = [c for c, _ in changes] + [K]
        seen = set()
        for i, (c, new_above) in enumerate(changes):
            label = "up" if new_above else "down"
            if label in seen:
                continue  # keep the first change in this direction only
            seen.add(label)
            end = bounds[i + 1]
            est = above if new_above else ~above
            act = (a == 1) if new_above else (a == 0)
            out[f"delay_{label}_first"] = _first_sustained(est, c, end, 1)
            out[f"delay_{label}_sust"] = _first_sustained(est, c, end, PERSIST_ESTIMATE)
            out[f"act_delay_{label}"] = _first_sustained(act, c, end, PERSIST_ACTION)
            w_end = min(c + IAE_WINDOW, end)
            out[f"IAE_{label}"] = np.abs(q[:, c:w_end] - qs[:, c:w_end]).sum(1)
        if start_above:
            out["engage_delay"] = _first_sustained(above, 0, bounds[0], PERSIST_ESTIMATE)

    # M1: ramp hysteresis at matched true reliability (rounds trough - j and trough + j)
    for col in ("hyst_area", "lag_down", "lag_up"):
        out[col] = nan
    if tr.ramp is not None:
        T, J = tr.ramp
        j = np.arange(1, J + 1)
        q_down, q_up, q_inf = q[:, T - j], q[:, T + j], qs[:, T - j]
        out["hyst_area"] = (q_down - q_up).mean(1)
        out["lag_down"] = (q_down - q_inf).mean(1)
        out["lag_up"] = (q_inf - q_up).mean(1)

    # M5: oracle-policy deviation and expected loss
    out["OPD"] = np.abs(q - qs).mean(1)
    out["OR"] = _masked_row_mean(q, ps < u)
    out["UR"] = _masked_row_mean(1 - q, ps > u)
    v_c, v_i = action_values(ps, m)
    out["exp_loss"] = (np.maximum(v_c, v_i) - (q * v_c + (1 - q) * v_i)).sum(1)

    # M6: accuracy
    out["brier_x"] = ((tr.x - o) ** 2).mean(1)
    out["brier_phat"] = ((ph - o) ** 2).mean(1)
    if tr.scheduled:
        out["track_phat"] = ((ph - ps) ** 2).mean(1)
        out["track_x"] = ((tr.x - ps) ** 2).mean(1)
    else:
        out["track_phat"] = out["track_x"] = nan

    # Axelrod-style profile, on the reliance probability (less noisy than realized actions)
    out["provocability"] = _masked_row_mean(1 - q[:, 1:], o[:, :-1] == 0)
    out["forgiveness"] = _masked_row_mean(q[:, 2:], (o[:, :-2] == 0) & (o[:, 1:-1] == 1))

    out["t_mean"] = tr.t.mean(1) if tr.t is not None else nan
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Pooled metrics (computed over all rounds of all runs; CIs by run-level bootstrap)
# ---------------------------------------------------------------------------------------------
POOLED_METRICS = ("CV", "CV_within", "AUROC", "AUROC_within", "conf_brier_gap", "murphy_rel", "murphy_res",
                  "CVaR10", "sched_slope", "sched_offset", "sched_q0", "sched_q1", "repair_break_ratio")


def pooled(tr: Trace, runs: Optional[pd.DataFrame] = None, idx=None) -> dict:
    """Pooled metrics of a trace (optionally of the runs idx), given its per_run frame."""
    if idx is not None:
        tr = tr.subset(idx)
        runs = None if runs is None else runs.iloc[idx]
    n, K = tr.n, tr.K
    ph, o, c = tr.phat.ravel(), tr.o.ravel(), tr.conc.ravel()
    res = {}

    # M3: one-step confidence informativeness
    neg_err = -(ph - o) ** 2
    res["CV"] = _spearman(c, neg_err)
    correct = (ph >= 0.5) == (o == 1)
    res["AUROC"] = _auroc(c, correct) if np.nanstd(c) > 0 else np.nan
    # Same two measures across runs within each round. All three Beta benchmarks' concentration is a
    # deterministic function of the round (2 + k, 2 + min(k, w), the discounted sum), so their pooled
    # values are a correlation of time with error; within a round theirs is 0 (0.5) by construction.
    res["CV_within"] = _within_round(tr.conc, -(tr.phat - tr.o) ** 2, _spearman, 0.0)
    res["AUROC_within"] = _within_round(tr.conc, (tr.phat >= 0.5) == (tr.o == 1), lambda s, y: _auroc(s, y > 0.5), 0.5)
    lo_thr, hi_thr = np.quantile(c, [1 / 3, 2 / 3])
    if hi_thr > lo_thr:
        res["conf_brier_gap"] = float((neg_err[c >= hi_thr]).mean() - (neg_err[c <= lo_thr]).mean())
    else:
        res["conf_brier_gap"] = np.nan

    # M6: Murphy decomposition of the decision estimate
    res["murphy_rel"], res["murphy_res"], _ = _murphy(ph, o)

    # Outcome: CVaR of total payoff across runs
    pi = np.sort(tr.pay.sum(1))
    res["CVaR10"] = float(pi[:max(1, int(np.ceil(CVAR_ALPHA * n)))].mean())

    # M7: reliance against the partner's collaboration rate over the previous W rounds
    W = SCHEDULE_WINDOW
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(tr.o, axis=1)], axis=1)
    r = ((cs[:, W:K] - cs[:, :K - W]) / W).ravel()      # rate over rounds k-W .. k-1, for k >= W
    qq = tr.q[:, W:].ravel()
    res["sched_slope"] = float(np.polyfit(r, qq, 1)[0]) if r.std() > 0 else np.nan
    level = np.rint(r * W).astype(int)
    cnt = np.bincount(level, minlength=W + 1)
    qbar = np.bincount(level, weights=qq, minlength=W + 1) / np.where(cnt > 0, cnt, 1)
    qbar[cnt == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res["sched_offset"] = float(np.nanmean(qbar - np.arange(W + 1) / W))
    res["sched_q0"], res["sched_q1"] = float(qbar[0]), float(qbar[W])

    # M1: repair / breakdown delay ratio (ratio of means over runs that complete both)
    res["repair_break_ratio"] = np.nan
    if runs is not None:
        both = runs[["delay_down_sust", "delay_up_sust"]].dropna()
        if len(both) and both["delay_down_sust"].mean() > 0:
            res["repair_break_ratio"] = float(both["delay_up_sust"].mean() / both["delay_down_sust"].mean())
    return res


@dataclass
class CellResult:
    """Everything computed for one agent x partner cell."""
    runs: pd.DataFrame            # per-run metrics
    summary: pd.DataFrame         # metric, value, lo, hi, n
    boot: pd.DataFrame            # bootstrap replicates of the pooled metrics (rows = replicates)


def score_cell(tr: Trace, n_boot: int = N_BOOT, seed: int = 0) -> CellResult:
    """Per-run metrics (mean, 95% CI over runs) and pooled metrics (run-level bootstrap CI)."""
    runs = per_run(tr)
    rows = []
    for col in runs.columns:
        v = runs[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(dict(metric=col, value=v.mean(), lo=v.mean() - Z95 * se, hi=v.mean() + Z95 * se,
                         n=len(v), n_runs=tr.n))
    point = pooled(tr, runs)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        boot = pd.DataFrame([pooled(tr, runs, rng.integers(0, tr.n, tr.n)) for _ in range(n_boot)])
    for col, val in point.items():
        if not np.isfinite(val):
            continue
        lo, hi = np.nanpercentile(boot[col], [2.5, 97.5]) if boot[col].notna().any() else (np.nan, np.nan)
        rows.append(dict(metric=col, value=val, lo=lo, hi=hi, n=tr.n, n_runs=tr.n))
    return CellResult(runs=runs, summary=pd.DataFrame(rows), boot=boot)


def compare_cells(a: CellResult, b: CellResult) -> pd.DataFrame:
    """
    Difference a - b for every metric both cells have, with a 95% CI.

    Per-run metrics use a Welch interval (the agents run on independent RNG streams); pooled metrics
    use the independent bootstrap replicates of the two cells.
    """
    rows = []
    for col in a.runs.columns.intersection(b.runs.columns):
        x = a.runs[col].dropna().to_numpy(float)
        y = b.runs[col].dropna().to_numpy(float)
        if len(x) < 2 or len(y) < 2:
            continue
        d = x.mean() - y.mean()
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        rows.append(dict(metric=col, diff=d, lo=d - Z95 * se, hi=d + Z95 * se))
    sa = a.summary.set_index("metric")["value"]
    sb = b.summary.set_index("metric")["value"]
    for col in POOLED_METRICS:
        if col not in sa or col not in sb:
            continue
        reps = (a.boot[col] - b.boot[col]).dropna()
        lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan)
        rows.append(dict(metric=col, diff=sa[col] - sb[col], lo=lo, hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# M1 controlled-shock harness (identical evidence sequences, no partner, no action noise)
# ---------------------------------------------------------------------------------------------
def _feed(agent, actions):
    for obs in actions:
        agent.update(int(obs))


def _steps_until(agent, obs: int, done: Callable[[float], bool], max_steps: int) -> float:
    """Feed obs repeatedly until done(p-hat); the number of observations needed (NaN if censored)."""
    for k in range(max_steps + 1):
        if done(agent.get_belief()):
            return float(k)
        agent.update(obs)
    return np.nan


def shock_profile(agent_factory: Callable, seeds=range(100), warmup: int = SHOCK_WARMUP,
                  depth: int = SHOCK_DEPTH, max_steps: int = SHOCK_MAX_STEPS,
                  payoff_matrix=None, inv_temp: float = INVERSE_TEMPERATURE):
    """
    Responses to scripted evidence from identical pre-event histories.

    The agents' updates read the partner's action only, so every agent sees exactly the same
    evidence; seeds matter only for the trust agent's signal noise. Returns (per-seed frame,
    summary Series) with the quantities in SHOCK_CATALOG.
    """
    m = as_payoff_matrix(payoff_matrix)
    u = stag_indifference_threshold(m)
    below, at_or_above = (lambda p: p < u), (lambda p: p >= u)

    def q(p):
        return float(reliance_prob(p, m, inv_temp))

    rows = []
    for s in seeds:
        np.random.seed(s)
        row = {"seed": s}
        ag = agent_factory()
        row["n_engage"] = _steps_until(ag, 1, at_or_above, max_steps)

        ag = agent_factory()
        _feed(ag, [1] * warmup)
        row["n_lose"] = _steps_until(ag, 0, below, max_steps)
        row["n_regain_min"] = _steps_until(ag, 1, at_or_above, max_steps) if np.isfinite(row["n_lose"]) else np.nan

        ag = agent_factory()
        _feed(ag, [1] * warmup + [0] * depth)
        row["n_regain_10"] = _steps_until(ag, 1, at_or_above, max_steps)

        ag = agent_factory()
        _feed(ag, [0] * warmup)
        row["n_build"] = _steps_until(ag, 1, at_or_above, max_steps)

        for sign, history, shock in (("neg", 1, 0), ("pos", 0, 1)):
            ag = agent_factory()
            _feed(ag, [history] * warmup)
            p0, t0 = ag.get_belief(), getattr(ag, "t", np.nan)
            ag.update(shock)
            p1, t1 = ag.get_belief(), getattr(ag, "t", np.nan)
            row[f"dp_{sign}"] = abs(p1 - p0)
            row[f"dq_{sign}"] = abs(q(p1) - q(p0))
            row[f"dt_{sign}"] = t0 - t1           # confidence lost to the prediction error
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    s = per_seed.drop(columns="seed").mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        s["regain_over_lose"] = s["n_regain_min"] / s["n_lose"]
        s["A_p"] = s["dp_neg"] / s["dp_pos"]
        s["A_q"] = s["dq_neg"] / s["dq_pos"]
        s["A_t"] = s["dt_neg"] / s["dt_pos"] if np.isfinite(s["dt_pos"]) and s["dt_pos"] != 0 else np.nan
    return per_seed, s
