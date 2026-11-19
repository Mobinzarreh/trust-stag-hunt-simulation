#!/usr/bin/env python3
"""
Guard rails for the trust-sensitive agent's specification. No Monte Carlo study, a few seconds.

    python scripts/verify_revised_spec.py

The default FocalAgent (the revised model of the paper) differs from the originally submitted agent in where the
asymmetric response lives and how confidence is scored; the decision stage is the same:
    nu      an unprovoked failed collaboration weighs nu = 3 in the belief memory (attribution: a defection that
            answers the agent's own defection weighs 1)
    psi(u)  a surprise (agent held back, partner collaborated) weighs psi = c (1 - u) / u, c = 1.5, in the belief memory
    graded  confidence scores the probability the belief gave to what happened (lambda = mu = 1, no category weights)
    eta     belief step 0.85 (submitted 0.7); t_max = 10 as submitted

Checks, each asserted:
  1. submitted_v1=True reproduces the submitted FocalAgent (scripts/reference/focal_agent_submitted_v1.py) bit for bit
     (every reported partner x 12 seeds, study game, trajectories and states).
  2. The decision stage is unchanged: p-hat = (eps + x t) / (2 eps + t), and the agent carries only (x, t) for the
     decision; the own-action history is bookkeeping of an observable input.
  3. nu: the belief-memory average equals a hand computation on a scripted history, a defection that answers the
     agent's own defection weighs 1 with attribution and nu without it, and nu = 1 is the parent.
  4. graded: with a hard forecast (x in {0, 1}) it equals the discounted hit rate to rounding; with a soft forecast it
     differs in the expected direction (a correct but unsure prediction scores below 1).
  5. update() without decide() (scripted evidence) falls back to the deterministic action and still records it.
  6. The default differs from the submitted agent (the revision does something) and consumes the same random draws
     as the submitted agent against a partner that ignores the agent (same noise stream).
  7. psi: a surprise weighs psi, an exposed betrayal nu, a provoked reply and a match 1 (hand computation); psi = 1 is the
     submitted behavior.
  8. t_max is a separate constant: AGENT_T_MAX = 10 at every gamma_x, and an explicit value wins.
  9. t has no category weights: with mu = lambda = 1 the confidence recursion equals the plain discounted mean of s.
 11. psi(u) = c (1 - u) / u is read from the agent's own payoff matrix: 2.25 at u = 0.4, 0.75 at u = 2/3, 0.375 at u = 0.8; an explicit
     surprise_weight wins; submitted_v1 gives 1.
 12. The default equals the explicit setting (psi 0.75, eta 0.85, t_max 10, nu 3) bit for bit on every reported partner at the study game, and
     the game-scaled rule equals the explicit psi at u = 0.4 and u = 0.8.
 13. The belief-driven partners still read the submitted ETA and TRUST_MAX.
 14. The trust agent's concentration ceiling 2 eps + t_max equals the benchmarks' (discounted 2 eps + 1/(1 - rho), window 2 eps + w): the
     comparison is concentration-matched.
(Check 10, an equivalence with an exploratory probe agent, is not part of this release.)
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import trust_bayesian_agent_comparison.agents  # noqa: E402,F401  (parent package for the relative imports below)
from trust_bayesian_agent_comparison.agents import FocalAgent  # noqa: E402
from trust_bayesian_agent_comparison.config import (  # noqa: E402
    AGENT_ETA, AGENT_T_MAX, ETA, EPS, NUM_ROUNDS, SURPRISE_SCALE, TRUST_MAX, payoff_matrix_for_u)
from trust_bayesian_agent_comparison.partners.registry import (  # noqa: E402
    DIAGNOSTIC_PARTNERS, MANUSCRIPT_PARTNERS, partner_factory)
from trust_bayesian_agent_comparison.simulation import run_agent_simulation, slot_rng  # noqa: E402

COLS = ["agent_belief", "agent_action", "partner_action", "agent_payoff", "trust_level", "signal"]
SUBMITTED = ROOT / "scripts" / "reference" / "focal_agent_submitted_v1.py"


def submitted_class():
    """The submitted FocalAgent, loaded beside the package so its relative imports resolve."""
    spec = importlib.util.spec_from_file_location("trust_bayesian_agent_comparison.agents.focal_submitted", SUBMITTED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FocalAgent


def run(agent, partner, game, seed):
    return run_agent_simulation(agent, partner_factory(partner, NUM_ROUNDS, payoff_matrix=game)(), NUM_ROUNDS, seed,
                                rng=slot_rng(seed, 1))


def check_submitted_reproduced():
    """Study game only: the submitted class predates the payoff_matrix argument."""
    Old = submitted_class()
    n, worst = 0, 0.0
    for p in MANUSCRIPT_PARTNERS + DIAGNOSTIC_PARTNERS:
        for i in range(12):
            a = run(Old(), p, None, 42 + i)
            b = run(FocalAgent(submitted_v1=True), p, None, 42 + i)
            worst = max(worst, float(np.abs(a[COLS].to_numpy(float) - b[COLS].to_numpy(float)).max()))
            n += 1
    assert worst == 0.0, worst
    npart = len(MANUSCRIPT_PARTNERS + DIAGNOSTIC_PARTNERS)
    print(f"1. submitted_v1=True equals the submitted FocalAgent: {n} runs ({npart} partners x 12 seeds, study "
          f"game), max |diff| {worst:.1e}. The 300-run check against the submitted CSVs is "
          f"scripts/reproduce_manuscript_tables.py --rerun")


def check_decision_stage():
    ag = FocalAgent()
    worst = 0.0
    for x in (0.0, 0.3, 0.8, 1.0):
        for t in (0.0, 2.0, 6.0, TRUST_MAX):
            ag.x, ag.t = x, t
            worst = max(worst, abs(ag.expected_p() - (EPS + x * t) / (2 * EPS + t)))
    assert worst < 1e-15, worst          # alpha/(alpha+beta) against the closed form: rounding only
    print(f"2. decision stage p-hat = (eps + x t)/(2 eps + t) unchanged (max |diff| {worst:.1e}); states: x, t")


def hand_signal(actions, own, nu, gamma, attribution, psi=1.0):
    """P_obs of the last round by direct computation."""
    n = len(actions)
    w = []
    for i in range(n):
        exposed = own[i] == 1 and actions[i] == 0
        provoked = attribution and i > 0 and own[i - 1] == 0
        surprise = own[i] == 0 and actions[i] == 1
        w.append(gamma ** (n - 1 - i) * (nu if exposed and not provoked else psi if surprise else 1.0))
    return sum(a * wi for a, wi in zip(actions, w)) / sum(w)


def check_nu():
    gamma = 0.9
    partner = [1, 1, 0, 1, 0, 1, 1, 0]
    own = [1, 1, 1, 0, 1, 1, 1, 1]          # round 4 is a reply to the agent's own defection in round 3
    for psi in (1.0, 1.5):
        for attribution in (True, False):
            ag = FocalAgent(exposed_betrayal_weight=3.0, surprise_weight=psi, attribution=attribution, noise_sigma=0.0,
                            u_i=0.5)
            x_prev = ag.x
            for k in range(len(partner)):
                ag._last_action = own[k]
                P = (hand_signal(partner[:k + 1], own[:k + 1], 3.0, gamma, attribution, psi) if k
                     else float(partner[0]))
                ag.update(partner[k])
                assert abs(ag.x - float(np.clip(x_prev + ag.eta * (P - x_prev), 0, 1))) < 1e-12, (psi, attribution, k)
                x_prev = ag.x
    # the reply (round index 4, own[3] == 0) is the only provoked exposed betrayal in this history
    with_att = hand_signal(partner, own, 3.0, gamma, True)
    without = hand_signal(partner, own, 3.0, gamma, False)
    assert with_att > without, (with_att, without)
    # nu = 1 and psi = 1 are the parent (explicit mu and t_max, since the revised defaults for those changed)
    np.random.seed(1)
    a = FocalAgent(exposed_betrayal_weight=1.0, surprise_weight=1.0, confidence_scoring="hit", loss_aversion=3.0,
                   lambda_surprise=0.5, t_max=TRUST_MAX, eta=ETA, noise_sigma=0.0)
    b = FocalAgent(submitted_v1=True, noise_sigma=0.0)
    for k, o in enumerate(partner * 5):
        a._last_action = b._last_action = own[k % len(own)]
        a.update(o)
        b.update(o)
    assert a.x == b.x and a.t == b.t
    print("3. nu: matches a hand computation (with and without attribution, psi = 1 and 1.5); the provoked reply weighs 1 "
          "with attribution; nu = psi = 1 is the parent")


def check_graded():
    rng = np.random.default_rng(3)
    seq = [int(v) for v in rng.integers(0, 2, 120)]
    hit, gra = (FocalAgent(submitted_v1=True, noise_sigma=0.0), FocalAgent(confidence_scoring="graded",
                                                                             loss_aversion=3.0, lambda_surprise=0.5,
                                                                             t_max=TRUST_MAX, noise_sigma=0.0))
    worst = 0.0
    for o in seq:
        hard = float(rng.integers(0, 2))     # force a hard forecast in both agents before each update
        hit.x = gra.x = hard
        hit._last_action = gra._last_action = 1
        hit.update(o)
        gra.update(o)
        worst = max(worst, abs(hit.t - gra.t))
    assert worst < 1e-9, worst
    soft = FocalAgent(confidence_scoring="graded", t_max=TRUST_MAX, noise_sigma=0.0)
    hard_ = FocalAgent(submitted_v1=True, noise_sigma=0.0)
    for _ in range(30):                      # a partner that always cooperates; the belief rises toward 1 but is soft
        for ag in (soft, hard_):
            ag._last_action = 1
            ag.update(1)
    assert soft.t < hard_.t, (soft.t, hard_.t)
    print(f"4. graded equals the hit rate for a hard forecast (max |t diff| {worst:.1e}); a soft forecast scores "
          f"below it (t {soft.t:.2f} against {hard_.t:.2f} after 30 collaborations)")


def check_fallback():
    ag = FocalAgent(noise_sigma=0.0)
    for _ in range(5):
        ag.update(1)
    assert len(ag.own_action_history) == 5 and set(ag.own_action_history) <= {0, 1}
    print(f"5. update() without decide() records the deterministic fallback action: {ag.own_action_history}")


def check_revision_acts():
    a = run(FocalAgent(), "StrategicCheater", None, 42)
    b = run(FocalAgent(submitted_v1=True), "StrategicCheater", None, 42)
    assert not np.array_equal(a[COLS].to_numpy(float), b[COLS].to_numpy(float))
    # a partner that ignores the agent and an agent that never meets an exposed betrayal: same noise stream
    c = run(FocalAgent(surprise_weight=1.0, eta=ETA), "AlwaysCooperate", None, 7)
    d = run(FocalAgent(submitted_v1=True), "AlwaysCooperate", None, 7)
    assert np.allclose(c["signal"], d["signal"], atol=1e-12)
    # psi cannot act against a partner that always collaborates (the weighted share is 1 whatever the weights), so the
    # check that psi does act uses a mixed record
    e, f = run(FocalAgent(), "Random", None, 7), run(FocalAgent(surprise_weight=1.0), "Random", None, 7)
    assert not np.allclose(e["signal"], f["signal"], atol=1e-12)
    print("6. the revised default differs from the submitted one on StrategicCheater, and its belief x is identical against "
          "AlwaysCooperate when psi = 1 and eta is the submitted one (no exposed betrayal to weigh, same noise stream); the "
          "game-scaled psi changes x against Random")


def check_psi():
    # a surprise (own D, partner C) weighs psi; an exposed unprovoked betrayal nu; a provoked reply and a match 1
    gamma = 0.9
    actions, own = [1, 1, 0, 1], [1, 1, 1, 0]            # rounds: match, match, unprovoked exposed betrayal, surprise
    psi, nu = 1.5, 3.0
    expect = (gamma ** 3 + gamma ** 2 + 0 + psi) / (gamma ** 3 + gamma ** 2 + nu * gamma + psi)
    got = hand_signal(actions, own, nu, gamma, True, psi)
    assert abs(expect - got) < 1e-15, (expect, got)
    assert hand_signal(actions, own, nu, gamma, True, 1.0) < got                  # psi > 1 raises the share of collaboration
    print(f"7. psi: surprise {psi}, exposed betrayal {nu}, match and provoked reply 1 (hand computation {got:.6f}); psi = 1 gives "
          f"{hand_signal(actions, own, nu, gamma, True, 1.0):.6f}")


def check_tmax_tie():
    for g in (0.8, 0.9, 0.95, 1.0):
        assert FocalAgent(memory_discount=g).t_max == AGENT_T_MAX, (g, FocalAgent(memory_discount=g).t_max)
    assert AGENT_T_MAX == 10.0
    assert FocalAgent(submitted_v1=True, memory_discount=0.8).t_max == TRUST_MAX
    assert FocalAgent(t_max=7.0).t_max == 7.0 and FocalAgent(submitted_v1=True, t_max=7.0).t_max == 7.0
    print("8. t_max is a separate constant: AGENT_T_MAX = 10 at every gamma_x (equal to TRUST_MAX), an explicit value wins")


def check_t_plain_mean():
    # mu = lambda = 1: D_k adds 1 every round, so N_k / D_k is the discounted mean of the scores s
    g = 0.9
    ag = FocalAgent(noise_sigma=0.0)
    rng = np.random.default_rng(11)
    N = D = 0.0
    t = ag.t
    worst = 0.0
    for _ in range(80):
        o = int(rng.integers(0, 2))
        x_prev = ag.x
        ag._last_action = int(rng.integers(0, 2))
        ag.update(o)
        s_k = x_prev if o == 1 else 1.0 - x_prev
        N, D = g * N + s_k, g * D + 1.0
        t = (1 - ag.trust_smoothing) * t + ag.trust_smoothing * ag.t_max * N / D
        worst = max(worst, abs(ag.t - t))
    assert worst < 1e-7, worst       # the agent divides by D + 1e-9, a guard against 0/0, so the match is to about 1e-9
    print(f"9. t is the smoothed discounted mean of the scores s, no category weights (max |diff| {worst:.1e})")




def check_psi_of_u():
    got = {}
    for u, want in ((0.4, 2.25), (2 / 3, 0.75), (0.8, 0.375)):
        game = None if abs(u - 2 / 3) < 1e-9 else payoff_matrix_for_u(u)
        got[round(u, 3)] = FocalAgent(payoff_matrix=game).surprise_weight
        assert abs(got[round(u, 3)] - want) < 1e-9, (u, got[round(u, 3)], want)
    assert SURPRISE_SCALE == 1.5
    assert FocalAgent(surprise_weight=2.0).surprise_weight == 2.0                    # explicit wins
    assert FocalAgent(surprise_scale=3.0).surprise_weight == 1.5                     # c = 3 gives psi = 1.5 at the study game
    assert FocalAgent(submitted_v1=True, surprise_scale=3.0).surprise_weight == 1.0  # submitted ignores the rule
    assert FocalAgent(submitted_v1=True, payoff_matrix=payoff_matrix_for_u(0.4)).surprise_weight == 1.0
    print(f"11. psi(u) = c (1 - u) / u read from the agent's own game: {got}; an explicit surprise_weight wins; submitted_v1 gives 1")


def check_default_equals_explicit():
    cand = dict(surprise_weight=0.75, eta=AGENT_ETA, t_max=AGENT_T_MAX, exposed_betrayal_weight=3.0)
    worst, n = 0.0, 0
    for p in MANUSCRIPT_PARTNERS + DIAGNOSTIC_PARTNERS:
        for i in range(6):
            a = run(FocalAgent(), p, None, 42 + i)
            b = run(FocalAgent(**cand), p, None, 42 + i)
            worst = max(worst, float(np.abs(a[COLS].to_numpy(float) - b[COLS].to_numpy(float)).max()))
            n += 1
    assert worst == 0.0, worst
    # at another game the rule must equal the explicit psi for that game (partners built for the game played)
    for u, psi in ((0.4, 2.25), (0.8, 0.375)):
        game = payoff_matrix_for_u(u)
        for p in ("TitForTat", "SuspiciousTFT", "Random", "BreakRepair", "StrategicCheater"):
            a = run(FocalAgent(payoff_matrix=game), p, game, 42)
            b = run(FocalAgent(payoff_matrix=game, **dict(cand, surprise_weight=psi)), p, game, 42)
            assert float(np.abs(a[COLS].to_numpy(float) - b[COLS].to_numpy(float)).max()) == 0.0, (u, p)
    print(f"12. FocalAgent() equals the explicit setting (psi 0.75, eta 0.85, t_max 10, nu 3) bit for bit: {n} runs ({len(MANUSCRIPT_PARTNERS + DIAGNOSTIC_PARTNERS)} partners x 6 seeds), "
          f"max |diff| {worst:.1e}; at u = 0.4 and 0.8 the rule equals the explicit psi (2.25, 0.375)")


def check_concentration_match():
    from trust_bayesian_agent_comparison.config import DISCOUNTED_BETA_RHO, SLIDING_WINDOW_W
    trust = 2 * EPS + FocalAgent().t_max
    discpp = 2 * EPS + 1.0 / (1.0 - DISCOUNTED_BETA_RHO)
    window = 2 * EPS + SLIDING_WINDOW_W
    assert abs(trust - discpp) < 1e-9 and abs(trust - window) < 1e-9, (trust, discpp, window)
    print(f"14. concentration ceiling: trust {trust:g} = DiscPP {discpp:g} = window {window:g} (the comparison is concentration-matched)")


def check_partners_unmoved():
    import inspect
    from trust_bayesian_agent_comparison.partners import belief_driven
    src = inspect.getsource(belief_driven)
    assert "ETA," in src and "AGENT_ETA" not in src and "AGENT_T_MAX" not in src
    assert ETA == 0.7 and TRUST_MAX == 10.0
    print("13. the belief-driven partners still read the submitted ETA = 0.7 and TRUST_MAX = 10")


if __name__ == "__main__":
    check_submitted_reproduced()
    check_decision_stage()
    check_nu()
    check_graded()
    check_fallback()
    check_revision_acts()
    check_psi()
    check_tmax_tie()
    check_t_plain_mean()
    check_psi_of_u()
    check_default_equals_explicit()
    check_partners_unmoved()
    check_concentration_match()
    print("\nAll guard rails passed.")
