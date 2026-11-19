"""
Configuration module for trust-bayesian agent comparison study.
Centralized constants and default parameters.
"""
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_timestamped_path(base_dir: Path, prefix: str = "", suffix: str = ".csv") -> Path:
    """Generate timestamped file path for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{prefix}{timestamp}{suffix}"

# ============================================================================
# GAME PARAMETERS
# ============================================================================
# Stag Hunt Payoff Matrix
PAYOFF_MATRIX = np.array([
    [[2, 2], [3, 0]],  # [Hare][Partner action][player_id]
    [[0, 3], [4, 4]]   # [Stag][Partner action][player_id]
], dtype=float)


def as_payoff_matrix(payoff_matrix=None) -> np.ndarray:
    """Return payoff_matrix (default PAYOFF_MATRIX) as a float array of shape (2, 2, 2)."""
    m = np.array(PAYOFF_MATRIX if payoff_matrix is None else payoff_matrix, dtype=float)
    if m.shape != (2, 2, 2):
        raise ValueError(f"payoff matrix must have shape (2, 2, 2), got {m.shape}")
    return m


def get_payoff(player1_strategy: int, player2_strategy: int, player_id: int, payoff_matrix=None) -> float:
    """Get payoff for a given strategy combination."""
    m = PAYOFF_MATRIX if payoff_matrix is None else payoff_matrix
    return float(m[int(player1_strategy), int(player2_strategy), int(player_id)])

def stag_indifference_threshold(payoff_matrix=None) -> float:
    """
    Calculate the normalized deviation loss (threshold) for cooperation.

    Returns the probability threshold at which the agent is indifferent
    between cooperating (Stag) and defecting (Hare) based on the payoff matrix.

    Formula: p* = (a00 - a10) / [(a00 - a10) + (a11 - a01)]
    where aij is the payoff for (agent_action=i, partner_action=j)

    Args:
        payoff_matrix: Payoff matrix indexed [agent_action, partner_action, player_id]
            (default: PAYOFF_MATRIX)

    Returns:
        float: Threshold probability in [0, 1]
    """
    a00 = get_payoff(0, 0, 0, payoff_matrix)
    a01 = get_payoff(0, 1, 0, payoff_matrix)
    a10 = get_payoff(1, 0, 0, payoff_matrix)
    a11 = get_payoff(1, 1, 0, payoff_matrix)

    den = (a00 - a10) + (a11 - a01)
    return float((a00 - a10) / den) if den != 0 else 0.5


def total_deviation_loss(payoff_matrix=None) -> float:
    """D = (a00 - a10) + (a11 - a01), the denominator of the normalized deviation loss u."""
    m = as_payoff_matrix(payoff_matrix)
    return float((m[0, 0, 0] - m[1, 0, 0]) + (m[1, 1, 0] - m[0, 1, 0]))


def best_response_payoff(payoff_matrix=None) -> np.ndarray:
    """Array [best payoff if partner plays 0, best payoff if partner plays 1]."""
    m = as_payoff_matrix(payoff_matrix)
    return np.array([m[:, o, 0].max() for o in (0, 1)])


def symmetric_payoff_matrix(R: float, S: float, M: float, P: float) -> np.ndarray:
    """
    Symmetric game with R = (C, C), S = (C, I), M = (I, C), P = (I, I) payoffs to the row player.

    Manuscript notation: the study matrix is R = 4, M = 3, P = 2, S = 0, satisfying the canonical
    Stag Hunt ordering R > M >= P > S. Index 1 is collaborate (Stag) and 0 is independence (Hare),
    as in PAYOFF_MATRIX.
    """
    return np.array([
        [[P, P], [M, S]],
        [[S, M], [R, R]],
    ], dtype=float)


def payoff_letters(payoff_matrix=None) -> dict:
    """(R, M, P, S) of a symmetric payoff matrix, in the manuscript's notation."""
    m = as_payoff_matrix(payoff_matrix)
    return {"R": float(m[1, 1, 0]), "M": float(m[0, 1, 0]),
            "P": float(m[0, 0, 0]), "S": float(m[1, 0, 0])}


MANUSCRIPT_FAMILY = "manuscript"
FIXED_D_FAMILIES = ("fixed_stag", "fixed_hare")
GAME_FAMILIES = (MANUSCRIPT_FAMILY,) + FIXED_D_FAMILIES
# regime -> (primary family, alt family that rescores the same runs; None = no companion defined)
GAME_REGIMES = {"manuscript": (MANUSCRIPT_FAMILY, None), "fixed_D": FIXED_D_FAMILIES}
U_SWEEP_GRID = (0.2, 0.3, 0.4, 0.5, 0.6, 2 / 3, 0.7, 0.8, 0.9)


def payoff_matrix_for_u(u: float, family: str = MANUSCRIPT_FAMILY) -> np.ndarray:
    """
    Symmetric Stag Hunt with normalized deviation loss u = (P - S) / ((P - S) + (R - M)).

    'manuscript' reproduces the published payoff tables and lets D vary; the two fixed_* families
    hold the total deviation loss D = 3. All three equal the study matrix at u = 2/3.
    """
    if not 0 < u < 1:
        raise ValueError("u must be in (0, 1)")
    if family == MANUSCRIPT_FAMILY:
        # M = 3, P = 2 throughout. Below the study threshold only the upside R moves (S stays 0);
        # above it only the downside S moves (R stays 4). Both branches meet at u = 2/3.
        if u <= 2 / 3:
            return symmetric_payoff_matrix(R=1 + 2 / u, S=0.0, M=3.0, P=2.0)
        # rounded so the published downside lands exactly (u = 0.8 -> S = -2, not -2.000000000000001);
        # 1 - u is the only subtraction here that loses precision, and 12 decimals is far below
        # any payoff difference that matters while keeping u exact to ~1e-13
        return symmetric_payoff_matrix(R=4.0, S=round(2 - u / (1 - u), 12), M=3.0, P=2.0)
    if family == "fixed_stag":
        return symmetric_payoff_matrix(R=4.0, S=0.0, M=1 + 3 * u, P=3 * u)
    if family == "fixed_hare":
        return symmetric_payoff_matrix(R=6 - 3 * u, S=2 - 3 * u, M=3.0, P=2.0)
    raise ValueError(f"Unknown game family '{family}'. Known: {GAME_FAMILIES}")

# Decision threshold for cooperation (dynamically calculated)
DECISION_THRESHOLD = stag_indifference_threshold()

# ============================================================================
# AGENT PARAMETERS
# ============================================================================
# Learning parameters for dual-state agent
# The agent maintains two state variables:
#   - x (signal): Expected cooperation rate, updated via EWMA
#   - t (trust): Confidence/precision in the signal, updated via consistency measure
ETA = 0.7                  # Learning rate for signal (x) update via EWMA: x_new = x + η*(obs - x)
LEARNING_RATE_T = 0.2      # Learning rate for trust (t) update: t_new = (1-α)*t + α*target
                           # Note: This is implemented as 'trust_smoothing' parameter in FocalAgent
NOISE_SIGMA = 0.02         # Gaussian noise for exploration

# Trust-based agent defaults
MEMORY_DISCOUNT = 0.9      # Recency weighting for signal update (γ_x)
TRUST_DISCOUNT = 0.9      # Recency weighting for trust update (γ_t)
TRUST_SMOOTHING = LEARNING_RATE_T  # Alias: smoothing factor = learning rate for trust
TRUST_MIN = 0.0            # Minimum trust level
TRUST_MAX = 10.0           # Maximum trust level
EPS = 1                  # Small constant for Beta distribution

# Asymmetric loss parameters
LOSS_AVERSION = 3.0        # Betrayal penalty multiplier (λ), submitted value
LAMBDA_SURPRISE = 0.5      # Surprise penalty multiplier (μ), submitted value

AGENT_LOSS_AVERSION = 1.0       # λ in the consistency measure: a violation counts as one miss (mu keeps the
                                # three-way categorization of principle 3)
EXPOSED_BETRAYAL_WEIGHT = 3.0   # ν: weight of an exposed betrayal in the belief memory (the submitted λ, relocated)
ATTRIBUTION = True              # weight only unprovoked exposed betrayals (a reply to the agent's own defection counts 1)
CONFIDENCE_SCORING = "graded"   # consistency scores the probability x gave to what happened ('hit' = submitted)

AGENT_LAMBDA_SURPRISE = 1.0     # mu in the consistency measure: t scores every round alike (a discounted mean of the
                                # probability the belief gave to what happened); submitted value 0.5 = LAMBDA_SURPRISE

AGENT_ETA = 0.85                # eta: step of the belief toward the weighted record (submitted 0.7 = ETA). Must stay <= 0.85:
                                # at 1.0 the stability signature is lost (Gradual Deterioration crossings +0.68 vs DiscPP)
AGENT_T_MAX = 10.0              # ceiling of confidence; at 16 it bought stable-partner payoff and the Distrust Lock, but broke the concentration match
SURPRISE_SCALE = 1.5            # c in psi(u) = c (1 - u) / u: the weight of a surprise is set by the stakes of the game the agent plays,
                                # read from its own payoff matrix (the route x_0 = 1 - u already takes). psi = 0.75 at the study
                                # game u = 2/3, 2.25 at u = 0.4, 0.375 at u = 0.8. Skeptical of unexpected cooperation when
                                # collaboration is risky, generous when it is cheap. c = 3 (psi = 1.5 at the study game) has NO
                                # repair signature there; 1.5 is the value that buys it

# Decision-making
STOCHASTIC = True          # Use probabilistic (logit) choice
INVERSE_TEMPERATURE = 2.0  # Exploration-exploitation trade-off

# Bayesian agent defaults
ALPHA_0 = EPS              # Prior cooperation count
BETA_0 = EPS               # Prior defection count

DISCOUNTED_BETA_RHO = MEMORY_DISCOUNT        # Forgetting factor rho (rho = 1 is the cumulative model)
SLIDING_WINDOW_W = 10                        # Window width w ~ 1 / (1 - DISCOUNTED_BETA_RHO)
BENCHMARK_RHO_GRID = (0.80, 0.90, 0.95)      # Sensitivity grid for rho
BENCHMARK_WINDOW_GRID = (5, 10, 20)          # Matching grid for w (~ 1 / (1 - rho))
RECENCY_BENCHMARK_KEY = f"disc_pp_rho{DISCOUNTED_BETA_RHO:.2f}"       # results/benchmarks/<key>/
RECENCY_BENCHMARK_LABEL = f"DiscPP rho={DISCOUNTED_BETA_RHO:.2f}"

CHANGEPOINT_HAZARD = 1 - DISCOUNTED_BETA_RHO   # h: prior prob. of a regime change per round (1/h = 10)
BENCHMARK_HAZARD_GRID = (0.05, 0.10, 0.20)     # Sensitivity grid for h (h = 0 is the cumulative model)

T_MAX_GRID = (5.0, 10.0, 16.0, 20.0, 50.0, 100.0)                                       # Confidence ceiling t_max
ROUNDS_GRID = (10, 20, 50, 100, 200, 500, 1000)                                 # Rounds per simulation K
INIT_X0_GRID = (0.2, 1 - DECISION_THRESHOLD, 0.5, DECISION_THRESHOLD, 0.9)    # Initial belief x_0
INIT_T0_GRID = (0.0, 1.0, 5.0, 10.0)                                         # Initial confidence t_0

FIXED_CONFIDENCE_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

OAT_DESIGN = {
    "theta":   ("inv_temp",        INVERSE_TEMPERATURE, 1.0,  3.0),
    "eta_x":   ("eta",             AGENT_ETA,           0.65, 1.0),
    "gamma_x": ("memory_discount", MEMORY_DISCOUNT,     0.8,  0.95),
    "gamma_t": ("trust_discount",  TRUST_DISCOUNT,      0.8,  0.95),
    "omega":   ("trust_smoothing", TRUST_SMOOTHING,     0.1,  0.3),
    "nu":      ("exposed_betrayal_weight", EXPOSED_BETRAYAL_WEIGHT, 1.5, 4.5),
    "c":       ("surprise_scale",  SURPRISE_SCALE,      0.75, 2.25),
    "sigma":   ("noise_sigma",     NOISE_SIGMA,         0.01, 0.03),
    "t_max":   ("t_max",           AGENT_T_MAX,         5.0,  20.0),
    "x_0":     ("u_i",             1 - DECISION_THRESHOLD, (1 - DECISION_THRESHOLD) / 2, 0.5),
    "t_0":     ("t_init",          0.0,                 None, 1.0),
}
# theta is shared with the benchmarks, so they are run at these values too (5 checks saturation)
OAT_THETA_BENCHMARK_GRID = (1.0, 2.0, 3.0, 5.0)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
NUM_ROUNDS = 100            # Default rounds per simulation
DEFAULT_SEED = 42          # Reproducibility seed

# ============================================================================
# SENSITIVITY ANALYSIS PARAMETERS
# ============================================================================
SENSITIVITY_SEEDS = (42, 43, 44)  # Multiple seeds for robustness

# Parameter grids
ETA_GRID = np.linspace(0.0, 1.0, 4)
MEMORY_DISCOUNT_GRID = np.linspace(0.0, 1.0, 4)
TRUST_DISCOUNT_GRID = np.linspace(0.0, 1.0, 4)
TRUST_SMOOTHING_GRID = np.linspace(0.0, 1.0, 4)
LOSS_AVERSION_GRID = np.linspace(1.0, 5.0, 5)
LAMBDA_SURPRISE_GRID = np.linspace(0.0, 1.0, 4)
INVERSE_TEMPERATURE_GRID = np.array([0.5, 2.0, 5.0, 10.0])  # Exploration-exploitation trade-off
INITIAL_X_GRID = np.array([0.2, 0.5, 0.66, 0.9])  # Initial signal values (corresponds to stag_indifference_threshold)
T_INIT_GRID = np.array([0.1, 0.5, 1.0, 5.0])  # Initial trust level (confidence in initial signal)

# Parallel processing
N_JOBS = -1                # Use all available cores
VERBOSE = 5               # Progress reporting level

# ============================================================================
# MONTE CARLO PARAMETERS
# ============================================================================
NUM_MONTE_CARLO_RUNS = 300  # Simulations per partner
MC_BASE_SEED = 42           # Base seed for MC runs

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
# Standard figure size
FIGURE_SIZE = (14, 8)
GRID_FIGURE_SIZE = (16, 12)

# Color palettes
TRUST_COLOR = '#2E86AB'
BAYESIAN_COLOR = '#A23B72'
PARTNER_COLOR = '#F18F01'
THRESHOLD_COLOR = '#C73E1D'

# Plot styling
DPI = 100
SAVE_FORMAT = 'png'
TRANSPARENT_BG = False
