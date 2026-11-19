#!/usr/bin/env bash
# Regenerate every simulation, table and figure of the paper into results/.
# Usage: bash reproduce_all.sh [python]      (default: python on PATH, with the package installed)
# Steps run in dependency order; each later step reads the outputs of earlier ones.
set -euo pipefail
PY="${1:-python}"
cd "$(dirname "$0")"

step() {
    local t0=$SECONDS
    echo "=== $* ==="
    "$PY" "$@"
    echo "=== done in $((SECONDS - t0)) s: $* ==="
}

step scripts/verify_revised_spec.py                          # specification checks (seconds)

# Main comparison (Section 4: Tables 4 to 6, Figure 1)
step scripts/run_benchmark_comparison.py --overwrite         # nine partners of the original study, four agents
step scripts/run_metric_panel.py --overwrite                 # diagnostic partners (Break and Repair, Reliability Ramp, Distrust Lock)
step scripts/run_metric_panel.py --overwrite --recency plain # the same with the baseline-discounting variant (Section 5.4)
step scripts/run_variant_comparison.py --overwrite           # per-run metrics of every agent on the twelve partners
step scripts/make_se_tables.py                               # Tables 4 to 6 (results/tables_se/)
step scripts/make_dynamics_figures.py                        # Figure 1

# Sensitivity (Section 5 and Appendix B)
step scripts/run_ablation_study.py --overwrite               # Table B1
step scripts/run_bounds_init_sensitivity.py --overwrite      # confidence ceiling and initialization runs
step scripts/run_oat_sensitivity.py --overwrite              # one-factor-at-a-time payoff sensitivity
step scripts/make_oat_tornado.py --gap --paper               # Figure 2
step scripts/run_recency_sweep.py --overwrite                # memory sweep of the benchmarks (Section 5.4)
step scripts/run_oat_behavioral.py --overwrite               # one-factor-at-a-time behavioral sensitivity
step scripts/run_game_structure_sweep.py --overwrite         # Table B2
step scripts/run_rounds_sensitivity.py --overwrite           # number of rounds, 10 to 1000
step scripts/check_persistence_length.py                     # persistence length of the delays, L = 1 to 10
step scripts/check_mc_convergence.py                         # number of Monte Carlo runs

# The originally submitted model against the values printed in the submitted manuscript
step scripts/reproduce_manuscript_tables.py --rerun
