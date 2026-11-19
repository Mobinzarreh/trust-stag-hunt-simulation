# Trust-sensitive and Bayesian agents in a repeated Stag Hunt

Simulation code for

> Zarreh, M. and Grogan, P.T. "Trust in the Design of Collaborative Systems: A Theoretical Model of Repeated
> Strategic Interaction." *Design Science* (manuscript DS-2026-0085).

Repository: https://github.com/Mobinzarreh/trust-stag-hunt-simulation

The repository contains the agents, the partner strategies, the simulation and analysis code, and the scripts that
regenerate every table and figure in the paper. Simulation outputs are not stored here. The scripts regenerate them
from fixed random seeds.

## Installation

Python 3.12 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # the exact versions used for the paper
pip install -e .
```

## Reproducing the paper

```bash
bash reproduce_all.sh
```

This runs every step below in order and writes to `results/` (tables, per-run data) and `results/figures/`.
The full run takes about 45 minutes on a 20-core machine (the simulations use every core through joblib) and
writes about 2.6 GB. Each script can also be run on its own (`python scripts/<name>.py`). Steps marked
"reads" use outputs of earlier steps.

| Paper | Script | Output |
|---|---|---|
| Specification checks | `verify_revised_spec.py` | printed (seconds) |
| Main runs, nine partners of the original study | `run_benchmark_comparison.py` | `results/benchmarks/` |
| Diagnostic partners (Break and Repair, Reliability Ramp, Distrust Lock) | `run_metric_panel.py`, then `run_metric_panel.py --recency plain` | `results/metric_panel/`, `results/metric_panel_plainDisc/` |
| Per-run metrics of all agents on the twelve partners | `run_variant_comparison.py` (reads the two above) | `results/variants/` |
| Tables 4, 5 and 6 | `make_se_tables.py` (reads `results/variants/`) | `results/tables_se/` |
| Figure 1 | `make_dynamics_figures.py` | `results/figures/paper_dynamics_combined.{png,pdf}` |
| Table B1 (mechanisms) | `run_ablation_study.py` | `results/ablation/` |
| Confidence ceiling and initialization runs | `run_bounds_init_sensitivity.py` | `results/bounds_init/` |
| Section 5.1 (initialization) and 5.3 (parameters) | `run_oat_sensitivity.py` (reads the two above) | `results/oat_sensitivity/` |
| Figure 2 | `make_oat_tornado.py --gap --paper` | `results/figures/paper_oat_tornado_gap.{png,pdf}` |
| Section 5.4 (memory of the benchmarks) | `run_recency_sweep.py` | `results/recency_sweep/` |
| Section 5.3 and Appendix B (behavioral sensitivity) | `run_oat_behavioral.py` (reads `oat_sensitivity`, `recency_sweep`, `variants`) | `results/oat_behavioral/` |
| Section 5.5 and Table B2 (game structure) | `run_game_structure_sweep.py` | `results/game_structure/` (regime `manuscript`, condition `coupled/coupled`) |
| Appendix B (number of rounds) | `run_rounds_sensitivity.py` | `results/rounds_sensitivity/` |
| Appendix B (persistence length of the delays) | `check_persistence_length.py` | `results/persistence_length/` |
| Number of Monte Carlo runs (N = 300) | `check_mc_convergence.py` | `results/convergence/` |
| Originally submitted model against the submitted tables | `reproduce_manuscript_tables.py --rerun` | printed |

Every simulation uses 300 Monte Carlo runs of 100 rounds. Run *i* uses seed `MC_BASE_SEED + i`. The trust-sensitive
agent draws from RNG slot 1 and the benchmarks from slot 2, so the two streams are independent, and the standard error
of a difference is `sqrt(SE_1^2 + SE_2^2)`. Several scripts assert that cells they share with an earlier step are
identical run for run.

## Code

```
trust_bayesian_agent_comparison/
  config.py              payoff matrix (R = 4, M = 3, P = 2, S = 0, u = 2/3), parameter defaults, sweep grids
  agents/
    focal_agent.py       FocalAgent: the trust-sensitive agent (belief x, confidence t)
    bayesian_agent.py    BayesianFocalAgent: cumulative Beta-Bernoulli benchmark (Cum)
    discounted_beta_agent.py
                         PriorPreservingDiscountedBetaAgent: the discounted benchmark (Disc, rho = 0.9);
                         DiscountedBetaAgent: the baseline-discounting variant (Section 5.4)
    sliding_window_agent.py
                         SlidingWindowBetaAgent: the sliding-window benchmark (SW, w = 10)
    base.py              shared decision rule (expected utility and logit choice)
  partners/              partner strategies; always build them through registry.partner_factory
  simulation/runner.py   one interaction between an agent and a partner
  analysis/              per-run metrics, behavioral measures, Monte Carlo batches
scripts/                 the steps in the table above
scripts/reference/       the originally submitted FocalAgent, used by verify_revised_spec.py
```

`FocalAgent()` is the model reported in the paper. `FocalAgent(submitted_v1=True)` is the model of the originally
submitted manuscript, and `partner_factory(..., submitted_v1=True)` gives the partner configuration of that version.
Both are kept so the submitted results stay reproducible.

The package also holds code that the paper does not report: the held-out and co-adaptive partners
(`partners/heldout.py`, `partners/learners.py`), change-point and Thompson-sampling benchmarks
(`agents/changepoint_agent.py`, `agents/thompson_agents.py`), and partners retired during the revision. They are kept
because the partner registry imports them. No script in `reproduce_all.sh` uses them in a reported result.

### Labels used in the code

| Label | Meaning |
|---|---|
| TS, version B | the trust-sensitive agent, `FocalAgent()` |
| version A | the originally submitted agent, `FocalAgent(submitted_v1=True)` |
| Disc, DiscPP, `disc_pp_rho0.90`, `discpp0.9` | discounted Bayesian agent with a fixed prior, rho = 0.9 |
| plain Disc, `discounted_rho0.90`, `disc0.9` | baseline-discounting variant (the prior decays too) |
| SW, `window_w10`, `win10` | sliding-window Bayesian agent, w = 10 |
| Cum, `cumulative` | cumulative Bayesian agent |
| B1 | the main runs of `run_benchmark_comparison.py` |
| B2 | the mechanism ablation, `run_ablation_study.py` |
| B4 | the ceiling and initialization runs, `run_bounds_init_sensitivity.py` |
| T9 | the memory sweep of the benchmarks, `run_recency_sweep.py` |
| u | normalized deviation loss, the belief at which collaborating and not collaborating pay the same |

## License

MIT, see [LICENSE](LICENSE).
