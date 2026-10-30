"""Make the figures of report.md from the Newton logs of the runs in output/.

    python notes/newton-convergence/make_figures.py [figure_dir]

The runs are the rerun of all configurations one at a time (output/newton_convergence_rerun),
whose iterates are those of the first batches, and the 60-step check of the full Jacobian
(output/newton_convergence_check), whose times stand in for the disturbed full-Jacobian rerun in
the cost figure. Writes SVG and PNG figures into figure_dir, by default plots/ next to this file.

The residual, contraction and iteration figures come in pairs, with and without the displacement
prediction, on the same axes and with the same colors and labels, to be shown side by side. The
J_0 runs without the prediction are those of fsi2_biharmonic_mixed, and those with it of
fsi2_simplified, which takes the same iterates as fsi2_biharmonic_mixed without the prediction.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from xfsi_solver.scripts.plot_newton_logs import (
    _RC,
    Run,
    plot_contraction_by_age,
    plot_cost,
    plot_iterations_over_time,
    plot_residual_history,
)

RERUN = Path("output/newton_convergence_rerun")
CHECK = Path("output/newton_convergence_check")
GAMMA = 0.2

figure_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "plots"
figure_dir.mkdir(parents=True, exist_ok=True)

full = Run("full Jacobian", RERUN / "fulljacobian")
mixed_fresh = Run("J_0, reassembled", RERUN / "mixed_none")
mixed_reused = Run(f"J_0, reused (γ={GAMMA})", RERUN / "mixed_0.2")
predicted_fresh = Run("J_0, reassembled", RERUN / "simplified_none")
predicted_reused = Run(f"J_0, reused (γ={GAMMA})", RERUN / "simplified_0.2")
# The full Jacobian has no prediction; it is the reference in both figures of a pair.
full_reference = Run("full Jacobian (no prediction)", RERUN / "fulljacobian")
with_prediction = [full_reference, predicted_fresh, predicted_reused]
without_prediction = [full, mixed_fresh, mixed_reused]

simplified_fresh = Run("simplified, reassembled", RERUN / "simplified_none_nopred")
simplified_reused = Run(f"simplified, reused (γ={GAMMA})", RERUN / "simplified_0.2_nopred")
cost_runs = [
    Run("full Jacobian (60 steps)", CHECK / "fulljacobian"),
    Run("mixed, reassembled", RERUN / "mixed_none"),
    Run(f"mixed, reused (γ={GAMMA})", RERUN / "mixed_0.2"),
    simplified_fresh,
    simplified_reused,
    Run("simplified, predicted, reassembled", RERUN / "simplified_none"),
    Run(f"simplified, predicted, reused (γ={GAMMA})", RERUN / "simplified_0.2"),
]

all_runs = with_prediction + without_prediction
max_it = max(int(run.iterations.max()) for run in all_runs)
# The residual figures stop at the last iteration that half the steps of a run reach.
last_it = 7
max_age = max(int(run.iteration_records()["age"].max()) for run in (predicted_reused, mixed_reused))
its_ylim = (min(run.iterations.min() for run in all_runs) - 0.5, max_it + 0.5)
pairs = {"with_prediction": ("With the prediction", with_prediction),
         "without_prediction": ("Without the prediction", without_prediction)}  # fmt: skip

with plt.rc_context(_RC):
    for suffix in ("svg", "png"):
        for name, (heading, runs) in pairs.items():
            _, fresh, reused = runs
            plot_residual_history(
                runs,
                figure_dir / f"residuals_{name}.{suffix}",
                title=f"{heading}: residual per iteration, median and IQR",
                last_it=last_it,
                ylim=(1.0e-10, 1.0e3),
            )
            plot_contraction_by_age(
                [reused],
                fresh,
                GAMMA,
                figure_dir / f"contraction_{name}.{suffix}",
                title=f"{heading}: contraction, median and IQR",
                max_age=max_age,
            )
            plot_iterations_over_time(
                runs,
                figure_dir / f"iterations_{name}.{suffix}",
                title=f"{heading}: tip displacement and iterations per step",
                its_ylim=its_ylim,
            )
        plot_cost(cost_runs, figure_dir / f"cost.{suffix}")
