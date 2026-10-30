"""Make the figures of report.md from the Newton logs of the runs in output/.

    python notes/newton-convergence/make_figures.py [figure_dir]

The runs are the rerun of all configurations one at a time (output/newton_convergence_rerun),
whose iterates are those of the first batches, and the 60-step check of the full Jacobian
(output/newton_convergence_check), whose times stand in for the disturbed full-Jacobian rerun in
the cost figure. Writes SVG and PNG figures into figure_dir, by default plots/ next to this file.
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

figure_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "plots"
figure_dir.mkdir(parents=True, exist_ok=True)

full = Run("full Jacobian", RERUN / "fulljacobian")
mixed_fresh = Run("J_0, reassembled", RERUN / "mixed_none")
mixed_reused = Run("J_0, reused (γ=0.2)", RERUN / "mixed_0.2")
simplified_fresh = Run("simplified, reassembled", RERUN / "simplified_none_nopred")
simplified_reused = Run("simplified, reused (γ=0.2)", RERUN / "simplified_0.2_nopred")
predicted_fresh = Run("simplified, predicted, reassembled", RERUN / "simplified_none")
predicted_reused = Run("simplified, predicted, reused (γ=0.2)", RERUN / "simplified_0.2")
full_check = Run("full Jacobian (60 steps)", CHECK / "fulljacobian")

with plt.rc_context(_RC):
    for suffix in ("svg", "png"):
        plot_residual_history([full, mixed_fresh, mixed_reused], figure_dir / f"jacobian_residuals.{suffix}")
        plot_iterations_over_time([full, mixed_fresh, mixed_reused], figure_dir / f"jacobian_iterations.{suffix}")
        plot_contraction_by_age([mixed_reused], mixed_fresh, 0.2, figure_dir / f"reuse_contraction.{suffix}")
        plot_residual_history(
            [simplified_fresh, predicted_fresh, simplified_reused, predicted_reused],
            figure_dir / f"prediction_residuals.{suffix}",
        )
        plot_iterations_over_time(
            [simplified_fresh, predicted_fresh, simplified_reused, predicted_reused],
            figure_dir / f"prediction_iterations.{suffix}",
        )
        plot_contraction_by_age(
            [predicted_reused], predicted_fresh, 0.2, figure_dir / f"prediction_contraction.{suffix}"
        )
        plot_cost(
            [full_check, mixed_fresh, mixed_reused, simplified_fresh, simplified_reused, predicted_fresh,
             predicted_reused],
            figure_dir / f"cost.{suffix}",
        )  # fmt: skip
