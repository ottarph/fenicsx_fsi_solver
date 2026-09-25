# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Monolithic ALE FSI2 solver with stiffened linear-elastic mesh motion.

The problem is that of :mod:`xfsi_solver.solvers.fsi2_harmonic_diffmesh`:
shared quadratic displacement/velocity on the whole mesh, linear pressure on
the fluid submesh, the same materials, boundary conditions, inflow ramp and
theta scheme. Only the mesh extension differs: the fluid displacement is
the non-incremental, Jacobian-stiffened linear-elastic extension of the
interface displacement on the reference configuration (see
:mod:`xfsi_solver.fsi.mesh_extension`), which replaces both the harmonic
volume term and its interface flux. The time driver, the direct and
fieldsplit linear solvers and the full/no-ALE Jacobians are shared; the
fieldsplit displacement preconditioner extracts the elastic fluid-interior
block from the preconditioning matrix.

Time steps use accepted-time semantics: step ``n`` imposes the inflow at,
and reports the state at, ``t0 + n dt`` through ``T`` (see
:func:`xfsi_solver.solvers.fsi2_harmonic_diffmesh._steps`).

Every run writes to ``--output-dir``:

``uv.bp``, ``p.bp``
    VTX output of ``(u, v)`` and ``p``.
``qoi.txt``
    ``t, drag, lift, A_x, A_y``.
``diagnostics.jsonl``
    One JSON record per accepted step: time, step size, Newton and Krylov
    iterations and convergence reasons, true linear residuals, timings,
    sampled geometry validity and field residuals; and a final record for a
    failed step.
``run.json``
    Configuration, software, mesh provenance, reference geometry and weight
    range, and the outcome.
``checkpoints/``
    Restart states (per rank, see ``--checkpoint-every``).

Example::

    conda run --no-capture-output -n xfsi_solver python -m \\
        xfsi_solver.solvers.fsi2_stiffened_elastic_diffmesh \\
        --mesh data/meshes/fsi2/mesh_sec.xdmf --T 15 --dt 0.0025 \\
        --linear-solver fieldsplit --jacobian-mode no_ale \\
        --mesh-stiffening-exponent 2.5 --mesh-poisson-ratio 0.3 \\
        --output-dir output/fsi2_stiffened_elastic
"""

import argparse
import hashlib
import json
import math
import platform
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import dolfinx as dfx
import dolfinx.fem.petsc  # noqa: F401
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.fsi.mesh_extension import (
    SAMPLE_RESOLUTION,
    StiffenedElasticMeshExtension,
    global_range,
    sample_points,
)
from xfsi_solver.solvers import fsi2_harmonic_diffmesh as shared
from xfsi_solver.solvers.fsi2_harmonic_diffmesh import (
    DIAGNOSTIC_FIELDS,
    PHYSICAL_MARKERS,
    FSIProblem,
    SolverConfig,
    StepMonitor,
)
from xfsi_solver.solvers.fsi2_harmonic_diffmesh_fieldsplit import FieldSplitConfig

SOLVER_NAME = "fsi2_stiffened_elastic_diffmesh"
OPTIONS_PREFIX = f"{SOLVER_NAME}_"


@dataclass
class MeshMotionConfig:
    """Parameters of the stiffened elastic mesh extension.

    Attributes:
        mesh_stiffening_exponent: ``chi`` of ``w = (j_star / j_0)^chi``.
        mesh_poisson_ratio: ``nu_m``.
        mesh_equation_scale: ``alpha``, the scale of the mesh equation relative
            to the solid kinematic equation in the shared displacement rows
            (the harmonic solver's ``alpha_u``).
        mesh_modulus: ``E_0``, a dimensionless normalization.
        mesh_quadrature_degree: ``None`` for the default of the cell type.
        mesh_weighting: ``"pointwise"`` or the experimental ``"cell_volume"``.
    """
    mesh_stiffening_exponent: float = 2.5
    mesh_poisson_ratio: float = 0.3
    mesh_equation_scale: float = 1e-9
    mesh_modulus: float = 1.0
    mesh_quadrature_degree: int | None = None
    mesh_weighting: str = "pointwise"

    def __post_init__(self):
        if not (math.isfinite(self.mesh_equation_scale) and self.mesh_equation_scale > 0.0):
            raise ValueError("mesh_equation_scale must be positive and finite")
        self.extension()

    def extension(self) -> StiffenedElasticMeshExtension:
        return StiffenedElasticMeshExtension(
            stiffening_exponent=self.mesh_stiffening_exponent, poisson_ratio=self.mesh_poisson_ratio,
            modulus=self.mesh_modulus, quadrature_degree=self.mesh_quadrature_degree,
            weighting=self.mesh_weighting)


def build_problem(mesh_path, dt_val, mesh_config: MeshMotionConfig | None = None) -> FSIProblem:
    """The FSI2 problem with the stiffened elastic mesh extension."""
    mesh_config = MeshMotionConfig() if mesh_config is None else mesh_config
    return shared.build_problem(mesh_path, dt_val, mesh_extension=mesh_config.extension(),
                                alpha_u=mesh_config.mesh_equation_scale)


def mesh_checksum(mesh_path) -> str:
    """SHA-256 of the XDMF file and its HDF5 data file (computed on rank 0)."""
    digest = None
    if MPI.COMM_WORLD.rank == 0:
        h = hashlib.sha256()
        xdmf = Path(mesh_path)
        for path in (xdmf, xdmf.with_suffix(".h5")):
            if path.exists():
                h.update(path.read_bytes())
        digest = h.hexdigest()
    return MPI.COMM_WORLD.bcast(digest, root=0)


def software_versions() -> dict:
    import basix
    import petsc4py

    return {"python": platform.python_version(), "dolfinx": dfx.__version__, "basix": basix.__version__,
            "ufl": ufl.__version__, "petsc4py": petsc4py.__version__,
            "petsc": ".".join(map(str, PETSc.Sys.getVersion())), "mpi": MPI.Get_library_version().splitlines()[0]}


class InvalidStateError(RuntimeError):
    """An accepted state failed the sampled geometry validity check."""


def deformation_measures(grad_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``det(I + grad u)`` and the condition number of ``I + grad u`` (2D), sample-wise."""
    F = grad_u + np.eye(2)
    det = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]
    frobenius2 = np.sum(F ** 2, axis=(-2, -1))
    # singular values s1 >= s2: s1^2 + s2^2 = |F|^2, s1 s2 = |det F|
    root = np.sqrt(np.maximum(frobenius2 ** 2 - 4.0 * det ** 2, 0.0))
    s1 = np.sqrt(0.5 * (frobenius2 + root))
    s2 = np.abs(det) / np.where(s1 > 0, s1, 1.0)
    with np.errstate(divide="ignore"):
        cond = np.where(s2 > 0, s1 / np.where(s2 > 0, s2, 1.0), np.inf)
    return det, cond


class GeometryMonitor(StepMonitor):
    """Sampled ALE validity, residual diagnostics and machine-readable run records.

    ``J_ALE = det(I + grad_X u)`` and the condition number of ``I + grad_X u``
    are sampled at the lattice :func:`~xfsi_solver.fsi.mesh_extension.sample_points`
    (``sample_resolution`` subintervals per parent-cell edge, vertices and
    edge points included) of every owned fluid and solid cell. This is a
    sampled validity check, not a proof of global bijectivity. Newton
    iterates with a sampled ``J_ALE <= 0`` or non-finite values are rejected
    as a SNES function domain error (``check_iterates``); an accepted state
    failing the check raises :class:`InvalidStateError`.
    """

    def __init__(self, output_dir, sample_resolution: int = SAMPLE_RESOLUTION, check_iterates: bool = True,
                 run_info: dict | None = None):
        self.output_dir = Path(output_dir)
        self.sample_resolution = sample_resolution
        self.check_iterates = check_iterates
        self.run_info = run_info or {}
        self.comm = MPI.COMM_WORLD
        self.current = None
        self.last_accepted = None
        self.rejected_iterates = 0

    @property
    def diagnostics_path(self):
        return self.output_dir / "diagnostics.jsonl"

    @property
    def run_path(self):
        return self.output_dir / "run.json"

    def _write_json(self, path, record, mode="w"):
        if self.comm.rank == 0:
            with open(path, mode) as f:
                f.write(json.dumps(record, default=_json_default) + ("\n" if mode == "a" else ""))

    def setup(self, problem, nonlinear_problem, linear_solver) -> dict:
        self.problem = problem
        self.snes = nonlinear_problem.solver
        self.linear_solver = linear_solver
        mesh = problem.mesh
        n_owned = mesh.topology.index_map(mesh.topology.dim).size_local
        points = sample_points(mesh.topology.cell_name(), self.sample_resolution)
        self._grad = {}
        for name in ("fluid", "solid"):
            cells = problem.cell_tags.find(PHYSICAL_MARKERS["ALE_fluid" if name == "fluid" else "solid"])
            cells = cells[cells < n_owned]
            expression = dfx.fem.Expression(ufl.grad(problem.u), points, comm=mesh.comm)
            self._grad[name] = (expression, cells, points.shape[0])
        self.alpha = float(problem.constants["alpha_u"].value)

        dofs = {name: space.dofmap.index_map.size_global * space.dofmap.index_map_bs
                for name, space in (("u", problem.U), ("v", problem.V), ("p", problem.P))}
        self.run = {
            "solver": SOLVER_NAME, "ranks": self.comm.size, "software": software_versions(),
            "mesh": {"path": problem.mesh_path, "sha256": mesh_checksum(problem.mesh_path),
                     "cells": mesh.topology.index_map(mesh.topology.dim).size_global,
                     "cell_type": mesh.topology.cell_name(), "geometry_degree": mesh.geometry.cmap.degree},
            "dofs": dofs | {"total": sum(dofs.values())},
            "mesh_extension": problem.mesh_extension_info,
            "sample_resolution": self.sample_resolution,
            "sample_points_per_cell": len(sample_points(mesh.topology.cell_name(), self.sample_resolution)),
            **self.run_info,
            "initial_geometry": self.geometry(),
            "status": "running",
        }
        if self.comm.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.diagnostics_path.write_text("")
        self._write_json(self.run_path, self.run)
        if self.comm.rank == 0:
            print(f"mesh motion: {json.dumps(self.run['mesh_extension'], default=_json_default)}")
        return {"run": self.run}

    def geometry(self) -> dict:
        """Sampled ``J_ALE`` and condition number of the deformation gradient in fluid and solid."""
        result = {}
        for name, (expression, cells, n_points) in self._grad.items():
            values = expression.eval(self.problem.mesh, cells) if cells.size else np.empty((0, n_points * 4))
            det, cond = deformation_measures(values.reshape(cells.size, n_points, 2, 2))
            det_min, det_max, det_bad = global_range(det, self.comm)
            _, cond_max, cond_bad = global_range(cond, self.comm)
            result[name] = {"J_min": det_min, "J_max": det_max, "cond_max": cond_max,
                            "nonfinite": det_bad + cond_bad}
        u_bad = self.comm.allreduce(int(np.sum(~np.isfinite(np.concatenate(
            [f.x.array for f in self.problem.solution])))), op=MPI.SUM)
        result["nonfinite_dofs"] = u_bad
        result["valid"] = bool(u_bad == 0 and all(result[k]["nonfinite"] == 0 and result[k]["J_min"] > 0.0
                                                  for k in ("fluid", "solid")))
        return result

    def check_iterate(self, problem) -> bool:
        if not self.check_iterates:
            return True
        valid = self.geometry()["valid"]
        if not valid:
            self.rejected_iterates += 1
        return valid

    def begin_step(self, t, dt):
        self.current = {"t": t, "dt": dt}

    def accepted(self, problem, step) -> dict:
        geometry = self.geometry()
        final = step.field_residuals[-1]
        initial = step.field_residuals[0]
        residuals = dict(zip(DIAGNOSTIC_FIELDS, final.tolist(), strict=True))
        record = {
            "t": step.t, "dt": step.dt, "accepted": True,
            "snes_iterations": step.snes_iterations, "snes_reason": step.converged_reason,
            "residual_initial": float(step.residual_history[0]), "residual_final": float(step.residual_history[-1]),
            "field_residuals_initial": dict(zip(DIAGNOSTIC_FIELDS, initial.tolist(), strict=True)),
            "field_residuals_final": residuals,
            # the mesh rows carry the factor alpha; kinematic = solid displacement rows,
            # momentum = velocity rows, incompressibility = pressure rows
            "scaled_residuals_final": {"mesh": residuals["u_fluid"] / self.alpha, "kinematic": residuals["u_solid"],
                                       "momentum": residuals["v"], "incompressibility": residuals["p"]},
            "ksp_iterations": [d["iterations"] for d in step.linear_solves],
            "ksp_reasons": [d["reason"] for d in step.linear_solves],
            "true_relative_residuals": [d["true_relative_residual"] for d in step.linear_solves],
            "timings": step.timings | {"step": step.time},
            "preconditioner_statistics": step.preconditioner_statistics,
            "drag": step.drag, "lift": step.lift, "A": step.tip_displacement.tolist(),
            "geometry": geometry,
            "rejected_iterates": self.rejected_iterates,
        }
        self._write_json(self.diagnostics_path, record, mode="a")
        if not geometry["valid"]:
            raise InvalidStateError(f"Invalid accepted state at t = {step.t}: {geometry}")
        self.last_accepted = record
        return record

    def failed(self, problem, error):
        reason = self.snes.getConvergedReason() if hasattr(self, "snes") else None
        history = self.snes.getConvergenceHistory()[0].tolist() if hasattr(self, "snes") else []
        ksp_reason = self.snes.getKSP().getConvergedReason() if hasattr(self, "snes") else None
        geometry = self.geometry() if hasattr(self, "_grad") else None
        record = {"t": None if self.current is None else self.current["t"],
                  "dt": None if self.current is None else self.current["dt"], "accepted": False,
                  "error": f"{type(error).__name__}: {error}", "snes_reason": reason, "ksp_reason": ksp_reason,
                  "residual_history": history, "geometry_of_last_iterate": geometry,
                  "rejected_iterates": self.rejected_iterates}
        if hasattr(self, "run"):
            self._write_json(self.diagnostics_path, record, mode="a")
            self.run |= {"status": "failed", "failure": record, "last_accepted": self.last_accepted,
                         "traceback": traceback.format_exception(error)[-3:]}
            self._write_json(self.run_path, self.run)

    def finish(self, result):
        steps = result.steps
        solves = [d for s in steps for d in s.linear_solves]
        geometry = [s.diagnostics["geometry"] for s in steps]
        self.run |= {
            "status": "completed",
            "t_final": steps[-1].t if steps else result.metadata.get("t0"),
            "steps": len(steps), "elapsed": result.elapsed,
            "newton_iterations": int(sum(s.snes_iterations for s in steps)),
            "krylov_iterations": int(sum(d["iterations"] for d in solves)),
            "max_krylov_iterations": int(max((d["iterations"] for d in solves), default=0)),
            "max_true_relative_residual": float(max((d["true_relative_residual"] for d in solves), default=0.0)),
            "min_J_fluid": min((g["fluid"]["J_min"] for g in geometry), default=None),
            "min_J_solid": min((g["solid"]["J_min"] for g in geometry), default=None),
            "max_cond_fluid": max((g["fluid"]["cond_max"] for g in geometry), default=None),
            "rejected_iterates": self.rejected_iterates,
            "timings": {k: float(sum(s.timings.get(k, 0.0) for s in steps)) for k in
                        (steps[0].timings if steps else {})},
        }
        self._write_json(self.run_path, self.run)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def state_metadata(mesh_path, dt_val, mesh_config: MeshMotionConfig) -> dict:
    """Restart metadata: a state is only valid for the same solver, mesh, time step and mesh law."""
    return {"solver": SOLVER_NAME, "mesh_sha256": mesh_checksum(mesh_path), "dt": dt_val,
            "mesh_motion": asdict(mesh_config)}


def solve(mesh_path, T, dt_val, output_dir, config: SolverConfig | None = None,
          mesh_config: MeshMotionConfig | None = None, *, initial_state=None, checkpoint_every: float | None = None,
          output_every: int = 4, sample_resolution: int = SAMPLE_RESOLUTION, check_iterates: bool = True):
    """Time step FSI2 with stiffened elastic mesh motion from ``t0`` (0, or the restart time) to ``T``.

    Returns the :class:`~xfsi_solver.solvers.fsi2_harmonic_diffmesh.SolveResult`;
    see the module docstring for the files written to ``output_dir``.
    """
    config = SolverConfig() if config is None else config
    mesh_config = MeshMotionConfig() if mesh_config is None else mesh_config
    output_dir = Path(output_dir)
    run_info = {"T": T, "dt": dt_val, "solver_config": asdict(config), "mesh_motion": asdict(mesh_config),
                "initial_state": None if initial_state is None else str(initial_state),
                "checkpoint_every": checkpoint_every, "output_every": output_every}
    monitor = GeometryMonitor(output_dir, sample_resolution=sample_resolution, check_iterates=check_iterates,
                              run_info=run_info)
    result = shared.solve(
        mesh_path, T, dt_val,
        output_path=str(output_dir / "uv.bp"), output_path_p=str(output_dir / "p.bp"),
        qoi_path=str(output_dir / "qoi.txt"), config=config,
        initial_state=initial_state, checkpoint_dir=output_dir / "checkpoints", checkpoint_every=checkpoint_every,
        problem_builder=lambda path, dt: build_problem(path, dt, mesh_config),
        options_prefix=OPTIONS_PREFIX, time_semantics="accepted", save_every=output_every, monitor=monitor,
        state_metadata=state_metadata(mesh_path, dt_val, mesh_config),
    )
    monitor.finish(result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--T", type=float, default=15.0, help="final time")
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--linear-solver", choices=shared.LINEAR_SOLVERS, default="fieldsplit")
    parser.add_argument("--jacobian-mode", choices=shared.JACOBIAN_MODES, default="no_ale")
    parser.add_argument("--preconditioner-mode", choices=shared.JACOBIAN_MODES, default=None)
    parser.add_argument("--displacement-fluid", default="cholesky",
                        help="fieldsplit solver of the fluid-interior displacement block")
    parser.add_argument("--snes-atol", type=float, default=SolverConfig.snes_atol)
    parser.add_argument("--snes-rtol", type=float, default=SolverConfig.snes_rtol)
    parser.add_argument("--snes-max-it", type=int, default=SolverConfig.snes_max_it)
    parser.add_argument("--snes-linesearch", default="none")
    parser.add_argument("--ksp-rtol", type=float, default=SolverConfig.ksp_rtol)
    parser.add_argument("--snes-monitor", action="store_true")
    parser.add_argument("--mesh-stiffening-exponent", type=float, default=MeshMotionConfig.mesh_stiffening_exponent)
    parser.add_argument("--mesh-poisson-ratio", type=float, default=MeshMotionConfig.mesh_poisson_ratio)
    parser.add_argument("--mesh-equation-scale", type=float, default=MeshMotionConfig.mesh_equation_scale)
    parser.add_argument("--mesh-quadrature-degree", type=int, default=None)
    parser.add_argument("--mesh-weighting", default="pointwise")
    parser.add_argument("--output-dir", default="output/fsi2_stiffened_elastic")
    parser.add_argument("--output-every", type=int, default=4, help="VTX output every N steps")
    parser.add_argument("--checkpoint-every", type=float, default=None, help="restart states every this time")
    parser.add_argument("--restart", default=None, help="state file (written by this solver) to continue from")
    parser.add_argument("--sample-resolution", type=int, default=SAMPLE_RESOLUTION)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = SolverConfig(
        jacobian_mode=args.jacobian_mode, preconditioner_mode=args.preconditioner_mode,
        linear_solver=args.linear_solver, snes_atol=args.snes_atol, snes_rtol=args.snes_rtol,
        snes_max_it=args.snes_max_it, snes_linesearch_type=args.snes_linesearch, snes_monitor=args.snes_monitor,
        ksp_rtol=args.ksp_rtol, fieldsplit=FieldSplitConfig(displacement_fluid=args.displacement_fluid))
    mesh_config = MeshMotionConfig(
        mesh_stiffening_exponent=args.mesh_stiffening_exponent, mesh_poisson_ratio=args.mesh_poisson_ratio,
        mesh_equation_scale=args.mesh_equation_scale, mesh_quadrature_degree=args.mesh_quadrature_degree,
        mesh_weighting=args.mesh_weighting)
    if MPI.COMM_WORLD.rank == 0:
        print(" ".join(sys.argv), flush=True)
    solve(args.mesh, args.T, args.dt, args.output_dir, config, mesh_config, initial_state=args.restart,
          checkpoint_every=args.checkpoint_every, output_every=args.output_every,
          sample_resolution=args.sample_resolution)


if __name__ == "__main__":
    main()
