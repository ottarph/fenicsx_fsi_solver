# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Measure the sizes of the state, the residual and the Jacobian blocks of
fsi2_biharmonic_fulljacobian_initialguess, by group of dofs, at every Newton iteration.

The groups are those of the solver's nest Jacobian, ``NEST_GROUPS``: v, p, u_s, z_int, u_int, z_f,
u_f. The rows of u_s and u_int hold the solid kinematic relation, those of u_f the mesh motion, and
Dirichlet rows stay in their group with their identity rows. For comparison with the estimates of
``dimensional_analysis_fsi2_biharmonic``.

Run with MPI, for example from the t = 16 snapshot of fsi2_simplified:

    mpirun -n 8 python -m xfsi_solver.scripts.block_scales_fsi2_biharmonic \\
        --initial-state-dir output/snapshots/fsi2_simplified/t16.0000 --steps 3 --out output/block_scales.json
"""

import argparse
import json

import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian_initialguess import NEST_GROUPS, PHYSICAL_MARKERS, solve


def _subvector_norms(vec: PETSc.Vec, index_set: PETSc.IS) -> dict:
    """The l2 and max norms of the entries of ``vec`` at the (owned) rows ``index_set``, and their
    number. Collective."""
    comm = vec.getComm().tompi4py()
    values = vec.getArray(readonly=True)[index_set.getIndices() - vec.getOwnershipRange()[0]]
    return {
        "l2": float(np.sqrt(comm.allreduce(float(values @ values), op=MPI.SUM))),
        "max": comm.allreduce(float(np.abs(values).max()) if values.size else 0.0, op=MPI.MAX),
        "n": comm.allreduce(values.size, op=MPI.SUM),
    }


def _block_stats(block: PETSc.Mat, cutoff: float = 1e-10) -> dict:
    """The largest absolute entry of ``block``, and the number and the geometric mean of its absolute
    entries above ``cutoff`` times the largest, which leaves out the zeros and roundoff in its
    pattern. Collective."""
    comm = block.getComm().tompi4py()
    _, _, values = block.getValuesCSR()
    magnitudes = np.abs(values)
    largest = comm.allreduce(float(magnitudes.max()) if magnitudes.size else 0.0, op=MPI.MAX)
    magnitudes = magnitudes[magnitudes > cutoff * largest] if largest > 0 else magnitudes[:0]
    count = comm.allreduce(magnitudes.size, op=MPI.SUM)
    log_sum = comm.allreduce(float(np.log10(magnitudes).sum()), op=MPI.SUM)
    return {"nnz": count, "max": largest, "geomean": 10 ** (log_sum / count) if count else 0.0}


def block_scales_hook(records: list):
    """A ``jacobian_callback`` that appends the sizes of the state, the residual and the Jacobian
    blocks, by group, to ``records``, one dict per Newton iteration."""

    def hook(snes: PETSc.SNES, nest_view) -> None:
        x = snes.getSolution()
        residual = snes.getFunction()[0]
        record = {"iteration": snes.getIterationNumber(), "state": {}, "residual": {}, "jacobian": {}}
        for name, index_set in zip(nest_view.names, nest_view.index_sets, strict=True):
            record["state"][name] = _subvector_norms(x, index_set)
            record["residual"][name] = _subvector_norms(residual, index_set)
        for row in nest_view.names:
            for col in nest_view.names:
                block = nest_view.block(row, col)
                if block is not None:
                    record["jacobian"][f"{row},{col}"] = _block_stats(block)
        records.append(record)

    return hook


def cell_sizes(mesh_path: str) -> dict:
    """The mean and the smallest diameter of the fluid and of the solid cells. Collective."""
    comm = MPI.COMM_WORLD
    with dolfinx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
    sizes = {}
    for name, marker in (("fluid", PHYSICAL_MARKERS["ALE_fluid"]), ("solid", PHYSICAL_MARKERS["solid"])):
        cells = cell_tags.find(marker)
        cells = cells[cells < mesh.topology.index_map(mesh.topology.dim).size_local]
        h = mesh.h(mesh.topology.dim, cells)
        count = comm.allreduce(h.size, op=MPI.SUM)
        sizes[name] = {
            "mean": comm.allreduce(float(h.sum()), op=MPI.SUM) / count,
            "min": comm.allreduce(float(h.min()) if h.size else np.inf, op=MPI.MIN),
            "cells": count,
        }
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/meshes/fsi2/mesh_sec.xdmf")
    parser.add_argument("--initial-state-dir", required=True)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--t0", type=float, default=16.0)
    parser.add_argument("--out", required=True, help="the JSON file of the records")
    args = parser.parse_args()

    stem = args.out.removesuffix(".json")
    records_by_step = []
    records = []
    step_times = []

    solve(
        mesh_path=args.mesh,
        T=args.t0 + args.steps * args.dt,
        dt_val=args.dt,
        output_path=f"{stem}.bp",
        output_path_p=f"{stem}_p.bp",
        qoi_path=f"{stem}_qoi.txt",
        newton_log_path=f"{stem}_newton.txt",
        initial_state_dir=args.initial_state_dir,
        jacobian_callback=block_scales_hook(records),
    )
    # Split the records into time steps at each Newton iteration 0.
    for record in records:
        if record["iteration"] == 0:
            records_by_step.append([])
            step_times.append(args.t0 + len(records_by_step) * args.dt)
        records_by_step[-1].append(record)

    sizes = cell_sizes(args.mesh)
    if MPI.COMM_WORLD.rank == 0:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "groups": list(NEST_GROUPS),
                    "cell_sizes": sizes,
                    "dt": args.dt,
                    "steps": [
                        {"t": t, "iterations": step} for t, step in zip(step_times, records_by_step, strict=True)
                    ],
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
