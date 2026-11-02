# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Transfer the state in an FSI2 checkpoint to another mesh of the same domain.

    python -m xfsi_solver.scripts.transfer_fsi2_checkpoint <source_mesh> <source_dir> <target_mesh> <target_dir> <dt>

Reads the latest checkpoint in ``source_dir``, written by a solver on ``source_mesh``, and writes
its state, interpolated onto ``target_mesh``, as a checkpoint in ``target_dir``, from which a solver
on ``target_mesh`` starts with ``initial_state_dir``. For starting a run on a refined mesh from the
state of a run on a coarser one, instead of spinning it up from rest.
"""

import argparse
from pathlib import Path

import dolfinx
import numpy as np
from mpi4py import MPI

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian import PHYSICAL_MARKERS
from xfsi_solver.tools.checkpoint import Checkpointer

# The fields of the FSI2 solvers: u and v on the whole mesh, p and z on the fluid submesh. z, of
# the mixed biharmonic mesh motion, is in the checkpoints of only some of the solvers.
ELEMENTS = {"u": ("P", 2, (2,)), "v": ("P", 2, (2,)), "p": ("P", 1), "z": ("P", 2, (2,))}
ON_FLUID = ("p", "z")


class _State:
    """The mesh of ``mesh_path``, its fluid submesh, and the FSI2 fields on them."""

    def __init__(self, mesh_path: str | Path, comm: MPI.Comm):
        with dolfinx.io.XDMFFile(comm, str(mesh_path), "r") as infile:
            self.mesh = infile.read_mesh()
            cell_tags = infile.read_meshtags(self.mesh, name="Cell tags")
        tdim = self.mesh.topology.dim
        self.fluid_mesh, self.fluid_cell_map, _, _ = dolfinx.mesh.create_submesh(
            self.mesh, tdim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
        )
        self.functions = {
            name: dolfinx.fem.Function(
                dolfinx.fem.functionspace(self.fluid_mesh if name in ON_FLUID else self.mesh, element), name=name
            )
            for name, element in ELEMENTS.items()
        }

    def checkpointer(self, directory: str | Path) -> Checkpointer:
        return Checkpointer(directory, self.mesh, [(self.fluid_mesh, self.fluid_cell_map)])


def _interpolate(target: dolfinx.fem.Function, source: dolfinx.fem.Function, padding: float) -> None:
    """``target`` <- ``source``, on another mesh of the same domain."""
    mesh = target.function_space.mesh
    index_map = mesh.topology.index_map(mesh.topology.dim)
    cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    data = dolfinx.fem.create_interpolation_data(target.function_space, source.function_space, cells, padding)
    target.interpolate_nonmatching(source, cells, data)
    target.x.scatter_forward()


def transfer_checkpoint(
    source_mesh_path: str | Path,
    source_dir: str | Path,
    target_mesh_path: str | Path,
    target_dir: str | Path,
    dt: float,
    padding: float = 1.0e-6,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> float:
    """Write the state of the latest checkpoint in ``source_dir`` (on ``source_mesh_path``),
    interpolated onto ``target_mesh_path``, as a checkpoint in ``target_dir``, and return its time.

    u and v are interpolated between the whole meshes, and p and z between their fluid submeshes, at
    the interpolation points of the target, which lie in a cell of the source up to ``padding``.
    The curved boundary of the cylinder is resolved differently on the two meshes, so a point near it
    may lie just outside the source. z is transferred only if the source holds it. The new checkpoint
    is step 0 of a run that starts at its time, with time step ``dt``; a run that starts from it
    with ``initial_state_dir`` may use any time step.
    """
    source, target = _State(source_mesh_path, comm), _State(target_mesh_path, comm)
    functions = list(source.functions.values())
    z = source.functions["z"]
    z.x.array[:] = np.nan  # left so if the checkpoint has no z
    t, _ = source.checkpointer(source_dir).read(functions, None, optional=["z"])
    has_z = comm.allreduce(not np.isnan(z.x.array).any(), op=MPI.LAND)
    names = [name for name in ELEMENTS if name != "z" or has_z]
    for name in names:
        _interpolate(target.functions[name], source.functions[name], padding)
    target.checkpointer(target_dir).write([target.functions[name] for name in names], t, 0, dt, t0=t)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source_mesh")
    parser.add_argument("source_dir")
    parser.add_argument("target_mesh")
    parser.add_argument("target_dir")
    parser.add_argument("dt", type=float)
    args = parser.parse_args()
    t = transfer_checkpoint(args.source_mesh, args.source_dir, args.target_mesh, args.target_dir, args.dt)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Wrote the state at t = {t} on {args.target_mesh} to {args.target_dir}")


if __name__ == "__main__":
    main()
