# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Quantities of interest of the FSI2 benchmark: drag, lift and the displacement of a point on the structure."""

from pathlib import Path

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI

from xfsi_solver.fsi.materials import Fluid
from xfsi_solver.tools.checkpoint import truncate_qoi_file

QOI_HEADER = "t\tdrag\tlift\tA_x\tA_y"


def find_point_dof(space: dolfinx.fem.FunctionSpace, point: np.ndarray, atol: float = 1e-6) -> int | None:
    """Local index of the owned node of ``space`` at ``point``, or ``None`` on the ranks that do not own it.

    Collective: raises if the node is not found on exactly one rank.
    """
    num_owned = space.dofmap.index_map.size_local
    coordinates = space.tabulate_dof_coordinates()[:num_owned, :]
    candidates = np.flatnonzero(np.all(np.isclose(coordinates, point, atol=atol), axis=1))
    num_found = space.mesh.comm.allreduce(len(candidates), op=MPI.SUM)
    assert num_found == 1, "None or multiple dofs found for measurement point"
    return candidates[0] if len(candidates) > 0 else None


def point_value(u: dolfinx.fem.Function, node: int | None) -> np.ndarray | None:
    """Value of ``u`` at the node found by :func:`find_point_dof`, on rank 0 (``None`` on the other ranks).

    Collective.
    """
    bs = u.function_space.dofmap.index_map_bs
    local = np.zeros(bs, dtype=np.float64)
    if node is not None:
        local[:] = u.x.array[bs * node : bs * (node + 1)]
    return u.function_space.mesh.comm.reduce(local, op=MPI.SUM, root=0)


def drag_lift_forms(
    u: dolfinx.fem.Function,
    v: dolfinx.fem.Function,
    p: dolfinx.fem.Function,
    nu_f: dolfinx.fem.Constant,
    rho_f: dolfinx.fem.Constant,
    measures: list[ufl.Measure],
    entity_maps: list[dolfinx.mesh.EntityMap],
) -> tuple[list[dolfinx.fem.Form], list[dolfinx.fem.Form]]:
    """Compiled forms of the drag and the lift on the fluid, as ``(drag_forms, lift_forms)``.

    The force is the fluid traction pulled back to the reference configuration,
    integrated over each of ``measures``, with one form per measure (the
    obstacle and the fluid side of the interface). The drag is the component
    along (-1, 0) and the lift the component along (0, 1).
    """
    mesh = u.function_space.mesh
    normal = ufl.FacetNormal(mesh)
    F = ufl.Identity(mesh.geometry.dim) + ufl.grad(u)
    transformed_normal = ufl.dot(ufl.inv(F.T), normal)
    traction = ufl.dot(Fluid.NS(u, v, p, nu_f, rho_f), transformed_normal)

    e_x = dolfinx.fem.Constant(mesh, (-1.0, 0.0))
    e_y = dolfinx.fem.Constant(mesh, (0.0, 1.0))

    def forms(direction):
        return [
            dolfinx.fem.form(ufl.dot(traction, direction) * ufl.det(F) * ds, entity_maps=entity_maps) for ds in measures
        ]

    return forms(e_x), forms(e_y)


def assemble_force(forms: list[dolfinx.fem.Form], comm: MPI.Intracomm) -> float | None:
    """Sum of the scalars assembled from ``forms``, on rank 0 (``None`` on the other ranks). Collective."""
    return comm.reduce(sum(dolfinx.fem.assemble_scalar(form) for form in forms))


def init_qoi_file(qoi_path: str | Path, comm: MPI.Intracomm, restart: bool, t: float, dt_val: float) -> None:
    """Create the QoI file with its header on rank 0.

    When restarting from a checkpoint at time ``t`` and the file exists, its
    rows after ``t`` are dropped instead, so new rows can be appended.
    """
    if comm.rank == 0:
        Path(qoi_path).parent.mkdir(parents=True, exist_ok=True)
        if restart and Path(qoi_path).exists():
            truncate_qoi_file(qoi_path, t, dt_val)
        else:
            with open(qoi_path, "wb") as f:
                np.savetxt(f, [], fmt="%.6e", delimiter="\t", header=QOI_HEADER)


def append_qoi_row(
    qoi_path: str | Path,
    comm: MPI.Intracomm,
    t: float,
    drag: float | None,
    lift: float | None,
    u_spot: np.ndarray | None,
) -> None:
    """Append the row ``t, drag, lift, *u_spot`` to the QoI file on rank 0, where the reduced values are."""
    if comm.rank == 0:
        with open(qoi_path, "ab") as f:
            np.savetxt(f, [[t, drag, lift, *u_spot]], fmt="%.6e", delimiter="\t")
