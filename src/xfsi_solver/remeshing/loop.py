# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Phase 4: the combined, still solver-free, standalone remeshing loop.

Steps a prescribed interface deformation (deformation.py) through
increasing amplitude, standing in for "time" (notes/remeshing/
implementation-plan.md §7 Phase 4). At each step: apply one incremental
bending step to the domain's *current* geometry; check mesh quality
(quality.py); if it has dropped below ``quality_threshold``, regenerate the
mesh from the current deformed geometry (discrete_mesh.py) and transfer a
field onto it (transfer.py) as a live demonstration of carrying state
across a remesh event; then continue.

The domain here is the *whole* FSI2 mesh -- solid flag and fluid channel,
regenerated together -- so the prescribed bending moves the flag's own
cells along with the fluid's, and the quality trigger watches both
subdomains rather than only the fluid.

No dolfinx.fem.petsc solver is used anywhere in this module -- the "field"
carried across remesh events is a passive demonstration field, not any
solved quantity (see ``run_prescribed_deformation_loop``'s
``carried_field`` argument).

Why deformation is applied incrementally, relative to the current mesh
--------------------------------------------------------------------------
An earlier version of this loop tried to track a "reference coordinates"
field (each current mesh node's true t=0 material position) across remesh
events via ``transfer.transfer_field``, so that
``deformation.prescribed_interface_deformation`` -- which depends on
distance from the *original* flag position -- could still be evaluated
correctly after a remesh. That doesn't work: transferring a field that is
literally the identity (a mesh's own geometry) via nonmatching
interpolation always just returns the query point itself again, regardless
of which source cell answers the query, so it carries zero information
about where that point "really" started out. Recovering true reference
position after a remesh would need actually solving an extension problem,
which is out of scope for this solver-free phase (see
``deformation.incremental_interface_deformation``'s docstring for the
full explanation).

Instead, ``deformation.incremental_interface_deformation`` decays with
distance from the mesh's *current* interface, which needs no tracking at
all -- it only needs the current mesh's own facet tags -- so this loop
tracks displacement purely relative to the current mesh (reset to zero at
every remesh, exactly like resetting the ALE displacement in the plan's
§5) rather than relative to any fixed reference.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import dolfinx as dfx
import numpy as np

from xfsi_solver.remeshing.deformation import incremental_interface_deformation
from xfsi_solver.remeshing.discrete_mesh import SizingField, regenerate_mesh
from xfsi_solver.remeshing.dof_geometry import geometry_to_dof_permutation, to_dof_order
from xfsi_solver.remeshing.domain import FsiDomain
from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS
from xfsi_solver.remeshing.quality import MeshQuality
from xfsi_solver.remeshing.transfer import transfer_field


@dataclass
class StepRecord:
    step_amplitude: float
    min_quality: float
    remeshed: bool


@dataclass
class LoopResult:
    domain: FsiDomain
    carried_field: dfx.fem.Function
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def n_remesh_events(self) -> int:
        return sum(step.remeshed for step in self.steps)


def _interface_points(domain: FsiDomain) -> np.ndarray:
    V = dfx.fem.functionspace(domain.mesh, ("CG", 1, (2,)))
    facets = domain.facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
    dofs = dfx.fem.locate_dofs_topological(V, 1, facets)
    return V.tabulate_dof_coordinates()[dofs, :2]


def _min_quality(mesh: dfx.mesh.Mesh, geometry_order_displacement: np.ndarray) -> float:
    """scaled_jacobian of ``mesh`` warped by a displacement given in
    geometry-node order (converted to the CG1 dof order pvmeshquality
    needs -- see dof_geometry.py)."""
    V = dfx.fem.functionspace(mesh, ("CG", 1, (2,)))
    perm = geometry_to_dof_permutation(mesh, V)
    u = dfx.fem.Function(V)
    u.x.array[:] = to_dof_order(geometry_order_displacement, perm).flatten()
    return float(MeshQuality(quality_measure="scaled_jacobian", fspace=V)(u).min())


def _signed_cell_areas(mesh: dfx.mesh.Mesh, geometry: np.ndarray) -> np.ndarray:
    """Signed area of each cell's corner triangle, with ``geometry`` giving
    the node positions to use (in geometry-node order)."""
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    corners = geometry[mesh.geometry.dofmaps[0][:num_cells, :3]]
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _has_inverted_cells(mesh: dfx.mesh.Mesh, geometry: np.ndarray) -> bool:
    """Cheap ground-truth check (signed corner-triangle area, independent of
    the scaled_jacobian trigger metric) for whether ``geometry`` -- the
    mesh's nodes moved to candidate physical positions -- has any
    actually-inverted cell.

    "Inverted" is a *change of sign* relative to the mesh's own undeformed
    orientation, not a negative area. The FSI2 mesh does not have one
    consistent orientation to compare against: the whole solid subdomain is
    stored with the opposite sign to the fluid (an inherited property of how
    ``scripts/create_mesh_FSI2.py`` builds the flag's curve loop, preserved
    through a remesh), so a plain ``signed_area <= 0`` test would declare
    every flag cell inverted on the undeformed mesh. ``quality.MeshQuality``
    takes the same per-cell-baseline approach for the same reason.

    Guards ``regenerate_mesh`` (discrete_mesh.py) against being asked
    to remesh an already-invalid, self-intersecting boundary, which was
    found empirically to be able to hang rather than fail cleanly (see that
    module's docstring) -- so this must be checked *before* attempting a
    remesh, not left to be discovered by a hang.
    """
    reference = _signed_cell_areas(mesh, mesh.geometry.x)
    return bool(np.any(np.sign(reference) * _signed_cell_areas(mesh, geometry) <= 0))


def run_prescribed_deformation_loop(
    initial_domain: FsiDomain,
    step_amplitudes: Sequence[float],
    quality_threshold: float = 0.35,
    sizing: SizingField | None = None,
    initial_field: Callable[[np.ndarray], np.ndarray] | None = None,
) -> LoopResult:
    """Step through ``step_amplitudes`` (each an *increment*), remeshing
    whenever quality drops below ``quality_threshold``.

    Args:
        initial_domain: a full FSI2 mesh (e.g. from
            ``domain.load_fsi2_domain``), assumed *undeformed*.
        step_amplitudes: a sequence of incremental bending amplitudes
            (each metres of *additional* tip deflection since the last
            step), standing in for successive time steps.
        quality_threshold: trigger a remesh once ``scaled_jacobian``'s
            minimum drops below this. Needs a *generous* margin above 0,
            not just "above 0": quality vs. amplitude was found empirically
            to fall off a cliff over a handful of steps once it starts
            degrading -- a too-low threshold combined with a not-small
            -enough step can let the *next* step land past actual
            inversion before a remesh ever gets a chance to fire. Past
            actual inversion this mechanism can no longer recover the mesh
            (discrete_mesh.py's docstring) and this function raises
            ``RuntimeError`` rather than letting ``regenerate_mesh``
            hang on it (see ``_has_inverted_cells``).
        sizing: graded sizing field to use for every regenerated mesh.
        initial_field: optional callable (dolfinx ``interpolate`` style)
            for a CG1 scalar "carried field" -- a stand-in for whatever a
            real solve would be carrying (velocity/pressure/...), used to
            demonstrate ``transfer.transfer_field`` moving *something*
            across each remesh event, not just driving the mesh's own
            deformation. Defaults to a smooth bump function.

    Returns:
        The final :class:`FsiDomain`, the final carried-field
        ``Function``, and a per-step quality/remesh log.

    Raises:
        RuntimeError: if the geometry at some step already has inverted
            cells by the time ``quality_threshold`` is checked -- lower the
            step amplitude and/or raise ``quality_threshold``.
    """
    if initial_field is None:
        initial_field = lambda x: np.exp(-(((x[0] - 0.4) / 0.3) ** 2) - (((x[1] - 0.2) / 0.1) ** 2))  # noqa: E731

    domain = initial_domain
    accumulated_displacement = np.zeros_like(domain.mesh.geometry.x[:, :2])

    V0 = dfx.fem.functionspace(domain.mesh, ("CG", 1))
    carried_field = dfx.fem.Function(V0)
    carried_field.interpolate(initial_field)

    result = LoopResult(domain=domain, carried_field=carried_field)

    for step_amplitude in step_amplitudes:
        interface_points = _interface_points(domain)
        increment = incremental_interface_deformation(step_amplitude, interface_points)(
            domain.mesh.geometry.x[:, :2].T
        )
        accumulated_displacement = accumulated_displacement + increment.T

        min_quality = _min_quality(domain.mesh, accumulated_displacement)
        remeshed = False

        if min_quality < quality_threshold:
            geometry = domain.mesh.geometry.x.copy()
            geometry[:, :2] += accumulated_displacement
            if _has_inverted_cells(domain.mesh, geometry):
                raise RuntimeError(
                    f"step_amplitude={step_amplitude}: the deformed geometry already has inverted cells "
                    f"(min_quality={min_quality:.4g} was only caught at this step). regenerate_mesh "
                    "cannot reliably remesh a self-intersecting boundary (see discrete_mesh.py). Use a "
                    "higher quality_threshold and/or a finer step_amplitude so a remesh triggers earlier."
                )
            new_domain = regenerate_mesh(domain, geometry, sizing=sizing)

            V_new = dfx.fem.functionspace(new_domain.mesh, ("CG", 1))
            carried_field = transfer_field(carried_field, V_new)

            domain = new_domain
            accumulated_displacement = np.zeros_like(domain.mesh.geometry.x[:, :2])
            remeshed = True

        result.steps.append(StepRecord(step_amplitude=step_amplitude, min_quality=min_quality, remeshed=remeshed))

    result.domain = domain
    result.carried_field = carried_field
    return result
