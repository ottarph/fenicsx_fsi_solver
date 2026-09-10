"""Write a genuinely remeshed (not just deformed) fluid domain to VTX, for
manual inspection in ParaView -- notes/remeshing/implementation-plan.md
Phase 4's loop produces a *new* dolfinx.mesh.Mesh object (different node
and cell counts) at each remesh event, not just moved points on a fixed
topology, so a single VTXWriter can't span a remesh event: "All Functions
for output must share the same mesh" (its own docstring). Each segment
(between remesh events) therefore gets its own VTX file, and within a
segment the mesh's own geometry is updated at every step (not just an
overlaid displacement field) via ``mesh_policy=VTXMeshPolicy.update``, so
the flag visibly bends in ParaView without needing a Warp-by-Vector filter.

Each new segment's *first* write reuses the previous segment's *last*
timestamp, rather than the next one, so that at that one instant both
segments have data and ParaView shows the old mesh's final (most
deformed) state and the new mesh's fresh, zero-displacement state
together -- otherwise the transition is an invisible cut between two
files that never overlap in time, and there is no way to see that the
new mesh really is a re-triangulation of the same domain shape the old
one had just reached, not a discontinuity.
"""

import dolfinx as dfx
import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.remeshing.deformation import incremental_interface_deformation
from xfsi_solver.remeshing.discrete_mesh import SizingField, regenerate_fluid_mesh
from xfsi_solver.remeshing.dof_geometry import geometry_to_dof_permutation, to_dof_order
from xfsi_solver.remeshing.fluid_domain import load_fsi2_fluid_domain
from xfsi_solver.remeshing.loop import _interface_points, _min_quality

# Coarse-ish on purpose to keep the test fast; not the production sizing.
TEST_SIZING = SizingField(size_near=0.02, size_far=0.06, distance=0.1)

# Calibrated against data/meshes/fsi2/mesh.xdmf (see
# notes/remeshing/implementation-plan.md): quality first drops below 0.35
# around the 11th step of 0.005, so 20 steps gives at least one remesh
# event -- i.e. at least two VTX segments -- without taking long to run.
STEP_AMPLITUDES = [0.005] * 20
QUALITY_THRESHOLD = 0.35


def test_vtx_output_across_remesh_events(output_dirs):
    """Step the fluid domain through a prescribed bending deformation,
    remeshing as needed, writing the *actually deforming* mesh plus the
    accumulated-displacement field to VTX at every step -- one file per
    remesh segment -- so the result can be opened in ParaView.
    """
    domain = load_fsi2_fluid_domain("data/meshes/fsi2/mesh.xdmf")
    accumulated_displacement = np.zeros_like(domain.mesh.geometry.x[:, :2])
    base_geometry = domain.mesh.geometry.x[:, :2].copy()

    segment_paths = []

    def open_segment_writer(mesh, segment_index):
        V = dfx.fem.functionspace(mesh, ("CG", 1, (2,)))
        displacement_field = dfx.fem.Function(V, name="u")
        path = output_dirs["pv"] / f"remeshing_demo_seg{segment_index}.bp"
        segment_paths.append(path)
        writer = dfx.io.VTXWriter(comm, str(path), [displacement_field], mesh_policy=dfx.io.VTXMeshPolicy.update)
        return writer, displacement_field, geometry_to_dof_permutation(mesh, V)

    segment = 0
    t = 0.0
    writer, displacement_field, perm = open_segment_writer(domain.mesh, segment)

    for step_amplitude in STEP_AMPLITUDES:
        interface_points = _interface_points(domain)
        increment = incremental_interface_deformation(step_amplitude, interface_points)(
            domain.mesh.geometry.x[:, :2].T
        )
        accumulated_displacement = accumulated_displacement + increment.T

        # Move the mesh's own geometry to the current deformed position for
        # this write, then restore it -- the rest of the pipeline (quality
        # check, regenerate_fluid_mesh) still expects domain.mesh.geometry.x
        # to be the fixed start-of-segment reference, exactly as in loop.py.
        domain.mesh.geometry.x[:, :2] = base_geometry + accumulated_displacement
        displacement_field.x.array[:] = to_dof_order(accumulated_displacement, perm).flatten()
        writer.write(t)
        domain.mesh.geometry.x[:, :2] = base_geometry
        t += 1.0

        min_quality = _min_quality(domain.mesh, accumulated_displacement)
        if min_quality < QUALITY_THRESHOLD:
            geometry = domain.mesh.geometry.x.copy()
            geometry[:, :2] += accumulated_displacement
            new_domain = regenerate_fluid_mesh(domain, geometry, sizing=TEST_SIZING)

            writer.close()
            domain = new_domain
            accumulated_displacement = np.zeros_like(domain.mesh.geometry.x[:, :2])
            base_geometry = domain.mesh.geometry.x[:, :2].copy()
            segment += 1
            writer, displacement_field, perm = open_segment_writer(domain.mesh, segment)

            # Overlap with the old segment's last timestamp (t - 1.0, since
            # t was already advanced above) so both meshes are visible at
            # once -- see the module docstring.
            displacement_field.x.array[:] = 0.0
            writer.write(t - 1.0)

    writer.close()

    assert segment >= 1, "expected at least one remesh event (at least two VTX segments) in this run"
    for path in segment_paths:
        assert path.exists()
        assert any(path.iterdir()), f"{path} is an empty ADIOS2 BP directory"
