# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Extract the FSI2 fluid-only domain as a standalone mesh.

See notes/remeshing/implementation-plan.md §2: the remeshing prototype
works on the fluid domain in isolation, so this module only ever needs the
``dolfinx.mesh.create_submesh`` extraction already used (for pressure) in
``solvers/fsi2_harmonic_diffmesh.py`` -- nothing else about that solver.
"""

from dataclasses import dataclass

import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm

from xfsi_solver.remeshing.markers import PHYSICAL_MARKERS


@dataclass
class FluidDomain:
    """A standalone fluid-only mesh with its boundary facets tagged.

    ``facet_tags`` values use the same numbering as ``PHYSICAL_MARKERS``
    (``solid_fluid_interface``, ``obstacle``, ``inflow``, ``outflow``,
    ``channel_side``); ``solid`` and ``solid_obstacle_interface`` never
    appear since there is no solid in this mesh.
    """

    mesh: dfx.mesh.Mesh
    facet_tags: dfx.mesh.MeshTags


def load_fsi2_fluid_domain(mesh_path: str) -> FluidDomain:
    """Load an FSI2 mesh XDMF file and extract just the fluid subdomain.

    Args:
        mesh_path: path to an XDMF file written by
            ``src/xfsi_solver/scripts/create_mesh_FSI2.py`` (e.g.
            ``data/meshes/fsi2/mesh.xdmf`` or ``mesh_sec.xdmf``), holding a
            mesh, "Cell tags", and "Facet tags" using ``PHYSICAL_MARKERS``.
    """
    with dfx.io.XDMFFile(comm, mesh_path, "r") as infile:
        mesh = infile.read_mesh()
        cell_tags = infile.read_meshtags(mesh, name="Cell tags")
        mesh.topology.create_connectivity(1, 2)
        facet_tags = infile.read_meshtags(mesh, name="Facet tags")

    return extract_fluid_domain(mesh, cell_tags, facet_tags)


def extract_fluid_domain(
    mesh: dfx.mesh.Mesh, cell_tags: dfx.mesh.MeshTags, facet_tags: dfx.mesh.MeshTags
) -> FluidDomain:
    """Extract the fluid-tagged cells of ``mesh`` as a standalone submesh."""
    fluid_mesh, cell_map, vertex_map, _ = dfx.mesh.create_submesh(
        mesh, mesh.topology.dim, cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
    )
    fluid_mesh.topology.create_connectivity(1, 2)
    fluid_facet_tags = dfx.mesh.transfer_meshtags_to_submesh(
        facet_tags, fluid_mesh, vertex_map, cell_map
    )
    return FluidDomain(mesh=fluid_mesh, facet_tags=fluid_facet_tags)
