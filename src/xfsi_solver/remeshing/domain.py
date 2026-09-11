# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The FSI2 domain: the whole mesh, both subdomains, all marked curves.

This replaces an earlier fluid-only version of this module, which pulled
the ``ALE_fluid`` cells out into a standalone submesh via
``dolfinx.mesh.create_submesh`` (notes/remeshing/implementation-plan.md §2
scoped the prototype to the fluid domain in isolation). Remeshing now
regenerates the *full* mesh -- solid flag and fluid channel together, with
the fluid-solid interface as an internal boundary shared by both -- so the
submesh extraction is gone and the cell tags, which the fluid-only version
had no use for, are now carried alongside the facet tags: they are what
tells ``discrete_mesh.regenerate_mesh`` which cells make up which
subdomain, and therefore which of the marked curves bound which surface.
"""

from dataclasses import dataclass

import dolfinx as dfx
from mpi4py.MPI import COMM_WORLD as comm


@dataclass
class FsiDomain:
    """A full FSI2 mesh with both its subdomains and all its curves tagged.

    ``cell_tags`` and ``facet_tags`` both use the ``markers.PHYSICAL_MARKERS``
    numbering: cells are ``solid`` or ``ALE_fluid``; facets are
    ``solid_fluid_interface`` (internal, between the two subdomains),
    ``solid_obstacle_interface`` (the flag's clamped root, on the cylinder),
    ``obstacle``, ``inflow``, ``outflow`` or ``channel_side``.
    """

    mesh: dfx.mesh.Mesh
    cell_tags: dfx.mesh.MeshTags
    facet_tags: dfx.mesh.MeshTags


def load_fsi2_domain(mesh_path: str) -> FsiDomain:
    """Load an FSI2 mesh XDMF file whole, with both tag sets.

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

    return FsiDomain(mesh=mesh, cell_tags=cell_tags, facet_tags=facet_tags)
