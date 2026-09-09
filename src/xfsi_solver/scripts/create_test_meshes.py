# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import gmsh

from xfsi_solver.scripts.create_mesh_DFG2d_alt import create_mesh as create_dfg2d_alt_mesh
from xfsi_solver.scripts.create_mesh_FSI2 import create_mesh as create_fsi2_mesh


def main():
    """Generate every coarse mesh the test suite (``tests/``) depends on.

    ``data/`` is gitignored, so these meshes don't ship with the repo and
    must be generated locally before running ``pytest`` for the first time.
    """

    # Each create_mesh() call below builds a gmsh model from scratch, so
    # gmsh.clear() is needed between calls in the same process to avoid
    # leftover geometry/physical-group state from the previous mesh (a
    # single call per process, as when running the individual
    # create_mesh_*.py scripts directly, never hits this).

    # data/meshes/fsi2/mesh_coarse.xdmf
    create_fsi2_mesh(fine=False, coarse=True, quads=False, semi_structured_quad=False, second_order=False)
    gmsh.clear()

    # data/meshes/fsi2/mesh_sec_coarse.xdmf
    create_fsi2_mesh(fine=False, coarse=True, quads=False, semi_structured_quad=False, second_order=True)
    gmsh.clear()

    # data/meshes/fsi2/mesh_quad_coarse.xdmf
    create_fsi2_mesh(fine=False, coarse=True, quads=True, semi_structured_quad=False, second_order=False)
    gmsh.clear()

    # data/meshes/dfg2d_alt/mesh_tri_coarse.xdmf
    create_dfg2d_alt_mesh(quads=False, coarse=True)
    gmsh.clear()

    # data/meshes/dfg2d_alt/mesh_quad_coarse.xdmf
    create_dfg2d_alt_mesh(quads=True, coarse=True)


if __name__ == "__main__":
    main()
