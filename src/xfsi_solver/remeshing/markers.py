# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

# Mirrors the ``PHYSICAL_MARKERS`` dict duplicated across every FSI2 solver
# and mesh-generation script (e.g. ``solvers/fsi2_harmonic_diffmesh.py``,
# ``scripts/create_mesh_FSI2.py``). Not yet unified repo-wide (see
# notes/remeshing/implementation-plan.md); the remeshing prototype code in
# this subpackage imports from here rather than adding another copy.
PHYSICAL_MARKERS = {
    "solid": 1,
    "ALE_fluid": 2,
    "solid_fluid_interface": 11,
    "obstacle": 21,
    "inflow": 22,
    "outflow": 23,
    "channel_side": 24,
    "solid_obstacle_interface": 25,
}
