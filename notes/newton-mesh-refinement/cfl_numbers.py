"""CFL numbers of the settled states of the refinement study, on the three meshes at dt = 0.0025.

    python notes/newton-mesh-refinement/cfl_numbers.py

Reads the state (u, v, p) of the first eigenvector sample of modes_ref<level>_dt0.0025 (t = 16.5625)
and prints, per mesh, the flow CFL number |v| dt / h on the fluid cells (|v| at the cell midpoint, h
the cell diameter, the mesh velocity left out) and the elastic CFL numbers c dt / h of the solid
(c_p and c_s the pressure and shear wave speeds of the solid of FSI2, with nu_s = 0.4).
"""

import dolfinx
import numpy as np
from analyse_modes import Run

DT = 0.0025
RHO_S, MU_S, NU_S = 1.0e4, 5.0e5, 0.4
LAMBDA_S = 2.0 * MU_S * NU_S / (1.0 - 2.0 * NU_S)
C_P, C_S = np.sqrt((LAMBDA_S + 2.0 * MU_S) / RHO_S), np.sqrt(MU_S / RHO_S)

if __name__ == "__main__":
    print(f"dt = {DT}, c_p = {C_P:.2f}, c_s = {C_S:.2f}")
    for level in (0, 1, 2):
        run = Run(f"modes_ref{level}_dt{DT:g}")
        state = run.read(run.samples()[0], 4)
        mesh, tdim = run.mesh, run.mesh.topology.dim
        cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        h = dolfinx.cpp.mesh.h(mesh._cpp_object, tdim, cells)
        midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
        speed = np.linalg.norm(state["v"].eval(midpoints, cells), axis=1)
        fluid, solid = run.fluid_cells, run.solid_cells
        cfl = speed[fluid] * DT / h[fluid]
        print(
            f"ref{level}: fluid h min {h[fluid].min():.2e}, max |v| {speed[fluid].max():.2f}, "
            f"|v| dt / h max {cfl.max():.2f}, 99th percentile {np.percentile(cfl, 99):.2f}, "
            f"median {np.median(cfl):.3f}; solid h min {h[solid].min():.2e}, "
            f"c_p dt / h {C_P * DT / h[solid].min():.1f}, c_s dt / h {C_S * DT / h[solid].min():.1f}"
        )
