"""Measure the dominant eigenvectors of J_0^-1 N written by the runs in output/newton_modes/.

    python notes/newton-mesh-refinement/analyse_modes.py [run ...]

Each run (by default every modes_ref<level>_dt<dt> in output/newton_modes/) wrote, with
fsi2_biharmonic_fulljacobian_initialguess and spectral_vectors_dir, the NEV dominant eigenpairs of
J_0^-1 N every 0.01 s, from the settled states of the refinement study. For every sample and mode,
this writes output/newton_modes/analysis/<run>_t<time>_mode<i>.npz, which make_figures.py plots.

N acts only on u in the fluid, and J_0 extends u from the interface into the fluid by the mesh
motion, so a mode is set by its interface displacement du_I: J_0^-1 N maps du_I to the response
(du, dv, dp, dz), whose interface displacement is lambda du_I. Each mode is therefore scaled to
||du_I||_L2(interface) = 1, with the complex phase that makes the real part of du_I largest, and
measured by

- its eigenvalue, and the L2 norms of du, dv and dp on the fluid and of du and dv on the solid;
- du_I and dv_I along the interface, in normal and tangential components, by arc length s from the
  lower beam-cylinder junction, along the bottom of the beam, up the tip and back along the top;
- the volume fluxes: the area change of the beam by du_I, and the fluxes of dv out of the beam and
  through the outflow;
- the cumulative fractions of the fluid integrals of |dv|^2 and dp^2, and of the solid integral of
  |dv|^2, by the distance of each cell from the beam and from the nearest of its four corners;
- for the first sample, the real parts of dv (interpolated to P1) and dp on the fluid, for field plots.
"""

import sys
from pathlib import Path

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI

from xfsi_solver.solvers.fsi2_biharmonic_fulljacobian import PHYSICAL_MARKERS
from xfsi_solver.tools.checkpoint import Checkpointer

ROOT = Path("output/newton_modes")
OUT = ROOT / "analysis"
ELEMENTS = {"u": ("P", 2, (2,)), "v": ("P", 2, (2,)), "p": ("P", 1), "z": ("P", 2, (2,))}
ON_FLUID = ("p", "z")
# The beam in the reference configuration: from the cylinder (centre (0.2, 0.2), radius 0.05) to
# x = 0.6, between y = 0.19 and 0.21.
X0, X1, Y0, Y1 = 0.2 + np.sqrt(0.05**2 - 0.01**2), 0.6, 0.19, 0.21
CORNERS = np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]])
BEAM_LENGTH = X1 - X0


def arc_length(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arc length s along the interface, and the outward unit normal of the solid, at points x."""
    s, n = np.empty(len(x)), np.empty((len(x), 2))
    tol = 1e-8
    bottom = np.abs(x[:, 1] - Y0) < tol
    top = np.abs(x[:, 1] - Y1) < tol
    tip = (np.abs(x[:, 0] - X1) < tol) & ~bottom & ~top
    if not np.all(bottom | top | tip):
        raise ValueError("Interface points off the beam outline")
    s[bottom], n[bottom] = x[bottom, 0] - X0, (0.0, -1.0)
    s[tip], n[tip] = BEAM_LENGTH + x[tip, 1] - Y0, (1.0, 0.0)
    s[top], n[top] = BEAM_LENGTH + (Y1 - Y0) + X1 - x[top, 0], (0.0, 1.0)
    return s, n


def distance_to_beam(x: np.ndarray) -> np.ndarray:
    dx = np.maximum(np.maximum(X0 - x[:, 0], x[:, 0] - X1), 0.0)
    dy = np.maximum(np.maximum(Y0 - x[:, 1], x[:, 1] - Y1), 0.0)
    return np.hypot(dx, dy)


def distance_to_corners(x: np.ndarray) -> np.ndarray:
    return np.min(np.linalg.norm(x[:, None, :2] - CORNERS[None], axis=2), axis=1)


class Run:
    """The mesh of a run, its fluid submesh, and the forms to measure its modes with."""

    def __init__(self, name: str):
        self.name = name
        level = int(name.split("_ref")[1].split("_")[0])
        self.dt = float(name.split("_dt")[1])
        self.theta = 0.5 + self.dt
        self.dir = ROOT / name
        mesh_path = f"data/meshes/fsi2/refinement/mesh_sec_ref{level}.xdmf"
        with dolfinx.io.XDMFFile(MPI.COMM_SELF, mesh_path, "r") as infile:
            self.mesh = infile.read_mesh()
            cell_tags = infile.read_meshtags(self.mesh, name="Cell tags")
            self.mesh.topology.create_connectivity(1, 2)
            facet_tags = infile.read_meshtags(self.mesh, name="Facet tags")
        tdim = self.mesh.topology.dim
        self.fluid_cells = cell_tags.find(PHYSICAL_MARKERS["ALE_fluid"])
        self.solid_cells = cell_tags.find(PHYSICAL_MARKERS["solid"])
        self.fluid_mesh, self.fluid_cell_map, _, _ = dolfinx.mesh.create_submesh(self.mesh, tdim, self.fluid_cells)
        self.spaces = {
            name: dolfinx.fem.functionspace(self.fluid_mesh if name in ON_FLUID else self.mesh, element)
            for name, element in ELEMENTS.items()
        }
        interface = facet_tags.find(PHYSICAL_MARKERS["solid_fluid_interface"])
        dofs = dolfinx.fem.locate_dofs_topological(self.spaces["u"], tdim - 1, interface)
        x = self.spaces["u"].tabulate_dof_coordinates()[dofs]
        s, n = arc_length(x)
        order = np.argsort(s)
        self.interface_dofs, self.s, self.normal = dofs[order], s[order], n[order]

        self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=cell_tags)
        self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=facet_tags)
        self.f = {name: dolfinx.fem.Function(space) for name, space in self.spaces.items()}
        self.entity_maps = [self.fluid_cell_map]
        u, v, p = self.f["u"], self.f["v"], self.f["p"]
        fluid, solid = self.dx(PHYSICAL_MARKERS["ALE_fluid"]), self.dx(PHYSICAL_MARKERS["solid"])
        dx_p = ufl.Measure("dx", domain=self.fluid_mesh)
        self.norm_forms = {
            "u_I": dolfinx.fem.form(ufl.inner(u("+"), u("+")) * self.dS(PHYSICAL_MARKERS["solid_fluid_interface"])),
            "v_I": dolfinx.fem.form(ufl.inner(v("+"), v("+")) * self.dS(PHYSICAL_MARKERS["solid_fluid_interface"])),
            "u_fluid": dolfinx.fem.form(ufl.inner(u, u) * fluid),
            "v_fluid": dolfinx.fem.form(ufl.inner(v, v) * fluid),
            "u_solid": dolfinx.fem.form(ufl.inner(u, u) * solid),
            "v_solid": dolfinx.fem.form(ufl.inner(v, v) * solid),
            "p_fluid": dolfinx.fem.form(p * p * dx_p),
        }
        # Volume fluxes: the outward normal of the solid on the interface, from the reference beam
        x = ufl.SpatialCoordinate(self.mesh)
        n_s = ufl.conditional(
            ufl.lt(abs(x[1] - Y0), 1e-8),
            ufl.as_vector((0.0, -1.0)),
            ufl.conditional(ufl.lt(abs(x[1] - Y1), 1e-8), ufl.as_vector((0.0, 1.0)), ufl.as_vector((1.0, 0.0))),
        )
        dS_I = self.dS(PHYSICAL_MARKERS["solid_fluid_interface"])
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=facet_tags)
        self.flux_forms = {
            "u_I": dolfinx.fem.form(ufl.inner(u("+"), n_s("+")) * dS_I),
            "abs_u_I": dolfinx.fem.form(abs(ufl.inner(u("+"), n_s("+"))) * dS_I),
            "v_I": dolfinx.fem.form(ufl.inner(v("+"), n_s("+")) * dS_I),
            "v_outflow": dolfinx.fem.form(v[0] * ds(PHYSICAL_MARKERS["outflow"])),
        }
        # Cell integrals, by a DG0 test function, and the midpoints of the cells
        W, W_f = (dolfinx.fem.functionspace(m, ("DG", 0)) for m in (self.mesh, self.fluid_mesh))
        w, w_f = ufl.TestFunction(W), ufl.TestFunction(W_f)
        self.cell_forms = {
            "v_fluid": dolfinx.fem.form(ufl.inner(v, v) * w * fluid),
            "v_solid": dolfinx.fem.form(ufl.inner(v, v) * w * solid),
            "p_fluid": dolfinx.fem.form(p * p * w_f * dx_p),
        }
        midpoints = {
            "mesh": dolfinx.mesh.compute_midpoints(
                self.mesh, tdim, np.arange(self._num_cells(self.mesh), dtype=np.int32)
            ),
            "fluid": dolfinx.mesh.compute_midpoints(
                self.fluid_mesh, tdim, np.arange(self._num_cells(self.fluid_mesh), dtype=np.int32)
            ),
        }
        # The DG0 dof of each cell, and its midpoint
        fluid_sub_cells = np.arange(self._num_cells(self.fluid_mesh))
        self.cell_points = {
            "v_fluid": (W.dofmap.list[self.fluid_cells, 0], midpoints["mesh"][self.fluid_cells]),
            "v_solid": (W.dofmap.list[self.solid_cells, 0], midpoints["mesh"][self.solid_cells]),
            "p_fluid": (W_f.dofmap.list[fluid_sub_cells, 0], midpoints["fluid"]),
        }
        # P1 interpolants for the field plots
        self.p1 = dolfinx.fem.functionspace(self.fluid_mesh, ("P", 1, (2,)))
        self.v_p1 = dolfinx.fem.Function(self.p1)
        sub_cells = np.arange(self._num_cells(self.fluid_mesh), dtype=np.int32)
        self.v_to_p1 = (self.fluid_cell_map.sub_topology_to_topology(sub_cells, inverse=False), sub_cells)

    @staticmethod
    def _num_cells(mesh: dolfinx.mesh.Mesh) -> int:
        index_map = mesh.topology.index_map(mesh.topology.dim)
        return index_map.size_local + index_map.num_ghosts

    def samples(self) -> list[Path]:
        return sorted((self.dir / "vectors").glob("t_*"))

    def eigenvalues(self) -> dict[float, np.ndarray]:
        rows = np.atleast_1d(np.genfromtxt(self.dir / f"{self.name}_spectrum.txt", names=True))
        nev = sum(1 for name in rows.dtype.names if name.startswith("abs_"))
        return {
            round(float(row["t"]), 6): np.array([row[f"re_{i}"] + 1j * row[f"im_{i}"] for i in range(1, nev + 1)])
            for row in rows
        }

    def read(self, sample: Path, nev: int) -> dict[str, dolfinx.fem.Function]:
        functions = {
            f"{name}_{part}_{i}": dolfinx.fem.Function(self.spaces[name], name=f"{name}_{part}_{i}")
            for i in range(1, nev + 1)
            for part in ("re", "im")
            for name in ELEMENTS
        }
        functions |= {name: dolfinx.fem.Function(self.spaces[name], name=name) for name in ("u", "v", "p")}
        Checkpointer(sample, self.mesh, [(self.fluid_mesh, self.fluid_cell_map)]).read(
            list(functions.values()), self.dt
        )
        return functions

    def norm(self, key: str) -> float:
        return np.sqrt(dolfinx.fem.assemble_scalar(self.norm_forms[key]))

    def complex_norms(self, mode: dict[str, np.ndarray]) -> dict[str, float]:
        """L2 norms of the complex mode, from those of its real and imaginary parts."""
        out = {}
        for key in self.norm_forms:
            total = 0.0
            for part in ("re", "im"):
                for name in ("u", "v", "p"):
                    self.f[name].x.array[:] = mode[f"{name}_{part}"]
                total += self.norm(key) ** 2
            out[key] = np.sqrt(total)
        return out

    def fluxes(self, mode: dict[str, np.ndarray]) -> dict[str, complex]:
        """The volume fluxes of the complex mode: the area change of the beam by du (int du.n_s ds, and
        int |du.n_s| ds of the real part), the flux of dv out of the beam and through the outflow."""
        values = {}
        for part in ("re", "im"):
            for name in ("u", "v"):
                self.f[name].x.array[:] = mode[f"{name}_{part}"]
            for key, form in self.flux_forms.items():
                values[f"{key}_{part}"] = dolfinx.fem.assemble_scalar(form)
        return {
            "flux_u_I": values["u_I_re"] + 1j * values["u_I_im"],
            "flux_abs_u_I": values["abs_u_I_re"],
            "flux_v_I": values["v_I_re"] + 1j * values["v_I_im"],
            "flux_v_outflow": values["v_outflow_re"] + 1j * values["v_outflow_im"],
        }

    def cumulative(self, mode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Distances and cumulative fractions of the cell integrals of |dv|^2 and dp^2 of the mode."""
        out = {}
        for key, form in self.cell_forms.items():
            cells, points = self.cell_points[key]
            values = 0.0
            for part in ("re", "im"):
                for name in ("v", "p"):
                    self.f[name].x.array[:] = mode[f"{name}_{part}"]
                values = values + dolfinx.fem.assemble_vector(form).array[cells]
            for distance_name, distance in (("beam", distance_to_beam), ("corner", distance_to_corners)):
                d = distance(points)
                order = np.argsort(d)
                out[f"{key}_{distance_name}_d"] = d[order]
                out[f"{key}_{distance_name}_fraction"] = np.cumsum(values[order]) / values.sum()
        return out

    def interface_trace(self, mode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {"s": self.s}
        for name in ("u", "v"):
            values = (mode[f"{name}_re"] + 1j * mode[f"{name}_im"]).reshape(-1, 2)[self.interface_dofs]
            out[f"{name}_n"] = np.sum(values * self.normal, axis=1)
            out[f"{name}_t"] = values[:, 0] * -self.normal[:, 1] + values[:, 1] * self.normal[:, 0]
        return out

    def fields(self, mode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The real parts of dv (interpolated to P1) and dp at the vertices of the fluid, on its triangles."""
        self.f["v"].x.array[:] = mode["v_re"]
        self.v_p1.interpolate(self.f["v"], cells0=self.v_to_p1[0], cells1=self.v_to_p1[1])
        triangles = self.p1.dofmap.list
        # P1 on the fluid submesh has the same dofs for p and the vector P1, cell by cell
        p_at = np.empty(len(self.v_p1.x.array) // 2)
        p_at[triangles.ravel()] = mode["p_re"][self.spaces["p"].dofmap.list.ravel()]
        return {
            "x": self.p1.tabulate_dof_coordinates()[:, :2],
            "triangles": triangles.copy(),
            "v": self.v_p1.x.array.reshape(-1, 2).copy(),
            "p": p_at,
        }

    def analyse(self) -> None:
        values = self.eigenvalues()
        nev = len(next(iter(values.values())))
        for k, sample in enumerate(self.samples()):
            t = float(sample.name[2:])
            functions = self.read(sample, nev)
            for i in range(1, nev + 1):
                lam = values[round(t, 6)][i - 1]
                if i > 1 and np.isclose(lam, np.conj(values[round(t, 6)][i - 2])) and lam.imag < 0:
                    continue  # the conjugate of the last one
                mode = {
                    f"{name}_{part}": functions[f"{name}_{part}_{i}"].x.array.copy()
                    for name in ELEMENTS
                    for part in ("re", "im")
                }
                # The phase that makes the real part of du_I largest, then ||du_I|| = 1
                u_I = (mode["u_re"] + 1j * mode["u_im"]).reshape(-1, 2)[self.interface_dofs].ravel()
                phase = -0.5 * np.angle(np.sum(u_I**2))
                rotation = np.exp(1j * phase)
                for name in ELEMENTS:
                    z = rotation * (mode[f"{name}_re"] + 1j * mode[f"{name}_im"])
                    mode[f"{name}_re"], mode[f"{name}_im"] = z.real, z.imag
                trace = self.interface_trace(mode)
                peak = np.argmax(np.abs(trace["u_n"]))
                sign = np.sign(trace["u_n"][peak].real) or 1.0
                norms = self.complex_norms(mode)
                scale = sign / norms["u_I"]
                for key in mode:
                    mode[key] *= scale
                norms = {key: value * abs(scale) for key, value in norms.items()}
                # On the solid, J_0 keeps u = dt theta v for the mode, as N has no solid rows
                solid = self.spaces["u"].dofmap.list[self.solid_cells]
                solid = (2 * np.unique(solid)[:, None] + np.arange(2)).ravel()
                kinematic = np.abs(mode["u_re"][solid] - self.dt * self.theta * mode["v_re"][solid]).max()
                out = {
                    "t": t,
                    "dt": self.dt,
                    "theta": self.theta,
                    "eigenvalue": lam,
                    "kinematic_error": kinematic / np.abs(mode["u_re"][solid]).max(),
                    **{f"norm_{key}": value for key, value in norms.items()},
                    **{f"trace_{key}": value for key, value in self.interface_trace(mode).items()},
                    **self.cumulative(mode),
                    **self.fluxes(mode),
                }
                if k == 0:
                    out |= {f"field_{key}": value for key, value in self.fields(mode).items()}
                OUT.mkdir(parents=True, exist_ok=True)
                np.savez(OUT / f"{self.name}_t{t:.4f}_mode{i}.npz", **out)
                print(
                    f"{self.name} t = {t:.4f} mode {i}: lambda = {lam:.3e}, |lambda| = {abs(lam):.3e}, "
                    + ", ".join(f"{key} {value:.3e}" for key, value in norms.items())
                    + f", kinematic error {out['kinematic_error']:.1e}"
                    + f", area change {out['flux_u_I']:.3e} of {out['flux_abs_u_I']:.3e}"
                    + f", flux out of beam {out['flux_v_I']:.3e}, through outflow {out['flux_v_outflow']:.3e}",
                    flush=True,
                )


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name for p in ROOT.glob("modes_ref*") if (p / "vectors").exists())
    for name in names:
        Run(name).analyse()
