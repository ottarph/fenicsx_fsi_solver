# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Moving-domain Taylor-Green vortex: a verification of the ALE fluid forms of ``FSIForms``.

The problem is the 2D vortex with an oscillating boundary of H. A. Kjeldsberg, J. Sundnes and
K. Valen-Sendstad, "A verified and validated moving domain computational fluid dynamics solver with
applications to cardiovascular flows", Int. J. Numer. Methods Biomed. Eng. 39 (2023) e3703, Section 2.4.1.
The velocity and pressure

    v(x, t) = (-sin(2 pi x_2), sin(2 pi x_1)) exp(-4 nu pi^2 t),
    p(x, t) = -cos(2 pi x_1) cos(2 pi x_2) exp(-8 nu pi^2 t),

solve the incompressible Navier-Stokes equations with no body force, for rho = 1 and any nu. The domain is
the image of the square Omega_0 = [-L/2, L/2]^2 under the prescribed displacement

    u_f(chi, t) = A sin(2 pi t / T_G) (sin(2 pi (chi_2 + L/2) / L), sin(2 pi (chi_1 + L/2) / L)),

which moves the boundary too, normal to itself, so the domain deforms. The velocity is given on the whole
boundary by the exact solution, at the moved position chi + u_f(chi, t).

The forms are those of the FSI solvers, ``FSIForms.build_residual``, on the reference square, here a mesh of
quadrilaterals all tagged as fluid. The solid subdomain is empty, so the solid terms vanish. The mesh
displacement u_f is interpolated from the formula above at every step, into the velocity space, so that its
interpolation error does not limit the velocity error; there is no mesh motion solve.

The velocity is given on the whole boundary, so the pressure is fixed only up to a constant. One pressure
degree of freedom is set to zero, and the pressure error is that of the pressures with their means removed.
The errors are the L2 errors on the current domain, pulled back to the reference square,

    ||e||^2 = int_{Omega_0} |e(chi)|^2 J dchi,  with the exact solution evaluated at x = chi + u_f(chi, t).

The pressure element may be continuous (``"Lagrange"``) or discontinuous (``"DPC"``, the full polynomials of
the degree on each quadrilateral, Q2/DPC1 being the Q2-P1disc pair). Neither A_P nor A_I takes a derivative
of the pressure, so a discontinuous pressure needs no facet terms. DPC is defined on the reference cell, which
loses accuracy on quadrilaterals that are not parallelograms; the mesh here is of squares, and the ALE map
only enters as the coefficient u_f, so the cells stay squares.
"""

import argparse
from pathlib import Path
from timeit import default_timer as timer

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI

from xfsi_solver.fsi.forms import FSIForms
from xfsi_solver.tools.convergence import check_converged
from xfsi_solver.tools.qoi import drag_lift_forms
from xfsi_solver.tools.quadrature import cap_quadrature_degree
from xfsi_solver.tools.vtkhdf import VTKHDFWriter

FLUID, SOLID = 1, 2

# The parameters of the paper: kinematic viscosity, amplitude and period of the mesh motion, and side length
NU = 0.025
AMPLITUDE = 0.08
PERIOD = 4.0
LENGTH = 1.0

# The element pairs of the studies: (velocity degree, pressure family, pressure degree)
ELEMENT_PAIRS = {
    "Q2Q1": (2, "Lagrange", 1),
    "Q2DPC1": (2, "DPC", 1),
    "Q3Q2": (3, "Lagrange", 2),
    "Q4Q3": (4, "Lagrange", 3),
}

# The errors at the final time in the study tables, in their columns after n and dt
ERROR_NAMES = ("error_v", "error_p", "error_p_theta", "error_force", "error_force_theta")

OUTPUT_DEGREE = 2  # the degree of the continuous Lagrange functions in the VTKHDF output


def mesh_displacement(chi, t, A=AMPLITUDE, T_G=PERIOD, L=LENGTH, lib=np):
    """The displacement u_f(chi, t) of the paper, eq. (20), as an array (``lib=np``) or UFL expression (``lib=ufl``)."""
    s = A * lib.sin(2.0 * np.pi * t / T_G)
    return (s * lib.sin(2.0 * np.pi * (chi[1] + L / 2) / L), s * lib.sin(2.0 * np.pi * (chi[0] + L / 2) / L))


def exact_velocity(x, t, nu=NU, lib=np):
    """The exact velocity at the physical point x, eq. (18), as a pair of components."""
    decay = lib.exp(-4.0 * nu * np.pi**2 * t)
    return (-lib.sin(2.0 * np.pi * x[1]) * decay, lib.sin(2.0 * np.pi * x[0]) * decay)


def exact_pressure(x, t, nu=NU, lib=np):
    """The exact pressure at the physical point x, eq. (19), for rho = 1."""
    return -lib.cos(2.0 * np.pi * x[0]) * lib.cos(2.0 * np.pi * x[1]) * lib.exp(-8.0 * nu * np.pi**2 * t)


def convergence_rates(h, errors):
    """The rates log(e_k+1 / e_k) / log(h_k+1 / h_k) between consecutive entries of ``h`` and ``errors``."""
    h, errors = np.asarray(h, dtype=float), np.asarray(errors, dtype=float)
    return np.log(errors[1:] / errors[:-1]) / np.log(h[1:] / h[:-1])


def _pin_pressure_dof(P):
    """The Dirichlet condition p = 0 at the first pressure dof of rank 0, which removes the constant mode.

    It does so only if the constant function has a nonzero coefficient on that dof, which is checked: it has
    on every dof of a Lagrange element and of the point-evaluation variants of DPC, but, for instance, only
    on the first dof of each cell of the Legendre variant of DPC.
    """
    comm = P.mesh.comm
    owns = comm.rank == 0 and P.dofmap.index_map.size_local > 0
    dofs = np.array([0] if owns else [], dtype=np.int32)
    one = dolfinx.fem.Function(P)
    one.x.array[:] = 0.0
    one.interpolate(lambda x: np.ones(x.shape[1]))
    removes_constant = comm.allreduce(bool(owns and abs(one.x.array[0]) > 0.5), op=MPI.LOR)
    if not removes_constant:
        raise ValueError(
            f"Setting one dof of the pressure element {P.ufl_element()} does not remove the constant pressure mode"
        )
    # A function value, not a constant one: the blocked NonlinearProblem assigns a condition to a block by its
    # function space, which a condition with a constant value and a space given separately does not match.
    zero = dolfinx.fem.Function(P)
    zero.x.array[:] = 0.0
    return dolfinx.fem.dirichletbc(zero, dofs)


def solve(
    n,
    T,
    dt_val,
    velocity_degree=2,
    pressure_family="Lagrange",
    pressure_degree=1,
    A=AMPLITUDE,
    T_G=PERIOD,
    nu=NU,
    theta_value=None,
    max_quadrature_degree=None,
    qoi_path=None,
    output_path=None,
    output_every=None,
    comm=MPI.COMM_WORLD,
    snes_converged_reason_on=False,
):
    """Solve the moving-domain Taylor-Green vortex on an ``n`` x ``n`` mesh of squares up to time ``T``.

    The velocity is continuous Lagrange of degree ``velocity_degree`` and the pressure ``pressure_family``
    (``"Lagrange"`` or ``"DPC"``) of degree ``pressure_degree``. ``theta_value`` is theta of the theta-scheme,
    by default 0.5 + dt as in the FSI solvers. ``max_quadrature_degree`` caps the quadrature degree of the
    residual (``tools.quadrature.cap_quadrature_degree``); the errors are integrated with a fixed degree.

    With ``qoi_path``, write a row of t, the velocity and pressure errors and the Newton iterations at every
    step. With ``output_path`` and ``output_every``, write v, p and u_f, interpolated into continuous Lagrange
    functions of degree 2, to a VTKHDF file every ``output_every`` steps; the mesh in the file is the reference
    square, which ParaView's Warp By Vector moves by u_f. A DPC pressure is interpolated cell by cell, a node
    shared by cells taking the value of one of them, so the output pressure is for viewing only.

    Return a dict with the time steps ``t``, and at each of them (the first at t = 0) the errors ``error_v``,
    ``error_p`` (against the exact pressure at t^{n+1}), ``error_p_theta`` (at t^{n+theta} on the domain at
    t^{n+theta}), ``error_force`` and ``error_force_theta`` (the force on the middle half of the bottom side by
    ``tools.qoi.drag_lift_forms``, with v^{n+1}, u_f^{n+1} and p^{n+1} against the exact force at t^{n+1}, and with
    the theta averages of v and u_f and p^{n+1} against the exact force at t^{n+theta}), and
    ``newton_iterations``; and the local dof values ``v`` and ``p`` at the final time.
    """
    if (output_path is None) != (output_every is None):
        raise ValueError("output_path and output_every go together")
    num_steps = round(T / dt_val)
    if not np.isclose(num_steps * dt_val, T):
        raise ValueError(f"T = {T} is not a whole number of time steps dt = {dt_val}")

    L = LENGTH
    mesh = dolfinx.mesh.create_rectangle(
        comm, [[-L / 2, -L / 2], [L / 2, L / 2]], [n, n], dolfinx.mesh.CellType.quadrilateral
    )
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    cells = np.arange(num_cells, dtype=np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cells, np.full(num_cells, FLUID, dtype=np.int32))

    cell = mesh.basix_cell()
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", cell, velocity_degree, shape=(2,)))
    P = dolfinx.fem.functionspace(mesh, basix.ufl.element(pressure_family, cell, pressure_degree))
    dv, dp = ufl.TestFunctions(ufl.MixedFunctionSpace(V, P))

    v, v_old = dolfinx.fem.Function(V, name="v"), dolfinx.fem.Function(V)
    p = dolfinx.fem.Function(P, name="p")
    u_f, u_f_old = dolfinx.fem.Function(V, name="u_f"), dolfinx.fem.Function(V)
    u_s_old = dolfinx.fem.Function(V)  # the solid terms vanish on the empty solid subdomain

    def displacement(t):
        return lambda chi: np.stack(mesh_displacement(chi, t, A, T_G, L))

    def boundary_velocity(t):
        def values(chi):
            d = mesh_displacement(chi, t, A, T_G, L)
            return np.stack(exact_velocity((chi[0] + d[0], chi[1] + d[1]), t, nu))

        return values

    # Dirichlet conditions: the exact velocity on the whole boundary, and one pressure dof
    mesh.topology.create_connectivity(tdim - 1, tdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    v_bc = dolfinx.fem.Function(V)
    bcs = [
        dolfinx.fem.dirichletbc(v_bc, dolfinx.fem.locate_dofs_topological(V, tdim - 1, boundary_facets)),
        _pin_pressure_dof(P),
    ]

    dt = dolfinx.fem.Constant(mesh, dt_val)
    theta = dolfinx.fem.Constant(mesh, 0.5 + dt_val if theta_value is None else theta_value)
    rho_f, nu_f = dolfinx.fem.Constant(mesh, 1.0), dolfinx.fem.Constant(mesh, nu)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    forms = FSIForms(
        dv=dv,
        dp=dp,
        dx_fluid=dx(FLUID),
        dx_solid=dx(SOLID),
        rho_f=rho_f,
        nu_f=nu_f,
        rho_s=dolfinx.fem.Constant(mesh, 1.0),
        mu_s=dolfinx.fem.Constant(mesh, 1.0),
        lambda_s=dolfinx.fem.Constant(mesh, 1.0),
        dt=dt,
        theta=theta,
    )
    residual = forms.build_residual(v, v_old, p, u_f, u_f_old, u_s_old)
    residual = cap_quadrature_degree(residual, max_quadrature_degree)
    # The Jacobian has no pressure-pressure block, and the assembler puts the unit diagonal of the pinned
    # pressure dof only into a block that exists. A pressure mass term with a zero coefficient (a Constant, so
    # that UFL does not drop it) creates the block and changes nothing else.
    residual += dolfinx.fem.Constant(mesh, 0.0) * p * dp * dx(FLUID)

    problem = dolfinx.fem.petsc.NonlinearProblem(
        ufl.extract_blocks(residual),
        [v, p],
        bcs=bcs,
        petsc_options_prefix="taylor_green_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_linesearch_type": "none",
            "snes_max_it": 20,
            "snes_atol": 1e-13,
            "snes_rtol": 1e-10,
            "snes_error_if_not_converged": False,
            "ksp_error_if_not_converged": False,
            **({"snes_converged_reason": None} if snes_converged_reason_on else {}),
        },
    )

    # The L2 errors on the current domain, pulled back to the reference square. The pressure of the
    # theta-scheme is also compared with the exact pressure at t^{n+theta} on the domain at t^{n+theta}, that of
    # the displacement theta u_f + (1 - theta) u_f_old: it is the Lagrange multiplier of the time step, and the
    # momentum equation weights its geometric factor J F^-T so (t^{n+1/2} for Crank-Nicolson).
    dx_error = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 2 * velocity_degree + 4})
    X = ufl.SpatialCoordinate(mesh)

    def assemble(form):
        return comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    t_exact = dolfinx.fem.Constant(mesh, 0.0)
    J = ufl.det(ufl.Identity(2) + ufl.grad(u_f))
    v_exact = ufl.as_vector(exact_velocity(X + u_f, t_exact, nu, lib=ufl))
    error_v_form = dolfinx.fem.form(ufl.inner(v - v_exact, v - v_exact) * J * dx_error)

    # The force of the fluid on the middle half of the bottom side of the domain (chi_2 = -L/2, |chi_1| <= L/4 in
    # the reference square; on a whole side the exact pressure force vanishes, cos(2 pi x_1) integrating to zero),
    # from tools.qoi.drag_lift_forms, the forms the FSI solvers compute drag and lift with, against the
    # same forms with the exact velocity and pressure. The pressure, fixed only up to a constant, is shifted so
    # that its mean is that of the exact pressure.
    bottom_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.isclose(x[1], -L / 2) & (np.abs(x[0]) <= L / 4 + 1e-12)
    )
    bottom_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, bottom_facets, np.full(len(bottom_facets), 1, np.int32))
    ds_bottom = ufl.Measure("ds", domain=mesh, subdomain_data=bottom_tags, metadata=dx_error.metadata())(1)

    def time_level_errors(u_d, v_d):
        """A function of the time t_d that returns the mean-free pressure error and the error of the force on the
        middle of the bottom side, for the discrete displacement ``u_d`` and velocity ``v_d`` of one time level
        (those of t^{n+1}, or their theta averages for t^{n+theta}), against the exact solution at t_d on the
        domain of ``u_d``."""
        t_d, mean_p, mean_p_exact = (dolfinx.fem.Constant(mesh, 0.0) for _ in range(3))
        J_d = ufl.det(ufl.Identity(2) + ufl.grad(u_d))
        p_exact = exact_pressure(X + u_d, t_d, nu, lib=ufl)
        v_exact_d = ufl.as_vector(exact_velocity(X + u_d, t_d, nu, lib=ufl))
        area = dolfinx.fem.form(J_d * dx_error)
        means = dolfinx.fem.form(p * J_d * dx_error), dolfinx.fem.form(p_exact * J_d * dx_error)
        error = dolfinx.fem.form(((p - mean_p) - (p_exact - mean_p_exact)) ** 2 * J_d * dx_error)
        p_shifted = p - mean_p + mean_p_exact
        force_forms = [
            [forms[0] for forms in drag_lift_forms(mesh, u_d, w, q, nu_f, rho_f, [ds_bottom])]
            for w, q in ((v_d, p_shifted), (v_exact_d, p_exact))
        ]

        def evaluate(t):
            t_d.value = t
            a = assemble(area)
            mean_p.value = assemble(means[0]) / a
            mean_p_exact.value = assemble(means[1]) / a
            force, force_exact = (np.array([assemble(form) for form in pair]) for pair in force_forms)
            return np.sqrt(assemble(error)), np.linalg.norm(force - force_exact)

        return evaluate

    errors_new = time_level_errors(u_f, v)
    errors_theta = time_level_errors(theta * u_f + (1.0 - theta) * u_f_old, theta * v + (1.0 - theta) * v_old)

    def errors(t, t_theta):
        """The velocity error at t, and the pressure and force errors at t and at t_theta."""
        t_exact.value = t
        error_p, error_force = errors_new(t)
        error_p_theta, error_force_theta = errors_theta(t_theta)
        return np.sqrt(assemble(error_v_form)), error_p, error_p_theta, error_force, error_force_theta

    writer = None
    if output_path is not None:
        V_out = dolfinx.fem.functionspace(mesh, ("Lagrange", OUTPUT_DEGREE, (2,)))
        P_out = dolfinx.fem.functionspace(mesh, ("Lagrange", OUTPUT_DEGREE))
        out = [
            dolfinx.fem.Function(V_out, name="v"),
            dolfinx.fem.Function(P_out, name="p"),
            dolfinx.fem.Function(V_out, name="u_f"),
        ]
        writer = VTKHDFWriter(output_path, mesh, out, cell_tags=cell_tags, degree=OUTPUT_DEGREE)

    def write_output(t, step):
        if writer is not None and step % output_every == 0:
            for f_out, f in zip(writer.functions, (v, p, u_f), strict=True):
                f_out.interpolate(f)
            writer.write(t)

    def write_qoi_row(*values):
        if qoi_path is not None and comm.rank == 0:
            with open(qoi_path, "a") as file:
                file.write("\t".join(f"{value:.10e}" for value in values) + "\n")

    if qoi_path is not None and comm.rank == 0:
        Path(qoi_path).parent.mkdir(parents=True, exist_ok=True)
        with open(qoi_path, "w") as file:
            file.write(
                f"# Moving-domain Taylor-Green vortex: n = {n}, dt = {dt_val}, theta = {theta.value}, "
                f"Q{velocity_degree}/{pressure_family}{pressure_degree}, A = {A}, T_G = {T_G}, nu = {nu}\n"
                "t\terror_v\terror_p\terror_p_theta\terror_force\terror_force_theta\tnewton_iterations\n"
            )

    # The initial state: the mesh is undeformed at t = 0
    t = 0.0
    u_f.interpolate(displacement(t))
    v.interpolate(boundary_velocity(t))
    names = ERROR_NAMES
    history = {"t": [t], "newton_iterations": [0]} | {name: [] for name in names}
    values = errors(t, t)
    for name, value in zip(names, values, strict=True):
        history[name].append(value)
    write_qoi_row(t, *values, 0)
    write_output(t, 0)

    start = timer()
    for step in range(1, num_steps + 1):
        t = step * dt_val
        v_old.x.array[:] = v.x.array
        u_f_old.x.array[:] = u_f.x.array
        u_f.interpolate(displacement(t))
        v_bc.interpolate(boundary_velocity(t))

        problem.solve()
        check_converged(problem, f"t = {t:.4f}")
        iterations = problem.solver.getIterationNumber()

        values = errors(t, t - (1.0 - theta.value) * dt_val)
        history["t"].append(t)
        for name, value in zip(names, values, strict=True):
            history[name].append(value)
        history["newton_iterations"].append(iterations)
        write_qoi_row(t, *values, iterations)
        write_output(t, step)

    if comm.rank == 0:
        print(
            f"n = {n}, dt = {dt_val}: "
            + ", ".join(f"{name} = {history[name][-1]:.4e}" for name in names)
            + f" at t = {t:.4f}, "
            f"{timer() - start:.1f} s",
            flush=True,
        )
    problem.solver.destroy()
    history = {key: np.array(value) for key, value in history.items()}
    history["v"], history["p"] = v.x.array.copy(), p.x.array.copy()
    return history


def _study(name, pairs, runs, out_dir, **solve_kwargs):
    """Run ``solve`` for each element pair and each (n, dt) of ``runs``, and write the errors at the final time
    with the rates between consecutive runs to ``out_dir/<name>_<pair>.txt``."""
    comm = solve_kwargs.get("comm", MPI.COMM_WORLD)
    for pair in pairs:
        velocity_degree, pressure_family, pressure_degree = ELEMENT_PAIRS[pair]
        rows = []
        for n, dt_val in runs:
            tag = f"{name}_{pair}_n{n}_dt{dt_val:g}"
            history = solve(
                n,
                dt_val=dt_val,
                velocity_degree=velocity_degree,
                pressure_family=pressure_family,
                pressure_degree=pressure_degree,
                qoi_path=out_dir / f"{tag}.txt",
                **solve_kwargs,
            )
            rows.append((n, dt_val, *(history[key][-1] for key in ERROR_NAMES)))
        rows = np.array(rows)
        step = 1.0 / rows[:, 0] if name.startswith("spatial") else rows[:, 1]
        rates = [
            np.concatenate([[np.nan], convergence_rates(step, rows[:, column])])
            for column in range(2, 2 + len(ERROR_NAMES))
        ]
        if comm.rank == 0:
            header = (
                f"{name}, {pair}, {', '.join(f'{k} = {v}' for k, v in solve_kwargs.items() if k != 'comm')}\n"
                "n\tdt\t" + "\t".join(ERROR_NAMES) + "\t" + "\t".join(f"rate_{name[6:]}" for name in ERROR_NAMES)
            )
            table = np.column_stack([rows, *rates])
            np.savetxt(out_dir / f"{name}_{pair}.txt", table, header=header, delimiter="\t", fmt="%.6e")
            print(f"\n{name}, {pair}\n{header}\n" + "\n".join("\t".join(f"{x:.4g}" for x in r) for r in table))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study", choices=["spatial", "temporal"])
    parser.add_argument("--pairs", nargs="+", choices=list(ELEMENT_PAIRS), default=None)
    parser.add_argument("--n", type=int, nargs="+", default=None, help="mesh sizes (spatial) or the mesh size")
    parser.add_argument("--dt", type=float, nargs="+", default=None, help="time steps (temporal) or the time step")
    parser.add_argument("--T", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None, help="theta, by default 0.5 + dt")
    parser.add_argument("--A", type=float, default=AMPLITUDE)
    parser.add_argument("--name", default=None, help="the name of the study files, by default the study")
    parser.add_argument("--out-dir", type=Path, default=Path("output/qoi/component_benchmarks/taylor_green"))
    args = parser.parse_args()

    if args.study == "spatial":
        pairs = args.pairs or ["Q2Q1", "Q2DPC1"]
        dt_val = (args.dt or [1e-3])[0]
        runs = [(n, dt_val) for n in (args.n or [8, 16, 32, 64])]
        T = args.T if args.T is not None else 1.0
    else:
        pairs = args.pairs or ["Q4Q3"]
        n = (args.n or [16])[0]
        runs = [(n, dt_val) for dt_val in (args.dt or [0.4, 0.2, 0.1, 0.05, 0.025])]
        T = args.T if args.T is not None else 2.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _study(args.name or args.study, pairs, runs, args.out_dir, T=T, A=args.A, theta_value=args.theta)


if __name__ == "__main__":
    main()
