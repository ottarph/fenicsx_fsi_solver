# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import dolfinx
import dolfinx.fem.petsc
import ufl

# Coefficients of the compressible three-term Mooney-Rivlin solid (Solid.MR3_energy), in Pa. Without a D1, the
# solvers take the bulk modulus from their St. Venant-Kirchhoff parameters
# (xfsi_solver.tools.parameters.mooney_rivlin_3_coefficients).
#
# An artificial benchmark material for FSI2, not a tissue fit: C10 + C01 = 2.5e5 Pa gives the initial shear
# modulus of the FSI2 solid, mu_s = 5e5 Pa, and with D1 from the FSI2 bulk modulus, K = 2e6 + 2 * 5e5 / 3 Pa,
# also its initial Poisson ratio 0.4, so it differs from the SVK solid at finite strain only. C11 is that of the
# "medium" fit of Yamamoto et al. (2025), below.
MR3_FSI2_MATCHED = {"C10": 0.0, "C01": 2.5e5, "C11": 1.8e6}
# The same without the third term, the two-term compressible Mooney-Rivlin solid with the same initial moduli.
MR3_FSI2_MATCHED_C11_0 = {"C10": 0.0, "C01": 2.5e5, "C11": 0.0}
# The fits to aneurysm wall tissue of Yamamoto et al., Impact of wall property and flow rate assumptions on
# simulations of flow-induced vibration of intracranial aneurysms, Int. J. Numer. Methods Biomed. Eng. (2025),
# Table 1, converted to Pa. The paper gives no numerical D1, and with the FSI2 bulk modulus these are much
# nearer incompressible than the FSI2 solid (initial Poisson ratio 0.4915 for "medium").
MR3_YAMAMOTO_2025 = {
    "soft": {"C10": 0.0, "C01": 2.0e3, "C11": 9.0e5},
    "medium": {"C10": 0.0, "C01": 2.0e4, "C11": 1.8e6},
    "stiff": {"C10": 0.0, "C01": 2.0e5, "C11": 1.8e6},
}


class Solid:
    def STVK(u: dolfinx.fem.Function, lambda_: dolfinx.fem.Constant, mu: dolfinx.fem.Constant):
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        E = 0.5 * (F.T * F - Id)
        J = ufl.det(F)
        sigma = ufl.inv(J) * F * (lambda_ * ufl.tr(E) * Id + 2.0 * mu * E) * F.T
        return sigma

    def IMR_elastic(u: dolfinx.fem.Function, c1: dolfinx.fem.Constant, c2: dolfinx.fem.Constant):
        """Elastic Cauchy stress of an incompressible Mooney-Rivlin material.

        For B = F F^T, return 2 c1 B - 2 c2 B^-1. The coefficients use the
        convention W = c1 (I1 - 3) + c2 (I2 - 3), with small-strain shear
        modulus 2 (c1 + c2). Setting c2 = 0 gives the neo-Hookean model.

        This law assumes det(F) = 1; the solver must impose that constraint
        separately. For a 2D displacement, return the in-plane stress of a
        plane-strain material with unit out-of-plane stretch.
        """
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        B = F * F.T
        return 2.0 * c1 * B - 2.0 * c2 * ufl.inv(B)

    def IMR_pressure(p: dolfinx.fem.Function):
        """Cauchy pressure stress -p I, with pressure positive in compression."""
        Id = ufl.Identity(p.ufl_domain().geometric_dimension)
        return -p * Id

    def IMR(
        u: dolfinx.fem.Function,
        p: dolfinx.fem.Function,
        c1: dolfinx.fem.Constant,
        c2: dolfinx.fem.Constant,
    ):
        """Full incompressible Mooney-Rivlin Cauchy stress, including pressure.

        Use IMR_elastic and IMR_pressure separately to time-average the elastic
        part while keeping pressure fully implicit. With this pressure
        convention, the undeformed stress-free state has p = 2 (c1 - c2).
        Convert to first Piola stress with J sigma F^-T, as for STVK.
        """
        return Solid.IMR_elastic(u, c1, c2) + Solid.IMR_pressure(p)

    def MR3_energy(F, C10, C01, C11, D1):
        """Strain energy per reference volume of the compressible three-term Mooney-Rivlin material.

        With J = det(F) and the isochoric left Cauchy-Green tensor Bbar = J^(-2/3) F F^T, whose invariants
        are I1bar = tr(Bbar) and I2bar = (tr(Bbar)^2 - tr(Bbar Bbar)) / 2,

            W = C10 (I1bar - 3) + C01 (I2bar - 3) + C11 (I1bar - 3) (I2bar - 3) + (J - 1)^2 / D1.

        ``F`` must be a 3 x 3 deformation gradient, the invariants being those of a three-dimensional
        tensor; for plane strain, embed the in-plane gradient as ``MR3_piola`` does. The Cij have units
        of stress and D1 of inverse stress; the initial shear and bulk moduli are mu0 = 2 (C10 + C01) and
        K0 = 2 / D1, and C11 does not enter them. The energy is defined only for J > 0: J^(-2/3) of a
        negative J is not real, which this function does not hide.
        """
        if ufl.shape(F) != (3, 3):
            raise ValueError(f"MR3_energy needs a 3 x 3 deformation gradient, not one of shape {ufl.shape(F)}")
        J = ufl.det(F)
        Bbar = J ** (-2.0 / 3.0) * F * F.T
        I1bar = ufl.tr(Bbar)
        I2bar = 0.5 * (ufl.tr(Bbar) ** 2 - ufl.tr(Bbar * Bbar))
        return C10 * (I1bar - 3.0) + C01 * (I2bar - 3.0) + C11 * (I1bar - 3.0) * (I2bar - 3.0) + (J - 1.0) ** 2 / D1

    def MR3_piola(u: dolfinx.fem.Function, C10, C01, C11, D1):
        """First Piola-Kirchhoff stress P = dW/dF of the compressible three-term Mooney-Rivlin material.

        P is obtained by UFL differentiation of ``MR3_energy`` with respect to the deformation gradient
        F = I + grad(u), marked as a UFL variable. P still depends on u through F, so the residual
        ``inner(P, grad(test)) dx`` can be differentiated again, by ``ufl.derivative``, for the Jacobian.

        For a 2D displacement this is plane strain: the in-plane gradient is embedded in the 3 x 3

            F3 = [[F[0, 0], F[0, 1], 0], [F[1, 0], F[1, 1], 0], [0, 0, 1]],

        the three-dimensional energy is differentiated with respect to F3, and the upper-left 2 x 2 block
        of P is returned. The out-of-plane stress P[2, 2] does not in general vanish and is not returned.
        The Cauchy stress is P F^T / J.
        """
        dim = u.ufl_shape[0]
        F = ufl.Identity(dim) + ufl.grad(u)
        if dim == 3:
            F3 = ufl.variable(F)
        elif dim == 2:
            F3 = ufl.variable(ufl.as_matrix([[F[0, 0], F[0, 1], 0.0], [F[1, 0], F[1, 1], 0.0], [0.0, 0.0, 1.0]]))
        else:
            raise ValueError(f"MR3_piola supports 2D (plane strain) and 3D displacements, not {dim}D")
        P3 = ufl.diff(Solid.MR3_energy(F3, C10, C01, C11, D1), F3)
        if dim == 3:
            return P3
        return ufl.as_matrix([[P3[0, 0], P3[0, 1]], [P3[1, 0], P3[1, 1]]])


class Fluid:
    def NS_pressure(p: dolfinx.fem.Function):
        Id = ufl.Identity(p.ufl_domain().geometric_dimension)
        sigma = -p * Id
        return sigma

    def NS_velocity(
        u: dolfinx.fem.Function, v: dolfinx.fem.Function, nu: dolfinx.fem.Constant, rho: dolfinx.fem.Constant
    ):
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        sigma = rho * nu * (ufl.grad(v) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(v).T)
        return sigma

    def NS_velocity_eulerian(v: dolfinx.fem.Function, nu: dolfinx.fem.Constant, rho: dolfinx.fem.Constant):
        sigma = rho * nu * (ufl.grad(v) + ufl.grad(v).T)
        return sigma

    def NS(
        u: dolfinx.fem.Function,
        v: dolfinx.fem.Function,
        p: dolfinx.fem.Function,
        nu: dolfinx.fem.Constant,
        rho: dolfinx.fem.Constant,
    ):
        return Fluid.NS_velocity(u, v, nu, rho) + Fluid.NS_pressure(p)

    def NS_eulerian(
        v: dolfinx.fem.Function, p: dolfinx.fem.Function, nu: dolfinx.fem.Constant, rho: dolfinx.fem.Constant
    ):
        return Fluid.NS_velocity_eulerian(v, nu, rho) + Fluid.NS_pressure(p)
