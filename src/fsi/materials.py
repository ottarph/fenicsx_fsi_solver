import dolfinx as dfx
import ufl


class Solid:

    def STVK(u: dfx.fem.Function, lambda_: dfx.fem.Constant, mu: dfx.fem.Constant):
        Id = ufl.Identity(u.ufl_shape[0])
        F = Id + ufl.grad(u)
        E = 0.5 * (F.T * F - Id)
        J = ufl.det(F)
        sigma = ufl.inv(J) * F * (lambda_ * ufl.tr(E) * Id + 2.0 * mu * E) * F.T
        return sigma
    

class Fluid:

    def NS_pressure(u: dfx.fem.Function, p: dfx.fem.Function):
        Id = ufl.Identity(p.ufl_domain().geometric_dimension())
        F = Id + ufl.grad(u)
        J = ufl.det(F)
        sigma = -J * p * Id * ufl.inv(F).T
        return sigma
    
    def NS_velocity(u: dfx.fem.Function, nu: dfx.fem.Constant, rho: dfx.fem.Constant):
        Id = ufl.Identity(u.ufl_shape)
        F = Id + ufl.grad(u)
        sigma = rho * nu * (ufl.grad(u) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(u).T)
        return sigma
    

material_parameters = {
    "solid": {
        "lambda": 2.0e6,
        "mu": 0.5e6,
        "rho": 1.0e4,
    },
    "fluid": {
        "nu": 1.0e-3,
        "rho": 1.0e3,
    }
}
    