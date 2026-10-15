"""Parameters of the CFD, CSM and FSI benchmarks of Turek & Hron.

S. Turek and J. Hron, "Proposal for numerical benchmarking of fluid-structure interaction between an elastic
object and laminar incompressible flow", Tables 4, 8 and 12.

All values are in SI units (the tables give them in scaled units, e.g. rho in 10^3 kg/m^3, mu_s in 10^6 kg/(m s^2),
nu_f in 10^-3 m^2/s). Keys follow the naming of the solvers:

- ``rho_s``: solid density [kg/m^3]
- ``nu_s``: solid Poisson ratio [-]
- ``mu_s``: solid shear modulus [kg/(m s^2)]
- ``E_s``: solid Young modulus [kg/(m s^2)]
- ``rho_f``: fluid density [kg/m^3]
- ``nu_f``: fluid kinematic viscosity [m^2/s]
- ``U_bar``: mean inflow velocity [m/s]
- ``g``: gravitational acceleration acting in -y direction on the solid only [m/s^2]
- ``beta``: density ratio rho_s / rho_f [-]
- ``Ae``: elasticity number E_s / (rho_f U_bar^2) [-]
- ``Re``: Reynolds number U_bar d / nu_f, with d = 2 r = 0.1 the cylinder diameter [-]
"""

# CFD tests: the flag is rigid, either by setting rho_s = 1e6 and mu_s = 1e12, or by solving the flow only with no-slip
# conditions on the flag interface. CFD1 is steady, CFD2 and CFD3 are periodic. (Table 4)
CFD_parameters = {
    "CFD1": {"rho_f": 1.0e3, "nu_f": 1.0e-3, "U_bar": 0.2, "Re": 20},
    "CFD2": {"rho_f": 1.0e3, "nu_f": 1.0e-3, "U_bar": 1.0, "Re": 100},
    "CFD3": {"rho_f": 1.0e3, "nu_f": 1.0e-3, "U_bar": 2.0, "Re": 200},
}

# CSM tests: the elastic beam alone, without the fluid, loaded by gravity on the solid only. CSM1 and CSM2 are steady,
# CSM3 is time dependent, starting from the undeformed configuration. (Table 8)
CSM_parameters = {
    "CSM1": {
        "rho_s": 1.0e3,
        "nu_s": 0.4,
        "mu_s": 0.5e6,
        "E_s": 1.4e6,
        "rho_f": 1.0e3,
        "nu_f": 1.0e-3,
        "U_bar": 0.0,
        "g": 2.0,
        "beta": 1,
        "Re": 0,
    },
    "CSM2": {
        "rho_s": 1.0e3,
        "nu_s": 0.4,
        "mu_s": 2.0e6,
        "E_s": 5.6e6,
        "rho_f": 1.0e3,
        "nu_f": 1.0e-3,
        "U_bar": 0.0,
        "g": 2.0,
        "beta": 1,
        "Re": 0,
    },
    "CSM3": {
        "rho_s": 1.0e3,
        "nu_s": 0.4,
        "mu_s": 0.5e6,
        "E_s": 1.4e6,
        "rho_f": 1.0e3,
        "nu_f": 1.0e-3,
        "U_bar": 0.0,
        "g": 2.0,
        "beta": 1,
        "Re": 0,
    },
}

# FSI tests: the full benchmark. FSI1 is steady, FSI2 and FSI3 are periodic. (Table 12)
FSI_parameters = {
    "FSI1": {
        "rho_s": 1.0e3,
        "nu_s": 0.4,
        "mu_s": 0.5e6,
        "rho_f": 1.0e3,
        "nu_f": 1.0e-3,
        "U_bar": 0.2,
        "beta": 1,
        "Ae": 3.5e4,
        "Re": 20,
    },
    "FSI2": {
        "rho_s": 10.0e3,
        "nu_s": 0.4,
        "mu_s": 0.5e6,
        "rho_f": 1.0e3,
        "nu_f": 1.0e-3,
        "U_bar": 1.0,
        "beta": 10,
        "Ae": 1.4e3,
        "Re": 100,
    },
    "FSI3": {
        "rho_s": 1.0e3,
        "nu_s": 0.4,
        "mu_s": 2.0e6,
        "rho_f": 1.0e3,
        "nu_f": 1.0e-3,
        "U_bar": 2.0,
        "beta": 1,
        "Ae": 1.4e3,
        "Re": 200,
    },
}
