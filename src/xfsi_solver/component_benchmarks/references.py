# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""The publications whose values the component benchmarks are compared with."""

# The benchmark definition, with the reference values of all the CSM, CFD and FSI tests.
TUREK_HRON_2006 = (
    "S. Turek and J. Hron. Proposal for numerical benchmarking of fluid-structure interaction between an elastic "
    "object and laminar incompressible flow. In H.-J. Bungartz and M. Schäfer (eds.), Fluid-Structure Interaction: "
    "Modelling, Simulation, Optimisation, LNCSE 53, pp. 371-385. Springer, 2006. doi:10.1007/3-540-34596-5_15"
)
TUREK_HRON_2006_SHORT = "Turek & Hron (2006)"

# The results of seven groups for FSI1 and FSI3, with different discretizations and solvers.
TUREK_ET_AL_2010 = (
    "S. Turek, J. Hron, M. Razzaq, H. Wobker and M. Schäfer. Numerical benchmarking of fluid-structure interaction: "
    "a comparison of different discretization and solution approaches. In H.-J. Bungartz, M. Mehl and M. Schäfer "
    "(eds.), Fluid Structure Interaction II, LNCSE 73, pp. 413-424. Springer, 2010. doi:10.1007/978-3-642-14206-2_15"
)
TUREK_ET_AL_2010_SHORT = "Turek et al. (2010)"


def print_references():
    print("References:")
    print(f"  {TUREK_HRON_2006_SHORT}: {TUREK_HRON_2006}")
    print(f"  {TUREK_ET_AL_2010_SHORT}: {TUREK_ET_AL_2010}")
