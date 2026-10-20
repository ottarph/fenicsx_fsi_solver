# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Print the comparison with the reference values and write the plots for the CSM, CFD and FSI benchmarks.

Runs compare_*_qois.py and plot_*.py for each benchmark, skipping the tests that have no QoI file, and lists the
publications the values are compared with (see references.py). The CFD and FSI QoI files can carry a suffix, such as
_fine for cfd_1_fine.txt. Run from the repository root:

    python -m xfsi_solver.component_benchmarks.run_analysis
    python -m xfsi_solver.component_benchmarks.run_analysis --cfd-suffix _fine_T20
"""

import argparse

from xfsi_solver.component_benchmarks import (
    compare_cfd_qois,
    compare_csm_qois,
    compare_fsi_qois,
    plot_cfd_forces,
    plot_csm_displacement,
    plot_fsi_qois,
)
from xfsi_solver.component_benchmarks.references import print_references


def print_title(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cfd-suffix", default="", help="suffix of the CFD QoI file names, e.g. _fine")
    parser.add_argument("--fsi-suffix", default="", help="suffix of the FSI QoI file names, e.g. _coarse")
    args = parser.parse_args()

    print_title("CSM: elastic beam alone")
    compare_csm_qois.compare_all()
    plot_csm_displacement.main()

    print_title(f"CFD: rigid flag (QoI files cfd_<n>{args.cfd_suffix}.txt)")
    compare_cfd_qois.compare_all(args.cfd_suffix)
    plot_cfd_forces.plot_all(args.cfd_suffix)

    print_title(f"FSI: elastic flag (QoI files fsi_<n>{args.fsi_suffix}.txt)")
    compare_fsi_qois.compare_all(args.fsi_suffix)
    plot_fsi_qois.plot_all(args.fsi_suffix)

    print()
    print_references()


if __name__ == "__main__":
    main()
