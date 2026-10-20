# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np

# Names of the columns of the FeatFlow point file, in order, as listed on the FeatFlow FSI benchmark
# reference page, with the names of our qoi files where they coincide. The drag and the lift of our
# runs are the sums of those on the beam and on the cylinder, with the opposite sign of the lift.
# The page doesn't name columns 3, 4, 9 and 10 (1-based), which are named by their column number.
REFERENCE_COLUMNS = [
    "t",
    "dt",
    "unknown_3",
    "unknown_4",
    "drag_beam",
    "lift_beam",
    "drag_cylinder",
    "lift_cylinder",
    "unknown_9",
    "unknown_10",
    "A_x",
    "A_y",
]


def translate_point_file(in_filename, out_filename):

    with open(in_filename, "r") as in_f, open(out_filename, "w") as out_f:
        out_f.write("# " + "\t".join(REFERENCE_COLUMNS) + "\n")
        for line in in_f:
            out_f.write(line.replace("  ", " "))


# Names of the columns of the reference file in the layout of our qoi files.
QOI_COLUMNS = ["t", "drag", "lift", "A_x", "A_y"]


def write_reference_qois(split_filename, out_filename):
    """Write the reference values of the split file written by :func:`translate_point_file` in the
    layout and with the conventions of our qoi files: the drag and the lift summed over the beam
    and the cylinder, the lift with our sign, and without the time step and the unnamed columns."""
    split = np.loadtxt(split_filename)
    col = {name: split[:, i] for i, name in enumerate(REFERENCE_COLUMNS)}
    qois = np.column_stack(
        [
            col["t"],
            col["drag_beam"] + col["drag_cylinder"],
            -(col["lift_beam"] + col["lift_cylinder"]),
            col["A_x"],
            col["A_y"],
        ]
    )
    np.savetxt(out_filename, qois, fmt="%.9e", delimiter="\t", header="\t".join(QOI_COLUMNS))


if __name__ == "__main__":
    translate_point_file("ref_fsi2.point", "data/fsi2_reference_split.txt")
    write_reference_qois("data/fsi2_reference_split.txt", "data/fsi2_reference.txt")

    ref = np.loadtxt("data/fsi2_reference.txt")

    t = ref[:, 0]

    drag = ref[:, 1]
    lift = ref[:, 2]
    xdisp = ref[:, 3]
    ydisp = ref[:, 4]

    import matplotlib.pyplot as plt

    plt.figure()

    key = "drag"
    val = {"drag": drag, "lift": lift, "xdisp": xdisp, "ydisp": ydisp}[key]
    lab = {"drag": "drag", "lift": "lift", "xdisp": "xdisp", "ydisp": "ydisp"}[key]

    plt.plot(t, val, "k-")
    plt.xlabel("time")
    plt.ylabel(lab)
    # plt.xlim([10.0, 10.8])
    plt.title("Reference")

    Path("output/figures/qoi/reference").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"output/figures/qoi/reference/{key}.pdf")
    plt.show()
