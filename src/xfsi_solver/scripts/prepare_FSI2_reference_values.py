# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import numpy as np
from pathlib import Path

def translate_point_file(in_filename, out_filename):

    with open(in_filename, 'r') as in_f, open(out_filename, 'w') as out_f:
        for line in in_f:
            out_f.write(line.replace("  ", " "))


if __name__ == "__main__":


    translate_point_file("ref_fsi2.point", "data/fsi2_reference.txt")

    
    ref = np.loadtxt("data/fsi2_reference.txt")

    t = ref[:,0]

    drag = ref[:,4] + ref[:,6]
    lift = ref[:,5] + ref[:,7]
    xdisp = ref[:,10]
    ydisp = ref[:,11]



    import matplotlib.pyplot as plt

    plt.figure()

    key = "drag"
    val = {"drag": drag, "lift": lift, "xdisp": xdisp, "ydisp": ydisp}[key]
    lab = {"drag": "drag", "lift": "lift", "xdisp": "xdisp", "ydisp": "ydisp"}[key]

    plt.plot(t, val, 'k-')
    plt.xlabel("time")
    plt.ylabel(lab)
    # plt.xlim([10.0, 10.8])
    plt.title("Reference")

    Path("output/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("output/figures/fsi2_referencedrag.pdf")
    plt.show()

