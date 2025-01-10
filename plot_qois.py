# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

qois_bih = np.loadtxt("output/fsi2_biharm_qoi.txt")
qois_harm = np.loadtxt("output/fsi2_harm_dm_qoi.txt")

assert Path("data/fsi2_reference.txt").exists(), "Download the reference data at https://wwwold.mathematik.tu-dortmund.de/~featflow/media/fsi/data/fsi2/0p0005/ref_fsi2.point, then run scripts/prepare_FSI2_reference_values.py"

ref = np.loadtxt("data/fsi2_reference.txt")
time_r = ref[:,0]
drag_r = ref[:,4] + ref[:,6]
lift_r = -ref[:,5] + -ref[:,7]
xdisp_r = ref[:,10]
ydisp_r = ref[:,11]


plt.figure(figsize=(10,5))
plt.plot(qois_bih[:,0], qois_bih[:,1], 'k-', label="biharmonic")
plt.plot(qois_harm[:,0], qois_harm[:,1], 'b--', label="harmonic")
plt.plot(time_r, drag_r, 'r:', label="reference")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Drag")
plt.legend()

plt.savefig("figures/drag.pdf")


plt.figure(figsize=(10,5))
plt.plot(qois_bih[:,0], qois_bih[:,2], 'k-', label="biharmonic")
plt.plot(qois_harm[:,0], qois_harm[:,2], 'b--', label="harmonic")
plt.plot(time_r, lift_r, 'r:', label="reference")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("lift")
plt.legend()

plt.savefig("figures/lift.pdf")


plt.figure(figsize=(10,5))
plt.plot(qois_bih[:,0], qois_bih[:,3], 'k-', label="biharmonic")
plt.plot(qois_harm[:,0], qois_harm[:,3], 'b--', label="harmonic")
plt.plot(time_r, xdisp_r, 'r:', label="reference")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Tip $x$-displacement [m]")
plt.legend()

plt.savefig("figures/x-disp.pdf")


plt.figure(figsize=(10,5))
plt.plot(qois_bih[:,0], qois_bih[:,4], 'k-', label="biharmonic")
plt.plot(qois_harm[:,0], qois_harm[:,4], 'b--', label="harmonic")
plt.plot(time_r, ydisp_r, 'r:', label="reference")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Tip $y$-displacement [m]")
plt.legend()

plt.savefig("figures/y-disp.pdf")

