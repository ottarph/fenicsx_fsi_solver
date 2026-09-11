# Copyright (C) 2025 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""FSI2 benchmark geometry constants.

Duplicated from (must be kept consistent with)
``src/xfsi_solver/scripts/create_mesh_FSI2.py:create_mesh`` (where they are
local variables, not importable). Used by the remeshing prototype
(``notes/remeshing/implementation-plan.md``) to build a prescribed
interface deformation, and by its tests to check a regenerated mesh's
resolution against the original's.
"""

import numpy as np

L = 2.5
H = 0.41
R = 0.05
C_X = 0.2
C_Y = 0.2
FLAG_LENGTH = 0.35
FLAG_THICKNESS = 0.02
A_X = 0.6
A_Y = 0.2

FLAG_LEFT = C_X + R * np.cos(np.arcsin(FLAG_THICKNESS / (2 * R)))
FLAG_RIGHT = A_X
FLAG_BOT = C_Y - FLAG_THICKNESS / 2
FLAG_TOP = C_Y + FLAG_THICKNESS / 2
