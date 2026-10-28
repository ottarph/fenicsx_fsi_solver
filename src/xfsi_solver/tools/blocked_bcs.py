# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Dirichlet conditions in the rows of a blocked (non-nest) PETSc matrix."""

from collections.abc import Sequence

import dolfinx
import numpy as np
from petsc4py import PETSc


def owned_dirichlet_rows(
    bcs_by_block: Sequence[Sequence[dolfinx.fem.DirichletBC]], offsets_owned: Sequence[int], first_row: int
) -> np.ndarray:
    """The global rows of the dofs that the conditions ``bcs_by_block[k]`` of block k constrain, owned by
    this rank, in a blocked matrix whose owned rows are numbered block by block from ``first_row``, with
    block k starting at ``offsets_owned[k]`` (the first part of the ``_blocks`` attribute of a blocked
    dolfinx vector).

    With ``J.zeroRows(rows, diag=1.0)``, this sets the unit diagonal of the Dirichlet rows of an
    assembled matrix J, whose other entries in these rows are already zero. Every constrained dof is
    owned by a rank that also locates it, so the owned rows of all ranks cover all of them.
    """
    rows = []
    for k, bcs in enumerate(bcs_by_block):
        for bc in bcs:
            dofs, num_owned = bc.dof_indices()  # unrolled local indices, owned first
            rows.append(first_row + offsets_owned[k] + dofs[:num_owned])
    return np.unique(np.concatenate(rows) if rows else np.empty(0)).astype(PETSc.IntType)
