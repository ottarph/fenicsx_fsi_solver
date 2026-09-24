# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Distributed index sets for PETSc field splits of DOLFINx block matrices.

A block matrix assembled by DOLFINx from a nested list of forms (``kind=None``)
is a single MPIAIJ matrix. Its rows are numbered through a local-to-global
map; the index sets here are derived from that map rather than from an
assumed storage order, and contain only the rows owned by each rank.

Index sets for a split nested inside another split (e.g. ``v`` inside
``(v, p)``) must be expressed in the numbering of the extracted submatrix.
PETSc numbers the rows of ``MatCreateSubMatrix(A, is)`` rank by rank in the
order of the entries of ``is``; :func:`nested_index_sets` reproduces that.
"""

from collections.abc import Sequence

import dolfinx as dfx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


def field_dof_rows(A: PETSc.Mat, spaces: Sequence[dfx.fem.FunctionSpace]) -> list[np.ndarray]:
    """Global row of every owned (block-expanded) DOF of each field, in local DOF order.

    Args:
        A: Matrix assembled by DOLFINx with row spaces ``spaces`` (a block
            matrix for several spaces).
        spaces: The row function spaces of ``A``, in block order.
    """
    local_sets = dfx.cpp.la.petsc.create_index_sets([(V.dofmap.index_map, V.dofmap.index_map_bs) for V in spaces])
    lgmap, _ = A.getLGMap()
    rstart, rend = A.getOwnershipRange()
    rows = []
    for V, local in zip(spaces, local_sets, strict=True):
        n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        field_rows = np.asarray(lgmap.apply(local.getIndices()[:n_owned]), dtype=PETSc.IntType)
        if np.any(field_rows < rstart) or np.any(field_rows >= rend):
            raise RuntimeError("Owned DOFs do not map to owned rows")
        rows.append(field_rows)
    return rows


def field_index_sets(A: PETSc.Mat, spaces: Sequence[dfx.fem.FunctionSpace]) -> list[PETSc.IS]:
    """Global index sets (sorted, with the space's block size) of the owned rows of each field.

    Args:
        A: Block matrix assembled by DOLFINx with row spaces ``spaces``.
        spaces: The row function spaces of ``A``, in block order.
    """
    sets = []
    for V, rows in zip(spaces, field_dof_rows(A, spaces), strict=True):
        index_set = PETSc.IS().createGeneral(np.sort(rows), comm=A.comm)
        index_set.setBlockSize(V.dofmap.index_map_bs)
        sets.append(index_set)
    return sets


def union_index_set(sets: Sequence[PETSc.IS]) -> PETSc.IS:
    """Sorted union of disjoint index sets on the same communicator."""
    indices = np.sort(np.concatenate([s.getIndices() for s in sets]))
    return PETSc.IS().createGeneral(indices.astype(PETSc.IntType), comm=sets[0].comm)


def nested_index_sets(parent: PETSc.IS, children: Sequence[PETSc.IS]) -> list[PETSc.IS]:
    """Express ``children`` (subsets of ``parent``) in the numbering of the ``parent`` submatrix.

    ``parent`` must be sorted, and every child entry must be an entry of
    ``parent`` on the same rank.
    """
    comm = parent.comm.tompi4py()
    parent_indices = parent.getIndices()
    if np.any(np.diff(parent_indices) <= 0):
        raise ValueError("Parent index set must be strictly increasing")
    offset = comm.exscan(parent_indices.size, op=MPI.SUM) or 0
    nested = []
    for child in children:
        child_indices = child.getIndices()
        local = np.searchsorted(parent_indices, child_indices)
        if np.any(local >= parent_indices.size) or np.any(parent_indices[np.minimum(local, parent_indices.size - 1)]
                                                          != child_indices):
            raise ValueError("Child index set is not a local subset of the parent index set")
        nested.append(PETSc.IS().createGeneral((offset + local).astype(PETSc.IntType), comm=parent.comm))
    return nested


def check_partition(sets: Sequence[PETSc.IS], rstart: int, rend: int) -> None:
    """Raise unless ``sets`` are disjoint and exactly cover the owned range ``[rstart, rend)``."""
    indices = np.concatenate([s.getIndices() for s in sets]) if sets else np.empty(0, dtype=PETSc.IntType)
    if indices.size != rend - rstart:
        raise ValueError(f"Index sets have {indices.size} entries for {rend - rstart} owned rows")
    if not np.array_equal(np.sort(indices), np.arange(rstart, rend)):
        raise ValueError("Index sets are not a disjoint cover of the owned rows")


def field_norms(vec: PETSc.Vec, sets: Sequence[PETSc.IS]) -> np.ndarray:
    """Euclidean norm of ``vec`` restricted to each index set."""
    norms = []
    for index_set in sets:
        sub = vec.getSubVector(index_set)
        norms.append(sub.norm())
        vec.restoreSubVector(index_set, sub)
    return np.array(norms)
