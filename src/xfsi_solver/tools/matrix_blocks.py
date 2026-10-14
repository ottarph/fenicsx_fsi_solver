# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Spy plots and block norm plots of distributed block matrices, for looking at Jacobians."""

from pathlib import Path

import numpy as np
from petsc4py import PETSc

MATRIX_BLOCKS_DIR = Path("output/figures/matrix_blocks")


def matrix_blocks_dir(script: str | Path) -> Path:
    """The folder for the matrix block plots of a solver script, named after the script."""
    return MATRIX_BLOCKS_DIR / Path(script).stem


def _gather(array: np.ndarray, comm) -> np.ndarray | None:
    """The arrays of all ranks concatenated on rank 0, in rank order, and None on the other ranks."""
    arrays = comm.gather(array, root=0)
    return np.concatenate(arrays) if comm.rank == 0 else None


def plot_matrix_blocks(
    mat: PETSc.Mat,
    partitions: dict[str, tuple[list[np.ndarray], list[str]]],
    title: str,
    output_dir: str | Path,
    t: float,
) -> None:
    """Save a spy plot and a block norm plot of ``mat`` for each partition of its rows and columns.

    ``partitions`` maps a label to ``(blocks, names)``: the global rows of each block that this
    rank owns, and the names of the blocks. The matrix is reordered block by block and saved as
    ``spy_<label>.png`` and ``block_norms_<label>.png`` in ``output_dir``. Collective: the
    nonzeros and the blocks are gathered to rank 0, which plots them, so only for small problems.
    """
    comm = mat.comm.tompi4py()
    row_start = mat.getOwnershipRange()[0]
    indptr, cols, vals = mat.getValuesCSR()
    rows = row_start + np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    nonzero = vals != 0.0
    rows, cols, vals = (_gather(a[nonzero], comm) for a in (rows, cols, vals))
    gathered = {label: ([_gather(b, comm) for b in blocks], names) for label, (blocks, names) in partitions.items()}
    if comm.rank != 0:
        return

    import matplotlib.colors
    import matplotlib.pyplot as plt
    import scipy.sparse

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = mat.getSize()[0]
    matrix = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    for label, (blocks, names) in gathered.items():
        perm = np.concatenate(blocks)
        spy = matrix[perm][:, perm].tocoo()
        spy.eliminate_zeros()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.spy(spy, markersize=0.05, color="black")
        bounds = np.cumsum([0] + [len(b) for b in blocks])
        for b in bounds[1:-1]:
            ax.axhline(b - 0.5, color="tab:red", linewidth=0.8)
            ax.axvline(b - 0.5, color="tab:red", linewidth=0.8)
        centers = 0.5 * (bounds[:-1] + bounds[1:])
        ax.set_xticks(centers, names)
        ax.set_yticks(centers, names)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlabel(f"{title} nonzeros at t = {t:.4f} ({spy.nnz} entries, {len(perm)} dofs)", fontsize="large")
        spy_path = output_dir / f"spy_{label}.png"
        fig.savefig(spy_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved spy plot to {spy_path} ({spy.nnz} nonzeros, {len(perm)} dofs)")

        # Frobenius norms of the blocks, with blocks without nonzeros left blank
        row_blocks = np.searchsorted(bounds, spy.row, side="right") - 1
        col_blocks = np.searchsorted(bounds, spy.col, side="right") - 1
        sq_norms = np.zeros((len(blocks), len(blocks)))
        np.add.at(sq_norms, (row_blocks, col_blocks), spy.data**2)
        block_norms = np.ma.masked_equal(np.sqrt(sq_norms), 0.0)

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        fig, ax = plt.subplots()
        im = ax.matshow(block_norms, cmap=cmap, norm=matplotlib.colors.LogNorm())
        fig.colorbar(im, ax=ax, label="Frobenius norm")
        for (i, j), norm in np.ndenumerate(block_norms):
            if norm is not np.ma.masked:
                ax.text(j, i, f"{norm:.2e}", ha="center", va="center", color="white", fontsize="small")
        ax.set_xticks(range(len(blocks)), names)
        ax.set_yticks(range(len(blocks)), names)
        ax.set_xlabel(f"{title} block norms at t = {t:.4f}", fontsize="large")
        norms_path = output_dir / f"block_norms_{label}.png"
        fig.savefig(norms_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved block norm plot to {norms_path}")
