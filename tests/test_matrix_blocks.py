import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.matrix_blocks import matrix_blocks_dir, plot_matrix_blocks


def test_matrix_blocks_dir_is_named_after_the_script():
    assert (
        str(matrix_blocks_dir("/a/b/fsi2_biharmonic_mixed.py")) == "output/figures/matrix_blocks/fsi2_biharmonic_mixed"
    )


def test_plot_matrix_blocks_saves_both_plots(tmp_path):
    # Block matrix [[2 I, 0], [I, 3 I]] on 4 + 4 rows, all on this rank
    n = 8
    mat = PETSc.Mat().createAIJ([n, n], nnz=2, comm=MPI.COMM_SELF)
    for i in range(4):
        mat.setValue(i, i, 2.0)
        mat.setValue(4 + i, i, 1.0)
        mat.setValue(4 + i, 4 + i, 3.0)
    mat.assemble()

    blocks = [np.arange(4), np.arange(4, 8)]
    plot_matrix_blocks(mat, {"test": (blocks, ["a", "b"])}, "Test matrix", tmp_path, t=0.0)
    mat.destroy()

    assert (tmp_path / "spy_test.png").exists()
    assert (tmp_path / "block_norms_test.png").exists()
