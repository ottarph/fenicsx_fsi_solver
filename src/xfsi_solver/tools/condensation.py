# Copyright (C) 2026 Ottar Hellan
#
# SPDX-License-Identifier: MIT

"""Condensation of groups of dofs out of a Jacobian held as a ``NestView``, for experiments.

The solvers with a ``jacobian_callback`` (fsi2_biharmonic_fulljacobian_initialguess and
fsi2_harmonic_fulljacobian_initialguess) pass their ``NestView`` to the hooks here. The groups are
named as in those solvers: ``v`` and ``p``; ``u_s``, the displacement on the solid cells off the
interface; ``u_int``, the displacement on the interface; ``u_f``, the other displacement dofs; and in
the biharmonic solver ``z_int`` and ``z_f``. ``condense_solid_exact`` also needs
``nest_view.data["v_rows"]``, which the solvers set when u and v have the same element.
"""

from timeit import default_timer as timer

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from xfsi_solver.tools.matrix_nest import NestView


def human_count(n: float) -> str:
    """``n`` with k for thousands and M for millions, such as 291.5k or 3.43M."""
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def _nonzeros(blocks) -> float:
    """The number of entries in the patterns of ``blocks``, a list of lists of Mats or None. Collective."""
    return sum(
        block.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM)["nz_used"] for row in blocks for block in row if block is not None
    )


def _nonzero_columns(mat: PETSc.Mat) -> PETSc.IS:
    """The columns of ``mat`` with an entry in its pattern on any rank, each in the IS on the rank
    that owns it in the column layout of ``mat``. Collective."""
    comm = mat.getComm()
    _, cols, _ = mat.getValuesCSR()
    cols = np.unique(np.concatenate(comm.tompi4py().allgather(np.unique(cols))))
    start, end = mat.getOwnershipRangeColumn()
    return PETSc.IS().createGeneral(cols[(cols >= start) & (cols < end)].astype(PETSc.IntType), comm=comm)


def _lu(A: PETSc.Mat) -> PETSc.KSP:
    """A KSP that solves with ``A`` by MUMPS LU."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


def _schur_blocks(nest_view: NestView, rest: list[str], updates: dict) -> tuple[list, list[PETSc.Mat]]:
    """The blocks of S = D - updates over the groups ``rest``: the blocks of the nest Jacobian, less
    ``updates[(a, b)]`` where given, which are used up. Returns the blocks and those made here."""
    blocks, made = [], []
    for a in rest:
        row = []
        for b in rest:
            block = nest_view.block(a, b)
            if (a, b) in updates:
                update = updates.pop((a, b))
                if block is None:
                    update.scale(-1.0)
                    block = update
                else:
                    block = block.duplicate(copy=True)
                    block.axpy(-1.0, update, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                    update.destroy()
                made.append(block)
            row.append(block)
        blocks.append(row)
    return blocks, made


def condense_u_s(nest_view: NestView, drop_tol: float = 1e-14) -> tuple[PETSc.Mat, list, list[PETSc.Mat]]:
    """Condense u_s out of the nest Jacobian: the Schur complement S = D - C A^-1 B of
    A = J[u_s, u_s], with B = J[u_s, rest], C = J[rest, u_s] and D = J[rest, rest], as a MatNest over
    the other groups of the nest, in their order and with the contiguous layout of a nest.

    Only the blocks of S that u_s couples to change: (v, v), (v, u_int), (u_int, v) and (u_int,
    u_int). The others are the blocks of the nest Jacobian themselves. A^-1 B is computed by MUMPS LU
    of A on the columns of B that have entries, as a dense matrix, whose entries below ``drop_tol``
    times its largest are dropped before it is multiplied by C, so that round-off does not fill S.
    Collective. Returns S, its blocks (None where zero) and the matrices made here, which the caller
    destroys with S.
    """
    rest = [name for name in nest_view.names if name != "u_s"]
    comm = nest_view.mat.getComm()
    ksp = _lu(nest_view.block("u_s", "u_s"))
    updates = {}  # (a, b): C_a A^-1 B_b, for the blocks of S that change
    for b in rest:
        B = nest_view.block("u_s", b)
        if B is None:
            continue
        rows = PETSc.IS().createStride(B.getLocalSize()[0], B.getOwnershipRange()[0], 1, comm=comm)
        cols = _nonzero_columns(B)
        B_dense = B.createSubMatrix(rows, cols).convert("dense")  # converted in place
        X = B_dense.duplicate()
        ksp.matSolve(B_dense, X)
        values = X.getDenseArray(readonly=False)
        largest = comm.tompi4py().allreduce(np.abs(values).max(initial=0.0), op=MPI.MAX)
        values[np.abs(values) < drop_tol * largest] = 0.0
        X.assemble()
        X_sparse = X.convert("aij", out=PETSc.Mat())
        # Q puts the columns of X back at the columns of B they were taken from.
        num_cols = cols.getLocalSize()
        Q = PETSc.Mat().createAIJ(
            ((num_cols, None), (B.getLocalSize()[1], None)),
            csr=(np.arange(num_cols + 1, dtype=PETSc.IntType), cols.getIndices(), np.ones(num_cols)),
            comm=comm,
        )
        for a in rest:
            C = nest_view.block(a, "u_s")
            if C is not None:
                CX = C.matMult(X_sparse)
                updates[(a, b)] = CX.matMult(Q)
                CX.destroy()
        for obj in (rows, cols, B_dense, X, X_sparse, Q):
            obj.destroy()
    ksp.destroy()
    blocks, made = _schur_blocks(nest_view, rest, updates)
    return PETSc.Mat().createNest(blocks, comm=comm), blocks, made


SOLID_GROUPS = ("u_s", "u_int")


def condense_solid_exact(nest_view: NestView) -> tuple[PETSc.Mat, list, list[PETSc.Mat], float]:
    """Condense the solid displacement, u_s and u_int, out of the nest Jacobian without inverting
    its mass matrix, by the kinematic relation of the solid.

    The rows of u_s and u_int are rho_s M (du - theta dt dv) on the solid, and those of the
    interface the same since the fluid is restricted out of them. u and v share their shape
    functions, so M_vu = M_u and du = theta dt dv + M^-1 r on the solid dofs: A^-1 B = K, where K
    takes v at the dof of each solid row, times -theta dt, and B = J[solid, v] is the only nonzero
    block of B (the interface rows keep zero entries of the fluid in their pattern). The factor is
    read from the Jacobian, as B[i, v(i)] / A[i, i], which is 0 on the Dirichlet rows. Then
    S = D - C K changes only in the columns of v, and C K is C with its columns moved from u to v at
    the same dofs, so the fill is cell-local: the fluid rows that reach u_int get blocks in v.

    Needs ``nest_view.data["v_rows"]``, as set by the solvers when u and v have the same element.
    Collective. Returns S, as a MatNest over the other groups, its blocks, the matrices made here,
    which the caller destroys with S, and |A K - B| / |B|, which is round-off when the relation
    holds.
    """
    rest = [name for name in nest_view.names if name not in SOLID_GROUPS]
    for g in SOLID_GROUPS:
        for b in rest:
            # The restriction zeroes the fluid entries of the interface rows but keeps their pattern.
            if b != "v" and nest_view.block(g, b) is not None and nest_view.block(g, b).norm() != 0.0:
                raise ValueError(f"The rows of {g} reach {b}, so they are not the kinematic relation alone")
    comm = nest_view.mat.getComm()
    v_is = nest_view.index_sets[nest_view.names.index("v")]
    v_rows, v_start = v_is.getIndices(), nest_view.block("v", "v").getOwnershipRangeColumn()[0]

    K = {}
    for g in SOLID_GROUPS:
        A, B = nest_view.block(g, g), nest_view.block(g, "v")
        # The column of v in the v block, at the dof of each row of g.
        positions = np.searchsorted(v_rows, nest_view.data["v_rows"][g])
        assert np.array_equal(v_rows[positions], nest_view.data["v_rows"][g])
        v_cols = (v_start + positions).astype(PETSc.IntType)
        indptr, cols, values = B.getValuesCSR()
        factor = np.zeros(len(v_cols))
        for i, col in enumerate(v_cols):
            row_cols = cols[indptr[i] : indptr[i + 1]]
            k = np.searchsorted(row_cols, col)
            if k < len(row_cols) and row_cols[k] == col:
                factor[i] = values[indptr[i] + k]
        factor /= A.getDiagonal().array
        K[g] = PETSc.Mat().createAIJ(
            ((len(v_cols), None), (B.getLocalSize()[1], None)),
            csr=(np.arange(len(v_cols) + 1, dtype=PETSc.IntType), v_cols, factor),
            comm=comm,
        )

    # A K - B, with A = J[solid, solid]
    residual, size = 0.0, 0.0
    for g in SOLID_GROUPS:
        AK = nest_view.block(g, "v").duplicate(copy=True)
        AK.scale(-1.0)
        size += AK.norm() ** 2
        for h in SOLID_GROUPS:
            if nest_view.block(g, h) is not None:
                product = nest_view.block(g, h).matMult(K[h])
                AK.axpy(1.0, product, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                product.destroy()
        residual += AK.norm() ** 2
        AK.destroy()

    updates = {}  # (a, v): C_a K
    for a in rest:
        for g in SOLID_GROUPS:
            C = nest_view.block(a, g)
            if C is None:
                continue
            CK = C.matMult(K[g])
            if (a, "v") in updates:
                updates[(a, "v")].axpy(1.0, CK, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
                CK.destroy()
            else:
                updates[(a, "v")] = CK
    for k in K.values():
        k.destroy()
    blocks, made = _schur_blocks(nest_view, rest, updates)
    return PETSc.Mat().createNest(blocks, comm=comm), blocks, made, np.sqrt(residual / size)


def check_condensation(nest_view: NestView, eliminated: list[str], S: PETSc.Mat) -> float:
    """The relative error of a condensation S of the nest Jacobian J that eliminates the groups
    ``eliminated``: for a random x and y = J x, S must map the other groups of x to
    y_rest - J[rest, E] J[E, E]^-1 y_E, where J[E, E] is solved with by MUMPS LU. Collective."""
    J = nest_view.mat
    comm = J.getComm()
    index_sets = dict(zip(nest_view.names, nest_view.index_sets, strict=True))
    rest = [name for name in nest_view.names if name not in eliminated]
    E = PETSc.IS().createGeneral(
        np.sort(np.concatenate([index_sets[name].getIndices() for name in eliminated])).astype(PETSc.IntType),
        comm=comm,
    )
    A = J.createSubMatrix(E, E)
    ksp = _lu(A)
    x, y = J.createVecs()
    x.setRandom()
    J.mult(x, y)
    y_E = y.getSubVector(E)
    w = y_E.duplicate()
    ksp.solve(y_E, w)
    y.restoreSubVector(E, y_E)

    # The vectors of S are plain vectors, with the groups at the index sets of the nest.
    x_rest, Sx = S.createVecs()
    expected = Sx.duplicate()
    S_index_sets, _ = S.getNestISs()
    for name, S_is in zip(rest, S_index_sets, strict=True):
        for full, condensed in ((x, x_rest), (y, expected)):
            sub, condensed_sub = full.getSubVector(index_sets[name]), condensed.getSubVector(S_is)
            sub.copy(condensed_sub)
            if condensed is expected:
                C = J.createSubMatrix(index_sets[name], E)
                Cw = condensed_sub.duplicate()
                C.mult(w, Cw)
                condensed_sub.axpy(-1.0, Cw)
                for obj in (C, Cw):
                    obj.destroy()
            full.restoreSubVector(index_sets[name], sub)
            condensed.restoreSubVector(S_is, condensed_sub)
    S.mult(x_rest, Sx)
    Sx.axpy(-1.0, expected)
    error = Sx.norm() / expected.norm()
    for obj in (E, A, ksp, x, y, w, x_rest, Sx, expected):
        obj.destroy()
    return error


def _report(snes, nest_view, label, eliminated, S, S_blocks, elapsed, extra=""):
    """Check the condensation S and print the error and the nonzeros of J, D and S on rank 0."""
    error = check_condensation(nest_view, eliminated, S)
    rest = [name for name in nest_view.names if name not in eliminated]
    nnz_J = _nonzeros(nest_view.blocks)
    nnz_D = _nonzeros([[nest_view.block(a, b) for b in rest] for a in rest])
    nnz_S = _nonzeros(S_blocks)
    num_dofs = sum(nest_view.index_sets[nest_view.names.index(name)].getSize() for name in eliminated)
    if nest_view.mat.getComm().getRank() == 0:
        print(
            f"  {label} ({num_dofs} dofs) in {elapsed:.3f} s at it {snes.getIterationNumber()}:"
            f" error {error:.1e}{extra}, nonzeros J {human_count(nnz_J)}, D {human_count(nnz_D)},"
            f" S {human_count(nnz_S)}",
            flush=True,
        )


def schur_u_s_jacobian_hook(snes: PETSc.SNES, nest_view: NestView) -> None:
    """Condense u_s out of the Jacobian with ``condense_u_s``, check it with ``check_condensation``
    and print the error and the nonzeros before and after, on rank 0."""
    start = timer()
    S, S_blocks, made = condense_u_s(nest_view)
    elapsed = timer() - start
    _report(snes, nest_view, "Schur condensed u_s", ["u_s"], S, S_blocks, elapsed)
    for obj in (S, *made):
        obj.destroy()


def exact_solid_condensation_hook(snes: PETSc.SNES, nest_view: NestView) -> None:
    """Condense u_s and u_int out of the Jacobian with ``condense_solid_exact``, check it with
    ``check_condensation`` and print the error, |A K - B| / |B| and the nonzeros, on rank 0."""
    start = timer()
    S, S_blocks, made, relation_error = condense_solid_exact(nest_view)
    elapsed = timer() - start
    _report(
        snes,
        nest_view,
        "exactly condensed u_s, u_int",
        list(SOLID_GROUPS),
        S,
        S_blocks,
        elapsed,
        extra=f", |A K - B| / |B| {relation_error:.1e}",
    )
    for obj in (S, *made):
        obj.destroy()


FLUID_ROWS = ("v", "p")
DISPLACEMENT_GROUPS = ("u_s", "u_int", "u_f")


def drop_ale_derivatives(nest_view: NestView) -> float:
    """Zero the derivatives of the momentum and mass equations with respect to the displacement, the
    blocks (v, p) x (u_s, u_int, u_f), in the Jacobian itself, ``nest_view.mat``, which Newton solves
    with, and copy it into the nest again.

    With ``condense_solid=True`` in fsi2_harmonic_fulljacobian_initialguess, the solid stress is a
    function of v, so these blocks hold only the fluid ALE derivatives: those of the geometry, J and
    F^-1, and of the mesh velocity, (u - u_old) / dt. What is left is J_0 of eq. (3) of
    docs/restricted-iterative-solver.md: (v, p) no longer depend on the displacement, and the
    displacement rows, the solid kinematic relation and the mesh equation, are linear, so each Newton
    step solves them exactly for the new v. This is the Jacobian of fsi2_simplified, which solves
    for (v, p) and sets the displacement from v after each linear solve, with the mesh motion of
    the solver. Without ``condense_solid``, the solid stress derivative is dropped too.

    The entries are zeroed as J - D_r J D_c, with D_r and D_c the diagonal masks of the rows of v, p
    and of the columns of u, so the pattern of J is kept. Collective. Returns the Frobenius norm of
    what was dropped relative to that of J.
    """
    J = nest_view.mat
    index_sets = dict(zip(nest_view.names, nest_view.index_sets, strict=True))
    start = J.getOwnershipRange()[0]
    row_mask, col_mask = J.createVecs()
    for mask, names in ((row_mask, FLUID_ROWS), (col_mask, DISPLACEMENT_GROUPS)):
        mask.set(0.0)
        for name in names:
            mask.array[index_sets[name].getIndices() - start] = 1.0
    dropped = J.duplicate(copy=True)
    dropped.diagonalScale(L=row_mask, R=col_mask)
    norm_dropped, norm_J = dropped.norm(), J.norm()
    J.axpy(-1.0, dropped, structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN)
    for obj in (dropped, row_mask, col_mask):
        obj.destroy()
    nest_view.update()
    return norm_dropped / norm_J


def J_0_jacobian_hook(snes: PETSc.SNES, nest_view: NestView) -> None:
    """Make the Jacobian Newton solves with J_0, by ``drop_ale_derivatives``, and print what was
    dropped, on rank 0. With ``condense_solid=True`` in fsi2_harmonic_fulljacobian_initialguess, this
    gives the Newton iteration of fsi2_simplified, with harmonic mesh motion."""
    relative = drop_ale_derivatives(nest_view)
    if nest_view.mat.getComm().getRank() == 0:
        print(
            f"  J_0 at it {snes.getIterationNumber()}: dropped (v, p) x (u_s, u_int, u_f),"
            f" |dropped| / |J| = {relative:.1e}",
            flush=True,
        )
