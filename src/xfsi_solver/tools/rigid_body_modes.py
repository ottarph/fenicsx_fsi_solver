import dolfinx
from petsc4py import PETSc


def rigid_body_modes(space):

    if space.value_shape == (2,):
        """Orthonormal rigid body modes of a 2D vector space: two translations and a rotation."""
        index_map, bs = space.dofmap.index_map, space.dofmap.index_map_bs
        num_dofs = index_map.size_local + index_map.num_ghosts
        x = space.tabulate_dof_coordinates()[:num_dofs]
        basis = [dolfinx.la.vector(index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(3)]
        b = [w.array.reshape(-1, bs) for w in basis]
        b[0][:, 0] = 1.0
        b[1][:, 1] = 1.0
        b[2][:, 0], b[2][:, 1] = -x[:, 1], x[:, 0]
        dolfinx.la.orthonormalize(basis)
        return PETSc.NullSpace().create(vectors=[dolfinx.la.petsc.create_vector_wrap(w) for w in basis])

    elif space.value_shape == (3,):
        """Orthonormal rigid body modes of a 3D vector space: three translations and three rotations."""
        index_map, bs = space.dofmap.index_map, space.dofmap.index_map_bs
        num_dofs = index_map.size_local + index_map.num_ghosts
        x = space.tabulate_dof_coordinates()[:num_dofs]
        basis = [dolfinx.la.vector(index_map, bs=bs, dtype=PETSc.ScalarType) for _ in range(6)]
        b = [w.array.reshape(-1, bs) for w in basis]
        b[0][:, 0] = 1.0
        b[1][:, 1] = 1.0
        b[2][:, 2] = 1.0
        b[3][:, 0], b[3][:, 1] = -x[:, 1], x[:, 0]  # rotation about the z-axis
        b[4][:, 1], b[4][:, 2] = -x[:, 2], x[:, 1]  # rotation about the x-axis
        b[5][:, 2], b[5][:, 0] = -x[:, 0], x[:, 2]  # rotation about the y-axis
        dolfinx.la.orthonormalize(basis)
        return PETSc.NullSpace().create(vectors=[dolfinx.la.petsc.create_vector_wrap(w) for w in basis])

    else:
        raise ValueError()
