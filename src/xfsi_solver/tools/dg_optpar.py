import dolfinx as dfx
import dolfinx.fem.petsc as dfpetsc
import ufl
import basix
import numpy as np

def get_dg_penalty_parameters_triangle(mesh: dfx.mesh.Mesh, a_val: float, p: int):
    """
    Local parameter selection proposed in 
    *Local parameter selection in the C^0 interior penalty method for the biharmonic equation*
    by Bringmann, Carstensen, and Streitberger, 2023.
    """
    assert mesh.basix_cell() == basix.CellType.triangle
    
    h = ufl.FacetArea(mesh)
    vol = ufl.CellVolume(mesh)

    alpha_base = dfx.fem.Constant(mesh, a_val)
    
    # Note that h("+") == h("-"), since h is the facet area, not cell volume.
    sigma = 3.0 * alpha_base * p * (p - 1) / 8.0 * h("+")**2 * 2 * ufl.avg(1 / vol)
    sigma_boundary = 3.0 * alpha_base * p * (p - 1) * h**2 / 2 * (1 / vol)

    return sigma, sigma_boundary

def get_dg_penalty_parameters_quadrilateral(mesh: dfx.mesh.Mesh, a_val: float, p: int, vol, h):
    """
    Local parameter selection proposed in 
    *Local parameter selection in the C^0 interior penalty method for the biharmonic equation*
    by Bringmann, Carstensen, and Streitberger, 2023.
    """
    assert mesh.basix_cell() == basix.CellType.quadrilateral

    alpha_base = dfx.fem.Constant(mesh, a_val)
    
    # Note that h("+") == h("-"), since h is the facet area, not cell volume.
    sigma = alpha_base * (p - 1)**2 * h("+")**2 * 2 * ufl.avg(1 / vol)
    sigma_boundary = 4.0 * alpha_base * (p - 1)**2 * h**2 * (1 / vol)

    return sigma, sigma_boundary


def compute_cell_volumes(mesh: dfx.mesh.Mesh):

    DG0 = dfx.fem.functionspace(mesh, ("DG", 0))
    v = ufl.TestFunction(DG0)
    b = dfx.fem.assemble_vector(dfx.fem.form(v * ufl.dx))

    vol = dfx.fem.Function(DG0, name="cell_volume")
    vol.x.array[:] = b.array[:]
    vol.x.scatter_forward()

    return vol


def compute_facet_areas(mesh):
    # Following the approach in 
    # https://fenicsproject.discourse.group/t/ufl-facetarea-for-quadrilaterals/17451/4

    tdim = mesh.topology.dim
    facets = dfx.mesh.locate_entities(mesh, tdim-1, lambda x: np.full_like(x[0], True, dtype=bool))
    submesh_facets, entity_map, _, _ = dfx.mesh.create_submesh(mesh, tdim-1, facets)

    Ve = dfx.fem.functionspace(submesh_facets, ("DG", 0))
    facet_area_h = dfx.fem.Function(Ve)
    child_facets = np.arange(facet_area_h.x.array.shape[0], dtype=np.int32)

    facet_area_h.x.array[:] = submesh_facets.h(submesh_facets.geometry.dim - 1, child_facets)
    facet_area_h.x.scatter_forward()

    return facet_area_h, entity_map



def main():

    import mpi4py.MPI as MPI
    import matplotlib.pyplot as plt

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    N = 16
    mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=dfx.mesh.CellType.quadrilateral)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    x0 = mesh.geometry.x * 1.0

    mesh.geometry.x[:, 0] += 0.03 * np.sin(2 * np.pi * x0[:, 1]) + 0.3 * x0[:,1]
    mesh.geometry.x[:, 1] += 0.03 * np.sin(1 * np.pi * x0[:, 0])

    LIMS = [-0.1, 1.5, -0.1, 1.3]


    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts

    print(f"r{mesh.comm.rank}: {num_cells_local = }", flush=True)
    MPI.COMM_WORLD.Barrier()

    vol = compute_cell_volumes(mesh)

    # 1. Extract volumes and set up the colormap / normalizer
    volumes = vol.x.array[:]
    # Normalize maps your min and max volume to a 0.0 - 1.0 scale
    min_vol = MPI.COMM_WORLD.allreduce(volumes.min(), op=MPI.MIN)
    max_vol = MPI.COMM_WORLD.allreduce(volumes.max(), op=MPI.MAX)
    norm = mcolors.Normalize(vmin=min_vol, vmax=max_vol)
    cmap = cm.viridis  # You can change this to 'plasma', 'inferno', 'coolwarm', etc.

    cycle = np.array([0, 1, 3, 2, 0])
    plt.figure()
    for c in range(num_cells_local):
        cell_vertices = mesh.geometry.dofmap[c,:]
        cell_coords = mesh.geometry.x[cell_vertices]
        
        # 2. Get the specific color for this cell's volume
        cell_color = cmap(norm(volumes[c]))
        
        # Plot the cell edges
        plt.plot(cell_coords[cycle,0], cell_coords[cycle,1], "k-")
        
        # 3. Use plt.fill to color the polygon interior
        plt.fill(cell_coords[cycle,0], cell_coords[cycle,1], color=cell_color, alpha=0.8)

    # 4. (Optional) Add a colorbar so you know what the colors mean
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([]) # Required for older matplotlib versions
    plt.colorbar(sm, ax=plt.gca(), label="Cell Volume")

    plt.xlim(*LIMS[:2])
    plt.ylim(*LIMS[2:])
    sname = f"figures/cellfigs/bcells_r{mesh.comm.rank}.png" if mesh.comm.size > 1 else "figures/cellfigs/bcells.png"
    plt.savefig(sname, dpi=300)



    facets = dfx.mesh.locate_entities(mesh, mesh.topology.dim - 1, lambda x: np.full_like(x[0], True, dtype=bool))
    submesh_facets, *_ = dfx.mesh.create_submesh(mesh, mesh.topology.dim-1, facets)

    fa, entity_map = compute_facet_areas(mesh)

    facet_lengths = fa.x.array[:]
    min_len = MPI.COMM_WORLD.allreduce(facet_lengths.min(), op=MPI.MIN)
    max_len = MPI.COMM_WORLD.allreduce(facet_lengths.max(), op=MPI.MAX)


    sub_norm = mcolors.Normalize(vmin=min_len, vmax=max_len)
    sub_cmap = cm.plasma  # You can change this to 'plasma', 'inferno', 'coolwarm', etc.


    plt.figure()

    submesh_cells = submesh_facets.geometry.dofmap
    submesh_x = submesh_facets.geometry.x

    for c in range(submesh_cells.shape[0]):
        sub_cell_vertices = submesh_cells[c,:]
        sub_cell_coords = submesh_x[sub_cell_vertices]
        
        # 2. Get the specific color for this cell's volume
        sub_cell_color = sub_cmap(sub_norm(fa.x.array[c]))
        
        # Plot the cell edges
        plt.plot(sub_cell_coords[[0,1],0], sub_cell_coords[[0,1],1], "-",
                 color=sub_cell_color, alpha=0.8)
        
    # Add the colorbar, remembering the ax argument!
    sm_sub = cm.ScalarMappable(cmap=sub_cmap, norm=sub_norm)
    sm_sub.set_array([])
    plt.colorbar(sm_sub, ax=plt.gca(), label="Facet Length")


    plt.xlim(*LIMS[:2])
    plt.ylim(*LIMS[2:])
    sname = f"figures/cellfigs/bfacets_r{mesh.comm.rank}.png" if mesh.comm.size > 1 else "figures/cellfigs/bfacets.png"
    plt.savefig(sname, dpi=300)


    ds, dS = ufl.Measure("ds", domain=mesh),  ufl.Measure("dS", domain=mesh)

    CG1 = dfx.fem.functionspace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(CG1), ufl.TestFunction(CG1)

    a = ufl.inner(fa("+") * ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS
    # form_a = dfx.fem.form(a) # KO
    form_a = dfx.fem.form(a, entity_maps=[entity_map])


    scal_p = MPI.COMM_WORLD.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(
        fa("+") * dS
    , entity_maps=[entity_map])), op=MPI.SUM)
    scal_m = MPI.COMM_WORLD.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(
        fa("-") * dS
    , entity_maps=[entity_map])), op=MPI.SUM)
    assert abs(scal_p - scal_m) < 1e-12, f"Facet area integrals on the + and - sides should be equal, but got {scal_p} and {scal_m}."

    scal_bd = MPI.COMM_WORLD.allreduce(dfx.fem.assemble_scalar(dfx.fem.form(
        fa("+") * ds
    , entity_maps=[entity_map])), op=MPI.SUM)
    if mesh.comm.rank == 0:
        print(f"{scal_bd = }")
        print(f"{scal_p = }")


    sigma, sigma_boundary = get_dg_penalty_parameters_quadrilateral(mesh, a_val=2.0, p=2, vol=vol, h=fa)

    a = ufl.inner(sigma * ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * dS
    # form_a = dfx.fem.form(a) # KO
    form_a = dfx.fem.form(a, entity_maps=[entity_map])

    b = ufl.inner(sigma_boundary * ufl.dot(ufl.grad(u), ufl.FacetNormal(mesh)), ufl.dot(ufl.grad(v), ufl.FacetNormal(mesh))) * ds
    form_b = dfx.fem.form(b, entity_maps=[entity_map])

    return


if __name__ == "__main__":
    main()

