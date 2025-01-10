# Copyright (C) 2023-2025 Jørgen S. Dokken, Ottar Hellan
#
# SPDX-License-Identifier: MIT

# Adapted https://github.com/jorgensd/dolfinx-tutorial/issues/158
# by Jørgen S. Dokken.


import dolfinx as dfx
import ufl
import numpy as np

def create_consistent_interior_facet_measure(mesh: dfx.mesh.Mesh, facet_tags: dfx.mesh.MeshTags,
                                             cell_tags: dfx.mesh.MeshTags, facet_tag_to_integrate: int, side_marker_cell_tag: int,
                                             new_tag: int) -> ufl.Measure:
    
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Get number of cells on process
    cell_map = mesh.topology.index_map(tdim)
    facet_map = mesh.topology.index_map(fdim)

    f_to_c = mesh.topology.connectivity(fdim, tdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)

    facets_to_integrate = facet_tags.find(facet_tag_to_integrate)

    # Compute integration entities for a single facet of a cell.
    # Each facet is represented as a tuple (cell_index, local_facet_index), where cell_index is local to process
    # local_facet_index is the local indexing of a facet for a given cell
    integration_entities = []
    for i, facet in enumerate(facets_to_integrate):
        
        # Only loop over facets owned by the process to avoid duplicate integration
        if facet >= facet_map.size_local:
            continue

        # Find cells connected to facet
        cells = f_to_c.links(facet)
        
        # Get value of cells
        marked_cells = cell_tags.values[cells]
        # Get the cell marked with side_marker_cell_tag
        correct_cell_list = np.flatnonzero(marked_cells == side_marker_cell_tag)
        assert len(correct_cell_list) == 1

        correct_cell = correct_cell_list[0]


        # Get local index of facet
        local_facets = c_to_f.links(cells[correct_cell])
        local_index_list = np.flatnonzero(local_facets == facet)
        assert len(local_index_list) == 1

        local_index = local_index_list[0]

        # Append integration entities
        integration_entity = (cells[correct_cell], local_index)
        integration_entities.extend(integration_entity)


    # Basically a flattened array of integration entities, without trouble of array-of-tuples.
    integration_entities = np.asarray(integration_entities, dtype=np.int32)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=[(new_tag, integration_entities)])


    return ds



if __name__ == "__main__":
    from mpi4py import MPI
    from pathlib import Path
    msh_path = Path("data/meshes/fsi2/mesh.xdmf")

    with dfx.io.XDMFFile(MPI.COMM_WORLD, msh_path, "r") as xdmf:
        msh = xdmf.read_mesh()
        msh.topology.create_connectivity(1, 2)
        cell_tags = xdmf.read_meshtags(msh, name= "Cell tags")
        facet_tags = xdmf.read_meshtags(msh, name= "Facet tags")

    solid_fluid_int_tag = 11
    new_tag = 14
    solid_tag = 1
    fluid_tag = 2


    my_ds_solid = create_consistent_interior_facet_measure(msh, facet_tags, cell_tags, 
                                    solid_fluid_int_tag, solid_tag, new_tag)
    print(f"{dfx.fem.assemble_scalar(dfx.fem.form( 
        ufl.inner( dfx.fem.Constant(msh, (1.0, 0.0)), ufl.FacetNormal(msh) ) * my_ds_solid(new_tag) 
    )):.3e}")
    print(f"{dfx.fem.assemble_scalar(dfx.fem.form( 
        ufl.inner( dfx.fem.Constant(msh, (0.0, 1.0)), ufl.FacetNormal(msh) ) * my_ds_solid(new_tag) 
    )):.3e}")
    
    print()

    my_ds_fluid = create_consistent_interior_facet_measure(msh, facet_tags, cell_tags, 
                                    solid_fluid_int_tag, fluid_tag, new_tag)
    print(f"{dfx.fem.assemble_scalar(dfx.fem.form( 
        ufl.inner( dfx.fem.Constant(msh, (1.0, 0.0)), ufl.FacetNormal(msh) ) * my_ds_fluid(new_tag) 
    )):.3e}")
    print(f"{dfx.fem.assemble_scalar(dfx.fem.form( 
        ufl.inner( dfx.fem.Constant(msh, (0.0, 1.0)), ufl.FacetNormal(msh) ) * my_ds_fluid(new_tag) 
    )):.3e}")


    eps = 1e-6
    x = ufl.SpatialCoordinate(msh)
    correct_normal = ufl.conditional(x[1] > 0.21-eps, ufl.as_vector([0.0, 1.0]), 
                        ufl.conditional(x[1] < 0.19+eps, ufl.as_vector([0.0, -1.0]), 
                            ufl.as_vector([1.0, 0.0])))
    
    int_right = 0.6
    r = 0.05
    h = 0.02
    C_y = C_x = 0.2
    # 0.5*h = r * np.sin(theta)
    theta = np.arcsin(0.5*h/r)
    int_left = C_x + np.cos(theta) * r

    correct_result = 2 * (int_right - int_left) + h

    print(f"{correct_result = :.3e}")
    print(f"my_ds_solid: {dfx.fem.assemble_scalar(dfx.fem.form( 
        ufl.inner( correct_normal, ufl.FacetNormal(msh) ) * my_ds_solid(new_tag)
    )):.3e}")
    print(f"my_ds_fluid: {dfx.fem.assemble_scalar(dfx.fem.form( 
        ufl.inner( correct_normal, ufl.FacetNormal(msh) ) * my_ds_fluid(new_tag)
    )):.3e}")

    fluid_mesh, fluid_ent_map, *_ = dfx.mesh.create_submesh(msh, 2, cell_tags.find(fluid_tag))
    integration_ent = np.full(msh.topology.index_map(2).size_local, -1, dtype=np.int32)
    integration_ent[fluid_ent_map] = np.arange(fluid_ent_map.shape[0], dtype=np.int32)
    U_fluid = dfx.fem.functionspace(msh, ("CG", 1, (2, )))
    u_fluid = dfx.fem.Function(U_fluid)
    u_fluid.x.array[:] = 1.0
    print(dfx.fem.assemble_scalar(dfx.fem.form(
        ufl.inner(ufl.FacetNormal(msh), u_fluid) * my_ds_fluid(new_tag), entity_maps={fluid_mesh: integration_ent},
    )))



