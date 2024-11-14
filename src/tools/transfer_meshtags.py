"""
Code written by Jørgen Dokken,
https://jsdokken.com/FEniCS-workshop/src/multiphysics/submeshes.html#transfering-meshtags-to-the-sub-mesh
"""


import dolfinx
import numpy as np
import numpy.typing as npt
def transfer_meshtags_to_submesh(
    mesh: dolfinx.mesh.Mesh,
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    sub_vertex_to_parent: npt.NDArray[np.int32],
    sub_cell_to_parent: npt.NDArray[np.int32]
)-> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.

    Args:
        mesh: Mesh containing the meshtags
        entity_tag: The meshtags object to transfer
        submesh: The submesh to transfer the `entity_tag` to
        sub_to_vertex_map: Map from each vertex in `submesh` to the corresponding
            vertex in the `mesh`
        sub_cell_to_parent: Map from each cell in the `submesh` to the corresponding
            entity in the `mesh`
    Returns:
        The entity tag defined on the submesh, and a map from the entities in the
        `submesh` to the entities in the `mesh`.

    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map
