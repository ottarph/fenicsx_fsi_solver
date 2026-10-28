# Partition weights for the iterative K_c solves: implementation log

Goal: test the partition-side fix proposed in the partition-weights report (the `mwe_partition_*.py`
scripts): weight the dual-graph edges at the solid, so that the beam is cut less and the Jacobi coupling
between ranks of PETSc's parallel SOR, the GAMG smoother on the velocity block, matters less
(notes/fsi3d/implementation-log.md, "Non-convergence on 26 or more ranks"). Open question 1 of the report:
do solid edge weights flatten the FGMRES counts against the rank count, and do they make less damped SOR
(omega closer to 1) robust again?

Everything here runs on DOLFINx main (`fenicsx-dev` and `.venv`, see `CLAUDE.md`): the edge weights are
not in 0.11.

## Set-up (2026-10-28)

- `tools/partitioning.py`: the functions of `mwe_partition_solid_edge_weights.py` (tag matching, the
  weighted graph partitioner), plus `read_mesh(comm, path, solid_edge_weight=None)`, which returns the mesh
  and its cell and facet tags as the solvers read them, and `partition_quality`, which counts the ranks
  with solid cells and the facets on rank boundaries (all, solid-solid, solid-fluid). With
  `solid_edge_weight=None` it is `XDMFFile.read_mesh`, unchanged on both versions; a weight raises
  `NotImplementedError` on 0.11.
- `create_mesh` needs a UFL mesh, not `coordinate_element`, or the mesh has no UFL domain and
  `ufl.Measure` fails; the MWEs never build forms, so they did not notice.
- `threeDfsi_c0ip.solve(partition_solid_edge_weight=...)`, and every run prints its partition at the
  start and in the summary.
- `jobs/threeDfsi_c0ip_partition_experiments.sh`: runs every combination of ranks, weights and SOR omega
  (`fieldsplit_v_mg_levels_pc_sor_omega` through `k_c_options`) one after the other on the workstation,
  from rest or from a checkpoint, and ends with a table of the partitions and the K_c iterations.
- `tests/test_partitioning.py`: on 3 ranks, the tags matched to the input cells agree with
  `read_meshtags`, weight 1 gives the default partition, and a weighted partition keeps all cells and
  tags (tet and hex meshes).

### ParMETIS, not SCOTCH

The MWEs default to SCOTCH. On the coarse tet mesh on 26 ranks, PT-SCOTCH gave a different partition in
every run, with or without weights, and even called directly by `create_mesh` without the wrapper (cut
facets 1958 to 2030 at weight 1). The default partitioner of `create_mesh` in these builds is ParMETIS,
which gave the same partition in every run, and with edge weight 1 exactly the default partition. So
`read_mesh` uses ParMETIS, and the comparison of weight None (or 1) against 10 changes only the weights.

One run with a SCOTCH partition at weight 10 (6 ranks, solid on 2 ranks, 27 solid-solid cuts) stopped in
the first residual of the first step with `DIVERGED_FUNCTION_NANORINF`. Its partition could not be
reproduced; the cell volumes and facet sizes of the C0IP penalty were finite on another SCOTCH partition,
and two more SCOTCH runs at weight 10 converged. Not explained; watch for it in the runs below, where it
would now be reproducible.

### Partitions of the coarse tet mesh (ParMETIS)

15,142 cells, 1,624 solid (10.7%). Ranks with solid cells, and facets on rank boundaries: solid-solid,
solid-fluid, all. Weight 1 gives the default (None) exactly, on all rank counts below.

| ranks | default | weight 10 | weight 100 |
|---|---|---|---|
| 6 | 4; 76, 8, 842 | 3; 54, 0, 948 | 2; 29, 0, 1110 |
| 24 | 11; 172, 24, 2128 | 8; 125, 0, 2150 | 7; 117, 0, 2169 |
| 26 | 11; 171, 23, 2189 | 8; 125, 1, 2191 | 8; 125, 0, 2244 |
| 28 | 11; 167, 15, 2228 | 9; 136, 1, 2233 | 8; 120, 0, 2242 |
| 30 | 14; 180, 17, 2276 | 10; 147, 1, 2341 | 10; 153, 0, 2418 |

Cell imbalance 1.01 to 1.02 throughout. On 24 to 30 ranks the beam is about 3 ranks' worth of cells, so it
cannot be kept on one rank; the weights take it from 11 to 14 ranks down to 7 to 10, cut 20 to 30% fewer
solid-solid facets and almost no interface facets, and leave the total cut within 1 to 6%. Weight 100
gains little over 10 there.

### Smoke tests

`threeDfsi_c0ip` on 6 ranks, coarse tet mesh, omega 0.4:

| start | weight | K_c iterations per solve |
|---|---|---|
| rest, 1 step | None | 32.0 (2 Newton iterations) |
| rest, 1 step | 10 | 30.0 |
| t = 10 (`threeDfsi_mixed_coarse` checkpoint), 1 step | 10 | 22.2 (5 Newton iterations) |

About 10 s per run, most of it set-up.
