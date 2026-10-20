# eX3 resources for xfsi_solver runs

Brief for agents preparing or running FSI simulations on the eX3 cluster (Simula).
Sources: <https://www.ex3.simula.no/resources> and Slurm output from the login node `srl-login1`,
checked 2026-10-19. Node states change constantly; re-check before planning a run.
Partition rules and software are on the eX3 wiki: <http://wiki.ex3.simula.no/>.

## Workload assumptions

- 3D monolithic ALE FSI with FEniCSx/PETSc, Krylov + fieldsplit. Only the scalar mesh-motion
  block (C0IP biharmonic component or Laplace) uses MUMPS, optionally telescoped.
- This is memory-bandwidth bound: pick nodes by memory channels per core and core count, not by RAM.
- CPU only. The GPUs and accelerators on eX3 are not used.

## Relevant partitions

| Partition  | Nodes     | CPU (sockets × cores)          | Phys. cores | RAM    | Notes |
|------------|-----------|--------------------------------|-------------|--------|-------|
| `genoaxq`  | n021      | 2 × 96 EPYC 9684X (Genoa-X)    | 192         | ?      | Best single node: 12 DDR5 channels/socket, ~1.1 GB L3/socket. 14-day limit. |
| `milanq`   | n013–016  | 2 × 64 EPYC 7763 (Milan)       | 128         | 2 TB   | Best for multi-node runs. Also hosts GPUs (n013–014: 2×A100, n015–016: 2×MI210), so GPU jobs hold a few cores. No time limit. |
| `defq`     | n001–004  | 2 × 32 (probably EPYC 7601 Naples) | 64      | 2 TB (page) | Default partition. Older; fine for sweeps. Hosts MI50/MI100 GPUs. |
| `xeonmaxq` | n022      | 2 × 56 Xeon Max 9480 (HBM)     | 112         | HBM + ? | Worth benchmarking for bandwidth-bound solves. Was down on 2026-10-19. |

Confirmed on n014: 2 NUMA domains (one per socket), logical CPUs n and n+128 are the same physical core,
~1.9 TB RAM available. Not useful: `rome16q`, `slowq`, `fpgaq`, `xeongold32q` (login/admin),
`virtq`, `nasq`, and all GPU/accelerator queues. ARM queues (`armq`, `huaq`, `aarchq`) would need
a separate linux-aarch64 conda env.

Compared with the local workstation (Threadripper 3970X, 32 cores, 4 DDR4 channels, ~100 GB/s,
125 GB RAM), a Milan node has ~4× the memory bandwidth, the Genoa-X node ~9× (spec-sheet estimates).

## Slurm configuration quirks

- `select/cons_tres` + `CR_CORE`: nodes are shared between jobs, the unit is a physical core, but each
  hardware thread counts as one "CPU". Without `--hint=nomultithread`, `--ntasks=128` can land on 64
  cores with 2 ranks each. Always use `--hint=nomultithread` and `srun --cpu-bind=cores`.
- Memory is not tracked (`RealMemory = 1`, no `_MEMORY` in `SelectTypeParameters`, mem limits
  `UNLIMITED`). Do not pass `--mem` (the job may never schedule). Other jobs on a shared node can use
  RAM and bandwidth; use `--exclusive` for timing runs and very large meshes.
- Node features are only `location=local`: select hardware by partition (`-p`) or node (`-w`), not
  `--constraint`.
- Keep each run on one node type. dolfinx balances cells per rank and Krylov iterations synchronise,
  so mixed node types run at the speed of the slowest.
- Spread ranks evenly over both sockets (default placement alternates; `--ntasks-per-socket=N` to be
  explicit). Fewer ranks than cores per node (e.g. 112 on Milan) is reasonable for a bandwidth-bound solver.

## Storage and environment

- Home (`/home/hellanottar`, NFS `master:/home`, 313 TB, 16% used on 2026-10-19), visible from all
  nodes: `miniforge3` with the conda env and code checkouts. For now, also put meshes, VTX `.bp`
  output, QoIs, checkpoints and Slurm logs here, one directory per run: runs in
  `/home/hellanottar/xfsi/runs/<run>`, logs in `/home/hellanottar/xfsi/logs`.
- Do not use D1 for now. `~/D1 -> /global/D1/homes/hellanottar` is shared global storage (BeeGFS,
  ~1 PB, 97% full on 2026-10-19). On 2026-10-19, appends to an existing file there failed from n003
  with `Remote I/O error` while new files could still be written. PETSc then recursed in its error
  handler, writing the error to the same failing stdout, and the job hung with growing memory.
  Move outputs back to D1 only once the eX3 admins have fixed that. D1 is not backed up
  and not for long-term storage.
- Use full paths (`/home/hellanottar/...`) in `#SBATCH` lines; `~` may not be expanded.
- The conda env uses conda-forge MPICH 5.0.1 (libfabric). Across nodes, check it uses InfiniBand, not
  TCP; alternatively use the cluster MPI via conda-forge's `external` MPI builds. If `srun` cannot
  wire up the ranks, try `srun --mpi=pmi2` or `mpirun` inside the job.
- Warm the FFCx JIT cache (`~/.cache/fenics`) with a short single-node run before large multi-node
  jobs. Do not add `-march=native` to JIT flags unless each node type gets its own cache directory.
- Generate meshes (gmsh on one rank) once beforehand; the solvers read XDMF in parallel.

## Useful commands (login node)

```bash
sinfo -s                                    # partitions and node counts (A/I/O/T)
sinfo -p genoaxq,milanq,defq,xeonmaxq -N -o "%8N %10P %8T %16C %4X %4Y %4Z %9m %25f %G"
sinfo -R                                    # why nodes are down
squeue -w n013,n014,n016,n021 -o "%8i %10u %9P %5C %12M %12l %b"
sbatch --test-only job.sh                   # estimated start time
```

## Job script template

```bash
#!/bin/bash
#SBATCH -p milanq
#SBATCH -N 2
#SBATCH --ntasks-per-node=112
#SBATCH --hint=nomultithread
#SBATCH -t 2-00:00:00
#SBATCH -o /home/hellanottar/xfsi/logs/%j.txt
##SBATCH --exclusive              # timing runs or very large meshes

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --cpu-bind=cores python -u <script> <args>
```

## Open questions

- RAM on the Genoa-X node and HBM mode on the Xeon Max node (`srun ... free -g; numactl -H`).
- Which MPI/modules eX3 recommends, and whether conda MPICH uses InfiniBand.
- Where the iterative solve saturates bandwidth: benchmark a short 3D case at several ranks per socket
  on each node type with `-log_view`.
