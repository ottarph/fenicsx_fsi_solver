#!/bin/bash -l
# Run the 3d FSI benchmark with xfsi_solver.solvers.threeDfsi_c0ip.solve() on the fine mesh on
# one eX3 node, up to t = 8 with dt = 0.004. As in Failer & Richter (Sect. 5.4), its checkpoint
# at t = 8 is the start of the runs over [8, 10] with dt = 0.004, 0.002 and 0.001, so do not
# restart this run beyond t = 8: that would replace the checkpoint.
#
#   sbatch jobs/threeDfsi_c0ip.sh
#
# The arguments of solve() are set below. Each can be overridden at submission without
# editing this file, e.g. a restart, or a separate short run:
#   sbatch --export=ALL,RESTART=True jobs/threeDfsi_c0ip.sh
#   sbatch --export=ALL,RUN_NAME=threeDfsi_c0ip_fine_short,T=0.02 -t 0-01:00:00 jobs/threeDfsi_c0ip.sh
# Values are inserted as Python expressions: True/False, None, numbers or arithmetic.
#
# Each run gets its own directory, RUN_DIR, which holds the QoIs, checkpoints, VTX output and
# a record of the code version and arguments.
#
# The meshes are not in git: copy data/meshes/fsi3d/mesh_sec_fine.{xdmf,h5} to the checkout
# on eX3 first.
#
# FEniCSx and PETSc come from the Spack environment fenicsx-v011
# (https://github.com/finsberg/ex3-spack-envs/tree/main/fenicsx-v0.11): OpenMPI with UCX for
# InfiniBand and Slurm support, PETSc 3.24 with MUMPS and hypre. xfsi_solver is not installed
# there; it is imported from the checkout through PYTHONPATH. The environment is built for
# x86_64_v3 (AVX2), which srl-login3 lacks, so check once on srl-login1 or on a compute node
# (srun), after activating the environment as below, that everything it needs imports:
#   PYTHONPATH=/home/hellanottar/code/xfsi_solver/src python -c \
#       "import xfsi_solver.solvers.threeDfsi_c0ip, xfsi_solver.scripts.plot_qois, io4dolfinx, scifem"
#
# The job takes a whole defq node (2 x 32 cores, 2 TB), with a rank on every core, split evenly
# over the two sockets, since the solve is bound by memory bandwidth; --exclusive keeps other
# jobs off its bandwidth. The fine mesh has about 3.0M unknowns, 47k per rank on 64 ranks;
# estimated 8 to 12 h for the 2000 steps to t = 8, not yet measured.
#SBATCH -J fsi3d-c0ip
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-socket=32
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH -t 2-00:00:00
#SBATCH -o /home/hellanottar/xfsi/logs/%x_%j.txt

# Before set -u, since the Spack shell setup may read unset variables
module use /cm/shared/spack-modules/modulefiles
module load spack
spack env activate fenicsx-v011

set -euo pipefail

REPO=/home/hellanottar/code/xfsi_solver
RUN_NAME=${RUN_NAME:-threeDfsi_c0ip_fine_t8}
RUN_DIR=/home/hellanottar/xfsi/runs/$RUN_NAME

# Arguments of solve()
MESH=${MESH:-$REPO/data/meshes/fsi3d/mesh_sec_fine.xdmf}
T=${T:-8.0}
DT=${DT:-0.004}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-250}
VTX_SAVE_EVERY=${VTX_SAVE_EVERY:-None}
GAMMA_REASSEMBLE=${GAMMA_REASSEMBLE:-0.2}
DIRECT_K_C_SOLVE=${DIRECT_K_C_SOLVE:-False}
MAX_QUADRATURE_DEGREE=${MAX_QUADRATURE_DEGREE:-8}
C0IP_A_VAL=${C0IP_A_VAL:-2.0}
MESH_MOTION_TELESCOPE_REDUCTION_FACTOR=${MESH_MOTION_TELESCOPE_REDUCTION_FACTOR:-None}
RESTART=${RESTART:-False}

if [[ ! -f $MESH ]]; then
    echo "Mesh $MESH not found; copy it to the checkout first" >&2
    exit 1
fi

# A fresh run must not overwrite an earlier one; a restart needs its checkpoints.
if [[ $RESTART == True ]]; then
    if [[ ! -d $RUN_DIR/checkpoints ]]; then
        echo "No checkpoints in $RUN_DIR to restart from" >&2
        exit 1
    fi
elif [[ -e $RUN_DIR ]]; then
    echo "$RUN_DIR exists; set RESTART=True or choose another RUN_NAME" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"/{pv,qoi,checkpoints}
cd "$RUN_DIR"

# Every rank runs the call from -c; a script on stdin would only reach rank 0
PY=$(cat <<EOF
from xfsi_solver.solvers.threeDfsi_c0ip import solve

solve(
    mesh_path="$MESH",
    T=$T,
    dt_val=$DT,
    output_path="$RUN_DIR/pv/$RUN_NAME.bp",
    output_path_p="$RUN_DIR/pv/${RUN_NAME}_p.bp",
    qoi_path="$RUN_DIR/qoi/$RUN_NAME.txt",
    checkpoint_dir="$RUN_DIR/checkpoints",
    checkpoint_every=$CHECKPOINT_EVERY,
    vtx_save_every=$VTX_SAVE_EVERY,
    restart=$RESTART,
    gamma_reassemble=$GAMMA_REASSEMBLE,
    direct_k_c_solve=$DIRECT_K_C_SOLVE,
    max_quadrature_degree=$MAX_QUADRATURE_DEGREE,
    c0ip_a_val=$C0IP_A_VAL,
    mesh_motion_telescope_reduction_factor=$MESH_MOTION_TELESCOPE_REDUCTION_FACTOR,
)
EOF
)

# Record which code and arguments this run (or restart) used
{
    echo "=== job ${SLURM_JOB_ID:-none} on $(hostname), $(date -Iseconds), ${SLURM_NTASKS:-?} ranks"
    git -C "$REPO" log -1 --format="%H %s"
    git -C "$REPO" status --short
    echo "$PY"
} >> run_info.txt
echo "$PY"

# Plot the QoIs of the whole run (with any earlier restarts) into RUN_DIR/figures. Failer &
# Richter give only the mean and amplitude over [9, 10], not a time series, so there is no
# reference curve. Run serially after the solve, since every rank of the solve would plot.
PLOT=$(cat <<EOF
from xfsi_solver.scripts.plot_qois import plot_qoi

for qoi, qoi_label in zip(
    ["drag", "lift", "B_x", "B_y", "B_z"],
    ["Drag", "Lift", "\$x\$-displacement of B [m]", "\$y\$-displacement of B [m]", "\$z\$-displacement of B [m]"],
):
    plot_qoi(
        qoi,
        qoi_file="$RUN_DIR/qoi/$RUN_NAME.txt",
        save_path="$RUN_DIR/figures/${SLURM_JOB_NAME:-none}_${SLURM_JOB_ID:-none}_" + qoi + ".pdf",
        label=None,
        title=None,
        xlabel="Time [s]",
        ylabel=qoi_label,
        time_interval=[0, $T],
    )
EOF
)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
# FFCx JIT cache on node-local disk, one per job
export XDG_CACHE_HOME=/tmp/$USER/fenics-cache-$SLURM_JOB_ID
# InfiniBand (and shared memory on the node) through UCX
export OMPI_MCA_pml=ucx

# Slurm does not confine jobs to their cores (at least on defq), so mpirun's own binding may
# put the ranks on other jobs' cores and fill one socket first. srun binds them to the cores
# Slurm allocated, which --ntasks-per-socket splits over the sockets.
#
# Plot also after a failed solve, to see how far it got, but still let the job fail with it
status=0
srun --mpi=pmix --cpu-bind=cores python -u -c "$PY" || status=$?

python -u -c "$PLOT"
exit $status
