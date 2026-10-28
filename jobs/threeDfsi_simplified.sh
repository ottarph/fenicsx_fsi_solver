#!/bin/bash -l
# Run the 3d FSI benchmark with xfsi_solver.solvers.threeDfsi_simplified.solve() on one eX3 node,
# by default on the medium mesh up to t = 10 with dt = 0.004.
#
#   sbatch jobs/threeDfsi_simplified.sh
#
# The arguments of solve() are set below. Each can be overridden at submission without
# editing this file, e.g. a restart, or a separate short run:
#   sbatch --export=ALL,RESTART=True jobs/threeDfsi_simplified.sh
#   sbatch --export=ALL,RUN_NAME=threeDfsi_simplified_short,T=0.02 -t 0-01:00:00 jobs/threeDfsi_simplified.sh
# Values are inserted as Python expressions: True/False, None, numbers or arithmetic.
#
# SOLVER selects the (v, p) solve: "iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS (the solver's
# default, the K_c options of threeDfsi_c0ip with a BoomerAMG strong threshold of 0.5),
# "mumps_on_solid", MUMPS_ON_SOLID (the same, with the velocity block split again into the solid
# dofs, by MUMPS LU, and the fluid dofs, by GAMG; run with requires_v_sub_IS=True), or "direct",
# DEFAULT_DIRECT_SOLVER_OPTIONS (MUMPS LU).
#
# With INITIAL_STATE_RUN, a new run starts from the latest checkpoint of that earlier run, with
# its own DT, and only reads from the earlier run's directory. Any 3d solver's checkpoints work,
# e.g. those of jobs/threeDfsi_c0ip.sh on the same mesh:
#   sbatch --export=ALL,RUN_NAME=threeDfsi_simplified_fine_dt0.002,MESH=/home/hellanottar/code/xfsi_solver/data/meshes/fsi3d/mesh_sec_fine.xdmf,INITIAL_STATE_RUN=threeDfsi_c0ip_fine_t8,DT=0.002,T=10.0 jobs/threeDfsi_simplified.sh
# Such a run that stopped is resumed with RESTART=True, without INITIAL_STATE_RUN.
#
# Each run gets its own directory, RUN_DIR, which holds the QoIs, checkpoints, VTX output, QoI
# plots and a record of the code version and arguments.
#
# The meshes are not in git: copy data/meshes/fsi3d/mesh_sec_<size>.{xdmf,h5} to the checkout
# on eX3 first.
#
# FEniCSx and PETSc come from the Spack environment fenicsx-v011, as in jobs/threeDfsi_c0ip.sh,
# which describes it and how to check once that the solver's imports resolve there. xfsi_solver
# is imported from the checkout through PYTHONPATH.
#
# The job takes a whole defq node (2 x 32 cores), with a rank on every core, split evenly over the
# two sockets, since the solve is bound by memory bandwidth; --exclusive keeps other jobs off its
# bandwidth. The run time on eX3 is not yet measured; the serial slow test on mesh_sec_coarse took
# 25 to 43 s per step.
#SBATCH -J fsi3d-simplified
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
RUNS=/home/hellanottar/xfsi/runs
RUN_NAME=${RUN_NAME:-threeDfsi_simplified_medium}
RUN_DIR=$RUNS/$RUN_NAME
INITIAL_STATE_RUN=${INITIAL_STATE_RUN:-}

# Arguments of solve()
MESH=${MESH:-$REPO/data/meshes/fsi3d/mesh_sec_medium.xdmf}
T=${T:-10.0}
DT=${DT:-0.004}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-250}
VTX_SAVE_EVERY=${VTX_SAVE_EVERY:-None}
GAMMA_REASSEMBLE=${GAMMA_REASSEMBLE:-0.2}
MAX_QUADRATURE_DEGREE=${MAX_QUADRATURE_DEGREE:-8}
MESH_MOTION_MAX_QUADRATURE_DEGREE=${MESH_MOTION_MAX_QUADRATURE_DEGREE:-8}
SOLVER=${SOLVER:-iterative}
RESTART=${RESTART:-False}

if [[ ! -f $MESH ]]; then
    echo "Mesh $MESH not found; copy it to the checkout first" >&2
    exit 1
fi

REQUIRES_V_SUB_IS=False
case $SOLVER in
    direct) SOLVER_OPTIONS=DEFAULT_DIRECT_SOLVER_OPTIONS ;;
    iterative) SOLVER_OPTIONS=DEFAULT_ITERATIVE_SOLVER_OPTIONS ;;
    mumps_on_solid)
        SOLVER_OPTIONS=MUMPS_ON_SOLID
        REQUIRES_V_SUB_IS=True
        ;;
    *)
        echo "SOLVER must be direct, iterative or mumps_on_solid, not $SOLVER" >&2
        exit 1
        ;;
esac

INITIAL_STATE_DIR=None
if [[ -n $INITIAL_STATE_RUN ]]; then
    if [[ ! -d $RUNS/$INITIAL_STATE_RUN/checkpoints ]]; then
        echo "No checkpoints in $RUNS/$INITIAL_STATE_RUN to start from" >&2
        exit 1
    fi
    INITIAL_STATE_DIR="\"$RUNS/$INITIAL_STATE_RUN/checkpoints\""
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
from xfsi_solver.solvers.threeDfsi_simplified import $SOLVER_OPTIONS, solve

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
    initial_state_dir=$INITIAL_STATE_DIR,
    gamma_reassemble=$GAMMA_REASSEMBLE,
    max_quadrature_degree=$MAX_QUADRATURE_DEGREE,
    mesh_motion_max_quadrature_degree=$MESH_MOTION_MAX_QUADRATURE_DEGREE,
    solver_options=$SOLVER_OPTIONS,
    requires_v_sub_IS=$REQUIRES_V_SUB_IS,
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

# Plot the QoIs of the whole run (with any earlier restarts) into RUN_DIR/figures/<job>, with a
# close-up of [9, 10], where Failer & Richter give the mean and amplitude; they give no time
# series, so there is no reference curve. Run serially after the solve, since every rank of the
# solve would plot.
PLOT=$(cat <<EOF
from xfsi_solver.scripts.plot_qois import QOI_LABELS_3D, plot_run_qois

plot_run_qois(
    "$RUN_DIR/qoi/$RUN_NAME.txt",
    $T,
    qoi_labels=QOI_LABELS_3D,
    reference_file=None,
    closeup_interval=(9.0, 10.0),
    figure_dir="$RUN_DIR/figures/${SLURM_JOB_NAME:-none}_${SLURM_JOB_ID:-none}",
)
EOF
)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
# FFCx JIT cache on node-local disk, one per job
export XDG_CACHE_HOME=/tmp/$USER/fenics-cache-$SLURM_JOB_ID
# InfiniBand (and shared memory on the node) through UCX
export OMPI_MCA_pml=ucx

# srun binds the ranks to the cores Slurm allocated (see jobs/threeDfsi_c0ip.sh)
#
# Plot also after a failed solve, to see how far it got, but still let the job fail with it
status=0
srun --mpi=pmix --cpu-bind=cores python -u -c "$PY" || status=$?

python -u -c "$PLOT"
exit $status
