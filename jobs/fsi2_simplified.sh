#!/bin/bash
# Run the FSI2 benchmark with xfsi_solver.solvers.fsi2_simplified.solve() on one eX3 node.
#
#   sbatch jobs/fsi2_simplified.sh
#
# The arguments of solve() are set below. Each can be overridden at submission without
# editing this file, e.g. a restart, or a separate short run:
#   sbatch --export=ALL,RESTART=True jobs/fsi2_simplified.sh
#   sbatch --export=ALL,RUN_NAME=fsi2_simplified_short,T=0.1 jobs/fsi2_simplified.sh
# Values are inserted as Python expressions: True/False, None, numbers or arithmetic.
#
# SOLVER selects the (v, p) solve: "direct", DEFAULT_DIRECT_SOLVER_OPTIONS (MUMPS LU, the
# solver's default), or "iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS (FGMRES with a Schur
# complement fieldsplit).
#
# MAX_QUADRATURE_DEGREE is 6 by default: down to that cap the QoIs did not change in the 7 printed
# digits from the uncapped run, at 22% less time per step (notes/restricted-cleanup,
# "Quadrature cap in the twice simplified solver"). None runs without a cap.
#
# With INITIAL_STATE_RUN, a new run starts from the latest checkpoint of that earlier run, with
# its own DT, and only reads from the earlier run's directory:
#   sbatch --export=ALL,RUN_NAME=fsi2_simplified_from16,INITIAL_STATE_RUN=fsi2_simplified,T=18.0 jobs/fsi2_simplified.sh
# Such a run that stopped is resumed with RESTART=True, without INITIAL_STATE_RUN.
#
# Each run gets its own directory, RUN_DIR, which holds the QoIs, checkpoints, VTX output, QoI
# plots and a record of the code version and arguments.
#SBATCH -J fsi2-simplified
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --hint=nomultithread
#SBATCH -t 0-06:00:00
#SBATCH -o /home/hellanottar/xfsi/logs/%x_%j.txt

set -euo pipefail

REPO=/home/hellanottar/code/xfsi_solver
RUNS=/home/hellanottar/xfsi/runs
RUN_NAME=${RUN_NAME:-fsi2_simplified}
RUN_DIR=$RUNS/$RUN_NAME
INITIAL_STATE_RUN=${INITIAL_STATE_RUN:-}

# Arguments of solve()
MESH=${MESH:-$REPO/data/meshes/fsi2/mesh_sec.xdmf}
T=${T:-15.0}
DT=${DT:-0.0025}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-400}
VTX_SAVE_EVERY=${VTX_SAVE_EVERY:-None}
GAMMA_REASSEMBLE=${GAMMA_REASSEMBLE:-0.2}
MAX_QUADRATURE_DEGREE=${MAX_QUADRATURE_DEGREE:-6}
SOLVER=${SOLVER:-direct}
RESTART=${RESTART:-False}

if [[ ! -f $MESH ]]; then
    echo "Mesh $MESH not found" >&2
    exit 1
fi

case $SOLVER in
    direct) SOLVER_OPTIONS=DEFAULT_DIRECT_SOLVER_OPTIONS ;;
    iterative) SOLVER_OPTIONS=DEFAULT_ITERATIVE_SOLVER_OPTIONS ;;
    *)
        echo "SOLVER must be direct or iterative, not $SOLVER" >&2
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
from xfsi_solver.solvers.fsi2_simplified import $SOLVER_OPTIONS, solve

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
    solver_options=$SOLVER_OPTIONS,
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

# Plot the QoIs of the whole run (with any earlier restarts) against the FSI2 reference into
# RUN_DIR/figures/<job>. Run serially after the solve, since every rank of the solve would plot.
PLOT=$(cat <<EOF
from xfsi_solver.scripts.plot_qois import plot_run_qois

plot_run_qois(
    "$RUN_DIR/qoi/$RUN_NAME.txt",
    $T,
    reference_file="$REPO/data/fsi2_reference.txt",
    figure_dir="$RUN_DIR/figures/${SLURM_JOB_NAME:-none}_${SLURM_JOB_ID:-none}",
)
EOF
)

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Single-node job: let Hydra start the ranks itself instead of through srun
export HYDRA_BOOTSTRAP=fork

# Plot also after a failed solve, to see how far it got, but still let the job fail with it
status=0
~/miniforge3/bin/conda run --no-capture-output -n xfsi_solver \
    mpiexec -n "$SLURM_NTASKS" -bind-to core \
    python -u -c "$PY" || status=$?

~/miniforge3/bin/conda run --no-capture-output -n xfsi_solver python -u -c "$PLOT"
exit $status
