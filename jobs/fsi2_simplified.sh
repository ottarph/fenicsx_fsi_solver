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
# solver's default), "iterative", DEFAULT_ITERATIVE_SOLVER_OPTIONS (FGMRES with a Schur
# complement fieldsplit, GAMG on the velocity block), or "mumps_on_solid", MUMPS_ON_SOLID (the
# same, with the velocity block split again into the solid dofs, by MUMPS LU, and the fluid dofs,
# by GAMG; run with requires_v_sub_IS=True).
#
# MAX_QUADRATURE_DEGREE is 6 by default: down to that cap the QoIs did not change in the 7 printed
# digits from the uncapped run, at 22% less time per step (32bd021:notes/restricted-cleanup,
# "Quadrature cap in the twice simplified solver"). None runs without a cap.
#
# With INITIAL_STATE_RUN, a new run starts from the latest checkpoint of that earlier run, with
# its own DT, and only reads from the earlier run's directory:
#   sbatch --export=ALL,RUN_NAME=fsi2_simplified_from16,INITIAL_STATE_RUN=fsi2_simplified,T=18.0 jobs/fsi2_simplified.sh
# Such a run that stopped is resumed with RESTART=True, without INITIAL_STATE_RUN.
#
# SNAPSHOT_TIMES keeps the state at the time steps closest to these times, each in
# RUN_DIR/snapshots/t<t>, named by the time reached (see xfsi_solver.tools.checkpoint.SnapshotWriter),
# besides the restart checkpoints every CHECKPOINT_EVERY steps. Separate the times by spaces, since
# sbatch ends a variable of --export at a comma:
#   sbatch --export=ALL,SNAPSHOT_TIMES="0.2 0.4 0.6" jobs/fsi2_simplified.sh
# A restart needs the same SNAPSHOT_TIMES to write the snapshots after its checkpoint. With
# INITIAL_STATE_SNAPSHOT, e.g. t0.4000, a new run starts from that snapshot of INITIAL_STATE_RUN
# instead of its latest checkpoint.
#
# Each run gets its own directory, RUN_DIR, which holds the QoIs, the Newton log (one row per
# Newton iterate, see xfsi_solver.tools.newton_log), checkpoints, snapshots, VTX output, QoI plots
# and a record of the code version and arguments.
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
INITIAL_STATE_SNAPSHOT=${INITIAL_STATE_SNAPSHOT:-}
SNAPSHOT_TIMES=${SNAPSHOT_TIMES:-}

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
    initial_state_path=$RUNS/$INITIAL_STATE_RUN/checkpoints
    if [[ -n $INITIAL_STATE_SNAPSHOT ]]; then
        initial_state_path=$RUNS/$INITIAL_STATE_RUN/snapshots/$INITIAL_STATE_SNAPSHOT
    fi
    if [[ ! -d $initial_state_path ]]; then
        echo "No checkpoints in $initial_state_path to start from" >&2
        exit 1
    fi
    INITIAL_STATE_DIR="\"$initial_state_path\""
elif [[ -n $INITIAL_STATE_SNAPSHOT ]]; then
    echo "INITIAL_STATE_SNAPSHOT needs INITIAL_STATE_RUN, the run whose snapshot to start from" >&2
    exit 1
fi

SNAPSHOT_DIR=None
SNAPSHOT_TIMES_TUPLE="()"
if [[ -n $SNAPSHOT_TIMES ]]; then
    for time in $SNAPSHOT_TIMES; do
        if [[ ! $time =~ ^[0-9.eE+-]+$ ]]; then
            echo "SNAPSHOT_TIMES must be numbers separated by spaces, not \"$SNAPSHOT_TIMES\"" >&2
            exit 1
        fi
    done
    SNAPSHOT_DIR="\"$RUN_DIR/snapshots\""
    # "0.2 0.4" -> (0.2,0.4,)
    SNAPSHOT_TIMES_TUPLE="($(echo $SNAPSHOT_TIMES | tr ' ' ','),)"
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
    newton_log_path="$RUN_DIR/qoi/${RUN_NAME}_newton.txt",
    checkpoint_dir="$RUN_DIR/checkpoints",
    checkpoint_every=$CHECKPOINT_EVERY,
    vtx_save_every=$VTX_SAVE_EVERY,
    restart=$RESTART,
    initial_state_dir=$INITIAL_STATE_DIR,
    snapshot_dir=$SNAPSHOT_DIR,
    snapshot_times=$SNAPSHOT_TIMES_TUPLE,
    gamma_reassemble=$GAMMA_REASSEMBLE,
    max_quadrature_degree=$MAX_QUADRATURE_DEGREE,
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
