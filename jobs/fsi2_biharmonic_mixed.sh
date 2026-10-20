#!/bin/bash
# Run the FSI2 benchmark with xfsi_solver.solvers.fsi2_biharmonic_mixed.solve() on one eX3 node.
#
#   sbatch jobs/fsi2_biharmonic_mixed.sh
#
# The arguments of solve() are set below. Each can be overridden at submission without
# editing this file, e.g. a restart, or a separate short run:
#   sbatch --export=ALL,RESTART=True jobs/fsi2_biharmonic_mixed.sh
#   sbatch --export=ALL,RUN_NAME=fsi2_mixed_short,T=0.1 jobs/fsi2_biharmonic_mixed.sh
# Values are inserted as Python expressions: True/False, None, numbers or arithmetic.
#
# Each run gets its own directory, RUN_DIR, which holds the QoIs, checkpoints, VTX output,
# matrix plots and a record of the code version and arguments.
#SBATCH -J fsi2-mixed
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH --hint=nomultithread
##SBATCH -t 1-00:00:00
#SBATCH -t 0-09:00:00
#SBATCH -o /home/hellanottar/xfsi/logs/%x_%j.txt

set -euo pipefail

REPO=/home/hellanottar/code/xfsi_solver
RUN_NAME=${RUN_NAME:-fsi2_biharmonic_mixed_fine}
RUN_DIR=/home/hellanottar/xfsi/runs/$RUN_NAME

# Arguments of solve()
MESH=${MESH:-$REPO/data/meshes/fsi2/mesh_fine_sec.xdmf}
T=${T:-15.0}
DT=${DT:-0.0025}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-400}
VTX_SAVE_EVERY=${VTX_SAVE_EVERY:-None}
PLOT_MATRICES=${PLOT_MATRICES:-False}
GAMMA_REASSEMBLE=${GAMMA_REASSEMBLE:-0.2}
DIRECT_K_C_SOLVE=${DIRECT_K_C_SOLVE:-True}
RESTART=${RESTART:-False}

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
# solve() writes the matrix plots to output/figures/ relative to the working directory
cd "$RUN_DIR"

# Every rank runs the call from -c; a script on stdin would only reach rank 0
PY=$(cat <<EOF
from xfsi_solver.solvers.fsi2_biharmonic_mixed import solve

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
    plot_matrices=$PLOT_MATRICES,
    restart=$RESTART,
    gamma_reassemble=$GAMMA_REASSEMBLE,
    direct_k_c_solve=$DIRECT_K_C_SOLVE,
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

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Single-node job: let Hydra start the ranks itself instead of through srun
export HYDRA_BOOTSTRAP=fork

~/miniforge3/bin/conda run --no-capture-output -n xfsi_solver \
    mpiexec -n "$SLURM_NTASKS" -bind-to core \
    python -u -c "$PY"
