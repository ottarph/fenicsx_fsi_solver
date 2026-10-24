#!/bin/bash -l
# Compare K_c preconditioner options of xfsi_solver.solvers.threeDfsi_c0ip.solve() in one job:
# every variant below runs a few time steps from the same checkpoint, one after the other on
# the same node and ranks, so that both the K_c iterations and the step times compare.
#
#   sbatch jobs/threeDfsi_c0ip_kc_experiments.sh
#
# Each variant is a name and a Python dict of PETSc options for the K_c solve (k_c_options in
# solve()), which replace the solver's own; None runs the solver as it is (on the velocity block,
# 3 SOR smoothing steps per level since 2026-10-22, 2 before, and omega 0.4 since 2026-10-23,
# 0.5 before). To run other
# variants without editing this file, write them to a file, one "name|dict" per line, and
# pass it as VARIANTS_FILE:
#   sbatch --export=ALL,VARIANTS_FILE=/home/hellanottar/xfsi/my_variants.txt jobs/threeDfsi_c0ip_kc_experiments.sh
#
# The variants start from the latest checkpoint of INITIAL_STATE_RUN, which they only read,
# and run up to T. The default is two steps from the checkpoint at t = 4 of fine_rank_test.
# Every variant writes its log and QoIs to EXP_DIR/<name>/, and the job log ends with a table
# of the mean K_c iterations per solve and the mean time per step after the first.
#
# The K_c iterations depend on how the mesh is split over the ranks, so the job takes 64 ranks,
# as the production run on defq. The variants then compare with each other, also in time, and
# in K_c iterations with the production run (about 58 per solve around t = 4.1). Whichever of
# the partitions has room first runs the job. --exclusive would keep other jobs off the node's
# memory bandwidth, at the cost of waiting for an empty node.
#SBATCH -J fsi3d-kc-exp
#SBATCH -p defq,milanq,genoaxq
#SBATCH -N 1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-socket=32
#SBATCH --hint=nomultithread
##SBATCH --exclusive
#SBATCH -t 0-06:00:00
#SBATCH -o /home/hellanottar/xfsi/logs/%x_%j.txt

# Before set -u, since the Spack shell setup may read unset variables
module use /cm/shared/spack-modules/modulefiles
module load spack
spack env activate fenicsx-v011

set -euo pipefail

REPO=/home/hellanottar/code/xfsi_solver
RUNS=/home/hellanottar/xfsi/runs
EXP_NAME=${EXP_NAME:-kc_exp_${SLURM_JOB_ID:-none}}
EXP_DIR=$RUNS/$EXP_NAME
INITIAL_STATE_RUN=${INITIAL_STATE_RUN:-fine_rank_test}
MESH=${MESH:-$REPO/data/meshes/fsi3d/mesh_sec_fine.xdmf}
T=${T:-4.008}
DT=${DT:-0.004}
# A variant that does not converge runs 20 Newton iterations, each with up to 200 FGMRES
# iterations, so stop it after this many seconds and go on with the next.
VARIANT_TIMEOUT=${VARIANT_TIMEOUT:-1800}
VARIANTS_FILE=${VARIANTS_FILE:-}

VARIANTS=(
    'base|None'
    'sweeps2|{"fieldsplit_v_mg_levels_ksp_max_it": 2}'
    'sweeps4|{"fieldsplit_v_mg_levels_ksp_max_it": 4}'
    'thresh005|{"fieldsplit_v_pc_gamg_threshold": 0.05}'
    'thresh01|{"fieldsplit_v_pc_gamg_threshold": 0.1}'
    'proceq1000|{"fieldsplit_v_pc_gamg_process_eq_limit": 1000}'
    'coarse5000|{"fieldsplit_v_pc_gamg_coarse_eq_limit": 5000}'
)
if [[ -n $VARIANTS_FILE ]]; then
    mapfile -t VARIANTS < <(grep -v -e '^\s*#' -e '^\s*$' "$VARIANTS_FILE")
fi

if [[ ! -f $MESH ]]; then
    echo "Mesh $MESH not found; copy it to the checkout first" >&2
    exit 1
fi
if [[ ! -d $RUNS/$INITIAL_STATE_RUN/checkpoints ]]; then
    echo "No checkpoints in $RUNS/$INITIAL_STATE_RUN to start from" >&2
    exit 1
fi
if [[ -e $EXP_DIR ]]; then
    echo "$EXP_DIR exists; choose another EXP_NAME" >&2
    exit 1
fi
mkdir -p "$EXP_DIR"

{
    echo "=== job ${SLURM_JOB_ID:-none} on $(hostname), $(date -Iseconds), ${SLURM_NTASKS:-?} ranks"
    git -C "$REPO" log -1 --format="%H %s"
    git -C "$REPO" status --short
    echo "from $INITIAL_STATE_RUN to T = $T with dt = $DT"
    printf '%s\n' "${VARIANTS[@]}"
} | tee "$EXP_DIR/run_info.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
# FFCx JIT cache on node-local disk, one per job, shared by the variants
export XDG_CACHE_HOME=/tmp/$USER/fenics-cache-$SLURM_JOB_ID
# InfiniBand (and shared memory on the node) through UCX
export OMPI_MCA_pml=ucx

for variant in "${VARIANTS[@]}"; do
    name=${variant%%|*}
    k_c_options=${variant#*|}
    dir=$EXP_DIR/$name
    mkdir -p "$dir"

    # Every rank runs the call from -c; a script on stdin would only reach rank 0
    PY=$(cat <<EOF
from xfsi_solver.solvers.threeDfsi_c0ip import solve

solve(
    mesh_path="$MESH",
    T=$T,
    dt_val=$DT,
    output_path="$dir/run.bp",
    output_path_p="$dir/run_p.bp",
    qoi_path="$dir/qoi.txt",
    initial_state_dir="$RUNS/$INITIAL_STATE_RUN/checkpoints",
    k_c_options=$k_c_options,
)
EOF
)
    echo "=== $name: $k_c_options, start $(date +%T)"
    # srun binds the ranks to the cores Slurm allocated (see jobs/threeDfsi_c0ip.sh)
    status=0
    timeout "$VARIANT_TIMEOUT" srun --mpi=pmix --cpu-bind=cores python -u -c "$PY" > "$dir/log.txt" 2>&1 || status=$?
    echo "=== $name: done $(date +%T), exit $status"
done

# One row per variant from the summary lines that solve() prints at the end
{
    printf '\n%-14s %-8s %-22s %-28s %s\n' variant result "K_c its per solve" "s per step after the first" k_c_options
    for variant in "${VARIANTS[@]}"; do
        name=${variant%%|*}
        log=$EXP_DIR/$name/log.txt
        its=$(grep -oP 'K_c iterations per solve: \K\S+' "$log" || echo -)
        step_time=$(grep -oP 'Time per step after the first: \K\S+' "$log" || echo -)
        result=ok
        grep -q "Traceback" "$log" && result=failed
        printf '%-14s %-8s %-22s %-28s %s\n' "$name" "$result" "$its" "$step_time" "${variant#*|}"
    done
} | tee "$EXP_DIR/summary.txt"
