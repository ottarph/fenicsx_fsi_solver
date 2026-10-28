#!/bin/bash
# FGMRES iterations of the (v, p) solve of xfsi_solver.solvers.threeDfsi_simplified.solve() against the
# partition of the mesh and the SOR omega of the GAMG smoother on the velocity block, on the workstation
# (not a Slurm job): do weighted dual-graph edges at the solid (partition_solid_edge_weight,
# tools/partitioning.py), which keep the beam on fewer ranks, flatten the FGMRES iterations against the
# rank count, and do they let SOR be damped less?
# (notes/partitioning/implementation-log.md)
#
# The edge weights need DOLFINx main, so this runs with the Python of the repository's .venv and the mpirun
# of the fenicsx-dev env (CLAUDE.md). Run it detached, as a long MPI run:
#
#   EXP_NAME=rest setsid nohup jobs/threeDfsi_simplified_partition_experiments.sh \
#       > output/logs/partition_kc_rest.txt 2>&1 < /dev/null &
#
# Every combination of RANKS, WEIGHTS and OMEGAS runs, one after the other, from rest (START=rest) or from
# the latest checkpoint in the directory START, up to T with time step DT. A weight of None is the default
# partition of XDMFFile.read_mesh, 1 the same partition through the weighted path (ParMETIS). Each
# variant writes its log and QoIs to EXP_DIR/<name>/, and the log ends with a table of the partitions and
# the FGMRES iterations.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
EXP_NAME=${EXP_NAME:-partition_kc_$(date +%Y%m%d_%H%M%S)}
EXP_DIR=${EXP_DIR:-$REPO/output/partition_kc/$EXP_NAME}
MESH=${MESH:-$REPO/data/meshes/fsi3d/mesh_sec_coarse.xdmf}
START=${START:-rest}
DT=${DT:-0.004}
# From rest: two steps, as in the diagnosis on 26 ranks (notes/fsi3d/implementation-log.md)
T=${T:-0.008}
RANKS=${RANKS:-24 26 28 30}
WEIGHTS=${WEIGHTS:-None 10}
OMEGAS=${OMEGAS:-0.4 1.0}
# A variant that does not converge runs 20 Newton iterations, each with up to 200 FGMRES iterations
VARIANT_TIMEOUT=${VARIANT_TIMEOUT:-600}
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
MPIRUN=${MPIRUN:-$HOME/miniforge3/envs/fenicsx-dev/bin/mpirun}

if [[ -e $EXP_DIR ]]; then
    echo "$EXP_DIR exists; choose another EXP_NAME" >&2
    exit 1
fi
if [[ $START != rest && ! -d $START ]]; then
    echo "No checkpoint directory $START" >&2
    exit 1
fi
mkdir -p "$EXP_DIR"

if [[ $START == rest ]]; then
    initial_state=None
else
    initial_state="\"$(cd "$START" && pwd)\""
fi

{
    echo "=== $(hostname), $(date -Iseconds)"
    git -C "$REPO" log -1 --format="%H %s"
    git -C "$REPO" status --short
    "$PYTHON" -c "import dolfinx; print('DOLFINx', dolfinx.__version__, dolfinx.git_commit_hash)"
    echo "mesh $MESH, start $START, T = $T, dt = $DT"
    echo "ranks: $RANKS; solid edge weights: $WEIGHTS; SOR omegas: $OMEGAS"
} | tee "$EXP_DIR/run_info.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

names=()
for ranks in $RANKS; do
    for weight in $WEIGHTS; do
        for omega in $OMEGAS; do
            name=r${ranks}_w${weight}_om${omega}
            names+=("$name")
            dir=$EXP_DIR/$name
            mkdir -p "$dir"
            # Every rank runs the call from -c; a script on stdin would only reach rank 0
            PY=$(cat <<EOF
from xfsi_solver.solvers.threeDfsi_simplified import DEFAULT_ITERATIVE_SOLVER_OPTIONS, solve

solve(
    mesh_path="$MESH",
    T=$T,
    dt_val=$DT,
    output_path="$dir/run.bp",
    output_path_p="$dir/run_p.bp",
    qoi_path="$dir/qoi.txt",
    initial_state_dir=$initial_state,
    solver_options=DEFAULT_ITERATIVE_SOLVER_OPTIONS | {"fieldsplit_v_0_mg_levels_pc_sor_omega": $omega},
    partition_solid_edge_weight=$weight,
)
EOF
)
            echo "=== $name: start $(date +%T)"
            status=0
            timeout "$VARIANT_TIMEOUT" "$MPIRUN" -n "$ranks" "$PYTHON" -u -c "$PY" > "$dir/log.txt" 2>&1 || status=$?
            echo "=== $name: done $(date +%T), exit $status"
        done
    done
done

# One row per variant from the lines that solve() prints
{
    printf '\n%-22s %-7s %-9s %-11s %-8s %-8s %-10s %-10s %s\n' variant result "solid on" "cut s-s" "cut s-f" \
        "cut all" "its/solve" "s/step>1" "FGMRES its/Newton its per step"
    for name in "${names[@]}"; do
        log=$EXP_DIR/$name/log.txt
        result=ok
        grep -qE "Traceback|did not converge|DIVERGED|Killed" "$log" && result=failed
        grep -q "Elapsed time" "$log" || result=failed
        partition=$(grep -m1 '^Partition' "$log")
        solid_on=$(grep -oP 'solid on \K\d+ of \d+' <<< "$partition" | tr -d ' ' | sed 's/of/\//')
        cut_all=$(grep -oP 'boundaries: \K\d+' <<< "$partition")
        cut_ss=$(grep -oP '\d+(?= solid-solid)' <<< "$partition")
        cut_sf=$(grep -oP '\d+(?= solid-fluid)' <<< "$partition")
        its=$(grep -oP 'Linear iterations per solve: \K\S+' "$log" || echo -)
        step_time=$(grep -oP 'Time per step after the first: \K\S+' "$log" || echo -)
        per_step=$(grep -oP 'linear iterations \K\d+ in \d+' "$log" | sed 's/ in /\//' | tr '\n' ' ')
        printf '%-22s %-7s %-9s %-11s %-8s %-8s %-10s %-10s %s\n' "$name" "$result" "${solid_on:--}" "${cut_ss:--}" \
            "${cut_sf:--}" "${cut_all:--}" "$its" "$step_time" "$per_step"
    done
} | tee "$EXP_DIR/summary.txt"
