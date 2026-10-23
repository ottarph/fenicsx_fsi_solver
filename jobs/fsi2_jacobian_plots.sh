#!/bin/bash
# Matrix plots of the three Jacobians of the restricted-test-function FSI2 formulation, in the
# order (u_S, v, p, z, u_I) of eq. (1) of docs/restricted-iterative-solver.md:
#
#   full         fsi2_biharmonic_fulljacobian: the Jacobian in the full state (u, v, p, z)
#   condensed    fsi2_biharmonic_mixed: u_S condensed out of the solid stress
#   approximate  fsi2_biharmonic_mixed: J_0, without the fluid ALE derivatives
#
# With FORMULATION=harmonic, the same plots of the harmonic version of the formulation, in the
# order (u_S, v, p, u_I), from fsi2_harmonic_fulljacobian (full, state (u, v, p)) and
# fsi2_harmonic_condensed (condensed and approximate). The default is FORMULATION=biharmonic.
#
# Each solver's solve() takes one time step from a developed checkpoint with plot_matrices=True,
# and plots the Jacobians at the state after that step. The result is OUT_DIR, by default
# output/figures/jacobian_plots/<short hash of COMMIT>/, with only
#
#   run_info.txt
#   spy_{full,condensed,approximate}.png          on the coarse mesh
#   block_norms_{full,condensed,approximate}.png  on mesh_sec
#
# (spy_harmonic_full.png and so on with FORMULATION=harmonic). The runs, with their logs and the
# other plots of the solvers, are in a temporary directory, and OUT_DIR is only made once all of
# them have succeeded. When a run fails, the end of its log is printed. Run it on a workstation
# from anywhere:
#
#   jobs/fsi2_jacobian_plots.sh
#   FORMULATION=harmonic jobs/fsi2_jacobian_plots.sh
#
# The solvers run from the source of COMMIT, exported into the temporary directory, which is
# removed when the script exits, and not from the checkout, so the plots stay tied to that commit
# whatever is checked out. The checkout is not changed. The meshes are
# read from the checkout. The checkpoints are not in git: the biharmonic defaults are the states
# at t = 15 that fsi2_biharmonic_mixed (at c3c6d01, dt = 0.0025) wrote in
# output/matrix_block_runs/. Any checkpoint of an FSI2 solver with (u, v, p, z) on the same
# mesh works; the solvers only read them. The harmonic defaults are the states at t = 7 that
# fsi2_harmonic_condensed (at e83a4d4, dt = 0.0025) wrote there, since plain harmonic mesh
# motion is expected to degenerate before t = 15 (min det F over the fluid on mesh_sec fell from
# 0.99 at t = 1 to 0.60 at t = 7). The harmonic solvers read (u, v, p) of any FSI2 checkpoint,
# also a biharmonic one, whose z they ignore.
#
# The ranks set how the dofs are numbered within each block, so they set the spy plots; on one
# rank, the dofs are in the order of the mesh. The block norms do not depend on them.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
FORMULATION=${FORMULATION:-biharmonic}
case $FORMULATION in
    biharmonic)
        default_commit=66b0885e02f07b5bbf3251bc6cf23c961186d412
        full_solver=fsi2_biharmonic_fulljacobian condensed_solver=fsi2_biharmonic_mixed
        runs=fsi2_mixed figure_prefix=""
        ;;
    harmonic)
        default_commit=e83a4d4e63555d610948eb3a6ae020cc20c3e02b
        full_solver=fsi2_harmonic_fulljacobian condensed_solver=fsi2_harmonic_condensed
        runs=fsi2_harmonic figure_prefix=harmonic_
        ;;
    *)
        echo "FORMULATION must be biharmonic or harmonic, not $FORMULATION" >&2
        exit 1
        ;;
esac
COMMIT=${COMMIT:-$default_commit}
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")
OUT_DIR=${OUT_DIR:-$REPO/output/figures/jacobian_plots/${COMMIT:0:7}}
COARSE_MESH=${COARSE_MESH:-$REPO/data/meshes/fsi2/mesh_sec_coarse.xdmf}
NORMAL_MESH=${NORMAL_MESH:-$REPO/data/meshes/fsi2/mesh_sec.xdmf}
COARSE_CHECKPOINTS=${COARSE_CHECKPOINTS:-$REPO/output/matrix_block_runs/${runs}_coarse/checkpoints}
NORMAL_CHECKPOINTS=${NORMAL_CHECKPOINTS:-$REPO/output/matrix_block_runs/${runs}_normal/checkpoints}
COARSE_RANKS=${COARSE_RANKS:-1}
NORMAL_RANKS=${NORMAL_RANKS:-8}
DT=${DT:-0.0025}

for path in "$COARSE_MESH" "$NORMAL_MESH"; do
    [[ -f $path ]] || { echo "Mesh $path not found" >&2; exit 1; }
done
for path in "$COARSE_CHECKPOINTS" "$NORMAL_CHECKPOINTS"; do
    [[ -d $path ]] || { echo "No checkpoints in $path to start from" >&2; exit 1; }
done
if [[ -e $OUT_DIR ]]; then
    echo "$OUT_DIR exists; remove it or choose another OUT_DIR" >&2
    exit 1
fi
# The exported source, the runs and the collected results until they are moved to OUT_DIR
WORK_DIR=$(mktemp -d)
# Also when a run fails or the script is stopped, which otherwise skips the EXIT trap
trap 'rm -rf "$WORK_DIR"' EXIT
trap 'exit 1' INT TERM HUP
git -C "$REPO" archive "$COMMIT" src | tar -x -C "$WORK_DIR"
RESULT_DIR=$WORK_DIR/result
mkdir -p "$RESULT_DIR"

{
    echo "=== $(hostname), $(date -Iseconds)"
    git -C "$REPO" log -1 --format="%H %s" "$COMMIT"
    echo "formulation: $FORMULATION"
    echo "coarse: $COARSE_MESH from $COARSE_CHECKPOINTS on $COARSE_RANKS ranks"
    echo "normal: $NORMAL_MESH from $NORMAL_CHECKPOINTS on $NORMAL_RANKS ranks"
    echo "dt = $DT"
} | tee "$RESULT_DIR/run_info.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$WORK_DIR/src

run() {
    local mesh_name=$1 mesh=$2 checkpoints=$3 ranks=$4 solver=$5 solver_kwargs=$6
    local dir=$WORK_DIR/runs/$mesh_name/$solver
    mkdir -p "$dir"
    # Every rank runs the call from -c; a script on stdin would only reach rank 0
    local py
    py=$(cat <<EOF
from pathlib import Path

import dolfinx
from mpi4py import MPI

import xfsi_solver
from xfsi_solver.solvers.$solver import solve
from xfsi_solver.tools.checkpoint import Checkpointer

# The editable install of the checkout must not shadow the exported source
assert Path(xfsi_solver.__file__).is_relative_to("$WORK_DIR/src"), xfsi_solver.__file__

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "$mesh", "r") as f:
    t0, _ = Checkpointer("$checkpoints", f.read_mesh()).latest_times()

solve(
    mesh_path="$mesh",
    T=t0 + $DT,
    dt_val=$DT,
    output_path="pv/run.bp",
    output_path_p="pv/run_p.bp",
    qoi_path="qoi.txt",
    initial_state_dir="$checkpoints",
    plot_matrices=True,
    $solver_kwargs
)
EOF
)
    echo "=== $mesh_name $solver: start $(date +%T)"
    # The plots go to output/figures/matrix_blocks/<solver>/ in the working directory
    if ! (cd "$dir" && timeout 1800 conda run --no-capture-output -n xfsi_solver \
        mpirun -n "$ranks" python -u -c "$py" > log.txt 2>&1); then
        echo "=== $mesh_name $solver: failed $(date +%T); end of its log:" >&2
        tail -n 40 "$dir/log.txt" >&2
        exit 1
    fi
    echo "=== $mesh_name $solver: done $(date +%T)"
}

for mesh_name in coarse normal; do
    if [[ $mesh_name == coarse ]]; then
        args=("$COARSE_MESH" "$COARSE_CHECKPOINTS" "$COARSE_RANKS")
    else
        args=("$NORMAL_MESH" "$NORMAL_CHECKPOINTS" "$NORMAL_RANKS")
    fi
    run "$mesh_name" "${args[@]}" "$full_solver" ""
    run "$mesh_name" "${args[@]}" "$condensed_solver" "direct_k_c_solve=True,"
done

plots() {
    local mesh_name=$1 kind=$2
    local runs=$WORK_DIR/runs/$mesh_name figures=output/figures/matrix_blocks
    cp "$runs/$full_solver/$figures/$full_solver/${kind}_eq1.png" \
        "$RESULT_DIR/${kind}_${figure_prefix}full.png"
    cp "$runs/$condensed_solver/$figures/$condensed_solver/${kind}_eq1.png" \
        "$RESULT_DIR/${kind}_${figure_prefix}condensed.png"
    cp "$runs/$condensed_solver/$figures/$condensed_solver/${kind}_approximate_eq1.png" \
        "$RESULT_DIR/${kind}_${figure_prefix}approximate.png"
}
plots coarse spy
plots normal block_norms

mkdir -p "$(dirname "$OUT_DIR")"
mv -T "$RESULT_DIR" "$OUT_DIR"
echo "=== plots in $OUT_DIR"
