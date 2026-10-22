#!/bin/bash
# Matrix plots of the three Jacobians of the restricted-test-function FSI2 formulation, in the
# order (u_S, v, p, z, u_I) of eq. (1) of docs/restricted-iterative-solver.md:
#
#   full         fsi2_biharmonic_fulljacobian: the Jacobian in the full state (u, v, p, z)
#   condensed    fsi2_biharmonic_mixed: u_S condensed out of the solid stress
#   approximate  fsi2_biharmonic_mixed: J_0, without the fluid ALE derivatives
#
# Each solver's solve() takes one time step from a developed checkpoint with plot_matrices=True,
# and plots the Jacobians at the state after that step. The script collects
#
#   OUT_DIR/figures/spy_{full,condensed,approximate}.png          on the coarse mesh
#   OUT_DIR/figures/block_norms_{full,condensed,approximate}.png  on mesh_sec
#
# and keeps every plot and log of the runs in OUT_DIR/<mesh>/<solver>/. Run it on a workstation
# from anywhere:
#
#   jobs/fsi2_jacobian_plots.sh
#
# The solvers run from the source of COMMIT, exported into OUT_DIR/src, and not from the
# checkout, so the plots stay tied to that commit whatever is checked out. The meshes are
# read from the checkout. The checkpoints are not in git: the defaults are the states at
# t = 15 that fsi2_biharmonic_mixed (at c3c6d01, dt = 0.0025) wrote in
# output/matrix_block_runs/. Any checkpoint of an FSI2 solver with (u, v, p, z) on the same
# mesh works; the solvers only read them.
#
# The ranks set how the dofs are numbered within each block, so they set the spy plots; on one
# rank, the dofs are in the order of the mesh. The block norms do not depend on them.

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=${COMMIT:-66b0885e02f07b5bbf3251bc6cf23c961186d412}
COMMIT=$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")
OUT_DIR=${OUT_DIR:-$REPO/output/jacobian_plots/${COMMIT:0:7}}
COARSE_MESH=${COARSE_MESH:-$REPO/data/meshes/fsi2/mesh_sec_coarse.xdmf}
NORMAL_MESH=${NORMAL_MESH:-$REPO/data/meshes/fsi2/mesh_sec.xdmf}
COARSE_CHECKPOINTS=${COARSE_CHECKPOINTS:-$REPO/output/matrix_block_runs/fsi2_mixed_coarse/checkpoints}
NORMAL_CHECKPOINTS=${NORMAL_CHECKPOINTS:-$REPO/output/matrix_block_runs/fsi2_mixed_normal/checkpoints}
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
mkdir -p "$OUT_DIR/figures"
git -C "$REPO" archive "$COMMIT" src | tar -x -C "$OUT_DIR"

{
    echo "=== $(hostname), $(date -Iseconds)"
    git -C "$REPO" log -1 --format="%H %s" "$COMMIT"
    echo "coarse: $COARSE_MESH from $COARSE_CHECKPOINTS on $COARSE_RANKS ranks"
    echo "normal: $NORMAL_MESH from $NORMAL_CHECKPOINTS on $NORMAL_RANKS ranks"
    echo "dt = $DT"
} | tee "$OUT_DIR/run_info.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$OUT_DIR/src

run() {
    local mesh_name=$1 mesh=$2 checkpoints=$3 ranks=$4 solver=$5 solver_kwargs=$6
    local dir=$OUT_DIR/$mesh_name/$solver
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
assert Path(xfsi_solver.__file__).is_relative_to("$OUT_DIR/src"), xfsi_solver.__file__

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
    (cd "$dir" && timeout 1800 conda run --no-capture-output -n xfsi_solver \
        mpirun -n "$ranks" python -u -c "$py" > log.txt 2>&1)
    echo "=== $mesh_name $solver: done $(date +%T)"
}

for mesh_name in coarse normal; do
    if [[ $mesh_name == coarse ]]; then
        args=("$COARSE_MESH" "$COARSE_CHECKPOINTS" "$COARSE_RANKS")
    else
        args=("$NORMAL_MESH" "$NORMAL_CHECKPOINTS" "$NORMAL_RANKS")
    fi
    run "$mesh_name" "${args[@]}" fsi2_biharmonic_fulljacobian ""
    run "$mesh_name" "${args[@]}" fsi2_biharmonic_mixed "direct_k_c_solve=True,"
done

plots() {
    local mesh_name=$1 kind=$2
    local figures=output/figures/matrix_blocks
    cp "$OUT_DIR/$mesh_name/fsi2_biharmonic_fulljacobian/$figures/fsi2_biharmonic_fulljacobian/${kind}_eq1.png" \
        "$OUT_DIR/figures/${kind}_full.png"
    cp "$OUT_DIR/$mesh_name/fsi2_biharmonic_mixed/$figures/fsi2_biharmonic_mixed/${kind}_eq1.png" \
        "$OUT_DIR/figures/${kind}_condensed.png"
    cp "$OUT_DIR/$mesh_name/fsi2_biharmonic_mixed/$figures/fsi2_biharmonic_mixed/${kind}_approximate_eq1.png" \
        "$OUT_DIR/figures/${kind}_approximate.png"
}
plots coarse spy
plots normal block_norms
echo "=== plots in $OUT_DIR/figures"
