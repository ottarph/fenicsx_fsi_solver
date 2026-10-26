#!/bin/bash
# Speed and accuracy of fsi2_biharmonic_mixed_twice_simplified with a capped quadrature degree.
#
# Runs the solver from the developed state in INITIAL_STATE (by default a copy of the t = 16
# checkpoint of its benchmark run) up to time T, once for each cap in CAPS, one run after the
# other on RANKS ranks, so that the timings are comparable. "none" is the run without a cap, with
# the degrees UFL estimates (up to 14 in the residual), which the other runs are compared to. For
# each cap, OUT_DIR gets
#
#   qoi_<cap>.txt   the QoIs from the start time on
#   log_<cap>.txt   the solver output, with the Newton iterations of each step and the time per step
#
# Run it from anywhere; it runs the solver of the checkout:
#
#   jobs/fsi2_quadrature_cap.sh
#   CAPS="none 4" RANKS=8 jobs/fsi2_quadrature_cap.sh

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CAPS=${CAPS:-none 10 8 6 5 4 3}
RANKS=${RANKS:-16}
T=${T:-18.0}
MESH=${MESH:-$REPO/data/meshes/fsi2/mesh_sec.xdmf}
INITIAL_STATE=${INITIAL_STATE:-$REPO/output/quadrature_cap/initial_state}
OUT_DIR=${OUT_DIR:-$REPO/output/quadrature_cap}

mkdir -p "$OUT_DIR"
cd "$REPO"
for cap in $CAPS; do
    if [ "$cap" = none ]; then degree=None; else degree=$cap; fi
    echo "$(date '+%F %T') cap $cap: started"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 conda run --no-capture-output -n xfsi_solver \
        mpirun -n "$RANKS" python -u -c "
from xfsi_solver.solvers.fsi2_biharmonic_mixed_twice_simplified import solve

solve(
    mesh_path='$MESH',
    T=$T,
    dt_val=0.0025,
    output_path=None,
    output_path_p=None,
    qoi_path='$OUT_DIR/qoi_$cap.txt',
    initial_state_dir='$INITIAL_STATE',
    max_quadrature_degree=$degree,
)
" >"$OUT_DIR/log_$cap.txt" 2>&1
    echo "$(date '+%F %T') cap $cap: $(grep 'Time per step' "$OUT_DIR/log_$cap.txt")"
done
echo "$(date '+%F %T') all done"
