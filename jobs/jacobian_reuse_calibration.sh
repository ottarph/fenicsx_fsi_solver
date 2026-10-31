#!/bin/bash
# Measure the cost of a Newton iteration of fsi2_simplified that assembles and refactorizes the
# Jacobian, and of one that reuses it, at several rank counts: short runs from the latest checkpoint
# of a sweep run (in the periodic regime), one at a time on the otherwise idle machine, with
# gamma_reassemble None (every iteration assembles) and 0.2 (most reuse). Run it detached:
#
#   setsid nohup bash jobs/jacobian_reuse_calibration.sh > output/logs/jrc.txt 2>&1 < /dev/null &
#
# Each run writes to $OUT/p<ranks>_g<gamma>/ and its log to output/logs/jrc_p<ranks>_g<gamma>.txt.

set -uo pipefail
cd "$(dirname "$0")/.."

RANK_COUNTS=${RANK_COUNTS:-"1 2 4 8 12 16 24 32"}
GAMMAS=${GAMMAS:-"none 0.2"}
INITIAL_STATE_DIR=${INITIAL_STATE_DIR:-output/jacobian_reuse_sweep/g0.2/checkpoints}
T=${T:-15.5} # 200 steps from the checkpoint at t = 15
OUT=${OUT:-output/jacobian_reuse_calibration}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT" output/logs
for ranks in $RANK_COUNTS; do
    for gamma in $GAMMAS; do
        run_dir=$OUT/p${ranks}_g$gamma
        if [[ -e $run_dir/stats.json ]]; then
            echo "$run_dir is done; skipping"
            continue
        fi
        rm -rf "$run_dir"
        log=output/logs/jrc_p${ranks}_g$gamma.txt
        echo "$(date -Iseconds) $ranks ranks, gamma $gamma"
        conda run --no-capture-output -n xfsi_solver \
            mpirun -n "$ranks" -bind-to core \
            python -u -m xfsi_solver.scripts.jacobian_reuse_sweep --gamma "$gamma" --run-dir "$run_dir" --T "$T" \
            --initial-state-dir "$INITIAL_STATE_DIR" --checkpoint-every 100000 \
            > "$log" 2>&1 < /dev/null || echo "FAILED: $ranks ranks, gamma $gamma (see $log)"
    done
done
echo "$(date -Iseconds) calibration done"
