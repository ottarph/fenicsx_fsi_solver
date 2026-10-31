#!/bin/bash
# Short runs of threeDfsi_simplified with several gamma_reassemble, from a developed state, one at a
# time on the otherwise idle machine, so their wall times compare. By default 88 steps (two periods
# of the beam, dt = 0.004) on the medium mesh from the threeDfsi_mixed_medium checkpoint at t = 10,
# on 28 ranks. Run it detached:
#
#   setsid nohup bash jobs/jacobian_reuse_3d.sh > output/logs/jr3d.txt 2>&1 < /dev/null &
#
# Each run writes to $OUT/g<gamma>/ and its log to output/logs/jr3d_g<gamma>.txt; a run whose
# stats.json exists is skipped.

set -uo pipefail
cd "$(dirname "$0")/.."

RANKS=${RANKS:-28}
GAMMAS=${GAMMAS:-"0.2 0.1 0.05 0.02 0.01 none"}
INITIAL_STATE_DIR=${INITIAL_STATE_DIR:-output/checkpoints/threeDfsi_mixed_medium}
T=${T:-10.352}
OUT=${OUT:-output/jacobian_reuse_3d}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT" output/logs
for gamma in $GAMMAS; do
    run_dir=$OUT/g$gamma
    if [[ -e $run_dir/stats.json ]]; then
        echo "$run_dir is done; skipping"
        continue
    fi
    rm -rf "$run_dir"
    log=output/logs/jr3d_g$gamma.txt
    echo "$(date -Iseconds) $RANKS ranks, gamma $gamma"
    conda run --no-capture-output -n xfsi_solver \
        mpirun -n "$RANKS" -bind-to core \
        python -u -m xfsi_solver.scripts.jacobian_reuse_sweep --solver threeDfsi_simplified --gamma "$gamma" \
        --run-dir "$run_dir" --T "$T" --initial-state-dir "$INITIAL_STATE_DIR" --checkpoint-every 100000 \
        > "$log" 2>&1 < /dev/null || echo "FAILED: gamma $gamma (see $log)"
done
echo "$(date -Iseconds) done"
