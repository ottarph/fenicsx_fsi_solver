#!/bin/bash
# Launch the sweep over gamma_reassemble of fsi2_simplified on the workstation: one detached run of
# xfsi_solver.scripts.jacobian_reuse_sweep per value of GAMMAS, each on RANKS ranks pinned to its own
# physical cores (logical CPUs 0-31 are the 32 physical cores; 32-63 their hyperthreads).
#
#   bash jobs/jacobian_reuse_sweep.sh
#
# Each run writes to $OUT/g<gamma>/, its log to output/logs/jrs_g<gamma>.txt and its PID (the process
# group leader, so `kill -- -<pid>` stops the run) to output/logs/jrs_g<gamma>.pid; LOG_PREFIX
# replaces jrs_. With INITIAL_STATE_DIR, the runs start from the latest checkpoint in that directory,
# e.g. a developed state:
#   GAMMAS="0.4 0.5" OUT=output/jacobian_reuse_restart LOG_PREFIX=jrr_ FIRST_CORE=24 \
#       INITIAL_STATE_DIR=output/jacobian_reuse_sweep/developed_t10 bash jobs/jacobian_reuse_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GAMMAS=${GAMMAS:-"none 0.01 0.02 0.05 0.08 0.1 0.13 0.16 0.2 0.25 0.3 0.4 0.5 0.6 0.8 1.0"}
RANKS=${RANKS:-2}
T=${T:-15.0}
OUT=${OUT:-output/jacobian_reuse_sweep}
FIRST_CORE=${FIRST_CORE:-0}
LOG_PREFIX=${LOG_PREFIX:-jrs_}
INITIAL_STATE_ARGS=()
if [[ -n ${INITIAL_STATE_DIR:-} ]]; then
    INITIAL_STATE_ARGS=(--initial-state-dir "$INITIAL_STATE_DIR")
fi

mkdir -p "$OUT" output/logs
core=$FIRST_CORE
for gamma in $GAMMAS; do
    run_dir=$OUT/g$gamma
    if [[ -e $run_dir ]]; then
        echo "$run_dir exists; skipping" >&2
        continue
    fi
    cores=$(seq -s, "$core" $((core + RANKS - 1)))
    core=$((core + RANKS))
    log=output/logs/${LOG_PREFIX}g$gamma.txt
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup \
        conda run --no-capture-output -n xfsi_solver \
        mpirun -n "$RANKS" -bind-to "user:$cores" \
        python -u -m xfsi_solver.scripts.jacobian_reuse_sweep --gamma "$gamma" --run-dir "$run_dir" --T "$T" \
        "${INITIAL_STATE_ARGS[@]}" > "$log" 2>&1 < /dev/null &
    echo $! > output/logs/${LOG_PREFIX}g$gamma.pid
    echo "gamma $gamma on cores $cores: PID $!, log $log"
done
