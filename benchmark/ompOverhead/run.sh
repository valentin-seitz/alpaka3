#!/usr/bin/env bash
# Sweeps OMP_NUM_THREADS and ALPAKA_OMP_TASKS_NUM and collects the CSV of ompOverhead.
# usage: run.sh <path/to/ompOverhead> [out.csv] ; override sweeps with THREADS="1 8 64" TASKS="0 16 256"
set -euo pipefail
bin=${1:?usage: run.sh <ompOverhead binary> [out.csv]}
out=${2:-ompOverhead.csv}
ncpu=$(nproc)
THREADS=${THREADS:-"1 $((ncpu / 2)) $ncpu"}
# 0 = unset (runtime default); cpuOmpBlocks ignores this variable, its rows are repeated per setting
TASKS=${TASKS:-"0 $ncpu $((ncpu * 4)) $((ncpu * 16))"}
first=1
for t in $THREADS; do
    for n in $TASKS; do
        echo "threads=$t tasks=$n" >&2
        if [ "$n" = 0 ]; then unset ALPAKA_OMP_TASKS_NUM; else export ALPAKA_OMP_TASKS_NUM=$n; fi
        if [ $first = 1 ]; then
            OMP_NUM_THREADS=$t "$bin" ${EXTRA_ARGS:-} >"$out"; first=0
        else
            OMP_NUM_THREADS=$t "$bin" ${EXTRA_ARGS:-} | tail -n +2 >>"$out"
        fi
    done
done
echo "wrote $out" >&2
