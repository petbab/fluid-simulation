#!/bin/bash

STATES=(dragon_settled dragon_falling fountain_active vortex_active \
        tilting_chaotic surface_settled BIG_falling BIG_settled)
declare -A BIN=(
  [dragon_settled]=dragon_collision
  [dragon_falling]=dragon_collision
  [fountain_active]=fountain
  [vortex_active]=vortex
  [tilting_chaotic]=tilting_box
  [surface_settled]=surface_tension
  [BIG_falling]=BIG
  [BIG_settled]=BIG
)

for s in "${STATES[@]}"; do
    echo "Tuning ${s}..."
    out="measurements/runs/e1_${s}"
    ./build/bin/${BIN[$s]} --headless \
      --snapshot-load measurements/snapshots/${s}.sphs \
      --tuning-budget 1.0 \
      --warmup-iters 50 --fixed-dt 0.01 \
      --stop iters=1700 \
      --ktt-output "${out}"
    sleep 10
done
