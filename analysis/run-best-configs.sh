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

for state in "${STATES[@]}"; do
    echo "state: ${state}"
    out="measurements/runs/e3_${state}"
    ./build/bin/${BIN[$state]} --headless \
    --snapshot-load measurements/snapshots/${state}.sphs \
    --frozen-config measurements/configs/${state}.json \
    --tuning-budget 0.0 \
    --warmup-iters 50 --fixed-dt 0.01 \
    --stop iters=200 \
    --ktt-output "${out}"
    sleep 5
done
