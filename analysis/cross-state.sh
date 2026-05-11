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


for tgt in "${STATES[@]}"; do
  for src in "${STATES[@]}"; do
    echo "src: ${src}, tgt: ${tgt}"
    out="measurements/runs/e2_src-${src}_tgt-${tgt}"
    ./build/bin/${BIN[$tgt]} --headless \
    --snapshot-load measurements/snapshots/${tgt}.sphs \
    --frozen-config measurements/configs/${src}.json \
    --tuning-budget 0.0 \
    --warmup-iters 50 --fixed-dt 0.01 \
    --stop iters=200 \
    --ktt-output "${out}"
    sleep 5
  done
done

