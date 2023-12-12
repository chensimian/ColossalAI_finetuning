#!/bin/bash
export OMP_NUM_THREADS=8
colossalai run --nproc_per_node 8 benchmark.py \
  --plugin gemini_auto \
  --config 13b \
  --grad_checkpoint \
  --xformers \
  --batch_size 5
