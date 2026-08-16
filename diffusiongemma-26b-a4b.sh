#!/bin/bash

CUDA_VISIBLE_DEVICES=0 llama-server \
  -hf unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M \
  --alias diffusiongemma-26b-a4b \
  --n-gpu-layers 99 \
  --ctx-size 131072 \
  --threads 16 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --parallel 1 \
  --flash-attn on \
  --jinja \
  --host 0.0.0.0 \
  --port 11434 \
  --temperature 0.65 \
  --top-p 0.95 \
  --repeat-penalty 1.1
