#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -hf unsloth/Ministral-3-8B-Instruct-2512-GGUF:UD-Q4_K_XL \
  --alias ministral3-8b \
  --n-gpu-layers 99 \
  --threads 16 \
  --ctx-size 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --jinja \
  --port 11434 \
  --temperature 0.7 \
  --top-p 0.95 \
  --repeat-penalty 1.1
