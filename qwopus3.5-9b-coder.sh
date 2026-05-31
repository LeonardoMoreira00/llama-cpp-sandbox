#!/bin/bash

CUDA_VISIBLE_DEVICES=0 llama-server \
  -hf Jackrong/Qwopus3.5-9B-Coder-GGUF:Q8_0 \
  --alias qwopus3.5-9b-coder \
  --n-gpu-layers 99 \
  --ctx-size 131072 \
  --threads 16 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --parallel 1 \
  --flash-attn on \
  --jinja \
  --port 11434 \
  --temperature 0.65 \
  --top-p 0.95 \
  --repeat-penalty 1.1
