#!/bin/bash
# 64 layers, 27B dense
CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M \
  --alias qwen3.6-27b-mtp \
  --n-gpu-layers 28 \
  --ctx-size 131072 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q4_1 \
  --cache-type-v q4_1 \
  --jinja \
  --flash-attn on \
  --port 11434 \
  --temperature 0.6 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  --parallel 1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2