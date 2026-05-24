#!/bin/bash
# 64 layers
#--n-gpu-layers 20 \
CUDA_VISIBLE_DEVICES=0,1  llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M \
  --ctx-size 262144 \
  --cache-type-k q4_1 \
  --cache-type-v q4_1 \
  --jinja \
  --flash-attn on \
  --port 11434 \
  --temperature 0.6 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2