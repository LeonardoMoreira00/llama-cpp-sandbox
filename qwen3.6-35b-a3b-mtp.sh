#!/bin/bash
# 64 layers, 35B total (3B active MoE)
CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M \
  --alias qwen3.6-35b-a3b-mtp \
  --n-gpu-layers 28  \
  --ctx-size 131072 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --jinja \
  --flash-attn on \
  --port 11434 \
  --temperature 0.6 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  --parallel 1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2