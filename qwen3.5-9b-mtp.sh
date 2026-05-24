#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1  llama-server \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  --n-gpu-layers 16 \
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
  --repeat-penalty 1.1 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2
  
