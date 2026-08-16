#!/bin/bash
# 30 layers, 25.2B total parameters, 3.8b active parameters, context length 256k tokens
CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
  --alias gemma-4-26b-a4b-it \
  --n-gpu-layers 30 \
  --threads 16 \
  --ctx-size 131072 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --tensor-split 51,20 \
  --jinja \
  --flash-attn on \
  --host 0.0.0.0 \
  --port 11434 \
  --temperature 0.65 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  --parallel 1