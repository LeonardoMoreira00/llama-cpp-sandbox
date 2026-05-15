#!/bin/bash

llama-server \
  -hf unsloth/Ministral-3-8B-Instruct-2512-GGUF:UD-Q4_K_XL \
  -ngl 20 \
  -c 32768 \
  -b 2048 \
  -ub 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -fa on \
  --jinja \
  --port 11434 \
  --temp 0.7 \
  --top-p 0.95 \
  --repeat-penalty 1.1
