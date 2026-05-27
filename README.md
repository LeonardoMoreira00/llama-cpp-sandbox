# Llama.cpp Server Configuration Guide

Configuration and tuning guide for running large language models with llama.cpp.

---

## Overview

This document explains the core configuration parameters for llama.cpp and how they affect inference performance and memory usage. The goal is to help you configure the server to run optimally on your specific hardware.

### Key Concepts

**GPU Offloading:** llama.cpp can split model computation between your GPU (fast, limited memory) and CPU (slower, more memory). The GPU processes layers in ~1-2ms per token; CPU layers take ~10-20ms per token.

**KV Cache:** The model stores information about previously processed tokens in a key-value cache. This cache grows linearly with context length and is critical for long conversations.

**Layers:** A transformer model consists of sequential processing layers (typically 32-96 layers depending on model size). Each layer can independently run on GPU or CPU.

---

## Critical Configuration Parameters

### `--n-gpu-layers` (GPU Layer Offloading)

**What it controls:**
The number of model layers to offload to the GPU. Each GPU layer processes significantly faster than a CPU layer, but consumes VRAM.

**Performance impact:**
- GPU layer: ~1-2ms per token (parallel computation across cores)
- CPU layer: ~10-20ms per token (depends on CPU core count)
- Every additional GPU layer improves throughput by ~1-3 tokens/second (depending on hardware)

**Memory impact:**
- Each layer costs approximately (model_size / total_layers) × VRAM
- For a 20B model with 40 layers: each layer ≈ 0.5 GB
- KV cache for GPU layers is stored in VRAM

**How to set:**
1. Start with a conservative value (e.g., 20-30)
2. Increase by 2-4 until you hit CUDA out of memory errors
3. Reduce by 2-4 from the crash point and use that as your final value
4. Monitor with `nvidia-smi` to verify VRAM usage stays below 90%

**Typical ranges:**
- RTX 4090 (24GB VRAM): 60-80 layers
- RTX 3090 (24GB VRAM): 40-60 layers
- RTX 4080 (16GB VRAM): 30-50 layers
- RTX 3060 (12GB VRAM): 20-35 layers
- RTX 4060 (8GB VRAM): 15-25 layers
- Integrated GPU (shared memory): 0-10 layers (if any)

---

### `--threads` (CPU Thread Count)

**What it controls:**
The number of CPU threads allocated for layers running on the CPU and for thread-based operations on GPU layers.

**Performance impact:**
- More threads = faster CPU computation for non-GPU layers
- Diminishing returns: doubling threads doesn't double speed due to memory bandwidth limits
- Using all cores fully can impact system responsiveness

**How to set:**
- Set to the number of physical CPU cores (not logical cores/SMT)
- On a 16-core CPU: `--threads 16`
- Use `nproc --all` to see total logical cores, divide by 2 if hyperthreading is significant
- Or set dynamically: `--threads $(nproc --physical 2>/dev/null || nproc)`

**Typical impact:**
- 4 cores: ~5-10 tok/s for CPU layers
- 8 cores: ~12-18 tok/s for CPU layers
- 16 cores: ~25-35 tok/s for CPU layers
- Scaling is roughly logarithmic due to memory bandwidth saturation

---

### `--ctx-size` (Context Window)

**What it controls:**
Maximum number of tokens the model can process in a single context. Determines how much conversation history the model can reference.

**Memory impact:**
- Linear relationship: doubling context doubles KV cache size
- KV cache size ≈ ctx_size × layers × 2 × head_dim × bytes_per_token
- With q4_0 quantization: each 1000 tokens ≈ 40-50 MB of KV cache
- Total context memory = model_size + (KV cache for GPU layers + KV cache for CPU layers)

**Practical values:**
- Short conversations: 4096 tokens (~1KB text per conversation)
- Moderate conversations: 8192-16384 tokens (~2-4KB text)
- Document analysis: 32768 tokens (~8KB+ text)
- Very long documents: 65536+ tokens (requires significant RAM)

**How to set:**
1. Start with 8192 as baseline
2. Increase if you need to reference longer documents
3. Monitor RAM usage with `free -h` during generation
4. Leave at least 2-4GB RAM free to avoid swapping
5. If swap usage appears (`Swap > 0` in free output), reduce context length

---

### `--batch-size` (Prompt Batch Size)

**What it controls:**
Maximum number of tokens to process together during prompt ingestion (prefill phase). Larger batches process long prompts faster but require more VRAM.

**Performance impact:**
- Larger batches: faster prompt processing, first token appears sooner
- Smaller batches: lower peak VRAM usage during prefill
- Decoding phase (token-by-token generation): mostly unaffected by batch size

**How to set:**
- Start with 512 for stability
- Increase by 256 while monitoring VRAM with `nvidia-smi` (keep below 95%)
- Typical range: 512-2048
- Don't set higher than your context window

**When it matters:**
- Significant impact when users paste long prompts
- Minimal impact for interactive chat (short prompts typed one at a time)

---

### `--ubatch-size` (Micro-batch Size)

**What it controls:**
Size of sub-batches within a batch. Smaller micro-batches reduce peak VRAM spikes by breaking computation into smaller steps.

**How to set:**
- Set to approximately 1/4 of `--batch-size`
- If `--batch-size 2048`, use `--ubatch-size 512`
- Reduces peak VRAM at cost of slightly slower prefill
- Typical range: 128-512

---

### `--cache-type-k` and `--cache-type-v` (KV Cache Quantization)

**What they control:**
Quantization format for the key (K) and value (V) components of the attention cache.

**Impact on memory and quality:**

| Format | VRAM Usage | Quality | Notes |
|--------|-----------|---------|-------|
| q8_0 | baseline | Highest precision | Unquantized storage |
| q4_0 | 50% of baseline | Imperceptible loss | Recommended choice |
| q4_1 | 50% of baseline | Imperceptible loss | Slightly different distribution |
| q3_0 | 37% of baseline | Minor loss | For extreme constraints |

**How to set:**
- Use `q4_0` for both (recommended)
- Provides 50% memory savings with negligible quality impact
- Only use q3_0 if running extremely low on VRAM

---

### `--flash-attn` (Flash Attention)

**What it controls:**
Whether to use the Flash Attention algorithm for computing attention. This is a fused kernel implementation that avoids materializing the full attention matrix.

**Impact:**
- Speed: 40-50% faster attention computation
- Memory: Significantly reduces peak VRAM during attention (important for long contexts)
- Quality: No impact (mathematically equivalent)

**How to set:**
- Set to `on` if your GPU supports it (most modern GPUs do)
- Most recent NVIDIA GPUs (RTX 20 series and newer) support it
- Older GPUs or AMD GPUs may not; the server will warn if unsupported

---

## Advanced Parameters

### `--n-cpu-moe` (CPU Threads for MoE Routing)

**What it controls:**
CPU thread allocation specifically for Mixture-of-Experts (MoE) expert routing operations.

**When to use:**
- Only for MoE models (e.g., Qwen-30B-A3B, Mixtral 8x7B)
- Dense models don't use this parameter
- MoE models have a router layer that selects which experts to activate for each token

**What it affects:**
- Router computation speed (which experts to activate)
- **Does NOT increase memory usage** - uses CPU threads already allocated via `--threads`
- Faster router = faster expert selection = lower latency

**How it works:**
A MoE model typically has:
1. Router layer: decides which N experts to activate (runs on CPU in llama.cpp)
2. Expert layers: selected experts perform actual computation (can be on GPU or CPU)

More `--n-cpu-moe` threads = faster router decision = overall faster inference for MoE models.

**How to set:**
- For MoE models: set to your `--threads` value or leave unset (will auto-tune)
- For dense models: omit this parameter entirely
- Example: `--threads 16 --n-cpu-moe 16`

**Performance impact:**
- Negligible if already using all available CPU threads
- Can improve MoE performance by 5-10% if threads were underutilized

---

### `--mmap` (Memory Mapping)

**What it controls:**
Whether to memory-map the model file instead of fully loading it into RAM.

**When mmap is enabled:**
- Model file is mapped to virtual memory
- Pages are loaded on-demand
- Reduces initial RAM requirement
- Slightly slower if model doesn't fit in RAM (reads from disk)

**How to set:**
- Default: enabled (on)
- Keep enabled for large models that don't fit in available RAM
- Disable (via `--no-mmap`) only if you have enough RAM for full model and want maximum speed

---

### `--numa` (NUMA Awareness)

**What it controls:**
Whether to use NUMA-aware memory allocation for multi-socket systems.

**When to use:**
- Only on systems with multiple CPU sockets (typically servers/workstations, not consumer desktops)
- Check with: `numactl --hardware` (should show multiple nodes)
- Consumer systems: leave disabled

**Impact:**
- Improves performance on NUMA systems by respecting socket locality
- Negligible impact on single-socket systems

---

### `--min-gpu-layers` (Minimum GPU Layers)

**What it controls:**
Minimum number of layers to keep on GPU even if VRAM would be exceeded elsewhere.

**Use case:**
- Rarely needed
- Only relevant for split-quantization models
- Most users should leave at default (1)

---

## Sampling Parameters

These control generation behavior, not performance:

### `--temperature` (Sampling Temperature)

Controls randomness of token selection:
- `0.0`: Always pick most likely token (deterministic, can be repetitive)
- `0.5`: Conservative sampling (good for facts, coding)
- `1.0`: Balanced (default for most tasks)
- `1.5+`: High randomness (creative but potentially incoherent)

**Recommended:**
- Facts/analysis: 0.3-0.5
- General chat: 0.7-0.8
- Creative writing: 0.9-1.2

---

### `--top-p` (Nucleus Sampling)

Controls diversity of token choices:
- `0.8`: Conservative, stick to most likely tokens
- `0.9`: Slightly more diverse
- `0.95`: Default, good balance
- `0.99`: High diversity, allow rare choices

**Recommended:** 0.9-0.95 for most tasks

---

### `--repeat-penalty` (Repetition Penalty)

Penalizes recently-used tokens to reduce repetition:
- `1.0`: Disabled
- `1.1`: Light penalty (default)
- `1.2-1.3`: Medium penalty (if model repeats)
- `1.5+`: Aggressive (may affect coherence)

**Recommended:** 1.1 default, increase to 1.2-1.3 if model repeats

---

### `--min-p` (Minimum Probability)

Discard tokens with very low probability:
- `0.0`: Disabled (default)
- `0.01-0.05`: Light filtering
- `0.05-0.1`: Medium filtering (reduces gibberish)

**Recommended:** 0.05 if model generates gibberish

---

## Memory Architecture

### VRAM Allocation Pattern

```
GPU VRAM (total available, e.g., 24GB)
├─ Model layers (n-gpu-layers × layer_size)
├─ KV cache (GPU-resident portion, grows with context)
├─ Attention computations (temporary, freed after use)
└─ CUDA runtime overhead (~0.5-1.0 GB)
```

### RAM Allocation Pattern

```
System RAM (total available, e.g., 64GB)
├─ CPU model layers ((total_layers - n-gpu-layers) × layer_size)
├─ KV cache (CPU-resident portion)
├─ Operating system / other processes (~2-3GB typical)
└─ Free headroom (should keep ≥ 2-4GB available)
```

### KV Cache Growth

KV cache grows linearly with context:
- 8192 tokens: ~1 GB (q4_0) to ~2 GB (q8_0)
- 16384 tokens: ~2 GB (q4_0) to ~4 GB (q8_0)
- 32768 tokens: ~4 GB (q4_0) to ~8 GB (q8_0)

Use q4_0 quantization for KV cache to minimize memory usage.

---

## Configuration Examples

### High-Performance Desktop (RTX 4090, 128GB RAM, 32-core CPU)

```bash
llama-server \
  --model model.gguf \
  --n-gpu-layers 80 \
  --threads 32 \
  --ctx-size 32768 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on
```

Expected: 60-100 tok/s, full context window

### Balanced Workstation (RTX 4070, 64GB RAM, 12-core CPU)

```bash
llama-server \
  --model model.gguf \
  --n-gpu-layers 40 \
  --threads 12 \
  --ctx-size 16384 \
  --batch-size 1024 \
  --ubatch-size 256 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on
```

Expected: 25-40 tok/s, good context window

### Constrained Laptop (RTX 3060, 16GB RAM, 8-core CPU)

```bash
llama-server \
  --model model.gguf \
  --n-gpu-layers 20 \
  --threads 8 \
  --ctx-size 8192 \
  --batch-size 512 \
  --ubatch-size 128 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --flash-attn on
```

Expected: 12-20 tok/s, moderate context window

---

## Tuning Workflow

### 1. Find GPU Layer Ceiling

```bash
# Start conservative
llama-server --model model.gguf --n-gpu-layers 10 ...

# Increase by 2-4 each test
# Watch nvidia-smi for VRAM usage
# Stop when you see "CUDA out of memory"
# Final setting: last successful value - 4
```

### 2. Verify Memory Headroom

Monitor during inference:
```bash
watch nvidia-smi          # VRAM should stay <90%
watch free -h            # Check RAM, verify Swap ≈ 0
```

### 3. Adjust Batch Size (If Needed)

If prefill is slow:
- Increase `--batch-size` by 256 until VRAM hits 95%
- If prefill crashes, reduce by 256

### 4. Benchmark Performance

```bash
# Count tokens generated in 10 seconds
# Typical metric: tokens/second
# Lower is normal for large models with many CPU layers
```

---

## Troubleshooting

### CUDA Out of Memory

**Solution:**
1. Lower `--n-gpu-layers` by 4-8
2. If persists, lower `--ctx-size` by 50%
3. If still fails, try `--batch-size 256`
4. Last resort: use smaller quantization (Q4_K_M → Q3_K_M)

### Slow Inference (< 10 tok/s)

**Check in order:**
1. Are many layers on CPU? (`--n-gpu-layers` too low)
   - Solution: Increase `--n-gpu-layers`
2. Is CPU bottlenecked? (`htop` shows idle cores)
   - Solution: Verify `--threads` equals physical core count
3. Is swap being used? (`free -h` shows Swap > 0)
   - Solution: Reduce `--ctx-size`
4. Is batch size too small? (prefill slow but decode fast)
   - Solution: Increase `--batch-size`

### Repetitive Output

**Solution:**
1. Increase `--repeat-penalty` to 1.2-1.3
2. Lower `--temperature` to 0.5-0.7
3. Enable `--min-p 0.05`

### First Token Slow, Rest Fast

This is prefill bottleneck (expected behavior for large models):
- Profiling prefill is secondary to decode performance
- To improve: lower `--ctx-size` if possible, or accept the latency
- Decode performance is more important for user experience

### Very Long Prefill Time With Long Prompts

**Solution:**
1. Increase `--batch-size` to process prompt faster
2. Increase `--n-gpu-layers` if VRAM allows
3. Reduce prompt length if acceptable for your use case

---

## Performance Monitoring

### Real-Time Metrics

```bash
# GPU utilization and memory
nvidia-smi

# System memory
free -h

# CPU cores active
top -p $(pgrep llama-server)
# or: htop
```

### Generation Metrics to Track

- **Tokens per second (tok/s):** Primary metric
  - Typical range: 5-100 tok/s depending on hardware
  - Scale roughly: (GPU layers) × (GPU speed) + (CPU layers) × (CPU speed)

- **First token latency:** Secondary metric
  - Should be < 1 second for interactive use
  - Dominated by prefill computation for long prompts

- **Memory efficiency:** Verify no swap usage
  - Check `free -h` output
  - Swap indicates insufficient RAM for current config

---

## Using with AI Coding Tools

Once llama-server is running (e.g. on port 11434), you can use it as a backend for Claude Code and OpenAI Codex CLI via its OpenAI-compatible API.

### Claude Code

```bash
# Point Claude Code at your local llama.cpp server
ANTHROPIC_BASE_URL=http://localhost:11434/v1 \
ANTHROPIC_API_KEY=dummy \
claude --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M
```

The model name must match the one loaded by llama-server. The API key can be any non-empty string since llama.cpp doesn't authenticate.

### OpenAI Codex CLI

```bash
# Point Codex at your local llama.cpp server
OPENAI_BASE_URL=http://localhost:11434/v1 \
OPENAI_API_KEY=dummy \
codex --model unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_M
```

### Notes

- Both tools expect an OpenAI-compatible `/v1/chat/completions` endpoint, which llama-server provides when started with `--jinja`.
- Use `--port` to change the server port if 11434 conflicts with another service (e.g. Ollama).
- For best results with coding tasks, use a model with large context (`--ctx-size 32768` or higher) and low temperature (`--temperature 0.3-0.6`).

---

## References

- **llama.cpp GitHub:** https://github.com/ggerganov/llama.cpp
- **Server documentation:** https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
- **Model zoo:** https://huggingface.co/models (filter for GGUF format)
- **Benchmarking:** Use `llama-bench` tool included with llama.cpp

