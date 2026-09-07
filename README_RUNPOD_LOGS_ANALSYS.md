# What vLLM actually does before it answers

Cold start is not “the container is Running.” It is a short, ordered movie: parse the model, pick CUDA kernels, pull weights, compile graphs, carve KV cache, then open an HTTP door.

This walkthrough uses one real boot of **vLLM 0.22.1** serving **Qwen2.5-7B-Instruct-AWQ** (GitHub Actions → Runpod, 2026-09-07). Clock starts at `13:15:37` UTC. The OpenAI API is up at `13:17:44`. First `/v1/completions` returns **200** at `13:17:51`.

How this file is used: the [RunPod Management](README.md#runpod-management) section in `README.md` (metrics, knobs, 10M/h Ready time, KV dashboards) is keyed off these lines. Commands to reproduce the boot: [`imp-commands-runpod.md`](imp-commands-runpod.md).

## Contents

- [The two-minute movie](#the-two-minute-movie)
- [Two processes, not one Python script](#two-processes-not-one-python-script)
- [The banner: which model, which knobs](#the-banner-which-model-which-knobs)
- [Hugging Face, safetensors, and AWQ](#hugging-face-safetensors-and-awq)
- [CUDA picks a team: NCCL, FlashAttention, FlashInfer](#cuda-picks-a-team-nccl-flashattention-flashinfer)
- [Weights leave the Hub and occupy VRAM](#weights-leave-the-hub-and-occupy-vram)
- [torch.compile, then CUDA graphs](#torchcompile-then-cuda-graphs)
- [KV cache: why 7B is not “7B of GPU”](#kv-cache-why-7b-is-not-7b-of-gpu)
- [The OpenAI door opens](#the-openai-door-opens)
- [First request: Triton still had homework](#first-request-triton-still-had-homework)
- [Where the two minutes went](#where-the-two-minutes-went)
- [What to remember](#what-to-remember)

---

## The two-minute movie

| Clock (UTC) | Beat |
|---|---|
| 13:15:37 | `vllm serve` prints the banner. HTTP is **not** up. |
| 13:15:54 | Architecture resolved: `Qwen2ForCausalLM`. Context cap: **4096**. |
| 13:16:10 | EngineCore (pid 348) starts the **V1** engine. |
| 13:16:12 | Attention backend: **FlashAttention 2**. |
| 13:16:13 → 13:17:03 | Hugging Face weight download: **50.8 s**, **5.19 GiB**. |
| 13:17:04 | Safetensors shards loaded into GPU in **0.61 s**. |
| 13:17:05 | Weights + runtime in VRAM: **5.29 GiB**. |
| 13:17:11 → 13:17:25 | `torch.compile` (Inductor): **19.79 s**. |
| 13:17:36 | KV cache sized: **14.43 GiB**, **270,144 tokens**. |
| 13:17:38 | CUDA graphs captured. Engine init done. |
| 13:17:44 | `Starting vLLM server on http://0.0.0.0:8000` |
| 13:17:48 | `GET /v1/models` → **200** |
| 13:17:51 | `POST /v1/completions` → **200** |

If you curl during the download or compile, you get 502. The GPU is busy becoming a language model, not refusing you.

---

## Two processes, not one Python script

The log splits into **APIServer pid=95** and **EngineCore pid=348**. That is vLLM’s V1 layout.

- **APIServer** talks HTTP: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, `/health`.
- **EngineCore** owns the GPU: load weights, attention, KV cache, sampling.

They meet over a local socket (`distributed_init_method=tcp://172.20.0.2:40033`, `backend=nccl`). For this lab `world_size=1` — one GPU, no tensor parallel. NCCL is still initialized so the same code path works when you later set `tensor_parallel_size=2`.

Serving is not “Flask plus PyTorch in one process.” It is a front door and an engine.

---

## The banner: which model, which knobs

```text
version 0.22.1
model   Qwen/Qwen2.5-7B-Instruct-AWQ

non-default args:
  host=0.0.0.0
  dtype=float16
  max_model_len=4096
  quantization=awq
  gpu_memory_utilization=0.9
  max_num_seqs=8
```

Read those as product decisions, not CLI trivia:

| Knob | Meaning |
|---|---|
| **Instruct-AWQ** | Chat-tuned Qwen, weights already **4-bit-ish** (AWQ). Smaller download, less VRAM than fp16 7B. |
| **dtype=float16** | Compute in FP16 on the GPU. Activations, not the stored AWQ integers. |
| **max_model_len=4096** | Hard cap on prompt + generated tokens. Shorter context → more concurrent requests in the same KV pool. |
| **gpu_memory_utilization=0.90** | Leave ~10% of the card for CUDA / fragmentation. The rest is weights + KV + graphs. |
| **max_num_seqs=8** | Scheduler will not run more than 8 sequences at once, even if KV cache could hold more. |
| **host=0.0.0.0** | Bind every interface so a proxy (Runpod `8000/http`) can reach the process. `127.0.0.1` would serve only itself. |

vLLM also warns that `--model` is going away; the new style is `vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ`. Same serving, cleaner CLI.

---

## Hugging Face, safetensors, and AWQ

Two Hub warnings appear (`HF_TOKEN`). This repo is **public**. Unauthenticated download still works; it is just slower and rate-limited. A token does not change the model. It changes how fast the 5.19 GiB arrives.

The API process **parses** safetensors **before** the engine finishes downloading:

```text
Parse safetensors files: 100% | 2/2
Resolved architecture: Qwen2ForCausalLM
```

Safetensors is not pickle. It is a mapped weight format: headers first, tensors later. Parsing two shard files tells vLLM the architecture without loading 5 GiB into VRAM yet.

Then a fork in the road:

```text
Detected that the model can run with awq_marlin,
however you specified quantization=awq explicitly, so forcing awq.
Use quantization=awq_marlin for faster inference
```

**AWQ** is how the weights were compressed. **Marlin** is a faster CUDA kernel that can *consume* those weights on many NVIDIA cards. We pinned `quantization=awq`, so vLLM honors the pin instead of auto-upgrading. For a lab that is honest: you get the slower-but-explicit kernel. For production, try `awq_marlin` and measure.

`enable_prefix_caching=True` and `enable_chunked_prefill=True` are on by default in this V1 engine. Shared prompt prefixes reuse KV. Long prompts are chunked so one whale request does not stall the batch.

---

## CUDA picks a team: NCCL, FlashAttention, FlashInfer

```text
device_config=cuda
Using FLASH_ATTN attention backend
Using FlashAttention version 2
Using FlashInfer for top-p & top-k sampling
```

Three different CUDA stories:

1. **Device is CUDA.** There is no CPU fallback for this serve. If the NVIDIA container runtime had failed (`cuda>=13.0`), this line would never appear.
2. **FlashAttention 2** wins over FlashInfer / Triton / FlexAttention for **attention** (the Q·Kᵀ·V math that is most of a transformer). FA2 is fused, IO-aware, and what you want on Ampere (RTX 3090).
3. **FlashInfer** is used for **sampling** (top-p / top-k), not attention. Sampling is tiny compared to attention, but it still runs on GPU so the decode loop does not bounce to CPU every token.

`kv_cache_dtype=auto` means KV stays in a dtype the engine picks (typically matching compute). FP8 KV is a later optimization; it is not on in this log.

---

## Weights leave the Hub and occupy VRAM

This is the long silence in the console — about **51 seconds** of download, then a blink of load:

```text
Time spent downloading weights for Qwen/Qwen2.5-7B-Instruct-AWQ: 50.804084 seconds
Checkpoint size: 5.19 GiB
Loading safetensors checkpoint shards: 100% | 2/2
Loading weights took 0.61 seconds
Model loading took 5.29 GiB memory and 52.678780 seconds
```

A 7B **dense** model in FP16 is ~14 GiB of weights. AWQ packed this one to **5.19 GiB** on disk and **5.29 GiB** on the GPU (weights + a little runtime). That is why a 24 GiB 3090 has room left for KV cache.

Load is fast because safetensors can stream into GPU memory. Download is slow because the Hub is the internet. Mount `/root/.cache/huggingface` and the **second** boot skips the 50 seconds.

`download_dir=None` means the default HF cache. The volume in this lab is that cache. The log’s “XFS, auto-prefetch disabled” line is vLLM saying: this is a local disk, not Lustre, so I will not prefetch like an HPC filesystem. Fine for a single GPU.

---

## torch.compile, then CUDA graphs

After weights sit in VRAM, vLLM still does not serve. It **compiles** the model:

```text
Using cache directory: .../torch_compile_cache/.../backbone for vLLM's torch.compile
Dynamo bytecode transform time: 6.49 s
Compiling a graph for compile range (1, 2048) takes 9.28 s
torch.compile took 19.79 s in total
```

**TorchDynamo** traces Python into a graph. **Inductor** turns that graph into fused CUDA kernels. Compile range `(1, 2048)` means: optimize for sequences from 1 token up to 2048 in that compiled bucket (the model still allows 4096; the rest is handled with other shapes / eager).

Then **CUDA graphs**: record a sequence of GPU launches and replay them with almost no CPU.

```text
Profiling CUDA graph memory: PIECEWISE=5 (largest=16), FULL=4 (largest=8)
Estimated CUDA graph memory: 0.35 GiB
Graph capturing finished in 1 secs, took 0.33 GiB
```

- **FULL** graphs: decode (one new token, batch 1–8).
- **PIECEWISE** graphs: mixed prefill+decode, batch up to 16.

`max_num_seqs=8` matches the FULL capture sizes `[1, 2, 4, 8, 16]` (16 is piecewise). Replay is why decode can stay in the hundreds of tokens/s on a 3090 after warmup. First boot pays ~20 s of compile; a later boot with the same cache directory pays much less.

`enforce_eager=False` in the engine config is the switch that *allows* this. Eager mode would skip graphs and start sooner, then run slower forever.

---

## KV cache: why 7B is not “7B of GPU”

The line that decides concurrency:

```text
Available KV cache memory: 14.43 GiB
GPU KV cache size: 270,144 tokens
Maximum concurrency for 4,096 tokens per request: 65.95x
```

Every generated token (and every prompt token) stores keys and values per layer. That store is the **KV cache**. Weights were 5.29 GiB. CUDA graphs ~0.33 GiB. **Most of the rest of the 90% GPU budget is KV.**

270,144 tokens of cache ÷ 4,096 tokens per request ≈ **66** full-length requests theoretically. `max_num_seqs=8` caps the scheduler at 8, so this card is KV-rich and scheduler-conservative. Turn `max_num_seqs` up only if you also watch p99 latency.

vLLM also notes that CUDA-graph memory profiling (default since 0.21) makes `--gpu-memory-utilization=0.90` behave like **0.885** of older math. To keep the same KV as pre-0.21, bump utilization to **0.9148**. Ignore this until you are squeezing the last gigabyte.

`--gpu-memory-utilization` does **not** mean “use 90% for weights.” Weights are fixed. Utilization is “how much of the card vLLM may claim,” and **KV gets the remainder**.

---

## The OpenAI door opens

```text
Supported tasks: ['generate']
Starting vLLM server on http://0.0.0.0:8000
Application startup complete.
GET /v1/models  200
```

Only now is “the server up.” Routes that matter for this lab:

| Route | Role |
|---|---|
| `GET /v1/models` | Is anyone home? Returns `Qwen/Qwen2.5-7B-Instruct-AWQ`. |
| `POST /v1/completions` | Raw prompt completion (what we curl). |
| `POST /v1/chat/completions` | Chat messages + template. |
| `GET /health` | Liveness. |
| `GET /metrics` | Prometheus counters. |

`GET /` is not in the list. Probes to `/` 404. That is not a crash.

Qwen’s `generation_config.json` overrides defaults (`temperature=0.7`, `top_p=0.8`, `top_k=20`, `repetition_penalty=1.05`). Our curl used `temperature=0`, which wins **for that request**. Config is the default, not a prison.

Chat template is detected as `string`. Completions API does not need it; chat API does.

---

## First request: Triton still had homework

```text
Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel.
This causes a latency spike; consider extending warmup to cover this shape/config.
POST /v1/completions  200
```

Warmup ran one profiling pass. The **first real** completions shape still compiled a Triton kernel (`_compute_slot_mapping_kernel`) — how tokens map into KV slots. That is a one-time JIT. The log even says it will warn on future surprise compiles.

Then the engine’s periodic line:

```text
Avg prompt throughput: 0.4 tokens/s
Avg generation throughput: 2.3 tokens/s
Running: 0 reqs
GPU KV cache usage: 0.0%
Prefix cache hit rate: 0.0%
```

Those averages are across a quiet window that included **one** 32-token completion. They are not the card’s peak. After the request, Running=0 and KV usage=0% — cache was freed. Prefix hit 0% because there was no second overlapping prompt.

**PASS** is the 200 on `/v1/completions` with a `"text"` field, not this throughput line.

---

## Where the two minutes went

From banner to `Application startup complete` ≈ **127 seconds**.

| Stage | ~Time | What you wait for |
|---|---|---|
| Import CUDA + resolve config | ~20 s | Python, cuda-bindings deprecation noise |
| **HF download** | **51 s** | Network. Cache this. |
| Copy weights to GPU | 1 s | Disk → VRAM |
| **torch.compile** | **20 s** | First boot. Cache this too. |
| KV profile + CUDA graphs | ~13 s | Must happen on this GPU |
| Bind HTTP | 2 s | FastAPI / Uvicorn |

The image (`vllm/vllm-openai:v0.22.1-cu129`) is **not** in this container log. Docker pull already finished before pid 95 printed the banner. If you only watch **container** logs, you miss a multi-gigabyte image extract. System logs show that pull; these logs start at `vllm serve`.

CUDA 12.9 vs 13 is also **before** this file. If `nvidia-container-cli` rejects the image, you never get a banner.

---

## What to remember

1. **Ready ≠ Running.** Ready is `Application startup complete` plus a 200 on `/v1/models`.
2. **Two clocks:** image pull (Docker) then model pull (Hugging Face). Both bill the GPU.
3. **AWQ is the reason 7B fits** in ~5.3 GiB of weights on a 24 GiB card.
4. **KV cache, not parameter count, sets concurrency.** Here 14.43 GiB of KV vs 5.29 GiB of weights.
5. **FlashAttention 2** is the attention kernel; **FlashInfer** is sampling; **NCCL** is ready for multi-GPU even when `world_size=1`.
6. **Compile and CUDA graphs** buy decode speed. First request may still JIT a Triton kernel.
7. **Second start** with a warm HF volume + compile cache skips the two slowest chapters.

The model did not “boot.” It was **assembled**: quantized weights, fused attention, a KV arena, recorded GPU graphs, then an OpenAI-shaped door on port 8000.
