# LLM-Inference-Serving-Azure-AKS

Provisioning a GPU on Azure AKS with Terraform, install the NVIDIA GPU Operator, and server Qwen2.5-7B-instruct-AWQ through vLLM's OpenAI-compatible API - a production shaped inference endpoint, reachable via curl /v1/completions

## Contents

- [Why self-hosting LLM Serving instead of calling Frontir Model(s) API?](#why-self-hosting-llm-serving-instead-of-calling-frontir-models-api)
- [Open Source vs Closed Source Models](#open-source-vs-closed-source-models)
- [vLLM Model Serving on Azure AKS](#vllm-model-serving-on-azure-aks)
  - [Why Kubernetes?](#why-kubernetes)
  - [Why AKS Specifically?](#why-aks-specifically)
- [The Driver Decision](#the-driver-decision)
  - [Why hand the driver to the Operator rather than the node image?](#why-hand-the-driver-to-the-operator-rather-than-the-node-image)
  - [Measured memory at --gpu-memory-utilization=0.90](#measured-memory-at---gpu-memory-utilization090)
- [Pinned Versions](#pinned-versions)
  - [This repo UV Setup on Macbook](#this-repo-uv-setup-on-macbook)
- [RunPod Management](#runpod-management)
  - [GitHub Actions](#github-actions)
  - [vLLM-specific metrics](#vllm-specific-metrics)
  - [Observability tools](#observability-tools)
  - [Configurations](#configurations)
  - [100k requests per hour](#100k-requests-per-hour)
  - [Cluster of vLLM, API router, load balancer](#cluster-of-vllm-api-router-load-balancer)
- [References](#references)

---

## Why self-hosting LLM Serving instead of calling Frontir Model(s) API?
* Cost at scale - per-token API pricing stops making sense past a certain volume
* Data and privcay - prompts never leave your network
* Model and latency control - run whatever model you want, tune the engine,  own the tail latency instead of inherting someone else's

## Open Source vs Closed Source Models
* Open Source - the weights are downloadable (Hugging Face and similar). You get the actual parameter file, can run it on your own GPU, fine-tune it, with no per-token fee. Examples: Llama, Qwen, Mistral, DeepSeek.

*Closed Source - you only get an API endpoint (GPT, Claude, Gemini). No weights, no self-hosting, you need a prompt over HTTPs and pay per token.

You cannot self-host a closed model - there are no weights to put on a GPU. That is why this repo serves Qwen, an open-weight model.

## vLLM Model Serving on Azure AKS

### Why Kubernetes?
A single GPU box running python -m vllm... would serve tokens too. Kubernetes earns its place on what comes after:


* Scheduling - GPUs become a countable resource (nvidia.com/gpu: 1), not just a machine (note that GPU is an expensive resource)
* Isolation - taints and source requests keep cheap pods off the expensive card and stop two containers fighting over one GPU
* Self-healing - the pod dies, it restarts behind a stable Service address
* Autoscaling - HPA/KEDA on the pod, node pool to zero between sessions
* Rollouts - canary and blue/green deployments are built-in primitives, no scripts
* Ecosystem - GPU Operator, DCGM Metrics, MIG and time-slicing all ship as Kubernetes components
* Portability - the same manifests run on any cloud's managed Kubernetes

### Why AKS Specifically?
Azure runs the control plane for free (sku_tier = "Free" ), and node pools are a first-class resource, so the GPU pool is one Terraform block.

## The Driver Decision
AKS can install an NVIDIA driver for you. This repo deliberately tells it not to.

```text
gpu_driver = "None"
driver.enabled = true
```

### Why hand the driver to the Operator rather than the node image?
The drivers lifecycle becomes decouple from the OS image. You can upgrade it or pin it per node pool without rebuilding an image drift between nodes stops being invisible, and the driver container toolkit, device plugin, node-feature-discovery and DCGM exporter all arrive as one Helm-versioned, mutually validated stack.

The cost is cold start. The driver DaemonSet must pull, install and pass health checks before the node can schedule the GPU pods at all. That is a real tradeoff, not a free win.

### Measured memory at --gpu-memory-utilization=0.90
| Item | VRAM |
| --- | --- |
| Total card | 15360 MiB |
| Allocated by vLLM (nvidia-smi) | 13605 MiB |
| Weights + runtime overhead | ~ 5.97 GiB |
| Available KV Cache | 7.32 GiB |

KV Cache - not weights - sets the concurrency ceiling. KV Cache is useful for concurrency.

## Pinned Versions
| Component | Pin |
| --- | --- |
| Terraform | ~> 1.15 |
| azurerm provider | ~> 4.0 |
| Kubernetes / Helm Providers | ~> 2.35 / ~> 2.17 |
| AKS Kubernetes Version | null AKS default for the region |
| GPU VM | Standard_NC4as_T4_v3 |
| GPU node OS | Ubuntu2204 (containerd 1.7 - see traps above) |
| System VM | Standard_D2s_v3 |
| GPU Operator Chart | v26.3.2, driver.enabled=true |
| vLLM image | vllm/vllm-openai:v0.22.1 (never: latest) |
| Model | Qwen/Qwen2.5-7B-Instract-AWQ (ungate - no HF token needed) |

Pinning the vLLM image matters more than it looks: :latest changes engine defaults under you, and a config that worked yesterday OOMs today. On a 15GiB card that margin is thin.

### This repo UV Setup on Macbook
```shell
uv add torch torchvision transformers streamlit requests pyairports
```
- Below command does not work on Mac because those packages are a Linux + NVIDIA GPU serving stack. 
- uv add tries to put them in the local .venv, and that is what blew up:
- vllm xformers Not installable — they need CUDA
- So Run vLLM on the AKS GPU node (the vllm/vllm-openai image), not in Mac laptop venv.

```shell
# Does not work on Macbook
uv add torch torchvision vllm transformers streamlit requests pyairports xformers 

```

## RunPod Management

Azure T4 quota blocked AKS (`NCasT4v3` 0/0). Same model and OpenAI API were served on a **Runpod community RTX 3090** instead. That is one Docker container, not Kubernetes. Image pin on GeForce hosts: `vllm/vllm-openai:v0.22.1-cu129` (plain `v0.22.1` is CUDA 13 and will not start).

CLI runbook: `imp-commands-runpod.md`. Terraform: `terraform/runpod/`. Cold-start log walkthrough: `README_RUNPOD_LOGS_ANALSYS.md`.

### GitHub Actions

Secret: **`RUNPOD_API_KEY`** (repo Settings → Secrets and variables → Actions).

| Workflow | What it does |
|---|---|
| `RUNPOD-ALL-SETUP-0001-EndToEnd` | List pods → `terraform apply` → curl `/v1/models` + `/v1/completions` |
| `RUNPOD-ALL-DESTROY-0001-EndToEnd` | `terraform destroy` → list pods (GPU bill → $0) |
| `RUNPOD-0001` … `0004` | Same steps, one workflow each |

Do not run Setup twice without Destroy. GPU bills from apply until destroy.

### vLLM-specific metrics

The server already exposes `GET /metrics` (Prometheus text, prefix `vllm:`). Your boot also logged engine stats every few seconds.

**Engine (is the GPU busy?)**

| Metric | Why it matters |
|---|---|
| `vllm:num_requests_running` / `_waiting` | In a batch vs queued. Waiting up → another replica or less load. |
| `vllm:kv_cache_usage_perc` | KV arena fill. Near 1.0 → preemption, not “CPU 80%.” |
| `vllm:prefix_cache_hits` / `_queries` | Shared-prompt reuse. High hits mean round-robin LB wastes KV. |
| `vllm:prompt_tokens_total` / `generation_tokens_total` | Real work. Scale on **tokens/s**, not request count. |

**Request SLOs (histograms)**

| Metric | Meaning |
|---|---|
| `vllm:time_to_first_token_seconds` | **TTFT** — prefill + queue. Chat UX. |
| `vllm:inter_token_latency_seconds` | **TPOT** — time per output token. Streaming. |
| `vllm:e2e_request_latency_seconds` | Whole request. |
| `vllm:request_queue_time_seconds` / `_prefill_` / `_decode_` | Where the seconds went. |
| `vllm:request_success_total` | `stop` vs `length` vs `abort`. |

Plus FastAPI HTTP counters on `/v1/completions`. After the server is up:

```bash
curl -sf "https://${POD_ID}-8000.proxy.runpod.net/metrics" | head
```

### Observability tools

| Layer | Tool |
|---|---|
| Scrape `/metrics` | Prometheus (or Grafana Alloy / OpenTelemetry Collector) |
| Dashboards | vLLM’s example Grafana board (TTFT, TPOT, KV %, running/waiting) |
| Traces | OpenTelemetry — `--otlp-traces-endpoint` (unset on the 2026-09-07 boot) |
| GPU hardware | DCGM Exporter (`gpu_memory_used`, SM util) — this is why AKS + GPU Operator still matters |
| Logs | Engine INFO; Triton JIT warnings on first request shapes |

### Configurations

Serve knobs already used in this lab: `gpu_memory_utilization=0.90`, `max_model_len=4096`, `max_num_seqs=8`, `quantization=awq`, `dtype=float16`.

| Extra | Effect |
|---|---|
| `--disable-log-stats` | Quieter logs; `/metrics` stays. |
| `--otlp-traces-endpoint` | Per-request traces. |
| `--collect-detailed-traces=model\|worker\|all` | Heavier traces for debugging. |
| `--generation-config vllm` | Ignore Qwen’s `generation_config.json` defaults. |
| `quantization=awq_marlin` | Faster AWQ kernels (the boot log suggested this). |
| Prefix caching (on in V1) | Pays off only if the **router** keeps similar prefixes on the same replica. |

`--gpu-memory-utilization` is not “90% for weights.” Weights are ~5.3 GiB on this AWQ 7B. The rest of the budget is **KV cache** (14.43 GiB on the 3090 boot) plus CUDA graphs.

### 100k requests per hour

100 000 ÷ 3 600 ≈ **27.8 requests/second** (~1 667/min). That is a *request* rate. Capacity is **tokens × batching**.

This lab replica (`max_num_seqs=8`, 32-token completions) can sit in the tens of RPS if prompts are short. If each request takes ~0.7–1 s of GPU time, 8 slots give roughly **8–12 RPS** sustained. **28 RPS of that shape is about 2–4 GPUs**, plus headroom for peaks (an hourly average hides a 3–5× spike). Long prompts or 512-token answers need more replicas, not a faster load balancer.

### Cluster of vLLM, API router, load balancer

Two different “clusters”:

1. **One replica, many GPUs** — `tensor_parallel_size` / `pipeline_parallel_size`. One HTTP endpoint. Use when the model does not fit one card (not this 7B AWQ).
2. **Many replicas** — what 100k/hour needs. Then something must sit in front.

| Front door | When |
|---|---|
| K8s Service / nginx / cloud LB (round-robin) | Simple. Fine at low QPS. **Breaks prefix cache** — each replica has its own KV. |
| vLLM `--data-parallel-size` + internal API-server scale-out | One logical engine group, internal LB. |
| Gateway API Inference Extension / **llm-d** / Envoy + Endpoint Picker | Production: route by **KV load + prefix locality**, not TCP round-robin. |
| Prefill/decode split | Huge traffic. Overkill for ~28 RPS of 7B. |

Yes, there is an API router. A generic L4/L7 balancer is only half of it. Inference-aware routing exists because **KV cache is sticky**. Round-robin is like spraying sessions across databases with no affinity.

For ~28 RPS of short 7B AWQ completions: start with **2–4 replicas** behind a Service, scrape `/metrics`, watch `kv_cache_usage_perc` and `num_requests_waiting`. If a shared system prompt must hit prefix cache, move to prefix-aware routing (llm-d / Inference Gateway).

A Runpod pod is **one replica, no router**. Kubernetes is what you add when you want a Service, HPA, DCGM, and a real front door.

## References
https://www.youtube.com/watch?v=EyXzfnAxCdA
https://github.com/shiqs90/vllm-serving-aks
https://github.com/vishakhasadhwani/llm-deployment-demo
https://docs.vllm.ai/en/latest/usage/metrics/
https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md
https://github.com/llm-d/llm-d

