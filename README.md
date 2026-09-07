# LLM-Inference-Serving-Azure-AKS

Provisioning a GPU on Azure AKS with Terraform, install the NVIDIA GPU Operator, and server Qwen2.5-7B-instruct-AWQ through vLLM's OpenAI-compatible API - a production shaped inference endpoint, reachable via curl /v1/completions

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

### References
https://www.youtube.com/watch?v=EyXzfnAxCdA
https://github.com/shiqs90/vllm-serving-aks
https://github.com/vishakhasadhwani/llm-deployment-demo

