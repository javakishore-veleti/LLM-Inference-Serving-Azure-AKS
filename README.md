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


### UV Setup on Macbook
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

