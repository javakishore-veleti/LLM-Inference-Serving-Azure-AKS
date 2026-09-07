# Important Commands — Runpod vLLM (same model as AKS)

Every command for serving `Qwen/Qwen2.5-7B-Instruct-AWQ` on Runpod, in order, with what it does and why.
Copy/paste-able. **You run these** — do not need Azure GPU quota.

Same payload as the AKS lab: image `vllm/vllm-openai:v0.22.1`, OpenAI-compatible `/v1/completions`.
GPU here is a community **RTX 3090 / 4090** (~16–24 GiB), not Azure T4.

---

## 0. Already running from an earlier agent create — terminate it yourself

A pod was created before you asked to run commands yourself. It bills until **you** delete it.

```bash
runpodctl pod list
runpodctl pod delete fq4som4vyjw3h3
```

If `runpodctl` is not installed yet, use the console: [Pods](https://console.runpod.io/pods) → `vllm-qwen25-7b-awq` → terminate.

A template `vllm-qwen25-7b-awq` (`uhotl56xrq`) may already exist. Reuse it in step 3, or ignore it and create a pod from `--image` instead.

---

## 1. One-time setup

### 1.1 Install the CLI

```bash
curl -sSL https://cli.runpod.net | bash
runpodctl update
runpodctl version
```

Homebrew alternative: `brew install runpod/runpodctl/runpodctl`. Then still `runpodctl update`.

### 1.2 API key

Create a key at [console.runpod.io/user/settings](https://console.runpod.io/user/settings).

```bash
export RUNPOD_API_KEY='paste-your-key-here'
runpodctl pod list
```

`pod list` returning `[]` (or existing pods) means auth works. `doctor` is interactive (API key + SSH) if you prefer that over `export`.

---

## 2. Pre-flight — pick a GPU that is in stock

```bash
runpodctl gpu list
runpodctl pod create --help
```

`gpu list` shows names you must pass to `--gpu-id` **exactly**. Prefer 16 GiB+ VRAM for this 7B AWQ model.

| GPU id | Why pick it |
|---|---|
| `NVIDIA GeForce RTX 4090` | 24 GiB, often in stock, community ~$0.34/hr |
| `NVIDIA GeForce RTX 3090` | 24 GiB, cheaper community ~$0.22/hr, can sell out |
| `NVIDIA RTX A5000` | 24 GiB, often cheapest community |

If create says *no instances available*, retry the next row. Do **not** pick H100/H200 for this lab.

---

## 3. Deploy — same vLLM image as AKS

Ports and start args must be set **at create**. You cannot add `8000/http` later without recreating.

```bash
runpodctl pod create \
  --name vllm-qwen25-7b-awq \
  --image vllm/vllm-openai:v0.22.1 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 1 \
  --cloud-type community \
  --disk 50 \
  --volume-size 30 \
  --volume-path /root/.cache/huggingface \
  --ports 8000/http \
  --args "--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 8 --host 0.0.0.0 --port 8000" \
  --wait
```

If 4090 is sold out, change only `--gpu-id` to `"NVIDIA GeForce RTX 3090"`.

`--wait` returns when SSH answers, **not** when the model is loaded. First boot still pulls ~11GB image + weights (~5–15 min).

Note the `id` in the JSON (example shape: `fq4som4vyjw3h3`). Export it:

```bash
POD_ID='paste-pod-id-here'
runpodctl pod get "$POD_ID"
```

Read `runtimeStatus` (usable) not only `desiredStatus` (`RUNNING` can still mean image pull).

Optional: reuse the saved template instead of `--image`:

```bash
runpodctl template list
runpodctl pod create --name vllm-qwen25-7b-awq --template-id uhotl56xrq --gpu-id "NVIDIA GeForce RTX 4090" --wait
```

---

## 4. Watch boot (image pull → weights → KV cache)

```bash
runpodctl pod logs "$POD_ID" --follow
```

System vs container:

```bash
runpodctl pod logs "$POD_ID" --source system     # pull / create / start
runpodctl pod logs "$POD_ID" --source container  # vLLM itself
```

Checkpoint lines (same idea as AKS `kubectl logs deploy/vllm`):

- `Uvicorn running on http://0.0.0.0:8000`
- `Available KV cache memory`
- `GPU KV cache size`
- `Maximum concurrency`

Ctrl-C the follow when those appear.

---

## 5. Verify serving — this is the PASS

Proxy URL is `https://<POD_ID>-8000.proxy.runpod.net`.

```bash
curl -sf "https://${POD_ID}-8000.proxy.runpod.net/v1/models"
```

Then the same completions call as AKS:

```bash
curl -s "https://${POD_ID}-8000.proxy.runpod.net/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","prompt":"Hello, my name is","max_tokens":32,"temperature":0}'
```

**PASS** = JSON with a `"text"` field of generated tokens.

If curl hangs: the proxy is up but vLLM is still loading. Keep watching `pod logs`. If you get 404/502, confirm `--ports 8000/http` was on **create**.

---

## 6. Teardown — run at the END of EVERY session

Community GPU bills while the pod exists.

```bash
runpodctl pod stop "$POD_ID"     # disk can still bill; GPU stops
runpodctl pod delete "$POD_ID"   # gone — this is the $0 state
runpodctl pod list
```

`stop` is a short break. **Done for the day → `delete`.**

Optional safety net on a later create (auto-delete after N hours — check current flag with `runpodctl pod create --help`):

```bash
runpodctl pod create --help | grep -i terminate
```

---

## Console path (same stack, no CLI)

1. [console.runpod.io/pods](https://console.runpod.io/pods) → **Deploy**
2. GPU: RTX 4090 or 3090, **Community**
3. Image: `vllm/vllm-openai:v0.22.1`
4. Container disk 50 GB, volume 30 GB mounted at `/root/.cache/huggingface`
5. Expose HTTP **8000**
6. Docker command / args:

```text
--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 8 --host 0.0.0.0 --port 8000
```

7. Connect tab → HTTP service on 8000 → same `curl` as section 5
8. Terminate when done

---

## Hub alternative (serverless, scale to zero)

Different API than AKS (`/run` not `/v1/completions`). Use only if you want workers that idle at $0.

```bash
runpodctl hub search vllm
runpodctl serverless create \
  --name vllm-qwen-awq \
  --hub-id runpod-workers/worker-vllm \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --env MODEL_NAME=Qwen/Qwen2.5-7B-Instruct-AWQ \
  --env QUANTIZATION=awq \
  --workers-min 0 \
  --wait
```

`--workers-min 0` (default) = no GPU bill while idle. `--workers-min 1` keeps a warm GPU **always billing**.

```bash
runpodctl serverless list
runpodctl serverless health <endpoint-id>
runpodctl serverless delete <endpoint-id>
```
