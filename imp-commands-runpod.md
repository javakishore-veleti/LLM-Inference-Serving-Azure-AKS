# Important Commands — Runpod vLLM (same model as AKS)

Every command for serving `Qwen/Qwen2.5-7B-Instruct-AWQ` on Runpod, in order, with what it does and why.
Copy/paste-able. **You run these** — do not need Azure GPU quota.

Same payload as the AKS lab: OpenAI-compatible `/v1/completions`, vLLM **v0.22.1**.
GPU here is a community **RTX 3090 / 4090** (~16–24 GiB), not Azure T4.

**Image pin on community GPUs:** `vllm/vllm-openai:v0.22.1-cu129` (CUDA 12.9).
Plain `v0.22.1` is CUDA 13.0 and will not start on typical community GeForce drivers (see trap under step 3).

## Contents

- [0. Delete the failed CUDA-13 3090](#0-delete-the-failed-cuda-13-3090--it-still-bills)
- [1. One-time setup](#1-one-time-setup)
  - [1.1 Install the CLI](#11-install-the-cli)
  - [1.2 API key](#12-api-key)
- [2. Pre-flight — pick a GPU](#2-pre-flight--pick-a-gpu-that-is-in-stock)
- [3. Deploy — CUDA 12.9 community](#3-deploy--vllm-v0221-on-cuda-129-community)
- [4. Watch boot](#4-watch-boot-image-pull--weights--kv-cache)
- [5. Verify serving](#5-verify-serving--this-is-the-pass)
- [6. Teardown](#6-teardown--run-at-the-end-of-every-session)
- [Console path](#console-path-same-stack-no-cli)
- [Hub alternative](#hub-alternative-serverless-scale-to-zero)
- [Deployment story — 2026-09-07](#deployment-story--2026-09-07-pass-then-deleted)

---

## 0. Delete the failed CUDA-13 3090 — it still bills

Pod `qckyjw1fehsqix` rented (`desiredStatus: RUNNING`) then died at container start:

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0
```

You cannot change `--image` on a live pod. Delete it, then recreate with `-cu129`.

```bash
runpodctl pod list
runpodctl pod delete qckyjw1fehsqix
runpodctl pod list
```

If the CLI is not ready yet, use the console: [Pods](https://console.runpod.io/pods) → terminate `llm-inf-serving-vllm-qwen25-7b-awq`.

A template `vllm-qwen25-7b-awq` (`uhotl56xrq`) may already exist and still points at CUDA 13. **Do not** reuse it until you edit the template image to `v0.22.1-cu129`. Prefer `--image` in step 3.

---

## 1. One-time setup

### 1.1 Install the CLI

On this Mac, Homebrew is the straightforward path:

```bash
brew install runpod/runpodctl/runpodctl
runpodctl update
runpodctl version
```

Official installer (writes under `/usr/local/bin`, so **bash** needs root, not curl):

```bash
curl -sSL https://cli.runpod.net | sudo bash
runpodctl version
```

`sudo curl … | bash` is wrong — only curl runs as root, bash stays your user, and you get `Please run as root with sudo.` Use `curl … | sudo bash`.

If `runpodctl` is still `command not found`, open a new terminal or add the install dir to `PATH`.

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

Live snapshot from this machine. Re-run `gpu list` before create — stock moves. **Sorted cheapest-first** for this 7B AWQ lab:

| `--gpu-id` | Cloud | ~$/hr | Stock notes |
|---|---|---|---|
| `NVIDIA GeForce RTX 3090` | community | **0.22** | **Cheapest that fits.** 24 GiB. Stock was EU-only. Skip `--public-ip` to keep more inventory. |
| `NVIDIA RTX 2000 Ada Generation` | secure | **0.24** | 16 GiB (same class as Azure T4). Low, EU. |
| `NVIDIA RTX A6000` | community | 0.33 | 48 GiB. Low; `US-TX-1`. |
| `NVIDIA GeForce RTX 4090` | community | 0.34 | 24 GiB. `--public-ip` sold out earlier. |
| `NVIDIA A40` | secure | 0.49 | 48 GiB. **High** stock — pay more to actually get a box. |

Do **not** pick H100/H200/B300/MI300X.

---

## 3. Deploy — vLLM v0.22.1 on CUDA 12.9 (community)

Ports, image, and start args must be set **at create**. You cannot add `8000/http` or swap the image later without deleting and recreating.

**Cheapest first: community RTX 3090** (~$0.22/hr). No `--public-ip` / `--wait` — those shrink community stock (4090 already sold out that way). Poll `pod get` instead.

**Trap — `cuda>=13.0`:** `vllm/vllm-openai:v0.22.1` (no suffix) is built on CUDA 13.0 and needs host driver **R580+**. Community 3090/4090 hosts usually have older GeForce drivers. You cannot upgrade the host driver. Symptom:

```text
error running prestart hook #0: ... nvidia-container-cli: requirement error:
unsatisfied condition: cuda>=13.0, please update your driver to a newer version,
or use an earlier cuda container
```

Fix: same vLLM version, CUDA 12.9 tag: `v0.22.1-cu129`. If *that* fails with `cuda>=12.9`, drop to `vllm/vllm-openai:v0.10.2` (CUDA 12.8) or pay for secure **A40** (datacenter driver; CUDA 13 often works). Do **not** start with `NVIDIA_DISABLE_REQUIRE` — that only skips the check; CUDA 13 still will not run on an old GeForce driver.

```bash
runpodctl pod create \
  --name llm-inf-serving-vllm-qwen25-7b-awq \
  --image vllm/vllm-openai:v0.22.1-cu129 \
  --gpu-id "NVIDIA GeForce RTX 3090" \
  --gpu-count 1 \
  --cloud-type COMMUNITY \
  --container-disk-in-gb 50 \
  --volume-in-gb 30 \
  --volume-mount-path /root/.cache/huggingface \
  --ports '8000/http' \
  --docker-args "--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 8 --host 0.0.0.0 --port 8000"
```

If that returns *no instances available*, next cheapest:

```bash
# 16 GiB T4-class, ~$0.24/hr
--gpu-id "NVIDIA RTX 2000 Ada Generation" --cloud-type SECURE

# 48 GiB community ~$0.33/hr
--gpu-id "NVIDIA RTX A6000" --cloud-type COMMUNITY

# 24 GiB community ~$0.34/hr (sold out earlier with --public-ip)
--gpu-id "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY

# High stock, ~$0.49/hr — pay extra to get a machine
--gpu-id "NVIDIA A40" --cloud-type SECURE --wait
```

Example A40 secure create (plain `v0.22.1` / CUDA 13 — datacenter driver is usually new enough):

```bash
runpodctl pod create \
  --name llm-inf-serving-vllm-qwen25-7b-awq \
  --image vllm/vllm-openai:v0.22.1 \
  --gpu-id "NVIDIA A40" \
  --gpu-count 1 \
  --cloud-type SECURE \
  --container-disk-in-gb 50 \
  --volume-in-gb 30 \
  --volume-mount-path /root/.cache/huggingface \
  --ports '8000/http,22/tcp' \
  --docker-args "--model Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq --dtype float16 --gpu-memory-utilization 0.90 --max-model-len 4096 --max-num-seqs 8 --host 0.0.0.0 --port 8000" \
  --wait
```

`--wait` returns when SSH answers, **not** when the model is loaded. First boot still pulls ~11GB image + weights (~5–15 min).

Note the `id` in the JSON. Export it:

```bash
POD_ID='paste-pod-id-here'
runpodctl pod get "$POD_ID"
```

Read `runtimeStatus` (usable) not only `desiredStatus` (`RUNNING` can still mean image pull).

Optional: reuse the saved template instead of `--image`:

```bash
runpodctl template list
runpodctl pod create --name llm-inf-serving-vllm-qwen25-7b-awq --template-id uhotl56xrq --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type COMMUNITY --public-ip --wait
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

If you see `cuda>=13.0` / `cuda>=12.9` in **system** logs, delete the pod and recreate with an older CUDA tag (step 3 trap). The GPU is billing the whole time the container cannot start.

Ctrl-C the follow when the Uvicorn / KV-cache lines appear.

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
3. Image: `vllm/vllm-openai:v0.22.1-cu129` (community GeForce). Use plain `v0.22.1` only on a secure datacenter GPU (A40) with a new enough driver.
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

---

## Deployment story — 2026-09-07 (PASS, then deleted)

Not Kubernetes. Same OpenAI payload as the AKS lab, on a Runpod community GPU.

| Step | What happened |
|---|---|
| 4090 + `--public-ip` `--wait` | `graphql_error`: no instances available |
| `qckyjw1fehsqix` — 3090, `vllm/vllm-openai:v0.22.1` | Rented CA @ $0.22/hr. Container never started: `nvidia-container-cli: cuda>=13.0` |
| `seatl8jx9azbbm` — 3090, `vllm/vllm-openai:v0.22.1-cu129` | Rented CA @ $0.22/hr (+ $0.011/hr disk/volume = **$0.23/hr**) |
| Logs | Image pull → weights ~5.19 GiB in ~13s → load 5.29 GiB VRAM → `Application startup complete` |
| `GET /` | 404 (Runpod probe; vLLM has no `/`) — ignore |
| `GET …/v1/models` | **200** — `Qwen/Qwen2.5-7B-Instruct-AWQ`, `max_model_len` 4096 |
| `POST …/v1/completions` | **200** — 32 tokens, `finish_reason: length`, fingerprint `vllm-0.22.1-…` |
| Teardown | `runpodctl pod delete seatl8jx9azbbm` → `pod list` = `[]` |

**PASS** = completions JSON with a `"text"` field from a Mac `curl` through `https://seatl8jx9azbbm-8000.proxy.runpod.net`.

**Cost:** ~8–15 min on the working pod at $0.23/hr ≈ **$0.03–$0.06**, plus whatever the failed CUDA-13 pod billed until it was deleted. GPU bill is $0 after delete.
