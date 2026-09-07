# Runpod via Terraform + GitHub Actions

Manual workflows. Same stack as the CLI lab (`v0.22.1-cu129` + Qwen 7B AWQ). This is **not** Kubernetes.

## Secret (once)

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Name | Value |
|---|---|
| `RUNPOD_API_KEY` | [console.runpod.io/user/settings](https://console.runpod.io/user/settings) |

## One-shot (Actions → workflow → Run workflow)

1. **RUNPOD-ALL-SETUP-0001-EndToEnd** — list pods → `terraform apply` → curl `/v1/models` + `/v1/completions`
2. **RUNPOD-ALL-DESTROY-0001-EndToEnd** — `terraform destroy` → list pods (GPU → $0)

GPU bills from apply until destroy. Do not run Setup twice without Destroy.

## Step-by-step (same secret, same Terraform)

1. **RUNPOD-0001-CheckPods** — lists existing pods
2. **RUNPOD-0002-Setup** — `terraform apply` only
3. **RUNPOD-0003-Verify** — curl only
4. **RUNPOD-0004-Destroy** — `terraform destroy` only

Destroy/verify pull `runpod-tfstate` from the latest successful **ALL-SETUP** or **0002-Setup** on this branch.

Terraform lives in `terraform/runpod/`. State is **not** committed; it is the GitHub Actions artifact (kept 7 days).
