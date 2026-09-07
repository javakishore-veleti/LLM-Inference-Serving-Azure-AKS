# Runpod via Terraform + GitHub Actions

Manual workflows. Same stack as the CLI lab (`v0.22.1-cu129` + Qwen 7B AWQ). This is **not** Kubernetes.

## Secret (once)

Repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Name | Value |
|---|---|
| `RUNPOD_API_KEY` | [console.runpod.io/user/settings](https://console.runpod.io/user/settings) |

## Run in order (Actions → workflow → Run workflow)

1. **RUNPOD-0001-CheckPods** — lists existing pods (should be `[]` before setup)
2. **RUNPOD-0002-Setup** — `terraform apply` (bills ~$0.22/hr)
3. **RUNPOD-0003-Verify** — curl `/v1/models` and `/v1/completions` (waits for first boot)
4. **RUNPOD-0004-Destroy** — `terraform destroy` (GPU → $0)

0003 and 0004 pull the `runpod-tfstate` artifact from the latest successful **RUNPOD-0002-Setup** run on the same branch. Do not re-run 0002 until 0004 has finished, or you can rent a second pod.

Terraform lives in `terraform/runpod/`. State is **not** committed; it is the GitHub Actions artifact (kept 7 days).
