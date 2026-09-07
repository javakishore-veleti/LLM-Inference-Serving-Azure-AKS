#!/usr/bin/env bash
# Prove vLLM OpenAI API on a Runpod HTTP proxy. Usage:
#   POD_ID=... ./scripts/runpod-verify.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
POD_ID="${POD_ID:?set POD_ID}"
BASE="https://${POD_ID}-8000.proxy.runpod.net"

echo "==> waiting for ${BASE}/v1/models (first boot can take 5–15 min)"
ok=0
for i in $(seq 1 90); do
  if curl -sf --max-time 15 "${BASE}/v1/models" >/dev/null; then
    ok=1
    break
  fi
  echo "    attempt ${i}/90 — not ready"
  sleep 10
done
[ "$ok" = 1 ] || { echo "FAIL: /v1/models never answered"; exit 1; }

echo "==> GET /v1/models"
curl -sf "${BASE}/v1/models" | (jq . 2>/dev/null || cat)
echo

echo "==> POST /v1/completions"
RESP=$(curl -sS --max-time 120 "${BASE}/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"Hello, my name is\",\"max_tokens\":32,\"temperature\":0}")
echo "$RESP" | (jq . 2>/dev/null || cat)
echo "$RESP" | grep -q '"text"' \
  && echo "PASS: API returned generated tokens." \
  || { echo "FAIL: no tokens in response."; exit 1; }
