output "pod_id" {
  value = runpod_pod.vllm.id
}

output "pod_name" {
  value = runpod_pod.vllm.name
}

output "desired_status" {
  value = runpod_pod.vllm.desired_status
}

output "cost_per_hr" {
  value = runpod_pod.vllm.cost_per_hr
}

output "proxy_base_url" {
  value = "https://${runpod_pod.vllm.id}-8000.proxy.runpod.net"
}

output "models_url" {
  value = "https://${runpod_pod.vllm.id}-8000.proxy.runpod.net/v1/models"
}

output "completions_url" {
  value = "https://${runpod_pod.vllm.id}-8000.proxy.runpod.net/v1/completions"
}
