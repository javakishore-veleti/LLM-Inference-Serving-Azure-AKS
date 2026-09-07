resource "runpod_pod" "vllm" {
  name       = var.pod_name
  image_name = var.image_name

  compute_type      = "GPU"
  cloud_type        = var.cloud_type
  gpu_count         = var.gpu_count
  gpu_type_ids      = var.gpu_type_ids
  gpu_type_priority = "custom"
  support_public_ip = false

  container_disk_in_gb = var.container_disk_in_gb
  volume_in_gb         = var.volume_in_gb
  volume_mount_path    = var.volume_mount_path

  ports = ["8000/http"]

  # Image ENTRYPOINT is `vllm serve`; these override CMD (same flags as the CLI runbook).
  docker_start_cmd = [
    "--model", var.model,
    "--quantization", "awq",
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.90",
    "--max-model-len", "4096",
    "--max-num-seqs", "8",
    "--host", "0.0.0.0",
    "--port", "8000",
  ]
}
