variable "pod_name" {
  type        = string
  description = "Runpod pod name (does not have to be unique)."
  default     = "llm-inf-serving-vllm-qwen25-7b-awq"
}

variable "image_name" {
  type        = string
  description = "vLLM OpenAI image. cu129 is required on community GeForce hosts."
  default     = "vllm/vllm-openai:v0.22.1-cu129"
}

variable "cloud_type" {
  type        = string
  description = "COMMUNITY (cheap) or SECURE."
  default     = "COMMUNITY"
}

variable "gpu_type_ids" {
  type        = list(string)
  description = "Cheapest-first fallbacks on the same cloud_type. Do not mix COMMUNITY 3090 with SECURE A40 here."
  default = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
  ]
}

variable "gpu_count" {
  type    = number
  default = 1
}

variable "container_disk_in_gb" {
  type    = number
  default = 50
}

variable "volume_in_gb" {
  type    = number
  default = 30
}

variable "volume_mount_path" {
  type    = string
  default = "/root/.cache/huggingface"
}

variable "model" {
  type    = string
  default = "Qwen/Qwen2.5-7B-Instruct-AWQ"
}
