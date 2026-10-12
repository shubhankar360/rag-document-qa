variable "project_id" {
  description = "GCP project to deploy into"
  type        = string
}

variable "region" {
  description = "Cloud Run and Artifact Registry region (Mumbai by default)"
  type        = string
  default     = "asia-south1"
}

variable "service_name" {
  type    = string
  default = "rag-document-qa"
}

variable "github_repo" {
  description = "owner/name of the repo allowed to deploy through Workload Identity Federation"
  type        = string
  default     = "shubhankar360/rag-document-qa"
}

variable "bootstrap_image" {
  description = "Image for the first apply, before CI has pushed one; CI owns the image after that"
  type        = string
  default     = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "max_instances" {
  description = "Upper bound on instances, which is also the upper bound on cost"
  type        = number
  default     = 2
}

variable "public" {
  description = "Allow unauthenticated requests"
  type        = bool
  default     = true
}

variable "enable_claude" {
  description = "Create a Secret Manager secret for ANTHROPIC_API_KEY and mount it; add the value with gcloud, not Terraform"
  type        = bool
  default     = false
}
