variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment tag (dev / staging / prod)"
  type        = string
  default     = "dev"
}

variable "openfda_api_key" {
  description = "openFDA API key (stored in Secrets Manager)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "desired_count" {
  description = "Number of ECS task instances to run"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum ECS auto-scaling capacity"
  type        = number
  default     = 4
}
