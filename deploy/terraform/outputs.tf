output "app_url" {
  description = "Public URL of the MAUDE NLP Classifier app"
  value       = "http://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL for Docker image pushes"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS Cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS Service name"
  value       = aws_ecs_service.app.name
}

output "checkpoints_bucket" {
  description = "S3 bucket for BERT checkpoints, MLflow runs, and accumulated data"
  value       = aws_s3_bucket.checkpoints.bucket
}

output "checkpoints_bucket_arn" {
  description = "ARN of the checkpoints S3 bucket"
  value       = aws_s3_bucket.checkpoints.arn
}

output "sagemaker_role_arn" {
  description = "IAM Role ARN for SageMaker BERT fine-tuning jobs"
  value       = aws_iam_role.sagemaker.arn
}

output "bert_task_definition_arn" {
  description = "ECS task definition ARN for GPU-based ClinicalBERT inference"
  value       = aws_ecs_task_definition.bert_inference.arn
}

output "mlflow_s3_uri" {
  description = "S3 URI to use as MLFLOW_TRACKING_URI for remote experiment logging"
  value       = "s3://${aws_s3_bucket.checkpoints.bucket}/mlruns"
}
