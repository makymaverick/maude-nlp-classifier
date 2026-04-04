################################################################################
# MAUDE NLP Classifier — AWS ECS/Fargate Terraform Config
#
# Resources created:
#   - ECR repository for Docker image
#   - ECS Cluster + Fargate Service
#   - Application Load Balancer
#   - Security Groups
#   - CloudWatch Log Group
#   - IAM Roles (task execution + task)
#   - Secrets Manager secret (openFDA API key)
################################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # S3 remote state — create the bucket first (bootstrap/README.md), then uncomment.
  # backend "s3" {
  #   bucket         = "maude-nlp-terraform-state-<account_id>"
  #   key            = "maude-nlp-classifier/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "maude-nlp-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name = "maude-nlp-classifier"
  tags = {
    Project     = "MAUDE-NLP"
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# ── Data sources ────────────────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" { state = "available" }

# ── VPC & Networking ────────────────────────────────────────────────────────
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = merge(local.tags, { Name = "${local.name}-vpc" })
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${local.name}-public-${count.index}" })
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.name}-igw" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = merge(local.tags, { Name = "${local.name}-rt" })
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── Security Groups ─────────────────────────────────────────────────────────
resource "aws_security_group" "alb" {
  name        = "${local.name}-alb-sg"
  description = "Allow HTTP/HTTPS inbound to ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

resource "aws_security_group" "ecs_task" {
  name        = "${local.name}-ecs-sg"
  description = "Allow traffic from ALB to ECS task"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = local.tags
}

# ── ECR Repository ───────────────────────────────────────────────────────────
resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }
  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── CloudWatch Logs ──────────────────────────────────────────────────────────
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name}"
  retention_in_days = 30
  tags              = local.tags
}

# ── IAM Roles ────────────────────────────────────────────────────────────────
resource "aws_iam_role" "ecs_execution" {
  name = "${local.name}-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_secrets" {
  name = "${local.name}-secrets-policy"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.openfda_key.arn
    }]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name}-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

# ── Secrets Manager ──────────────────────────────────────────────────────────
resource "aws_secretsmanager_secret" "openfda_key" {
  name                    = "maude/openfda_api_key"
  recovery_window_in_days = 7
  tags                    = local.tags
}

resource "aws_secretsmanager_secret_version" "openfda_key" {
  secret_id     = aws_secretsmanager_secret.openfda_key.id
  secret_string = var.openfda_api_key
}

# ── ECS Cluster ──────────────────────────────────────────────────────────────
resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

# ── ECS Task Definition ──────────────────────────────────────────────────────
resource "aws_ecs_task_definition" "app" {
  family                   = local.name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = local.name
    image     = "${aws_ecr_repository.app.repository_url}:latest"
    essential = true

    portMappings = [{
      containerPort = 8501
      hostPort      = 8501
      protocol      = "tcp"
    }]

    secrets = [{
      name      = "OPENFDA_API_KEY"
      valueFrom = aws_secretsmanager_secret.openfda_key.arn
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 20
    }
  }])

  tags = local.tags
}

# ── Application Load Balancer ────────────────────────────────────────────────
resource "aws_lb" "main" {
  name               = local.name
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  tags               = local.tags
}

resource "aws_lb_target_group" "app" {
  name        = local.name
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/_stcore/health"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
  tags = local.tags
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ── ECS Service ──────────────────────────────────────────────────────────────
resource "aws_ecs_service" "app" {
  name            = local.name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_task.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = local.name
    container_port   = 8501
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  tags = local.tags
}

# ── Auto Scaling ─────────────────────────────────────────────────────────────
resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.max_capacity
  min_capacity       = var.desired_count
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${local.name}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

# ── S3 — Model Checkpoints, MLflow Artifacts, Accumulated Data ───────────────
resource "aws_s3_bucket" "checkpoints" {
  bucket        = "${local.name}-checkpoints-${data.aws_caller_identity.current.account_id}"
  force_destroy = false
  tags          = merge(local.tags, { Purpose = "model-checkpoints" })
}

resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "checkpoints" {
  bucket                  = aws_s3_bucket.checkpoints.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "expire-old-checkpoints"
    status = "Enabled"

    filter { prefix = "checkpoints/" }

    noncurrent_version_expiration { noncurrent_days = 30 }
  }

  rule {
    id     = "expire-mlruns"
    status = "Enabled"

    filter { prefix = "mlruns/" }

    expiration { days = 90 }
  }
}

# Grant ECS task role read access to the checkpoints bucket
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${local.name}-task-s3-policy"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.checkpoints.arn,
        "${aws_s3_bucket.checkpoints.arn}/*",
      ]
    }]
  })
}

# ── SageMaker IAM Role for BERT fine-tuning ───────────────────────────────────
resource "aws_iam_role" "sagemaker" {
  name = "${local.name}-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "${local.name}-sagemaker-s3-policy"
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"]
      Resource = [
        aws_s3_bucket.checkpoints.arn,
        "${aws_s3_bucket.checkpoints.arn}/*",
      ]
    }]
  })
}

# Grant SageMaker role access to the Secrets Manager secret (for OPENFDA_API_KEY)
resource "aws_iam_role_policy" "sagemaker_secrets" {
  name = "${local.name}-sagemaker-secrets-policy"
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = aws_secretsmanager_secret.openfda_key.arn
    }]
  })
}

# ── GPU ECS Task Definition — ClinicalBERT batch inference ───────────────────
# Uses g4dn family via ECS Capacity Provider (EC2, not Fargate) for GPU access.
# This task is NOT launched continuously — it is invoked on-demand for
# batch reclassification of the full accumulated dataset.
resource "aws_ecs_task_definition" "bert_inference" {
  family                   = "${local.name}-bert"
  network_mode             = "awsvpc"
  requires_compatibilities = ["EC2"]        # GPU requires EC2 launch type
  cpu                      = "4096"         # 4 vCPU
  memory                   = "16384"        # 16 GB
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "${local.name}-bert"
    image     = "${aws_ecr_repository.app.repository_url}:bert-latest"
    essential = true

    resourceRequirements = [{
      type  = "GPU"
      value = "1"
    }]

    portMappings = [{
      containerPort = 8501
      hostPort      = 8501
      protocol      = "tcp"
    }]

    secrets = [{
      name      = "OPENFDA_API_KEY"
      valueFrom = aws_secretsmanager_secret.openfda_key.arn
    }]

    environment = [
      { name = "BERT_CHECKPOINT_S3_BUCKET", value = aws_s3_bucket.checkpoints.bucket },
      { name = "MLFLOW_TRACKING_URI",       value = "s3://${aws_s3_bucket.checkpoints.bucket}/mlruns" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs-bert"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 180   # BERT model download takes longer
    }
  }])

  tags = merge(local.tags, { ModelType = "ClinicalBERT" })
}

# CloudWatch alarm: alert if ECS BERT task exits unexpectedly
resource "aws_cloudwatch_metric_alarm" "bert_task_stopped" {
  alarm_name          = "${local.name}-bert-task-stopped"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "TaskCount"
  namespace           = "ECS/ContainerInsights"
  period              = 60
  statistic           = "Minimum"
  threshold           = 0
  alarm_description   = "BERT ECS task stopped unexpectedly"
  treat_missing_data  = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = local.name
  }
}
