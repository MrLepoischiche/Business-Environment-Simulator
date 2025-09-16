# Banking Simulator - Multi-Cloud Infrastructure
terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket  = "banking-simulator-terraform-state"
    key     = "infrastructure/terraform.tfstate"
    region  = "eu-west-1"
    encrypt = true
    
    dynamodb_table = "banking-simulator-terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "regions" {
  description = "Deployment regions"
  type = map(object({
    primary = bool
    cloud   = string
    region  = string
  }))
  default = {
    "eu-west-1" = {
      primary = true
      cloud   = "aws"
      region  = "eu-west-1"
    }
    "westeurope" = {
      primary = false
      cloud   = "azure"
      region  = "westeurope"
    }
    "europe-west1" = {
      primary = false
      cloud   = "gcp"
      region  = "europe-west1"
    }
  }
}

variable "cluster_config" {
  description = "Kubernetes cluster configuration"
  type = object({
    node_count     = number
    node_size      = string
    max_nodes      = number
    auto_scaling   = bool
  })
  default = {
    node_count   = 5
    node_size    = "m5.xlarge"  # AWS equivalent
    max_nodes    = 100
    auto_scaling = true
  }
}

# Data sources
data "aws_availability_zones" "available" {
  count = contains([for r in var.regions : r.cloud], "aws") ? 1 : 0
  state = "available"
}

# AWS Infrastructure
module "aws_infrastructure" {
  source = "./modules/aws"
  count  = contains([for r in var.regions : r.cloud], "aws") ? 1 : 0
  
  environment     = var.environment
  region         = [for k, v in var.regions : v.region if v.cloud == "aws"][0]
  cluster_config = var.cluster_config
  
  vpc_cidr = "10.0.0.0/16"
  
  tags = {
    Environment = var.environment
    Project     = "banking-simulator"
    Cloud       = "aws"
  }
}

# Azure Infrastructure  
module "azure_infrastructure" {
  source = "./modules/azure"
  count  = contains([for r in var.regions : r.cloud], "azure") ? 1 : 0
  
  environment     = var.environment
  location       = [for k, v in var.regions : v.region if v.cloud == "azure"][0]
  cluster_config = var.cluster_config
  
  tags = {
    Environment = var.environment
    Project     = "banking-simulator"
    Cloud       = "azure"
  }
}

# GCP Infrastructure
module "gcp_infrastructure" {
  source = "./modules/gcp"
  count  = contains([for r in var.regions : r.cloud], "gcp") ? 1 : 0
  
  environment     = var.environment
  region         = [for k, v in var.regions : v.region if v.cloud == "gcp"][0]
  cluster_config = var.cluster_config
  
  labels = {
    environment = var.environment
    project     = "banking-simulator"
    cloud       = "gcp"
  }
}

# Global DNS and CDN
module "global_dns" {
  source = "./modules/dns"
  
  environment = var.environment
  
  # Primary endpoint
  primary_endpoint = try(module.aws_infrastructure[0].load_balancer_dns, 
                        try(module.azure_infrastructure[0].load_balancer_dns,
                           module.gcp_infrastructure[0].load_balancer_dns))
  
  # Failover endpoints
  failover_endpoints = compact([
    try(module.azure_infrastructure[0].load_balancer_dns, ""),
    try(module.gcp_infrastructure[0].load_balancer_dns, ""),
    try(module.aws_infrastructure[0].load_balancer_dns, "")
  ])
  
  domain_name = "banking-simulator.com"
}

# Global CDN for Frontend
module "global_cdn" {
  source = "./modules/cdn"
  
  environment = var.environment
  
  origins = compact([
    try(module.aws_infrastructure[0].s3_bucket_domain, ""),
    try(module.azure_infrastructure[0].storage_domain, ""),
    try(module.gcp_infrastructure[0].bucket_domain, "")
  ])
  
  domain_name = "app.banking-simulator.com"
}

# Shared Database Cluster (Primary region only)
module "database_cluster" {
  source = "./modules/database"
  
  environment = var.environment
  
  # Deploy in primary region
  provider_config = [for k, v in var.regions : {
    cloud  = v.cloud
    region = v.region
  } if v.primary][0]
  
  cluster_config = {
    postgres = {
      instance_class = "db.r6g.xlarge"
      instances      = 3
      storage_gb     = 500
      backup_days    = 30
    }
    redis = {
      node_type   = "cache.r6g.large"
      num_shards  = 3
      replicas    = 1
    }
    influxdb = {
      instance_type = "m5.large"
      storage_gb    = 200
      retention     = "30d"
    }
  }
}

# Monitoring Stack (Deployed in all regions)
module "monitoring_stack" {
  source = "./modules/monitoring"
  count  = length(var.regions)
  
  environment = var.environment
  region     = values(var.regions)[count.index].region
  cloud      = values(var.regions)[count.index].cloud
  
  prometheus_config = {
    retention_days = 30
    storage_gb     = 100
    replicas       = 2
  }
  
  grafana_config = {
    replicas = 2
    plugins  = ["grafana-piechart-panel", "grafana-worldmap-panel"]
  }
}

# Outputs
output "cluster_endpoints" {
  description = "Kubernetes cluster endpoints"
  value = merge(
    try({ aws = module.aws_infrastructure[0].cluster_endpoint }, {}),
    try({ azure = module.azure_infrastructure[0].cluster_endpoint }, {}),
    try({ gcp = module.gcp_infrastructure[0].cluster_endpoint }, {})
  )
}

output "database_endpoints" {
  description = "Database cluster endpoints"
  value = module.database_cluster.endpoints
  sensitive = true
}

output "application_urls" {
  description = "Application URLs"
  value = {
    api_endpoint  = module.global_dns.api_url
    app_url      = module.global_cdn.app_url
    monitoring   = { for i, region in values(var.regions) : region.region => module.monitoring_stack[i].grafana_url }
  }
}

output "deployment_info" {
  description = "Deployment information"
  value = {
    environment      = var.environment
    deployed_regions = keys(var.regions)
    primary_region   = [for k, v in var.regions : k if v.primary][0]
    timestamp       = timestamp()
  }
}