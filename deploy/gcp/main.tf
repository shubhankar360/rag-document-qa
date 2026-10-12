locals {
  apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
  ]
}

resource "google_project_service" "apis" {
  for_each           = toset(local.apis)
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = var.service_name
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# The service runs as its own identity with no roles beyond reading the key.
resource "google_service_account" "runtime" {
  account_id   = "rag-qa-runtime"
  display_name = "rag-document-qa runtime"
}

# Terraform creates the secret but never sees its value, so the key stays out
# of state:  printf %s "$KEY" | gcloud secrets versions add anthropic-api-key --data-file=-
resource "google_secret_manager_secret" "anthropic" {
  count     = var.enable_claude ? 1 : 0
  secret_id = "anthropic-api-key"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "runtime_reads_key" {
  count     = var.enable_claude ? 1 : 0
  secret_id = google_secret_manager_secret.anthropic[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "app" {
  name                = var.service_name
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account                  = google_service_account.runtime.email
    max_instance_request_concurrency = 20
    timeout                          = "30s"

    # scale to zero when idle; max_instances caps the bill
    scaling {
      min_instance_count = 0
      max_instance_count = var.max_instances
    }

    containers {
      image = var.bootstrap_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        initial_delay_seconds = 2
        period_seconds        = 3
        failure_threshold     = 10

        http_get {
          path = "/health"
        }
      }

      dynamic "env" {
        for_each = var.enable_claude ? [1] : []
        content {
          name = "ANTHROPIC_API_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.anthropic[0].secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }

  # CI deploys new images with gcloud; without this, the next terraform apply
  # would roll the service back to whatever image Terraform last knew about.
  lifecycle {
    ignore_changes = [
      template[0].containers[0].image,
      client,
      client_version,
    ]
  }

  depends_on = [google_project_service.apis, google_secret_manager_secret_iam_member.runtime_reads_key]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.public ? 1 : 0
  name     = google_cloud_run_v2_service.app.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
