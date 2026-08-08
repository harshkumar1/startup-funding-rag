#!/usr/bin/env bash
# Deploy the RAG MCP skeleton to Google Cloud Run.
#
# Usage (run from anywhere; this script cd's into its own directory):
#   export GCP_PROJECT=your-project-id
#   export GCP_REGION=us-central1
#   ./deploy_cloud_run.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-rag-mcp}"
REPO="${ARTIFACT_REPO:-rag-mcp}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:latest"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "Set GCP_PROJECT or run: gcloud config set project PROJECT_ID"
  exit 1
fi

echo "Project:  ${PROJECT}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE}"
echo "Image:    ${IMAGE}"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT}"

gcloud artifacts repositories describe "${REPO}" \
  --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker \
       --location="${REGION}" \
       --project="${PROJECT}" \
       --description="RAG MCP images"

echo "Building and pushing image (Cloud Build)..."
gcloud builds submit \
  --project="${PROJECT}" \
  --config=cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_REPO=${REPO},_SERVICE=${SERVICE}" \
  --timeout=1200s \
  .

echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --platform=managed \
  --memory=512Mi \
  --cpu=1 \
  --timeout=300 \
  --concurrency=4 \
  --min-instances=0 \
  --max-instances=2 \
  --port=8080 \
  --set-env-vars="MCP_HOST=0.0.0.0" \
  --allow-unauthenticated

URL="$(gcloud run services describe "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format='value(status.url)')"

echo ""
echo "Deployed."
echo "  Service URL:  ${URL}"
echo "  MCP endpoint: ${URL}/mcp"
echo ""
echo "Security: publicly invokable — lock down before real use."
