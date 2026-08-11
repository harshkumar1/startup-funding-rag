#!/usr/bin/env bash
# Deploy startup-funding-rag to Google Cloud Run.
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
SERVICE="${SERVICE_NAME:-startup-funding-rag}"
REPO="${ARTIFACT_REPO:-startup-funding-rag}"

# Immutable version tag for this build — same scheme as the GHCR publish
# workflow (short git SHA). "local" if run outside a git checkout (e.g. a
# tarball) so the build never fails just because a SHA isn't available.
if git rev-parse --git-dir >/dev/null 2>&1; then
  SHA="$(git rev-parse --short=7 HEAD)"
  # A dirty tree means Cloud Build would upload uncommitted edits alongside
  # this SHA — the tag would then lie about what commit's contents actually
  # got built. Refuse by default; ALLOW_DIRTY=1 opts out for quick local
  # testing where that mismatch doesn't matter.
  if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY:-}" != "1" ]]; then
    echo "Uncommitted changes present — refusing to tag a build :${SHA} that" >&2
    echo "wouldn't match what 'git checkout ${SHA}' actually gives you." >&2
    echo "Commit/stash first, or re-run with ALLOW_DIRTY=1 to override." >&2
    exit 1
  fi
else
  SHA="local"
fi
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${SHA}"

if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "Set GCP_PROJECT or run: gcloud config set project PROJECT_ID"
  exit 1
fi

echo "Project:  ${PROJECT}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE}"
echo "Image:    ${IMAGE}  (also tagged :latest)"

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
       --description="startup-funding-rag images"

echo "Building and pushing image (Cloud Build)..."
gcloud builds submit \
  --project="${PROJECT}" \
  --config=cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_REPO=${REPO},_SERVICE=${SERVICE},_SHA=${SHA}" \
  --timeout=1200s \
  .

echo "Deploying to Cloud Run..."
# Deploy the immutable SHA tag, not :latest — so what's actually running is
# always traceable back to the exact commit that built it (`gcloud run
# services describe` will show this exact image ref, no need to guess what
# :latest happened to point to when it was deployed).
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
  --set-env-vars="API_HOST=0.0.0.0" \
  --allow-unauthenticated

URL="$(gcloud run services describe "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --format='value(status.url)')"

echo ""
echo "Deployed."
echo "  Image:        ${IMAGE}"
echo "  Service URL:  ${URL}"
echo "  API docs:     ${URL}/docs"
echo ""
echo "Security: publicly invokable — lock down before real use."
