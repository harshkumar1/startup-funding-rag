#!/bin/bash
set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INSTANCE_NAME="startup-funding-deployment"
ZONE="us-central1-a"
MACHINE_TYPE="e2-small"
TAGS="http-server,https-server,rag-node"
USERNAME=""

# Set your GitHub Repository URL here (or pass as env var GITHUB_REPO_URL)
# For Public Repos (HTTPS):  https://github.com/USERNAME/startup_funding_deployment.git
# For Private Repos (HTTPS): https://<TOKEN>@github.com/USERNAME/startup_funding_deployment.git
# For Private Repos (SSH):   git@github.com:USERNAME/startup_funding_deployment.git
GITHUB_REPO_URL="${GITHUB_REPO_URL:-git@github.com:USERNAME/startup_funding_deployment.git}"

# Extract repository directory name automatically from URL
REPO_NAME=$(basename "${GITHUB_REPO_URL}" .git)
REPO_DIR="${REPO_NAME:-startup_funding_deployment}"

# TUNNEL_TYPE options: "gcp" (default - SSH Reverse Tunnel) or "ngrok"
TUNNEL_TYPE="${TUNNEL_TYPE:-gcp}"

echo "========================================================"
echo " 🚀 GCP Deployment Script with MCP Server Support"
echo " Instance Name: ${INSTANCE_NAME}"
echo " Tunnel Type:   ${TUNNEL_TYPE}"
echo " Repo Name:     ${REPO_DIR}"
echo "========================================================"

# 1. Check if VM exists in GCP
echo "--> [1/5] Checking instance '${INSTANCE_NAME}'..."
INSTANCE_EXISTS=$(gcloud compute instances list --filter="name=${INSTANCE_NAME}" --format="value(name)" 2>/dev/null || echo "")

if [ -z "$INSTANCE_EXISTS" ]; then
    echo "--> Creating instance (${MACHINE_TYPE} in ${ZONE})..."
    gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --tags="${TAGS}"
    echo "✅ Instance created successfully."
else
    echo "ℹ️ Instance '${INSTANCE_NAME}' already exists."
fi

# 2. Start VM if stopped
STATUS=$(gcloud compute instances describe "${INSTANCE_NAME}" --zone="${ZONE}" --format="value(status)" 2>/dev/null || echo "UNKNOWN")
if [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "STOPPED" ]; then
    echo "--> Starting instance '${INSTANCE_NAME}'..."
    gcloud compute instances start "${INSTANCE_NAME}" --zone="${ZONE}"
fi

# 3. Firewall Rules for 7860 (UI), 8000 (FastAPI), and 8001 (MCP SSE)
echo "--> [2/5] Configuring firewall rules for HTTP/API/MCP..."
if ! gcloud compute firewall-rules describe allow-rag-app &>/dev/null; then
    gcloud compute firewall-rules create allow-rag-app \
        --allow=tcp:7860,tcp:8000,tcp:8001 \
        --target-tags=rag-node \
        --description="Allow Streamlit UI, FastAPI REST API, and MCP SSE Server traffic"
    echo "✅ Firewall rule 'allow-rag-app' created."
else
    echo "ℹ️ Firewall rule 'allow-rag-app' already exists. Verifying open ports..."
    gcloud compute firewall-rules update allow-rag-app --allow=tcp:7860,tcp:8000,tcp:8001 2>/dev/null || true
fi

# 4. Remote SSH Execution, Code Sync, and Container Launch
echo "--> [3/5] Syncing latest repository code and building container..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" --command="
    set -e

    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null || true

    if ! command -v git &> /dev/null || ! command -v docker &> /dev/null; then
        echo '--> Installing Git & Docker...'
        sudo apt-get update -y
        sudo apt-get install -y git docker.io docker-compose
        sudo systemctl enable --now docker
        sudo usermod -aG docker \$USER
    fi

    if [ ! -f /swapfile ]; then
        echo '--> Creating 2GB Swap file...'
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
    fi

    if [ -d '${REPO_DIR}/.git' ]; then
        echo '--> Pulling latest changes from GitHub...'
        cd '${REPO_DIR}'
        git pull
    else
        echo '--> Cloning repository from GitHub...'
        rm -rf '${REPO_DIR}'
        GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git clone '${GITHUB_REPO_URL}' '${REPO_DIR}'
        cd '${REPO_DIR}'
    fi

    if [ -f start.sh ]; then
        chmod +x start.sh
    fi

    echo '--> Building Docker image...'
    docker build -t startup-rag-app .

    echo '--> Launching Docker container...'
    docker stop rag-app-container 2>/dev/null || true
    docker rm rag-app-container 2>/dev/null || true

    ENV_FLAG=\"\"
    if [ -f .env ]; then
        ENV_FLAG=\"--env-file .env\"
    fi

    docker run -d \
      --name rag-app-container \
      --restart always \
      --add-host=host.docker.internal:host-gateway \
      \$ENV_FLAG \
      -p 7860:7860 \
      -p 8000:8000 \
      -p 8001:8001 \
      startup-rag-app
"

# 5. Reverse Tunnel Setup
echo "--> [4/5] Establishing local Milvus tunnel (${TUNNEL_TYPE})..."
if [ "$TUNNEL_TYPE" = "gcp" ]; then
    echo "Starting background GCP SSH reverse tunnel (Local 19530 -> GCP VM 19530)..."
    gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" -- -f -N -R 19530:localhost:19530 || true
    echo "✅ GCP SSH reverse tunnel established."
fi

# 6. Fetch Public IP
VM_IP=$(gcloud compute instances list --filter="name=${INSTANCE_NAME}" --format="value(networkInterfaces.accessConfigs.natIP)" 2>/dev/null || echo "YOUR_VM_IP")

echo "========================================================"
echo " 🎉 MCP-Enabled Deployment Complete!"
echo " 🌐 Streamlit UI:  http://${VM_IP}:7860"
echo " ⚡ FastAPI REST:  http://${VM_IP}:8000"
echo " 🔌 MCP SSE Host:  http://${VM_IP}:8001/sse"
echo "========================================================"