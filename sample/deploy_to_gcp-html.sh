#!/bin/bash
set -e

# ==============================================================================
# CONFIGURATION FOR HTML/JS + FASTAPI GCP DEPLOYMENT
# ==============================================================================
INSTANCE_NAME="startup-funding-deployment"
ZONE="us-central1-a"
MACHINE_TYPE="e2-small"
TAGS="http-server,https-server,rag-node"
USERNAME =""

# Set your GitHub Repository URL here (or pass as env var GITHUB_REPO_URL)
# For Public Repos (HTTPS):  https://github.com/USERNAME/startup_funding_deployment.git
# For Private Repos (HTTPS): https://<TOKEN>@github.com/USERNAME/startup_funding_deployment.git
# For Private Repos (SSH):   git@github.com:USERNAME/startup_funding_deployment.git
GITHUB_REPO_URL="${GITHUB_REPO_URL:-git@github.com:USERNAME/startup_funding_deployment.git}"

# Auto-derive repository folder name (e.g., startup_funding_deployment)
REPO_DIR=$(basename "${GITHUB_REPO_URL}" .git)

# TUNNEL_TYPE options: "gcp" (default) or "ngrok"
TUNNEL_TYPE="${TUNNEL_TYPE:-gcp}"

# ==============================================================================
# .ENV CONFIGURATION INSTRUCTIONS FOR RESPECTIVE TUNNEL TYPES
# ==============================================================================
# Option 1: TUNNEL_TYPE="gcp" (Free SSH Reverse Tunnel - Recommended)
#   GROQ_API_KEY=your_groq_api_key
#   MILVUS_HOST=host.docker.internal
#   MILVUS_PORT=19530
#   REDIS_HOST=localhost
#   REDIS_PORT=6380
#
# Option 2: TUNNEL_TYPE="ngrok" (Requires verified Ngrok account with TCP)
#   GROQ_API_KEY=your_groq_api_key
#   MILVUS_HOST=4.tcp.ngrok.io
#   MILVUS_PORT=12345
#   REDIS_HOST=localhost
#   REDIS_PORT=6380
# ==============================================================================

echo "========================================================"
echo " GCP Deployment Script (HTML/JS Web UI + FastAPI)"
echo " Instance Name: ${INSTANCE_NAME}"
echo " Tunnel Type:   ${TUNNEL_TYPE}"
echo " Repo URL:      ${GITHUB_REPO_URL}"
echo " Repo Dir:      ${REPO_DIR}"
echo "========================================================"

# 1. Check if instance exists in GCP
echo "--> [1/5] Checking if instance '${INSTANCE_NAME}' exists..."
INSTANCE_EXISTS=$(gcloud compute instances list --filter="name=${INSTANCE_NAME}" --format="value(name)" 2>/dev/null || echo "")

if [ -z "$INSTANCE_EXISTS" ]; then
    echo "--> Instance '${INSTANCE_NAME}' not found. Creating instance (${MACHINE_TYPE} in ${ZONE})..."
    gcloud compute instances create "${INSTANCE_NAME}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --tags="${TAGS}"
    echo "✅ Instance '${INSTANCE_NAME}' created successfully."
else
    echo "ℹ️ Instance '${INSTANCE_NAME}' already exists."
fi

# 2. Start VM if stopped
STATUS=$(gcloud compute instances describe "${INSTANCE_NAME}" --zone="${ZONE}" --format="value(status)" 2>/dev/null || echo "UNKNOWN")
if [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "STOPPED" ]; then
    echo "--> Starting instance '${INSTANCE_NAME}'..."
    gcloud compute instances start "${INSTANCE_NAME}" --zone="${ZONE}"
fi

# 3. Ensure Firewall Rules Exist (Port 8000 for HTML UI & FastAPI)
echo "--> [2/5] Checking/creating firewall rules..."
if ! gcloud compute firewall-rules describe allow-rag-html-app &>/dev/null; then
    gcloud compute firewall-rules create allow-rag-html-app \
        --allow=tcp:8000 \
        --target-tags=rag-node \
        --description="Allow HTML Web UI and FastAPI REST traffic on port 8000"
    echo "✅ Firewall rule 'allow-rag-html-app' created for port 8000."
else
    echo "ℹ️ Firewall rule 'allow-rag-html-app' already exists."
fi

# Ensure IAP SSH firewall rule exists for GCP Tunneling
if [ "$TUNNEL_TYPE" = "gcp" ]; then
    if ! gcloud compute firewall-rules describe allow-iap-ssh &>/dev/null; then
        gcloud compute firewall-rules create allow-iap-ssh \
            --allow=tcp:22 \
            --source-ranges=35.191.0.0/16 \
            --description="Allow IAP SSH tunneling" 2>/dev/null || true
        echo "✅ IAP SSH firewall rule ensured."
    fi
fi

# 4. Clone or pull code from GitHub on VM and launch container
echo "--> [3/5] Fetching latest code from GitHub on VM and launching application..."
gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" --command="
    set -e

    # Install git and docker if missing
    if ! command -v git &> /dev/null || ! command -v docker &> /dev/null; then
        echo '--> Installing Git & Docker...'
        sudo apt-get update -y
        sudo apt-get install -y git docker.io docker-compose
        sudo systemctl enable --now docker
        sudo usermod -aG docker \$USER
    fi

    # Configure 2 GB Swap Memory if not present
    if [ ! -f /swapfile ]; then
        echo '--> Creating 2 GB Swap file...'
        sudo fallocate -l 2G /swapfile
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
    fi

    # Ensure GitHub host key is added to known_hosts to prevent SSH prompt hangs
    mkdir -p ~/.ssh
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null || true

    # Clone or Pull Repository
    if [ -d '${REPO_DIR}/.git' ]; then
        echo '--> Pulling latest changes from GitHub...'
        cd '${REPO_DIR}'
        git pull
    else
        echo '--> Cloning repository from GitHub into ${REPO_DIR}...'
        rm -rf '${REPO_DIR}'
        git clone '${GITHUB_REPO_URL}' '${REPO_DIR}'
        cd '${REPO_DIR}'
    fi

    # Make start script executable
    if [ -f start.sh ]; then
        chmod +x start.sh
    fi

    # Build Docker Image
    echo '--> Building Docker image...'
    docker build -t startup-rag-html-app .

    # Stop and remove old container
    docker stop rag-app-container 2>/dev/null || true
    docker rm rag-app-container 2>/dev/null || true

    ENV_FLAG=\"\"
    if [ -f .env ]; then
        ENV_FLAG=\"--env-file .env\"
    fi

    # Launch container exposing only port 8000
    echo '--> Launching Docker container on port 8000...'
    docker run -d \
      --name rag-app-container \
      --restart always \
      --add-host=host.docker.internal:host-gateway \
      \$ENV_FLAG \
      -p 8000:8000 \
      startup-rag-html-app
"

# 5. Setup Reverse Tunnel
echo "--> [4/5] Setting up reverse tunnel (${TUNNEL_TYPE})..."

if [ "$TUNNEL_TYPE" = "gcp" ]; then
    echo "========================================================"
    echo " 🔌 GCP SSH Reverse Tunnel Mode"
    echo " Local Milvus Port 19530 -> GCP VM Port 19530"
    echo " Ensure .env on VM has: MILVUS_HOST=host.docker.internal"
    echo "========================================================"
    echo "Starting background SSH tunnel..."
    gcloud compute ssh "${INSTANCE_NAME}" --zone="${ZONE}" -- -f -N -R 19530:localhost:19530 || true
    echo "✅ GCP SSH Reverse tunnel established!"

elif [ "$TUNNEL_TYPE" = "ngrok" ]; then
    echo "========================================================"
    echo " 🌐 Ngrok Tunnel Mode"
    echo " Ensure .env has: MILVUS_HOST=<your-ngrok-host> & MILVUS_PORT=<your-ngrok-port>"
    echo "========================================================"
fi

# 6. Fetch Public IP
VM_IP=$(gcloud compute instances list --filter="name=${INSTANCE_NAME}" --format="value(networkInterfaces.accessConfigs.natIP)" 2>/dev/null || echo "YOUR_VM_IP")

echo "========================================================"
echo " 🎉 GCP Deployment Complete!"
echo " 🌐 Web UI & REST API: http://${VM_IP}:8000"
echo "========================================================"
