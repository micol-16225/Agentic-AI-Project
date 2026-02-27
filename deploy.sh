#!/bin/bash

# --- CONFIGURATION ---
REGION="us-east-1"
ACCOUNT_ID="883099622443"
REPO_NAME="biostat-auditor"
IMAGE_TAG="latest"
FULL_REPO_URL="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}"

echo "🚀 Starting Deployment for ${REPO_NAME}..."

# 1. CLEANUP: Remove that accidental file named after a command
echo "🧹 Cleaning up local directory..."
rm -f "ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 883099622443.dkr.ecr.us-east-1.amazonaws.com"

# 2. AWS AUTHENTICATION
echo "🔐 Authenticating with Amazon ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${FULL_REPO_URL}

# 3. DOCKER BUILD
echo "📦 Building Docker image..."
docker build -t ${REPO_NAME} .

# 4. TAGGING
echo "🏷️ Tagging image for ECR..."
docker tag ${REPO_NAME}:${IMAGE_TAG} ${FULL_REPO_URL}:${IMAGE_TAG}

# 5. PUSH TO CLOUD
echo "☁️ Pushing image to AWS ECR..."
docker push ${FULL_REPO_URL}:${IMAGE_TAG}

echo "✅ Success! Your cloud environment is now updated."