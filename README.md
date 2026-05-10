# Guardian AI: Agentic Biostatistical Protocol Auditor

Guardian AI is a specialized RAG-based (Retrieval-Augmented Generation) application designed to audit clinical trial protocols. It utilizes an agentic framework to simulate an FDA reviewer persona, identifying regulatory gaps and biostatistical omissions through a rigorous "skeptical" lens.

## Core Features
FDA Reviewer Persona: Simulates regulatory scrutiny to detect "violations by omission."

Dual-Library RAG: Integrates statutory regulations (FDA/ICH) and academic statistical rigor.

Protocol Optimizer: Suggests technical improvements based on identified gaps.

High-Performance Infrastructure: Pre-baked embeddings for rapid, memory-efficient cloud deployment.

### Tech Stack

LLM: Llama 3 (via Groq)

Embeddings: all-MiniLM-L6-v2 / BioBERT

Backend: Python 3.11, PyTorch

Frontend: Streamlit

Infrastructure: Docker, AWS ECR, AWS App Runner

## Installation & Local Setup

### Clone the Repository:
Bash

git clone https://github.com/micol-16225/Agentic-AI-Project

cd "To deploy"


### Environment Configuration:
Create a .env file in the root directory:

Code snippet

GROQ_API_KEY=your_api_key_here


### Build and Run via Docker:
Bash

#Rebuild the image (updates baked embeddings) 

docker build -t auditor-agent-v2:latest . 

#Stop old instances to free the port 

docker ps -q --filter ancestor=auditor-agent-v2:latest | xargs -r docker stop 

#Run the container

docker run --env-file .env -p 8501:8501 auditor-agent-v2:latest



Your app will be available at http://localhost:8501.

## ☁️  AWS Deployment (MLOps)
The project is containerized to ensure consistency across environments. To optimize memory usage on AWS, embeddings are pre-computed during the build phase.
### Build and Tag Image:
Bash

docker build -t auditor-agent-v2 .
docker tag auditor-agent-v2:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/auditor-agent-v2:latest


### Push to ECR:
Bash

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/auditor-agent-v2:latest


### Cloud Execution:
Deployed on AWS App Runner, providing a scalable, managed environment for the auditing agent.

## 📂 Key Files in this “To Deploy” Repo

lifecycle_agent4.py: Core agentic logic and RAG retrieval system.

generate_embeddings.py: Automation script for semantic tensor generation.

statutory_truth_with_ids.csv: The regulatory "Source of Truth" knowledge base.

Dockerfile: Multi-stage build configuration optimized for CPU-based cloud environments.

