# Enterprise Migration Copilot

AI-powered multi-source enterprise data transformation platform.
Converts SQL, HiveQL, PL/SQL, and Oracle Stored Procedures to PySpark
using fine-tuned LLMs, a 6-agent validation pipeline, and hybrid RAG.

> Live Demo: coming soon
> First request may take 30-60 seconds due to Render free tier cold start.

## Status
Currently in active development — 26-day build in progress.

## Stack
- Fine-tuning: QLoRA + PEFT + trl on Colab free T4
- Agents: LangGraph
- RAG: ChromaDB + BM25 + sentence-transformers
- Backend: FastAPI on Render free tier
- Frontend: React on Vercel free tier
- Observability: LangSmith + Prometheus + Grafana
- CI/CD: GitHub Actions
- Models: HuggingFace Inference API free tier
- Total cost: Zero

## Run Locally
`ash
git clone https://github.com/praveenkumar993/enterprise-migration-copilot
cd enterprise-migration-copilot
cp .env.example .env
# Add your HF_TOKEN and LANGCHAIN_API_KEY to .env

# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
`

## Architecture
Coming soon with benchmark results.
