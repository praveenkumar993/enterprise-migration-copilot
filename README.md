# ⚡ Enterprise Migration Copilot

> AI-powered platform that converts legacy SQL, HiveQL, PL/SQL, and Stored Procedures to production-ready PySpark — with built-in validation, PII detection, risk scoring, and a 6-agent LangGraph pipeline.

Built as a zero-cost portfolio project targeting real fintech migration pain: legacy data pipelines that need to move to Spark/Databricks without a $500K consulting engagement.

---

## Live Demo

| Service | URL |
|---|---|
| 🌐 Frontend | *Coming soon — deploying to Vercel* |
| ⚙️ API | *Coming soon — deploying to Render* |
| 📊 Dataset | [praveends/enterprise-migration-dataset](https://huggingface.co/datasets/praveends/enterprise-migration-dataset) |
| 🤗 Best Model | [migration-copilot-phi-3-5-mini-instruct](https://huggingface.co/praveends/migration-copilot-phi-3-5-mini-instruct) |

---

## What It Does

Paste any legacy SQL/HiveQL/PL/SQL/Stored Procedure → get back:

- ✅ **Generated PySpark code** via fine-tuned Phi-3.5-mini (57% benchmark pass rate)
- ✅ **Validation score** — syntax check, DataFrame ops check, semantic alignment
- ✅ **Risk report** — PII column detection (PAN, Aadhaar, card numbers, email), compliance flags, risk score 0-10
- ✅ **Migration notes** — procedural constructs needing manual review, anti-patterns, performance suggestions
- ✅ **Language auto-detection** — SQL, HiveQL, PL/SQL, or Stored Procedure (T-SQL)

---

## Architecture

```
Source Code (SQL / HiveQL / PL/SQL / Stored Procedure)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   Parser Layer                          │
│  sqlglot-based IR  ·  Language Router  ·  Dialect hints │
└─────────────────────────┬───────────────────────────────┘
                          │  Unified IR
                          ▼
┌─────────────────────────────────────────────────────────┐
│              6-Agent LangGraph Pipeline                 │
│                                                         │
│  ① Analyzer      → migration strategy, complexity      │
│  ② RAG Retrieve  → hybrid ChromaDB + BM25 context      │
│  ③ Migrator      → fine-tuned LLM code generation      │
│  ④ Validator     → syntax + semantic validation         │
│  ⑤ Optimizer     → anti-pattern detection              │
│  ⑥ Risk/Compliance → PII detection, risk score         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
              FastAPI  ·  React UI  ·  Prometheus/Grafana
```

---

## Benchmark Results

Evaluated on **480 held-out scripts** (120 per language × 4 difficulties), never seen during training.

### Base vs Fine-tuned

| Model | SQL | HiveQL | PL/SQL | SP | Overall |
|---|---|---|---|---|---|
| DeepSeek-1.3B base | 0% | 0% | 0% | 0% | 0% |
| DeepSeek-1.3B fine-tuned | 29% | 32% | 27% | 20% | 27% |
| Qwen2.5-1.5B base | 0% | 0% | 0% | 0% | 0% |
| Qwen2.5-1.5B fine-tuned | 48% | 61% | 50% | 22% | 45% |
| Phi-3.5-mini base | 0% | 0% | 0% | 0% | 0% |
| **Phi-3.5-mini fine-tuned** ⭐ | **64%** | **74%** | **57%** | **32%** | **57%** |

**Key finding:** Fine-tuning turns 0% → 27-57% across all model families. The base models score 0% because they don't follow the instruction prompt format — fine-tuning teaches both the format AND the migration task simultaneously.

### Why Not Just Use GPT-4?

The frontier benchmark is in progress. Early results show our best fine-tuned model (Phi-3.5-mini at 57%) gets within ~15-18% of frontier model performance at **zero inference cost** after training.

---

## Dataset

**1,312 validated SQL→PySpark training pairs**, publicly available on HuggingFace.

| Source | Raw Generated | Valid Pairs | Pass Rate |
|---|---|---|---|
| Claude (hand-crafted) | 300 | 299 | 99.67% |
| Ollama/qwen2.5-coder (local) | 1,533 | 1,013 | 79.58% |
| **Total (after dedup)** | — | **1,312** | — |

**Distribution:** 4 source languages × 4 difficulty tiers (easy/medium/hard/expert), ~300-340 examples per language.

**7-check validation pipeline** filters every pair:
- Required fields present
- Valid language/difficulty enums
- Minimum content length
- PySpark syntax (`ast.parse`)
- Real DataFrame operations present
- No placeholder text
- Source-to-PySpark token alignment (60% threshold)

---

## Fine-tuning

3 models fine-tuned using QLoRA/LoRA on Google Colab T4 GPU (free tier):

| Model | Adapter Size | Train Loss | Eval Loss | Time |
|---|---|---|---|---|
| DeepSeek-1.3B | 25.2 MB | 0.258 | 0.329 | 19 min |
| Qwen2.5-1.5B | 17.5 MB | 0.307 | 0.344 | 21 min |
| **Phi-3.5-mini** | **12.6 MB** | **2.133** | **0.294** | 54 min |

All adapters are LoRA weights only (~12-25MB each), merged with the base model at inference time.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Parsing** | sqlglot (SQL/HiveQL), regex + AST (PL/SQL, T-SQL) |
| **IR** | Unified JSON schema across all 4 source languages |
| **Orchestration** | LangGraph (9-node StateGraph) + LangSmith tracing |
| **RAG** | ChromaDB (vector) + BM25 (keyword), Reciprocal Rank Fusion |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **LLM** | Fine-tuned Phi-3.5-mini via HuggingFace Inference API |
| **Fine-tuning** | PEFT LoRA (r=16, alpha=32), SFTTrainer, bitsandbytes |
| **API** | FastAPI + Pydantic + Prometheus instrumentation |
| **Frontend** | React 18, inline CSS, typewriter animation |
| **Observability** | Prometheus metrics (7 custom counters/histograms) + Grafana dashboard |
| **Dataset gen** | Ollama (qwen2.5-coder:1.5b), delimiter-based prompting, 85% yield rate |
| **CI** | GitHub Actions (pytest, 43 tests, all green) |

---

## Project Structure

```
enterprise-migration-copilot/
├── backend/
│   ├── agents/          # 6 agents: analyzer, migrator, validator, reviewer, optimizer, risk
│   ├── parsers/         # SQL, HiveQL, PL/SQL, SP parsers + language router
│   ├── rag/             # ChromaDB + BM25 hybrid retriever
│   ├── pipeline/        # LangGraph 9-node StateGraph
│   ├── utils/           # IR builder, Prometheus metrics
│   └── tests/           # 43 tests, all passing
├── frontend/            # React UI — editor, PySpark output, 3 result cards
├── dataset_gen/         # Generation pipeline, validator, merge/push scripts
├── finetuning/          # Colab notebook, model cards
├── benchmark/           # 480-script evaluation, leaderboard, failure analysis
├── monitoring/          # Prometheus config, Grafana dashboard
└── .github/workflows/   # CI — pytest on push
```

---

## Running Locally

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in your tokens
cp .env.example .env

uvicorn main:app --reload
# API: http://localhost:8000
# Metrics: http://localhost:8000/metrics
# Docs: http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm start
# UI: http://localhost:3000
```

### Observability (requires Docker)

```bash
docker compose up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill in:

```env
HF_TOKEN=hf_...              # HuggingFace token (write access)
HF_USERNAME=praveends        # Your HuggingFace username
LANGCHAIN_API_KEY=lsv2_...   # LangSmith tracing (optional)
LANGCHAIN_PROJECT=enterprise-migration-copilot
LANGCHAIN_TRACING_V2=true
```

---

## Test Suite

```bash
cd backend
pytest tests/ -v
# 43 tests, all passing
```

Tests cover: parsers (12), IR builder (8), RAG retrieval (8), all 6 agents (10), full pipeline integration (5).


---

## Author

**Praveen Kumar** — Data Engineer transitioning into ML Engineering

- 🔗 [LinkedIn](https://linkedin.com/in/praveenkumar993)
- 🤗 [HuggingFace](https://huggingface.co/praveends)
- 💻 [GitHub](https://github.com/praveenkumar993)

---

*Built entirely on free-tier infrastructure: local 8GB RAM machine, Google Colab T4 GPU, HuggingFace free hosting. Zero cloud spend.*