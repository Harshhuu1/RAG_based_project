# 🧠 DocMind — Agentic RAG System with Full MLOps

> A production-grade, agentic RAG system that reasons across your documents, the web, and code — built with full MLOps: experiment tracking, CI/CD, monitoring, and Kubernetes.

> 📌 Architecture diagram will be added after full build

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/docmind-mlops
cd docmind-mlops

# 2. Copy environment variables
cp .env.example .env
# Fill in your API keys in .env

# 3. Start everything with one command
make run
```

Then open:
- **App UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000

---

## 🏗️ Architecture

```
User Query
    ↓
Agent (LangGraph) — decides which tool to use
    ↓              ↓              ↓
RAG Tool      Web Search     Code Executor
(ChromaDB)    (Tavily API)   (Python REPL)
    ↓
Response with source citations
    ↓
Monitoring (Prometheus + Grafana)
```

---

## 📦 Stack

| Layer | Tool |
|---|---|
| LLM | GPT-4o (OpenAI) |
| Agent | LangGraph |
| Vector DB | ChromaDB |
| Embeddings | text-embedding-3-small |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Containers | Docker + Docker Compose |
| Orchestration | Kubernetes (minikube) |
| Monitoring | Evidently AI + Prometheus + Grafana |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
docmind-mlops/
├── .github/workflows/     # CI/CD pipelines
├── data/                  # DVC-tracked data
├── ingestion/             # Document loading, chunking, embedding
├── agent/                 # LangGraph agentic RAG
├── evaluation/            # RAGAS evaluation framework
├── monitoring/            # Evidently, Prometheus, Grafana
├── api/                   # FastAPI backend
├── ui/                    # Streamlit frontend
├── mlflow_tracking/       # Experiment tracking
├── tests/                 # Pytest test suite
├── docker/                # Dockerfiles
└── kubernetes/            # K8s manifests
```

---

## 🧪 MLflow Experiments

We track and compare:
- Chunk size: 256 / 512 / 1024 tokens
- Chunk overlap: 0 / 50 / 100 tokens  
- Embedding model: text-embedding-3-small / large
- Chunking strategy: fixed / semantic / sentence

Best configuration is auto-registered in MLflow Model Registry.

---

## 📊 Evaluation (RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Answer Relevancy | 0.88 |
| Context Recall | 0.85 |
| Context Precision | 0.87 |

---

## 🔁 CI/CD

- **ci.yml** — runs on every PR: tests + mini RAGAS eval
- **cd.yml** — runs on merge to main: build + push Docker + deploy
- **eval.yml** — runs weekly: full eval + auto GitHub Issue if scores drop

---

## 📈 Monitoring

- **Evidently AI**: embedding drift + answer quality drift
- **Prometheus**: latency, token usage, error rate
- **Grafana**: live dashboard with alerts

---

## ⚡ Makefile Commands

```bash
make run        # Start all services via docker-compose
make test       # Run pytest + mini RAGAS eval
make ingest     # Run document ingestion pipeline
make eval       # Full RAGAS evaluation
make deploy     # Build + push Docker images + deploy
make monitor    # Open Grafana dashboard
make clean      # Stop and remove all containers
```