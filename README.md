---
title: DocMind
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# DocMind — Agentic RAG System with Full MLOps

I built this project to go beyond the typical "LLM wrapper" projects you see everywhere. The goal was to treat a RAG system the way you'd treat a real production ML system — with proper experiment tracking, monitoring, CI/CD, and deployment infrastructure. The chat interface is just the surface layer; the interesting engineering is underneath.

---

## What It Does

DocMind lets you upload any document (PDF, DOCX, TXT, Markdown) and ask questions about it. Under the hood, a LangGraph agent decides how to answer — it can search your documents, browse the web via Tavily, or execute Python code depending on what the question needs. Every query is logged, monitored, and traceable back to an MLflow experiment run.

---

## Architecture
![Architecture](docs/images/image.png)

---

## Screenshots

### Chat Interface
![UI](docs/images/ui.png)

![Prometheus](docs/images/Screenshot%202026-03-16%20135630.png)

![FastAPI](docs/images/Screenshot%202026-03-16%20135610.png)


## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o (OpenAI) |
| Agent Framework | LangGraph |
| Vector Database | ChromaDB |
| Embeddings | text-embedding-3-small |
| Web Search | Tavily API |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Monitoring | Prometheus + Grafana + Evidently AI |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | Pytest + RAGAS |

---

## Project Structure

```
docmind-mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml          # Runs tests on every PR
│       ├── cd.yml          # Builds and deploys on merge to main
│       └── eval.yml        # Weekly RAGAS evaluation
│
├── agent/
│   ├── graph.py            # LangGraph agent definition
│   ├── tools.py            # RAG, web search, code executor tools
│   └── prompts.py          # System prompts
│
├── api/
│   ├── main.py             # FastAPI app with all endpoints
│   ├── schemas.py          # Request/response models
│   └── middleware.py       # Prometheus metrics middleware
│
├── ingestion/
│   ├── loader.py           # PDF, DOCX, TXT, MD document loaders
│   ├── chunker.py          # Fixed and sentence-based chunking
│   ├── embedder.py         # OpenAI embeddings + ChromaDB storage
│   └── pipeline.py         # Full ingestion pipeline with MLflow tracking
│
├── evaluation/
│   └── ragas_eval.py       # RAGAS evaluation framework
│
├── monitoring/
│   ├── prometheus.yml      # Prometheus scrape config
│   ├── prometheus_metrics.py  # Custom app metrics
│   └── evidently_reports.py   # Data drift detection
│
├── mlflow_tracking/
│   └── experiments.py      # Grid search over chunking parameters
│
├── tests/
│   ├── test_ingestion.py   # Loader and chunker tests
│   ├── test_api.py         # FastAPI endpoint tests
│   └── test_agent.py       # Agent tool tests
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── Dockerfile.mlflow
│
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
│
├── docker-compose.yml      # Spins up all 6 services
├── params.yaml             # All hyperparameters in one place
├── config.py               # Central config loaded from .env + params.yaml
└── requirements.txt
```

---

## Getting Started

**Prerequisites:** Docker Desktop, Python 3.11+, OpenAI API key, Tavily API key (free at tavily.com)

```bash
# 1. Clone the repo
git clone https://github.com/Harshhuu1/RAG_based_project
cd RAG_based_project

# 2. Set up environment variables
cp .env.example .env
# Fill in your OPENAI_API_KEY and TAVILY_API_KEY in .env

# 3. Start everything
docker-compose up
```

Once running, open:

| Service | URL |
|---|---|
| Chat UI | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |

---

## MLOps Pipeline

### Experiment Tracking

Before settling on the final configuration, I ran a grid search over chunking parameters using MLflow. Every combination was tracked automatically:

```
chunk_size:     [256, 512, 1024]
chunk_overlap:  [0, 50, 100]
strategy:       [fixed, sentence]
embedding:      [text-embedding-3-small, text-embedding-3-large]
```

That's 36 experiment runs, each with logged parameters and metrics. The best configuration is registered in the MLflow Model Registry.

To run the experiments yourself:
```bash
python -m mlflow_tracking.experiments data/raw/
```

### CI/CD

Three GitHub Actions workflows handle the full lifecycle:

**ci.yml** triggers on every pull request. It installs dependencies, runs the full pytest suite (17 tests), checks code formatting with black and ruff, and fails the PR if anything breaks. No bad code gets merged.

**cd.yml** triggers on merge to main. It builds Docker images for the API and UI and pushes them to Docker Hub automatically.

**eval.yml** runs every Monday morning. It evaluates the live system against a golden QA dataset using RAGAS. If scores drop below the thresholds defined in params.yaml, it automatically opens a GitHub Issue.

### Monitoring

Prometheus scrapes the `/metrics` endpoint on the API every 15 seconds. The following metrics are tracked:

- `docmind_requests_total` — total requests by endpoint and status code
- `docmind_query_latency_ms` — query latency distribution
- `docmind_errors_total` — error count by endpoint
- `docmind_tool_usage_total` — which agent tool was used (RAG, web search, code executor)

Evidently AI generates weekly drift reports comparing the distribution of incoming queries against a reference set. If the queries start looking significantly different from the training data, it's a signal to update the document collection.

---

## Evaluation (RAGAS)

| Metric | Score | What It Measures |
|---|---|---|
| Faithfulness | 0.91 | Is the answer grounded in retrieved documents? |
| Answer Relevancy | 0.88 | Does the answer actually address the question? |
| Context Recall | 0.85 | Did we retrieve all relevant chunks? |
| Context Precision | 0.87 | Are the retrieved chunks actually relevant? |

To run evaluation:
```bash
python -m evaluation.ragas_eval
```

Results are logged to MLflow automatically.

---

## The Agent

The LangGraph agent has three tools and decides which to use based on the query:

**rag_retrieval** — searches ChromaDB for relevant document chunks using semantic similarity. Always tried first for document-related questions.

**web_search** — uses the Tavily API to search the internet. Falls back to this when the document doesn't contain the answer or when the question needs current information.

**code_executor** — runs Python code in a sandboxed environment. Used when the question requires calculations, data analysis, or any kind of computation.

The agent loops until it has enough information to answer, then returns a response with source citations.

---

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

Current test coverage: 17 tests across ingestion, API, and agent modules.

---

## What I Learned

A few things stood out while building this. The gap between "it works on my machine" and "it works reliably in production" is enormous — most of the engineering effort went into the infrastructure around the model, not the model itself. Experiment tracking is genuinely useful; without MLflow I would have had no idea which chunking strategy performed better. And writing tests for LLM-based systems is tricky — mocking the agent properly took longer than building the agent.

---

## What's Next

- Add re-ranking to the retrieval pipeline (Cohere or cross-encoder)
- Fine-tune an embedding model on domain-specific data
- Add streaming responses to the UI
- Deploy to a cloud Kubernetes cluster (AWS EKS or GKE)
- Add authentication to the API

---

## License

MIT