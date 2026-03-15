import time
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from fastapi import FastAPI, UploadFile, File, HTTPException
from api.schemas import(
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,

)

from api.middleware import metrics_middleware
from agent.graph import query_agent
from ingestion.pipeline import run_pipeline
from config import settings

#__App setup 
app=FastAPI(
    title="DocMind API",
    description="Agentic Rag system with full MLOps",
    version="1.0.0",
)

#add cors middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

#add metrics middleware
app.middleware("http")(metrics_middleware)

#mount prometheus

metrics_app=make_asgi_app()
app.mount("/metrics",metrics_app)


# What this means:

# FastAPI(title=...) → creates the app with auto generated docs at /docs
# CORSMiddleware → allows the Streamlit frontend to talk to this API from a different port
# allow_origins=["*"] → accepts requests from any origin. In production you'd restrict this to your frontend URL
# app.mount("/metrics", metrics_app) → exposes Prometheus metrics at /metrics endpoint — this is what Prometheus scrapes every 15 seconds

#__Endpoints______
@app.get("/health",response_model=HealthResponse)
async def health_check():
    """check if API is running"""
    return HealthResponse(status="healthy")

@app.post("/query",response_model=QueryResponse)
async def query(request:QueryRequest):
    """Send a question to the agent """
    logger.info(f"Query received: '{request.question[:50]}'")
    start_time=time.time()

    try:
        answer=query_agent(request.question)
        latency_ms=(time.time()-start_time)*1000

        return QueryResponse(
            answer=answer,
            question=request.question,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Query failed:{e}")
        raise HTTPException(status_code=500,detail=str(e))

@app.post("ingest", response_model=IngestResponse)
async def ingest(request:IngestRequest):
    """ingest documents into chromadb"""
    logger.info(f"INgestiion requested for :{request.path}")

    try:
        result=run_pipeline(
            path=request.path,
            experiment_name=request.experiment_name,
        )
        return IngestResponse(
            total_documents=result["total_documents"],
            total_chunks=result["total_chunks"],
            total_embedded=results["total_embedded"],)
    
    except Exception as e:
        logger.error(f"Ingestion failed:{e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """upload a file and ingest it."""
    logger.info(f"file upload received:{file.filename}")

    #save file to data/raw
    file_path=f"data/raw/{file.filename}"
    with open(file_path,"wb") as f:
        content=await file.read()
        f.write(content)

    #ingest the uploaded file
    results=run_pipeline(path=file_path)
    return IngestResponse(
        total_documents=results["total_documents"],
        total_chunks=results["total_chunks"],
        total_embedded=results["total_embedded"],   
    )

# What this means:

# @app.get("/health") → simple health check, used by Docker and Kubernetes to verify app is running
# @app.post("/query") → main endpoint, takes a question and returns agent's answer with latency
# @app.post("/ingest") → triggers ingestion pipeline for a given path
# @app.post("/upload") → accepts file uploads directly, saves to data/raw/ and ingests automatically
# HTTPException(status_code=500) → returns proper error responses instead of crashing