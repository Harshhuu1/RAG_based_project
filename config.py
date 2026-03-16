import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

PARAMS_PATH=Path(__file__).parent / "params.yaml"

def load_params()-> dict:
    with open(PARAMS_PATH,"r") as f:
        return yaml.safe_load(f)
PARAMS = load_params()

#--- ENVIRONMENT settings----

class Settings(BaseSettings):
    #api keys
    openai_api_key: str=Field(... ,env="OPENAI_API_KEY")
    tavily_api_key: str=Field(... , env="TAVILY_API_KEY")

    #chromaDB

    chroma_host: str=Field(default="localhost",env="CHROMA_HOST")
    chroma_port: int=Field(default=8001, env="CHROMA_PORT")
    chroma_collection_name: str= Field(default="docmind",env="CHROMA_COLLECTION_NAME")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="docmind-rag", env="MLFLOW_EXPERIMENT_NAME")

    #app
    api_host: str=Field(default="0.0.0.0" ,env="API_HOST")
    api_port:   int=Field(default=8000 , env="API_PORT")
    log_level:str=Field(default="INFO",env="LOG_LEVEL")
    env: str=Field(default="development",env="ENV")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

#--Typed param accessors---
class IngestionConfig:
    chunk_size: int =PARAMS["ingestion"]["chunk_size"]
    chunk_overlap: int=PARAMS["ingestion"]["chunk_overlap"]
    chunking_strategy: str=PARAMS["ingestion"]["chunking_strategy"]
    supported_formats: list =PARAMS["ingestion"]["supported_formats"]

class EmbeddingConfig:
    model: str=PARAMS["embeddings"]["models"]
    batch_size: int =PARAMS["embeddings"]["batch_size"]
    dimensions: int=PARAMS["embeddings"]["dimensions"]

class RetrievalConfig:
    top_k: int=PARAMS["retrieval"]["top_k"]
    score_threshold=PARAMS["retrieval"]["score_threshold"]

class AgentConfig:
    model: str=PARAMS["agent"]["model"]
    temperature: float=PARAMS["agent"]["temperature"]
    max_tokens: int =PARAMS["agent"]["max_tokens"]
    max_iterations:int=PARAMS["agent"]["max_iterations"]
    tools:list=PARAMS["agent"]["tools"]

class EvaluateConfig:
    golden_dataset_path:str=PARAMS["evaluation"]["golden_dataset_path"]
    min_faithfulness:float=PARAMS["evaluation"]["min_faithfulness"]
    min_answer_relevancy=PARAMS["evaluation"]["min_answer_relevancy"]

class MonitoringConfig:
    latency_threshold_ms: int = PARAMS["monitoring"]["latency_threshold_ms"]
    error_rate_threshold: float = PARAMS["monitoring"]["error_rate_threshold"]
# MLflow


# ── Singleton instances ───────────────────────────────
settings = Settings()
ingestion_cfg = IngestionConfig()
embedding_cfg = EmbeddingConfig()
retrieval_cfg = RetrievalConfig()
agent_cfg = AgentConfig()
eval_cfg = EvaluateConfig()
monitoring_cfg = MonitoringConfig()