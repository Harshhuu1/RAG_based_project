from pathlib import Path
from loguru import logger
import mlflow 
from ingestion.loader import load_document,load_directory
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents
from config import settings, ingestion_cfg,embedding_cfg


def run_pipeline(path: str, experiment_name: str = None) -> dict:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)

    mlflow.end_run()  # Close any existing run first
    with mlflow.start_run():
        #log parameters to MLflow
        mlflow.log_params({
            "chunk_size": ingestion_cfg.chunk_size,
            "chunk_overlap": ingestion_cfg.chunk_overlap,
            "chunking_strategy":ingestion_cfg.chunking_strategy,
            "embedding_model" : embedding_cfg.model,
            "input_path":path

        })

    #step 1 -Load
    logger.info("step 1/3 -Loading documents...")
    input_path=Path(path)
    if input_path.is_dir():
        docs=load_directory(input_path)
    else:
        docs=load_document(input_path)
    
    #step -2
    logger.info("step- 2/3 -chunking documents.")
    chunks=chunk_documents(docs)

    #step 3-Embed
    logger.info("step 3/3 - Embedding and storing...")
    total_embedded=embed_documents(chunks)

    #log metrics to MLFlow
    mlflow.log_metrics({
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "total_embed": total_embedded,
        "avg_chunks_per_doc":len(chunks)/ max(len(docs),1),

    })

    results={
        "total_documents":len(docs),
        "total_chunks":len(chunks),
        "total_embedded":total_embedded,
    }
    logger.success(f"Pipeline complete! {results}")
    return results
if __name__=="__main__":
    import sys
    
    if len(sys.argv)<2:
        print("Usage :python-m ingestion.pipeline <path_to_file_or_directory>")
        print("Example :python 0m ingestion.pipeline data/raw/")
        sys.exit(1)

    path=sys.argv[1]
    results=run_pipeline(path)
    print(f"\n pipeline result:")
    print(f"  Documents loaded:{results['total_documents']}")
    print(f" chunks created : {results['total_chunks']}")
    print(f"   chunks embedded : {results['total_embedded']}")