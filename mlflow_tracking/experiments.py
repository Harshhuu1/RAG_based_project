from typing import List , Dict , Any
from loguru import logger
from itertools import product ## generates the all combinations automatically
import mlflow 
import yaml
from pathlib import Path
from config import settings


#experiment parameter grid________

EXPERIMENT_GRID={
    "chunk_size": [256,512,1024],
    "chunk_overlap": [0,50,100],
    "chunking_strategy": ["fixed", "sentence"],
    "embedding_model":[
        "text-embedding-3-small",
        "text-embedding-3-large"
    ],
}
# here itertools will combines these values in every possible combinations like grid search cv


def run_single_experiment(
        params:Dict[str,Any],
        test_data_path:str,

)-> Dict[str,Any]:
    """ Run a single experminet with given params and return metrics."""

    #update params.yaml with current experiment params
    params_path=Path("params.yaml")
    with open(params_path,"r") as f:
        current_params=yaml.safe_load(f)
    
    current_params["ingestion"]["chunk_size"]=params["chunk_size"]
    current_params["ingestion"]["chunk_overlap"]=params["chunk_overlap"]
    current_params["ingestion"]["chunking_strategy"]=params["chunking_strategy"]
    current_params["embeddings"]["model"]=params["embedding_model"]

    with open(params_path,"w") as f:
        yaml.dump(current_params,f)

    import importlib
    import config
    importlib.reload(config)

    #run pipeline with new param

    from ingestion.pipeline import run_pipeline
    results =run_pipeline(test_data_path)

    return results



#Takes one combination of params and runs the full pipeline with it
#Updates params.yaml with the new values before each run
#importlib.reload(config) → reloads config so it picks up the new params.yaml values
#Returns the results so we can log them to MLflow

def run_all_experiments(test_data_path:str)->Dict[str,Any]:
    """Run all perimeter combinations and track mlflow"""

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    #generate all combinations

    keys=list(EXPERIMENT_GRID.keys())
    values=list(EXPERIMENT_GRID.values())
    combinations=list(product(*values))

    logger.info(f"Running {len(combination)} experiments...")
     
    best_run_id=None
    best_chunk =0
    all_results=[]

    for i , combo in enumerate(combinations):
        params=dict(zip(keys,combo))

        logger.info(f"Experiment {i+1}/{len(combinations)}; {params}")

        with mlflow.start_run(run_name=f"experiment_{i+1}"):
            
            # log parameters

            mlflow.log_params(params)

            try:

                #run experiment
                result = run_single_experiment(params,test_data_path)

                #log metrics
                mlflow.log_metrics({
                    "total_chunks": result["total_chunks"],
                    "total_documents": result["total_documents"],
                    "total_embedded": result["total_embedded"],
                })

                #tack best run

                if result["total_chunks"]> best_chunk:
                    best_chunks= result["total_chunks"]
                    best_run_id=mlflow.active_run().info.run_id

                all_results.append({
                    "params": params,
                    "results":result,
                    "run_id": mlflow.active_run().info.run_id
                })

                logger.success(f"experiment {i+1} completee : {result}")

            except Exception as e:
                logger.error(f"Experiment {i+1} failed : {e}")
                mlflow.log_param("error",str(e))
        logger.success(f"All experiments complete! Best run : {best_run_id}")
        return{
            "best_run_id":best_run_id,
            "total_experiments":len(combinations),
            "all_results":all_results,
        }
    


# product(*values) → generates all 36 combinations automatically
# Each combination runs in its own mlflow.start_run() so they're all tracked separately
# Tracks the best run based on total chunks created
# try/except → if one experiment fails, it logs the error and continues with the rest

if __name__=="__main__":
    import sys

    if len(sys.argv)<2:
        print("Usage: python -m mlflow_tracking.experiments <path-to-test-data>")
        print("Example:python -m mlflow_tracking_experiments data/raw")

    test_data_path=sys.argv[1]

    print("\n starting experiments [mlflow]")
    print(f" test data:{test_data_path}")
    print(f" total_combinations: {len(list(product(*EXPERIMENT_GRID.values())))}")
    print(f" Mlflow_UI:{settings.mlflow_tracking_uri}\n")

    result=run_all_experiments(test_data_path)

    print("experiments complete")
    print(f" total runs : {result['total_experiments']}")
    print(f" best run id : {result['best_run_id']}")
    print(f" best run id : {result['best_run_id']}")
    print(f" view result : {settings.mlflow_tracking_uri}")