import json
import mlflow 
from loguru import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_prescision,
)

from langchain_openai import ChatOpenAI , OpenAIEmbeddings

from agent.graph import query_agent
from ingestion.embedder import search_documents
from config import settings, eval_cfg

# What this means:

# ragas → the evaluation framework specifically built for RAG systems
# faithfulness → checks if answer is grounded in retrieved context
# answer_relevancy → checks if answer addresses the question
# context_recall → checks if relevant chunks were retrieved
# context_precision → checks if retrieved chunks are actually relevant
# We import MLflow to log evaluation scores automatically


def load_golden_dataset()->list:
    """Lost golden QA pairs from JSON file."""
    with open(eval_cfg.golden_dataset_path,"r") as f:
        data=json.load(f)
    logger.info(f"Loaded {len(data)} golden QA pairs")
    return data

def  prepare_ragas_dataset(golden_data:list) ->Dataset:
    """Run each quuestion through the agent and prepare dataset for RAGAS evaluation."""

    questions=[]
    answers=[]
    contexts=[]
    ground_truths=[]

    for item in goldenn_data:
        question=item["question"]
        ground_truth=item["ground_truth"]

        logger.info(f"Preprocessing:'{question[:50]}' ")

        #get agent answer 
        answer=query_agent(question)

        #get retrieved contexts

        retrieved_docs=search_documents(questions)
        context=[doc.page_content for doc in retrieved_docs]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question":questions,
        "answer":answers,
        "contexts":contexts,
        "ground_truth":ground_truths,
    })


def run_evaluation() -> dict:
    """Run full RAGAS evaluation and log to MLflow"""

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(setting.mlflow_experiment_name)

    with mlflow.start_run(run_name="ragas_evaluation"):

        #load dataset
        logger.info("Loading golden dataset...")
        golden_data=load_golden_dataset()

        #perpare dataset
        logger.info("Running questions through agent....")
        dataset= prepare_ragas_dataset(golden_data)

        #Run RAGAS evaluation

        logger.info("Running RAGAS evaluation...")
        results=evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_prescision,
                context_recall,
            ],
        )
        scores={
            "faithfulness":results["faithfulness"],
            "answer_relevancy":results["answer_relevancy"],
            "context_prescision":results["context_prescision"],
            "context_recall":results["context_recall"],

        }

        #  log to mlflow
        mlflow.log_metrics(scores)

        #check against toresholds

        passed=all([
            scores["faithfulness"]>=eval_cfg.min_faithfulness,
            scores["answer_relevancy"]>=eval_cfg.min_answer_relevancy,

        ])

        mlflow.log_param("evaluation_passed",passed)

        logger.success(f"Evaluation complete:{scores}")
        logger.info("fevaluation passed:{passed}")

        return {
            "scores":scores,
            "passed":passed,
        }


# What this means:

# Runs the full evaluation pipeline inside an MLflow run
# evaluate() → RAGAS calculates all 4 metrics automatically
# Checks scores against thresholds from params.yaml
# passed → boolean that tells CI/CD whether to proceed with deployment
# This is what your GitHub Actions CI pipeline will call on every PR!

if __name__ == "__main__":

    print("\n📊 Starting RAGAS Evaluation...")
    print(f"   Golden dataset: {eval_cfg.golden_dataset_path}")
    print(f"   Min faithfulness: {eval_cfg.min_faithfulness}")
    print(f"   Min answer relevancy: {eval_cfg.min_answer_relevancy}")
    print(f"   MLflow: {settings.mlflow_tracking_uri}\n")

    results = run_evaluation()

    print(f"\n✅ Evaluation Results:")
    print(f"   Faithfulness      : {results['scores']['faithfulness']:.3f}")
    print(f"   Answer Relevancy  : {results['scores']['answer_relevancy']:.3f}")
    print(f"   Context Recall    : {results['scores']['context_recall']:.3f}")
    print(f"   Context Precision : {results['scores']['context_precision']:.3f}")
    print(f"\n   Passed: {'✅ YES' if results['passed'] else '❌ NO'}")
    print(f"   View in MLflow: {settings.mlflow_tracking_uri}")