import pandas as pd
from loguru import logger
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TextOverviewPreset,
)

from evidently.metrics import (
    ColumnDriftMetric,
)
from datetime import datetime
from pathlib import Path


#--- drift detection----

def generate_drift_report(
        reference_data:list,
        current_data: list,
        report_path:str="monitoring/reports"
)->dict:
    """compare refrence data vs current data to detect drift
    refrence data=queries from last week
    current data=queries from this week"""

    #convert to DataFrame
    reference_df=pd.DataFrame({
        "question": reference_data,
        "question_length":[len(q) for q in current_data],
    })

    current_df=pd.DataFrame({
        "question":current_data,
        "question_length":[len(q) for q in current_data],
    })

    #build evidently report

    report=Report(metrics=[
        DataDriftPreset(),
        ColumnDriftMetric(column_name="question_length"),
    ])

    report.run(
        reference_data=reference_df,
        current_data=current_data,
    )

    #save report to HTML file
    Path(report_path).mkdir(parents=True,exist_ok=True)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path=f"{report_path}/drift_report_{timestamp}.html"
    report.save_html(output_path)

    #get drift result
    report_dict=report.as_dict()
    drift_detected=report_dict["metrics"][0]["result"]["dataset_drift"]

    logger.info(f"Drift report saved{output_path}")
    logger.info(f"drift detected:{drift_detected}")
    return {
        "drift_detected": drift_detected,
        "report_path":output_path,
        "timestamp":timestamp,
    }


#answer qualityb tracking___
def track_answer_quality(
        questions:list,
        answers:list,
        report_path:str="monitoring/reports"
)->str:
    """generate a text overview report for answer qualtiy"""
    df=pd.DataFrame({
        "question":questions,
        "answer":answers,
        "answer_length":[len(a) for a in answers],
            "question_length":[len(q) for q in questions],   
              })
    
    report=Report(metrics=[
        TextOverviewPreset(column_name="answer"),

    ])

    report.run(
        reference_data=None,
        current_data=df,
    )

    #save report 
    Path(report_path).mkdir(parents=True, exist_ok=True)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path=f"{report_path}/quality_report_{timestamp}.html"
    report.save_html(output_path)

    logger.success(f"quality report saved:{output_path}")
    return output_path

if __name__=="__main__":
    #Example usage with dummy data
    reference=[
        "What is RAG?",
        "How does chromaDB work?",
        "What is Langchain?",
    ]

    current=[
        "What is RAG?",
        "What is RAG?",
        "How does ChromaDB work?",
        "What is MLflow?",
        "How do I deploy a model?",
    ]

    print("\n generating drift report...")
    results=generate_drift_report(reference,current)
    print(f" drift detected: {results['drift_detected']}")
    print(f" report saved :{results['report_path']}")
