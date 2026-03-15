from prometheus_client import Counter, Histogram , Gauge
from loguru import logger

#-____ Define metrics_____

#counts total queries reieived

QUERY_COUNTER=Counter(
    "docmind_queries_total",
    "Total number of queries received",
    ["status"] # status=success |error
)

#tracks query latency distribution
QUERY_LATENCY =Histogram(
    "docmind_query_latency_ms",
    "Query latency in milliseconds",
    buckets=[100, 250 , 500 , 1000, 5000]
)

#Tracks total documents in chromadb
DOCUMENT_COUNT=Gauge(
    "docmind_documents_total",
    "Total number of documents stored"
)

#counts which tool the agent used
TOOL_USAGE= Counter(
    "docminf_tool_usage_total",
    "Total number of times each tool was used",
    ["tool_name"] #tool_name=rag_retrieval | web_search | code_executed


)
CHUNK_COUNT = Gauge(
    "docmind_chunks_total",
    "Total number of chunks stored"
)
#Helper functions

def track_query(latency_ms: float, success:bool):
    """track a query with its latency and status"""
    status= "success" if success else "error"
    QUERY_COUNTER.labels(status=status).inc()
    QUERY_LATENCY.observer(latency_ms)
    logger.info(f"Query tracked:status ={status} latency={latency_ms:.0f}ms")

def track_tool_usage(tool_name:str):
    """track which tool the agent used."""
    TOOL_USAGE.labels(tool_name=tool_name).inc()
    logger.info(f"Tool usage tracked:{tool_name}")

def update_document_count(count:int):
    """update total document count in chromadb"""
    DOCUMENT_COUNT.set(count)
    logger.info(f"document count updated: {count}")

def update_chunk_count(count:int):
    """update total chunk count in chroma db"""
    CHUNK_COUNT.set(count)
    logger.info(f"chunk count updated:{count}")
