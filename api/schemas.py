#  What is schemas.py?
# Schemas define the shape of data going in and out of the API. Think of them as contracts — if you send wrong data, FastAPI automatically 
# rejects it with a clear error message.

from pydantic import BaseModel ,Field
from typing import Optional, List
from datetime import datetime


#_____ Request schemas____

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    chat_history: Optional[list] = Field(default=[])

class IngestRequest(BaseModel):
    """Request schema for ingestion documents"""
    path: str=Field(..., description="Path to file or directory")
    experiment_name: Optional[str]=None


#___Responses schemas___

class QueryResponse(BaseModel):
    """Response schema for agent queries"""
    answer:str
    question:str
    latency_ms:float
    timestamp:datetime=Field(default_factory=datetime.utcnow)

class IngestResponse(BaseModel):
    """Response schema for document ingestiion"""
    total_documents:int
    total_chunks: int 
    total_embedded: int 
    timestamp:datetime=Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    """Response schema for health check"""

    status:str
    version:str="1.0.0"
    timestamp:datetime=Field(default_factory=datetime.utcnow)


# What this means:

# BaseModel → pydantic model that auto validates data
# Field(..., min_length=1) → ... means required, min_length=1 means can't be empty
# Field(default=5, ge=1, le=20) → default is 5, must be between 1 and 20
# default_factory=datetime.utcnow → automatically sets current timestamp