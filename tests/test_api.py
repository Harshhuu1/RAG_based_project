import pytest 
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from api.main import app

#____Test client____
#Test client lets us test FastApi without running a server
client =TestClient(app)

#___health check tests

def test_health_check():
    """test that health endpoint returs healthy staus"""
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="healthy"

def test_health_check_has_timestamp():
    """Test that health response includes timestamp"""
    response=client.get("/health")
    assert "timestamp" in response.json()

#___query tests___
def test_query_empty_question():
    """Test that empty question returns validation error.."""
    response=client.post("/query",json={"question":""})
    assert response.status_code==422

def test_query_too_long_question():
    """test that question over 1000 chars returns validation error"""
    response=client.post("/query", json={"question":"x"*1001})
    assert response.status_code==422
@patch("api.main.query_agent")
def test_query_success(mock_agent):
    """tets successful query return correct response"""
    #Mock the agent so we dont need real api keys in ci
    mock_agent.return_value="This is a test answer."

    response=client.post("/query",json={"question":"What is RAG?"})

    assert response.status_code==200
    assert response.json()["answer"]=="This is a test answer."
    assert response.json()["question"]=="What is RAG?"
    assert "latency_ms" in response.json()

@patch("api.main.query_agent")
def test_query_agent_failure(mock_agent):
    """test that agent failure returns 500 error"""
    #Mock the agent to raise an exfeption
    mock_agent.side_effect=Exception("Agent failed")

    response=client.post("/query",json={"question":"what is RAG?"})
    assert response.status_code==500

# What this means:

# TestClient → tests FastAPI without actually running a server
# @patch("api.main.query_agent") → replaces the real agent with a fake one during testing. This means tests run without needing OpenAI API keys!
# mock_agent.return_value → sets what the fake agent returns
# mock_agent.side_effect = Exception → makes the fake agent crash to test error handling
# 422 → FastAPI's validation error code when request data is invalid