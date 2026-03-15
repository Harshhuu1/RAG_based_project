import pytest 
from unittest.mock import patch , MagicMock
from langchain_core.documents import Document

#___RAG tool tests____
@patch("agent.tools.search_documents")
def test_rag_retrieval_success(mock_search):
    """Test RAg tool returns formatted results"""

    #Mock chromaDB search results

    mock_search.return_value = [
        Document(
            page_content="RAG stands for Retrieval Augmented Generation.",
            metadata={"source": "test.txt","chunk_index":0}
        )
    ]

    from agent.tools import rag_retrieval
    result = rag_retrieval.invoke("What is RAG?")

    assert "RAG stands for Retrieval Augmented Generation" in result
    # assert "test.txt" in result

@patch("agent.tools.search_documents")
def test_rag_retrieval_no_results(mock_search):
    """Test RAg tool handles empty results"""
    mock_search.return_value =[]
    from agent.tools import rag_retrieval
    result = rag_retrieval.invoke("unknown query")

    assert "No relevant documents found" in result

#___code executor tests_____
def test_code_executor_success():
    """Test code executor runs valid python code"""
    from agent.tools import code_executor
    result=code_executor.invoke("print('hello world')")
    assert "hello world" in result

def test_code_executor_handles_error():
    """test code executor handles invalid code grwcefully"""
    from agent.tools import code_executor
    result=code_executor.invoke("invalid python code!!!")
    assert "Error" in result

def test_code_executor_math():
    """test code execuutoe can do calculations"""
    from agent.tools import code_executor
    result=code_executor.invoke("print(2+2)")
    assert "4" in result