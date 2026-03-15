from typing import List
from loguru import logger
from langchain.tools import tool
from langchain_core.documents import Document
from tavily import TavilyClient

from ingestion.embedder import search_documents
from config import settings

# tools decorator make any normal python function to act as a agentic .function

@tool 
def rag_retrieval(query:str)->str:
    """search through upload documents to find the relevant information.
    Use this tool when the question is about the content of uploaded documents"""

    logger.info(f"Rag tool called with query :'{query[:50]}")
    docs: List[Document]=search_documents(query)
    if not docs:
        return "No relevant documents found for this query."
    
    #format results with citations
    results=[]
    for i ,doc in enumerate(docs):
        source=doc.metadata.get("source","unknown")
        chunk_index=doc.metadata.get("chunk_index",i)
        results.append(
            f"[source {i+1}:source -chunk {chunk_index}] \n {doc.page_content}")
    return "\n\n" .join(results)

# The docstring is very important — the agent reads it to decide when to use this tool
# search_documents(query) → calls our ChromaDB search function
# Formats results with source citations so the agent can tell users where the answer came from

@tool 
def web_search(query:str)->str:

    """search the internet for current information.
    use this tool when the question requires up to date information
    or when the answer is not found in the uploaded documetns"""

    logger.info(f"web search tool called with query :'{query[:50]}' ")
    client=TavilyClient(api=settings.tavily_api_key)
    response=client.search(
        query=query,
        search_depth="basic",
        max_results=5,
    )
    if not response["results"]:
        return "No web result found for this query"
    #format result
    result=[]
    for i , result in enumerate(response["results"]):
        result.append(
            f"[web source {i+1}: {result['url']}]\n{result['content']}"
        )
    
    
    return "\n\n".join(result)

@tool
def code_executor(code:str)->str:
    """execute python code and return the output.
    use this tool when the questin requires the calculatiions,
    data analysis , or any kind of comoputiation"""

    logger.info(f"code execution tool called")

    #capture output
    import io
    import sys
    import traceback

    old_stdout=sys.stdout
    sys.stdout=buffer=io.StringIO()

    try:
        exec(code,{"__builtins__":__builtins__})
        output=buffer.getvalue()
        if not output:
            output="code executed successfully with no ouput."
        logger.success(f"code executed successfully")
        return f"output:\n{output}"
    
    except Exception as e:
        error =traceback.format_exc()
        logger.error(f"code execution failed:{e}")
        return f"Error:\n{error}"
    finally:
        sys.stdout=old_stdout

#list of all tools

TOOLS=[rag_retrieval,web_search,code_executor]