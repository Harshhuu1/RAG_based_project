from typing import List
from loguru import logger
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import settings , embedding_cfg, retrieval_cfg

def get_embeddings():
    """FGet OpenAI embeddings model."""
    return OpenAIEmbeddings(
        model=embedding_cfg.model,
        openai_api_key=settings.openai_api_key,

    )

def get_vectorstore():
    """Connect to ChromaDB and return vectorstore."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        client_settings=chromadb.Settings(
            chroma_api_impl="rest",
            chroma_server_host=settings.chroma_host,
            chroma_server_http_port=settings.chroma_port,
        ),
    )

    logger.info(f"Connected to ChromaDB at {settings.chroma_host}:{settings.chroma_port}")
    return vectorstore


def embed_documents(chunks:List[Document])->int:
    """Embed chunks and store in ChromaDB"""
    if not chunks:
        logger.warning("No chunks to embed!")
        return 0
    
    vectorstore=get_vectorstore()
    logger.info(f"Embedding {len(chunks)} chunks....")
    
    #add documents in batches
    batch_size=embedding_cfg.batch_size
    for i in range(0,len(chunks),batch_size):
        batch=chunks[i:i+batch_size]
        vectorstore.add_documents(batch)
        logger.info(f"Embedded batch {i// batch_size+1} /{len(chunks)// batch_size+1}")
    
    logger.success(f"Successfully embedded {len(chunks)} into chromaDB")
    return len(chunks)
def search_documents(query:str, top_k:int=None)->List[Document]:
    """Search chromaDB for relevant chunks."""
    vectorstore=get_vectorstore()

    k=top_k or retrieval_cfg.top_k
    results=vectorstore.similarity_search(
        query=query,
        k=k,
    )
    logger.info(f"Retrieved {len(results)} chunks for query : '{query[:50]}...")
    return results
