from typing import List
from loguru import logger
from langchain.schema import Document

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, #fixed size chunking stretegy
    SentenceTransformersTokenTextSplitter, # sentence aware chunking strategy
)

from config import ingestion_cfg

def chunk_fixed(docs:List[Document])-> List[Document]:
    """split documents into fixed size chunks"""
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=ingestion_cfg.chunk_size,
        chunk_overlap=ingestion_cfg.chunk_overlap,
        length_function=len,
        separators=["\n\n","\n","."," ",""],
    )

    chunks=splitter.split_documents(docs)
    logger.info(f"Fixed chunking : {len(docs)} docs-> {len(chunks)} chunks")
    return chunks
def chunk_sentence(docs: List[Document])-> List[Document]:
    """split documents into sentence aware chunks """
    splitter=SentenceTransformersTokenTextSplitter(
        chunk_overlap=ingestion_cfg.chunk_overlap,
        tokens_per_chunk=ingestion_cfg.chunk_size
    )

    chunks=splitter.split_documents(docs)
    logger.info(f"sentence chunking:{len(docs)} docs->{len(chunks)} chunks")
    return chunks

def chunk_documents(docs:List[Document])->List[Document]:
    """chunk documents using strategy defined in params.yaml"""
    strategy=ingestion_cfg.chunking_strategy
    logger.info(f"Chunking strategy:{strategy}")
    if strategy == "fixed":
        chunks= chunk_fixed(docs)
    elif strategy== "sentence":
        chunks=chunks_sentence(docs)
    else:
        logger.warning(f"unknown strategy '{strategy}',falling back to fixed")
        chunks=chunk_fixed(docs)
    
    #add chunk to metadata

    for i , chunk in enumerate(chunks):
        chunk.metadata["chunk_index"]=i
        chunk.metadata["total_chunks"]=len(chunks)
    
    logger.success(f"chunking complete:{len(chunks)} total chunks")
    return chunks