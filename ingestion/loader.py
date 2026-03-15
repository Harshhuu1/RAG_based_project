from pathlib import Path
from typing import List
from loguru import logger

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from config import ingestion_cfg

# Map file extensions to loader classes
Loader_MAP={
    ".pdf":PyPDFLoader,
    ".docx":Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader
}

def load_document(file_path:str | Path)-> List[Document]:
    """Load a single document and return the list of lagchain document."""
    path=Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix=path.suffix.lower()
    if suffix not in ingestion_cfg.supported_formats:
        raise ValueError(f"unsupported_format:{suffix}. Support:{ingestion_cfg.supported_formats}")
    loader_cls=Loader_MAP.get(suffix)
    loader=loader_cls(str(path))
    docs=loader.load()

    for doc in docs:
        doc.metadata.update({
            "source":path.name,
            "file_path":str(path),
            "file_type":suffix,

        })
    logger.success(f"Loaded {len(docs)} pages(s) from {path.name}")
    return docs

def load_directory(dir_path:str |Path)->List[Document]:
    """recursively load all supported documents from a directory"""
    dir_path=Path(dir_path)

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    all_docs=[]
    supported=set(ingestion_cfg.supported_formats)

    files = [f for f in dir_path.rglob("*") if f.suffix.lower() in supported]

    for file_path in files:
        try:
            docs=load_document(file_path)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"failed to load {file_path}: {e}")
    logger.success(f"Total document loaded:{len(all_docs)}")

    return all_docs