import pytest 
from pathlib import Path
from ingestion.loader import load_document, load_directory
from ingestion.chunker import chunk_documents

#_____Fixtures_____
@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary text file for testing."""
    file=tmp_path /"test.txt"
    file.write_text(
        "This is a test document."
        "It has multiple sentences."
        "We use it to test our loader and chunker."

    )
    return file 

@pytest.fixture
def sample_directory(tmp_path):
    """Create a temporary directory with multiple files."""
    file1 = tmp_path / "doc1.txt"
    file2 = tmp_path / "doc2.txt"
    file1.write_text("Document one content.")
    file2.write_text("Document two content.")
    return tmp_path

#___ Loader tests____
def test_load_document_txt(sample_txt_file):
    """Test loading a single txt file."""
    docs=load_document(sample_txt_file)
    assert len(docs)>0
    assert docs[0].page_content!=""
    assert docs[0].metadata["file_type"]==".txt"

def test_load_content_not_found():
    """Test that filenotfounderror is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_document("nonexistent_file.txt")

def test_load_document_unsupported_format(tmp_path):
    """test that valueerror is raised for unsupported formats"""
    file=tmp_path/"test.csv"
    file.write_text("col1,col2\n1,2")
    with pytest.raises(ValueError):
        load_document(file)
def test_load_directory(sample_directory):
    """Test laoding all files from a directory"""
    docs=load_directory(sample_directory)
    assert len(docs)>=2


#___chunker tests
def test_chunk_documents(sample_txt_file):
    """Test that chunker produces chunks from documents"""
    docs=load_document(sample_txt_file)
    chunks=chunk_documents(docs)
    assert "chunk_index" in chunks[0].metadata
    assert "total_chunks" in chunks[0].metadata

def test_chunk_empty_documents():
    """Test chunker handles epty documents list"""
    chunks=chunk_documents([])
    assert chunks==[]



# What this means:

# Each function starting with test_ is automatically discovered and run by pytest
# assert → checks that something is true, fails the test if not
# pytest.raises(FileNotFoundError) → verifies that the correct error is raised
# We test both happy paths (normal usage) and edge cases (missing files, wrong formats)
# These are the tests that run in your CI pipeline on every PR!