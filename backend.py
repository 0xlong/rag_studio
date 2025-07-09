import os
from typing import List, Dict, Any
import streamlit as st
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter, RecursiveJsonSplitter, HTMLHeaderTextSplitter, PythonCodeTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader
)
from langchain.prompts import PromptTemplate
from data_info import get_proposition_prompt_text
import json
from langchain_community.docstore import InMemoryDocstore  # Import for in-memory docstore for FAISS
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from datetime import datetime
from rank_bm25 import BM25Okapi
import random  # Import random for random selection

# Dictionary mapping file extensions to their corresponding loaders
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader
}

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a single document from a file path.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        List[Dict[str, Any]]: List of documents with their content and metadata
    """
    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Check if file type is supported
    if file_extension not in LOADER_MAPPING:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # Get appropriate loader
    loader_class = LOADER_MAPPING[file_extension]
    
    try:
        # Initialize loader and load document
        loader = loader_class(file_path)
        documents = loader.load()
        
        # Convert documents to dictionary format
        processed_docs = []
        for doc in documents:
            processed_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return processed_docs
    
    except Exception as e:
        raise Exception(f"Error loading document: {str(e)}")

def chunking_fixed_size(
    documents: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    splitter_type: str = "Character Splitter"
) -> List[Dict[str, Any]]:
    
    """
    Chunk a list of documents using a fixed-size strategy with either CharacterTextSplitter or TokenTextSplitter.
    
    Args:
        documents (List[Dict[str, Any]]): List of documents to chunk
        chunk_size (int): Size of each chunk in characters or tokens
        chunk_overlap (int): Number of characters or tokens to overlap between chunks
        splitter_type (str): Type of splitter to use ("Character Splitter" or "Token Splitter")
        
    Returns:
        List[Dict[str, Any]]: List of chunked documents
    """

    # Initialize text splitter based on type
    if splitter_type == "Character Splitter":
        # Character splitter splits text by character count
        text_splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    else:  # Token Splitter
        # Token splitter uses tiktoken to split by tokens (more accurate for LLMs)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",  # OpenAI's encoding
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # Process documents and return chunked results
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_docs.append({
                "content": chunk,
                "metadata": doc["metadata"]
            })
    
    # --- Save chunked_docs to JSON file for inspection or later use ---
    os.makedirs('data/chunking', exist_ok=True)  # Ensure the directory exists
    with open('data/chunking/chunking_fixed_size.json', 'w', encoding='utf-8') as f:
        # Save the chunked documents as a JSON array
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    # This saves the chunked documents for later use or debugging
    return chunked_docs

def chunking_recursive(
    documents: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    separators: List[str] = ["\n\n", ". ", " ", ""]  # Default separators
) -> List[Dict[str, Any]]:
    """
    Chunk documents using RecursiveCharacterTextSplitter with multiple separators.
    
    Args:
        documents (List[Dict[str, Any]]): List of documents to chunk
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        separators (List[str]): List of separators to use for splitting text, in order of priority
        
    Returns:
        List[Dict[str, Any]]: List of chunked documents
    """
    # Initialize text splitter with the selected separators
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,  # Use the list of separators
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Process documents and return chunked results
    chunked_docs = []
    for doc in documents:   # Process each document
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_docs.append({
                "content": chunk,
                "metadata": doc["metadata"]
            })
    
    # --- Save chunked_docs to JSON file for inspection or later use ---
    os.makedirs('data/chunking', exist_ok=True)  # Ensure the directory exists
    with open('data/chunking/chunking_recursive.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    return chunked_docs

def chunking_by_doc_type(
    documents: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int
) -> List[Dict[str, Any]]:
    """
    Chunk documents based on their type using specialized LangChain splitters.
    Each document type uses a dedicated splitter that understands its structure.
    
    Args:
        documents (List[Dict[str, Any]]): List of documents to chunk
        chunk_size (int): Base size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks
        
    Returns:
        List[Dict[str, Any]]: List of chunked documents with preserved structure
    """
    chunked_docs = []
    
    for doc in documents:
        # Extracting document type from metadata of LangChain Document object
        doc_type = doc["metadata"].get("source", "").lower()
        content = doc["content"]
        print(doc_type)

        # Determine document type and apply appropriate splitter
        if ".md" in doc_type or ".markdown" in doc_type:
            # MarkdownHeaderTextSplitter splits by headers and maintains hierarchy
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            chunks = splitter.split_text(content)
            
        elif ".json" in doc_type:
            # RecursiveJsonSplitter understands JSON structure and splits by objects
            splitter = RecursiveJsonSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(content)
            
        elif ".html" in doc_type:
            # HTMLHeaderTextSplitter splits by HTML headers and maintains structure
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
            splitter = HTMLHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            chunks = splitter.split_text(content)
            
        elif any(ext in doc_type for ext in [".py", ".js", ".java", ".cpp", ".c", ".go"]):
            # PythonCodeTextSplitter understands code structure and splits by functions/classes
            splitter = PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_text(content)
            
        else:
            # Default to RecursiveCharacterTextSplitter for unknown types
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = splitter.split_text(content)
        
        # Add chunks to result with metadata
        for chunk in chunks:
            chunk_metadata = {
                **doc["metadata"],
                "chunk_type": doc_type.split(".")[-1] if "." in doc_type else "unknown"
            }
            if hasattr(chunk, 'metadata'):
                chunk_metadata.update(chunk.metadata)
            
            chunked_docs.append({
                "content": chunk.page_content if hasattr(chunk, 'page_content') else chunk,
                "metadata": chunk_metadata
            })
    
    # --- Save chunked_docs to JSON file for inspection or later use ---
    os.makedirs('data/chunking', exist_ok=True)  # Ensure the directory exists
    with open('data/chunking/chunking_by_doc_type.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    return chunked_docs 

def chunking_sentence_window(
    documents: List[Dict[str, Any]],
    window_size: int = 2
) -> List[Dict[str, Any]]:
    """
    Chunk documents using a sentence window approach. This strategy:
    1. Splits text into sentences
    2. Creates chunks that include the target sentence plus surrounding sentences (window)
    3. Each sentence becomes a target sentence with its context window
    
    Args:
        documents (List[Dict[str, Any]]): List of documents to chunk
        window_size (int): Number of sentences to include before and after the target sentence
        
    Returns:
        List[Dict[str, Any]]: List of chunked documents with sentence context
    """
    # Import required for sentence splitting
    from nltk.tokenize import sent_tokenize
    import nltk
    
    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    chunked_docs = []
    
    for doc in documents:
        # Split text into sentences
        sentences = sent_tokenize(doc["content"])
        
        # Process each sentence with its window
        for i, target_sentence in enumerate(sentences):
            # Calculate window boundaries
            start_idx = max(0, i - window_size)
            end_idx = min(len(sentences), i + window_size + 1)
            
            # Get the window of sentences
            window_sentences = sentences[start_idx:end_idx]
            window_text = " ".join(window_sentences)
            
            # Add the chunk with metadata
            chunked_docs.append({
                "content": window_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_type": "sentence_window",
                    "window_size": window_size,
                    "target_sentence": target_sentence
                }
            })
    
    # --- Save chunked_docs to JSON file for inspection or later use ---
    os.makedirs('data/chunking', exist_ok=True)  # Ensure the directory exists
    with open('data/chunking/chunking_sentence_window.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    return chunked_docs 

def chunking_semantic(
    documents: List[Dict[str, Any]],
    embedding_model: str = "huggingface",  # Options: "huggingface", "openai", "google"
    embedding_model_name: str = None,  # Will use default model for each provider if None
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 0.95,
    buffer_size: int = 1
) -> List[Dict[str, Any]]:
    """
    Chunk documents using semantic similarity to determine natural breakpoints.
    This approach creates chunks based on the semantic meaning of the text rather than
    arbitrary character or token counts.
    
    Args:
        documents (List[Dict[str, Any]]): List of documents to chunk
        embedding_model (str): Embedding provider to use ("huggingface", "openai", "google")
        embedding_model_name (str): Specific model name for the selected provider
        breakpoint_threshold_type (str): Type of threshold to use
        breakpoint_threshold_amount (float): Threshold value
        buffer_size (int): Number of sentences to include before and after breakpoints
        
    Returns:
        List[Dict[str, Any]]: List of semantically chunked documents
    """
    from langchain_experimental.text_splitter import SemanticChunker

    # Define default models for each provider
    DEFAULT_MODELS = {
        "huggingface": {
            "default": "sentence-transformers/all-MiniLM-L6-v2",
            "options": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2"
            ]
        },
        "openai": {
            "default": "text-embedding-3-small",
            "options": [
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]
        },
        "google": {
            "default": "models/embedding-001",
            "options": [
                "models/embedding-001",
                "models/text-embedding-004"
            ]
        }
    }
    
    # Use default model if none specified
    if embedding_model_name is None:
        embedding_model_name = DEFAULT_MODELS.get(embedding_model, {}).get("default")
    
    # Get API keys from Streamlit secrets
    try:
        api_keys = st.secrets["api_keys"]
    except Exception as e:
        raise ValueError(f"Error loading API keys from secrets: {str(e)}")
    
    # Initialize the appropriate embedding model with API keys
    if embedding_model == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            huggingfacehub_api_token=api_keys.get("huggingface_api_key")
        )

    elif embedding_model == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=api_keys.get("openai_api_key")
        )

    elif embedding_model == "google":
        # Map user-friendly model names to the required Google API format
        if embedding_model_name == "embedding-001":
            embedding_model_name = "models/embedding-001"
        elif embedding_model_name == "text-embedding-004":
            embedding_model_name = "models/text-embedding-004"
        # Now instantiate the Google embedding model with the correct name
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=api_keys.get("google_api_key")
        )

    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
    
    # Create semantic chunker with specified parameters
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
        buffer_size=buffer_size
    )
    
    # Process documents and return chunked results
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            chunked_docs.append({
                "content": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_type": "semantic",
                    "embedding_model": embedding_model,
                    "embedding_model_name": embedding_model_name,
                    "breakpoint_threshold_type": breakpoint_threshold_type,
                    "breakpoint_threshold_amount": breakpoint_threshold_amount,
                    "buffer_size": buffer_size
                }
            })
    
    # --- Save chunked_docs to JSON file for inspection or later use ---
    os.makedirs('data/chunking', exist_ok=True)  # Ensure the directory exists
    with open('data/chunking/chunking_semantic.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    return chunked_docs 

def chunking_propositions(
    documents: List[Dict[str, Any]],
    llm_provider: str,  # e.g., 'openai', 'huggingface', 'google'
    llm_model_name: str,  # e.g., model name string
    prompt_text: str = None  # Optional: custom prompt text
) -> list:
    """
    Extracts atomic factoids (propositions) from a given text using an LLM and a specific prompt.

    Args:
        documents (List[Dict[str, Any]]): List of documents to chunk
        llm_provider (str): The LLM provider to use (e.g., 'openai', 'huggingface', 'google').
        llm_model_name (str): The model name for the selected provider.
        prompt_text (str, optional): Custom prompt text. If not provided, uses the default prompt file.

    Returns:
        list: A list of strings.
    """
    # 1. Get API keys from Streamlit secrets
    api_keys = st.secrets["api_keys"]

    # 2. Instantiate the correct LLM based on provider
    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=llm_model_name,
            openai_api_key=api_keys.get("openai_api_key")
        )
    elif llm_provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub
        llm = HuggingFaceHub(
            repo_id=llm_model_name,
            huggingfacehub_api_token=api_keys.get("huggingface_api_key")
        )
    elif llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            google_api_key=api_keys.get("google_api_key")
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    # 3. Set the prompt template (from user or file as default)
    if prompt_text is not None:
        template_str = prompt_text
    else:
        template_str = get_proposition_prompt_text()

    # 4. Create the prompt using LangChain's PromptTemplate
    prompt_design = PromptTemplate(
        input_variables=["input"],
        template=template_str)
    # every document is a dictionary with content key where document text is stored

    # 5. Iterate over all documents
    factoids = [] # list for storing factoids
    for doc in documents:
        print(doc['metadata'],"")

        # every prompt shall be fed with document text, document is a dictionary with content key where document text is stored
        prompt = prompt_design.format(input=doc['content'])

        # 6. Call the LLM to get the response
        response = llm.invoke(prompt)

        # 7. Parse the response into a list of factoids (one per line)
        for line in response.content.strip().split('\n'):
            line = line.replace('"', '').strip()
            factoids.append(line)

    # 7. Save the factoids to a CSV file (for later use or inspection)
    with open("data/chunking/factoids.csv", 'a', encoding='utf-8') as f:
        for fact in factoids:
            f.write(fact + "\n")

    # 8. Return the list of factoids
    return factoids

def embed_dense(
    chunked_docs: List[Dict[str, Any]],
    provider: str,  # 'openai', 'huggingface', 'google'
    model_name: str,  # model name string
    embedding_dimensions: int
) -> List[Dict[str, Any]]:
    """
    Embed chunked documents using a dense (neural) embedding model.
    Adds an 'embedding' key to each chunk dict.
    Stores all embeddings in a file in data/embeddings/embeddings_dense.json.
    Args:
        chunked_docs: List of dicts with 'content' and 'metadata'.
        provider: Which provider to use ('openai', 'huggingface', 'google').
        model_name: Model name for the provider.
        embedding_dimensions: The dimension of the embedding vectors.
    Returns:
        List of dicts with 'embedding' key added (dense vector).
    """
    # Get API keys from Streamlit secrets
    api_keys = st.secrets["api_keys"]

    # Choose and load the embedding model
    if provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        base_embedder = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_keys.get("openai_api_key"),
            dimensions=embedding_dimensions
        )
    elif provider == "Hugging Face":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        base_embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            huggingfacehub_api_token=api_keys.get("huggingface_api_key")
        )
    elif provider == "Google":
        # Map user-friendly model names to the required Google API format
        if model_name == "embedding-001":
            model_name = "models/embedding-001"
        elif model_name == "text-embedding-004":
            model_name = "models/text-embedding-004"

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        base_embedder = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_keys.get("google_api_key"),
            output_dimensionality=embedding_dimensions
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Extract all texts to embed
    texts = [chunk['content'] for chunk in chunked_docs]

    # Get dense embeddings for all texts (batch)
    # This will call the embedding model and return a list of vectors
    embeddings = base_embedder.embed_documents(texts)

    # Add embedding to each chunk dict
    for chunk, emb in zip(chunked_docs, embeddings):
        chunk['embedding'] = emb  # emb is a list/array of floats
        # Now each chunk has its embedding

    # --- Save all chunked_docs with embeddings to a JSON file ---
    os.makedirs('data/embeddings/dense', exist_ok=True)  # Ensure the directory exists
    with open(f'data/embeddings/dense/embeddings_dense_{provider}_{model_name.replace("/", "_")}.json', 'w', encoding='utf-8') as f:
        # Save the list of dicts as a JSON array
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)

    return chunked_docs

def embed_sparse(
    chunked_docs: List[Dict[str, Any]],
    method: str = 'tfidf',  # 'tfidf' or 'bm25'
    k1: float = 1.5,        # BM25 parameter: term frequency scaling
    b: float = 0.75         # BM25 parameter: length normalization
) -> List[Dict[str, Any]]:
    """
    Embed chunked documents using a sparse (classic IR) embedding model.
    Adds an 'embedding' key to each chunk dict.
    Args:
        chunked_docs: List of dicts with 'content' and 'metadata'.
        method: 'tfidf' (default) or 'bm25'.
        k1: BM25 parameter controlling term frequency scaling (higher = more boost for frequent terms).
        b: BM25 parameter controlling length normalization (0 = no normalization, 1 = full normalization).
    Returns:
        List of dicts with 'embedding_sparse' key added (sparse vector, usually numpy array or list).
    """
    # Import scikit-learn for TF-IDF, rank_bm25 for BM25
    if method == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Extract all texts
        texts = [chunk['content'] for chunk in chunked_docs]
        # Fit TF-IDF on all texts
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)  # X is a sparse matrix
        # Add embedding to each chunk (convert to dense for simplicity)
        for i, chunk in enumerate(chunked_docs):
            chunk['embedding_sparse'] = X[i].toarray()[0].tolist()  # Convert to list of floats
    elif method == 'bm25':
        # BM25 needs tokenized input
        from rank_bm25 import BM25Okapi
        # Tokenize each chunk (simple whitespace split)
        tokenized = [chunk['content'].split() for chunk in chunked_docs]
        # BM25Okapi takes k1 and b as parameters, which control scoring behavior
        bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        # For each chunk, get its BM25 vector (score vs all docs)
        for i, chunk in enumerate(chunked_docs):
            # Score this chunk against all others (self-similarity is highest)
            scores = bm25.get_scores(tokenized[i])
            chunk['embedding_sparse'] = scores.tolist()  # List of floats
    else:
        raise ValueError(f"Unsupported sparse embedding method: {method}")

    # --- Save all chunked_docs with embeddings to a JSON file ---
    os.makedirs('data/embeddings/sparse', exist_ok=True)  # Ensure the directory exists
    with open(f'data/embeddings/sparse/embeddings_sparse_{method}.json', 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)
    # This file can be loaded later for search or analysis

    return chunked_docs

def add_to_vector_store(
    docs: List[Dict[str, Any]],
    vector_store_type: str,  # e.g., 'faiss', 'chroma', 'pinecone', 'weaviate', 'milvus'
    vector_store_config: dict = None,  # Optional config for API keys, index names, etc.
    embedding_provider: str = None,    # provider name (e.g., 'OpenAI', 'Hugging Face', 'Google')
    embedding_model_name: str = None,   # model name string
):
    """
    Adds embeddings from chunked_docs to the selected vector store.
    ---
    Deep Explanation:
    This function takes a list of document chunks (with their embeddings) and stores them in a vector database (vector store).
    Vector stores allow you to search for similar documents using vector math (like cosine similarity) instead of just keywords.
    
    - FAISS: Fast, local, and open-source. Good for prototyping and small/medium datasets. No cloud needed.
    - Chroma: Local, persistent, and easy to use. Good for local apps and demos.
    - (Other types like Pinecone, Weaviate, Milvus are not implemented here, but are cloud or scalable options.)
    
    The function also supports different index types for FAISS:   
      - 'flat': Brute-force, exact search (slow for large data, but always correct)
      - 'ivf': Inverted File, clusters vectors for faster search (approximate, needs training)
      - 'hnsw': Graph-based, very fast and scalable (approximate)
    
    Storage types:
      - 'in_memory': Fast, but data is lost when the app stops
      - 'local_disk': Data is saved to disk and persists between runs
    
    The function also manages a docstore (stores the actual text and metadata) and a mapping from vector index to docstore id.
    """
    # Always get API keys from Streamlit secrets for security and consistency
    api_keys = st.secrets["api_keys"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # embedding dimension from embedding section of first document from docs list (embedded_docs variable in app.py)
    dim = len(docs[0]['embedding'])

    # 1. Extract texts, embeddings, and metadatas from chunked_docs
    docs_text = [doc['content'] for doc in docs]  # The original docs text
    embeddings = [doc['embedding'] for doc in docs]  # The dense/sparse vectors
    metadatas = [doc['metadata'] for doc in docs]  # Any metadata (source, chunk type, etc.)

    # 2. Choose the vector store based on user input
    vector_store_type = vector_store_type.lower()
    vector_store_config = vector_store_config or {}

    # 3. Create embedding_function for both FAISS and Chroma ---
    if embedding_provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        embedding_function = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=api_keys.get("openai_api_key"),
            dimensions=dim
        )
    elif embedding_provider == "Hugging Face":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            huggingfacehub_api_token=api_keys.get("huggingface_api_key")
        )
    elif embedding_provider == "Google":
        if embedding_model_name == "embedding-001":
            embedding_model_name = "models/embedding-001"
        elif embedding_model_name == "text-embedding-004":
            embedding_model_name = "models/text-embedding-004"
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=api_keys.get("google_api_key"),
            dimensions=dim
        )
    else:
        raise ValueError(f"Unsupported provider: {embedding_provider}. Choose from 'OpenAI', 'Hugging Face', or 'Google'.")

    # distance metrics for building a vector store - cosine, inner product (ip) or l2 (euclidean)
    distance_metric = vector_store_config.get('distance_metric', 'l2')
    embeddings_np = np.array(embeddings).astype('float32')
    # normalization of vectors for cosine distance cause distance_metric ip == cosine but with normalized vectors
    if distance_metric == 'cosine':
        # Normalize vectors for cosine similarity (required for METRIC_INNER_PRODUCT)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / np.maximum(norms, 1e-10)

    # chroma vector store setup
    if vector_store_type == 'chroma':
        from langchain_community.vectorstores import Chroma

        # Add timestamp to chroma_dir
        # every vector store creation we create new chroma db instance and save it on disk, cause if we reuse the setting with embedding dimension will be taken
        chroma_dir = f'data/vector_stores/chroma/chroma_{embedding_provider}_{embedding_model_name}_{dim}_{timestamp}'
        
        # distance metric for building a vector store - cosine, ip (inner product) or l2 (euclidean), by default l2


        # Chroma db initialization
        vector_store = Chroma(
            embedding_function=embedding_function,
            persist_directory=chroma_dir,
            collection_metadata={"hnsw:space": distance_metric}
        )

        # Add doc content and metadata to empty Chroma vector store
        vector_store.add_texts(
            texts=docs_text,
            metadatas=metadatas
        )
        return vector_store

    # FAISS vector store setup
    elif vector_store_type == 'faiss':

        from langchain_community.vectorstores import FAISS
        import faiss

        # Extract index type from config, default to 'flat'
        index_type = vector_store_config.get('index_type', 'flat').lower()

        # --- Index type selection ---
        if index_type == 'flat':
            # Flat index: brute-force, exact search
            if distance_metric == 'l2':
                faiss_index = faiss.IndexFlatL2(dim)
            elif distance_metric == 'ip':
                faiss_index = faiss.IndexFlatIP(dim)  # Inner product (dot product)
            elif distance_metric == 'cosine':
                faiss_index = faiss.IndexFlatIP(dim)  # Use inner product for normalized vectors
            else:
                raise ValueError(f"Unsupported distance_metric: {distance_metric}. Choose 'l2', 'ip', or 'cosine'.")
            
        elif index_type == 'ivf':
            nlist = vector_store_config.get('nlist', 100)  # Number of clusters
            if distance_metric == 'l2':
                quantizer = faiss.IndexFlatL2(dim)
                faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            elif distance_metric == 'ip' or distance_metric == 'cosine':
                quantizer = faiss.IndexFlatIP(dim)
                faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                raise ValueError(f"Unsupported distance_metric: {distance_metric}. Choose 'l2', 'ip', or 'cosine'.")
            # Train the IVF index
            if not faiss_index.is_trained:
                faiss_index.train(embeddings_np)

        elif index_type == 'hnsw':
            # HNSW index: fast, graph-based, scalable (approximate)
            m = vector_store_config.get('hnsw_m', 32)  # Number of neighbors in the graph
            faiss_index = faiss.IndexHNSWFlat(dim, m)
        else:
            # If the user provides an unsupported index type, raise an error with explanation
            raise ValueError(f"Unsupported FAISS index_type: {index_type}. Choose from 'flat', 'ivf', or 'hnsw'.")

        # --- Pass embedding_function to FAISS ---
        vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(docs_text, embeddings)),
            embedding=embedding_function,
            metadatas=metadatas
        )

        vector_store.index = faiss_index

        # Add the vectors to the FAISS index (this is the actual storage step)
        embeddings_faiss = np.array([doc['embedding'] for doc in docs]).astype('float32')
        faiss_index.add(embeddings_faiss)

        # if user choose docstore local disk, faiss is saved to locally
        if vector_store_config.get('docstore_type') == 'local_disk':
            vector_store.save_local(f'data/vector_stores/faiss/{timestamp}/faiss_index')

        # Return the vector store object (handle)
        return vector_store

    else:
        # Explain why this error is raised: only supported vector store types are allowed
        raise ValueError(f"Unsupported vector store type: {vector_store_type}. Only 'faiss' and 'chroma' are implemented.")
    
def query_transform(query: str, mode: str) -> list:
    """
    Transforms the user query according to the selected scheme using an LLM.

    This function takes a user's query and a transformation mode, then uses an OpenAI model
    to generate different versions of the query. This helps improve the retrieval accuracy
    of vector databases by providing more diverse and contextually relevant search terms.

    Args:
        query (str): The original user query.
        mode (str): The transformation mode. It can be one of the following:
                    - 'no_transformation': Returns the original query.
                    - 'multi_query': Generates multiple alternative queries.
                    - 'hyde': Creates a hypothetical document that answers the query.
                    - 'step_back_prompting': Generates a more general, higher-level question.
    
    Returns:
        list: A list of transformed queries.
    """

    # We need the OpenAI API key to make requests. It is securely loaded from Streamlit's secrets manager.
    try:
        google_api_key = st.secrets["api_keys"].get("google_api_key")
        if not google_api_key:
            raise ValueError("Google API key not found in secrets.")
    except Exception as e:
        raise ValueError(f"Error loading Google API keys from secrets: {str(e)}")

    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key,
        temperature=0.7
    )

    # If no transformation is needed, we just return the original query in a list.
    if mode == "no_transformation":
        return [query]
    # For multi-query, we ask the LLM to generate several different versions of the original query.
    # This helps find documents that might use different phrasing but have the same meaning.
    if mode == "multi_query":
        # The prompt guides the LLM to generate 3 alternative queries.

        template = (
            "You are an AI assistant. Your task is to generate five different versions of the given user query "
            "to retrieve relevant documents from a vector database. By generating multiple perspectives on the user "
            "query, we can overcome some of the limitations of distance-based similarity search. "
            "Provide these alternative queries separated by newlines.\n"
            "Original query: {query}"
        )
        prompt = PromptTemplate(input_variables=["query"], template=template)


    # For HyDE (Hypothetical Document Embeddings), we generate a fake document that could answer the user's query.
    # This hypothetical document is often more effective for similarity search than the short query itself.
    elif mode == "hyde":
        template = (
            "You are an AI assistant. Please write a hypothetical passage that could answer the user's question. "
            "The passage should be plausible and on-topic, but it does not need to be factually correct. "
            "This passage will be used to find similar documents in a vector database.\n"
            "Question: {query}"
        )
        prompt = PromptTemplate(input_variables=["query"], template=template)

    # For step-back prompting, we create a more general, higher-level question.
    # This helps retrieve documents that provide broader context, which can be useful for answering the original specific query.
    elif mode == "step_back_prompting":
        template = (
            "You are an AI assistant. Your task is to perform 'step-back' prompting. "
            "Given the user's question, generate a more general, higher-level question that provides "
            "broader context. This step-back question will be used to retrieve general context documents, "
            "which will then be used to answer the original, more specific question.\n"
            "Original question: {query}\n"
            "Step-back question:"
        )
        prompt = PromptTemplate(input_variables=["query"], template=template)
    
    # If the mode is unknown, we just return the original query without any changes.
    else:    
        return query
    
    # execute LLM chain and return response 
    chain = prompt | llm
    response = chain.invoke({"query": query})
    return [response.content]

def retrieve_dense(
    query: str,
    vector_store,
    top_k: int = 3,
    score_threshold: float = 0.0,
    keywords: list = None,  # New argument for keyword filtering
    search_type: str = "similarity_score_threshold",  # New: search type, default as before
    fetch_k: int = 20,      # New: number of candidates to fetch for MMR
    lambda_mult: float = 0.5  # New: diversity vs relevance for MMR
) -> list:
    """
    Retrieve the most relevant documents from the vector store for a given query.

    Args:
        query (str): The user's search query.
        vector_store: The vector store object (e.g., FAISS, Chroma).
        top_k (int): Number of top documents to retrieve.
        score_threshold (float): Minimum similarity score to return a result.
        keywords (list): List of keywords to filter documents (only for Chroma).
        search_type (str): Type of search ('similarity_score_threshold', 'mmr', etc.).
        fetch_k (int): For MMR, number of candidates to fetch before reranking.
        lambda_mult (float): For MMR, tradeoff between relevance and diversity (0-1).

    Returns:
        list: List of retrieved documents (LangChain Document objects with page_content and metadata).
    """
    # --- Explanation for junior devs ---
    # This function now supports both standard similarity search and MMR (Maximal Marginal Relevance).
    # MMR helps to get results that are both relevant and diverse, not just similar to the query.
    # You can control the search type and its parameters using the new arguments.
    
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.vectorstores import Chroma, FAISS

    if vector_store is not None:
        if isinstance(vector_store, Chroma):

            # Prepare search_kwargs (parameters) for retriever
            chroma_params = {}

            # If using MMR, set up MMR-specific parameters
            if search_type == "mmr":
                # MMR needs k (final results), fetch_k (candidates), lambda_mult (diversity)
                chroma_params = {
                    "k": top_k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
                # If keywords are provided, add filter
                if keywords and isinstance(keywords, list) and len(keywords) > 0:
                    chroma_params["where_document"] = {
                        "$or": [{"$contains": kw} for kw in keywords]
                    }
            else:
                # Default: similarity search with optional score threshold and keyword filter
                chroma_params = {
                    "k": top_k,
                    "score_threshold": score_threshold
                }
                if keywords and isinstance(keywords, list) and len(keywords) > 0:
                    chroma_params["where_document"] = {
                        "$or": [{"$contains": kw} for kw in keywords]
                    }
            try:
                # Create a retriever from the vector store with the desired search parameters
                retriever = vector_store.as_retriever(
                    search_type=search_type,  # Use the search type specified by the user
                    search_kwargs=chroma_params)
                results = retriever.invoke(query)
                return results
            except TypeError:
                print(f"Error whil retrieveing docs with Chroma: {TypeError}")

        elif isinstance(vector_store, FAISS):
            try:
                # For FAISS, support both similarity and MMR search types
                faiss_params = {}
                if search_type == "mmr":
                    faiss_params = {
                        "k": top_k,
                        "fetch_k": fetch_k,
                        "lambda_mult": lambda_mult
                    }
                else:
                    faiss_params = {
                        "k": top_k,
                        "score_threshold": score_threshold
                    }
                retriever = vector_store.as_retriever(
                    search_type=search_type,
                    search_kwargs=faiss_params
                )
                results = retriever.invoke(query)
                return results
            except TypeError:
                print(f"Error whil retrieveing docs with FAISS: {TypeError}")

        else:
            print(f"Current vector store: {type(vector_store).__name__} (unknown type)")
    else:
        print("No vector store loaded in session state.")

def retrieve_hybrid(
    query: str,
    embedding_provider, embedding_model_name, # params for creating dense model
    embedding_sparse_method, embedding_sparse_kparam, embedding_sparse_bparam,
    dense_chunks: list,
    sparse_chunks: list,
    alpha: float = 0.5,  # Weight for dense vs sparse
    top_k: int = 3
) -> list:
    """
    Hybrid retrieval: combine dense and sparse scores for each chunk.
    """
    # Input validation with detailed error messages
    if not dense_chunks:
        raise ValueError("Dense chunks are empty. Please create dense embeddings first.")
    if not sparse_chunks:
        raise ValueError("Sparse chunks are empty. Please create sparse embeddings first.")
    if len(dense_chunks) != len(sparse_chunks):
        raise ValueError(f"Mismatch: {len(dense_chunks)} dense chunks vs {len(sparse_chunks)} sparse chunks. Please use the same chunking file for both embeddings.")
    
    # Validate that chunks have the required embeddings
    if 'embedding' not in dense_chunks[0]:
        raise ValueError("Dense chunks do not contain 'embedding' field. Please create dense embeddings first.")
    if 'embedding_sparse' not in sparse_chunks[0]:
        raise ValueError("Sparse chunks do not contain 'embedding_sparse' field. Please create sparse embeddings first.")

    dim = len(dense_chunks[0]['embedding'])

    # creating dense model
    if embedding_provider == "OpenAI":
        from langchain_openai import OpenAIEmbeddings
        dense_model = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=st.secrets["api_keys"].get("openai_api_key"),
            dimensions=dim
        )
    elif embedding_provider == "Hugging Face":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        dense_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            huggingfacehub_api_token=st.secrets["api_keys"].get("huggingface_api_key")
        )
    elif embedding_provider == "Google":
        if embedding_model_name == "embedding-001":
            embedding_model_name = "models/embedding-001"
        elif embedding_model_name == "text-embedding-004":
            embedding_model_name = "models/text-embedding-004"
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        dense_model = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=st.secrets["api_keys"].get("google_api_key"),
            dimensions=dim
        )
    else:
        raise ValueError(f"Unsupported provider: {embedding_provider}. Choose from 'OpenAI', 'Hugging Face', or 'Google'.")

    # 1. Compute dense embedding for query
    dense_query_emb = dense_model.embed_query(query)

    # 2. Compute sparse scores using the same approach as retrieve_sparse
    sparse_scores = []
    if embedding_sparse_method == 'tfidf':
        # Use the same TF-IDF approach as retrieve_sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [chunk['content'] for chunk in sparse_chunks]
        # Fit a new TF-IDF vectorizer on the document texts
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        # Transform the query into the same vector space
        query_vec = vectorizer.transform([query]).toarray()[0]
        # Compute dot product between query_vec and each doc's embedding
        sparse_scores = [np.dot(query_vec, chunk['embedding_sparse']) for chunk in sparse_chunks]
        
    elif embedding_sparse_method == 'bm25':
        # Use the same BM25 approach as retrieve_sparse
        from rank_bm25 import BM25Okapi
        # Tokenize each document (simple whitespace split)
        tokenized_docs = [chunk['content'].split() for chunk in sparse_chunks]
        # Initialize BM25 with user-specified k1 and b parameters
        bm25 = BM25Okapi(tokenized_docs, k1=embedding_sparse_kparam, b=embedding_sparse_bparam)
        # Tokenize the query
        tokenized_query = query.split()
        # Compute BM25 scores for the query against all documents
        sparse_scores = bm25.get_scores(tokenized_query)
    else:
        raise ValueError(f"Unsupported sparse embedding method: {embedding_sparse_method}")

    # 3. Compute scores for each chunk
    results = []
    for i, chunk in enumerate(dense_chunks):
        # Dense similarity (cosine)
        dense_emb = np.array(chunk['embedding'])
        dense_score = np.dot(dense_query_emb, dense_emb) / (np.linalg.norm(dense_query_emb) * np.linalg.norm(dense_emb) + 1e-10)
        
        # Get sparse score for this chunk
        sparse_score = sparse_scores[i]
        
        # Normalize sparse score to 0-1 range for fair combination with dense score
        # This is important because sparse and dense scores can have very different scales
        if len(sparse_scores) > 1:
            sparse_score_normalized = (sparse_score - min(sparse_scores)) / (max(sparse_scores) - min(sparse_scores) + 1e-10)
        else:
            sparse_score_normalized = sparse_score
        
        # Combine dense and sparse scores using alpha
        final_score = alpha * dense_score + (1 - alpha) * sparse_score_normalized
        results.append((final_score, chunk))

    # 4. Sort and return top-k
    results.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in results[:top_k]]

def retrieve_sparse(
        query: str, 
        sparse_chunks: list, 
        top_k: int = 3, 
        embedding_sparse_method: str = 'tfidf', 
        bm25_k1: float = 1.5, 
        bm25_b: float = 0.75) -> list:
    """
    Retrieve the most relevant documents using sparse embeddings (BM25 or TF-IDF).
    Args:
        query (str): The user's search query (keywords).
        sparse_chunks (list): List of dicts with 'content' and 'embedding_sparse'.
        top_k (int): Number of top documents to return.
        embedding_sparse_method (str): Which sparse method to use ('tfidf' or 'bm25').
        bm25_k1 (float): BM25 parameter k1 (term frequency scaling).
        bm25_b (float): BM25 parameter b (length normalization).
    Returns:
        list: Top-k most relevant chunks (dicts).
    """
    import numpy as np
    # If there are no chunks or no sparse embeddings, return empty list
    if not sparse_chunks or 'embedding_sparse' not in sparse_chunks[0]:
        print('No docs found.')
        return []

    # Explicitly use the method provided by the user (from Streamlit app)
    if embedding_sparse_method.lower() == 'tfidf':
        # --- TF-IDF retrieval logic ---
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [chunk['content'] for chunk in sparse_chunks]
        # Fit a new TF-IDF vectorizer on the document texts
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        # Transform the query into the same vector space
        query_vec = vectorizer.transform([query]).toarray()[0]
        # Compute dot product between query_vec and each doc's embedding
        scores = [np.dot(query_vec, chunk['embedding_sparse']) for chunk in sparse_chunks]

    elif embedding_sparse_method.lower() == 'bm25':
        # --- BM25 retrieval logic ---
        from rank_bm25 import BM25Okapi
        # Tokenize each document (simple whitespace split)
        tokenized_docs = [chunk['content'].split() for chunk in sparse_chunks]
        # Initialize BM25 with user-specified k1 and b parameters
        bm25 = BM25Okapi(tokenized_docs, k1=bm25_k1, b=bm25_b)
        # Tokenize the query
        tokenized_query = query.split()
        # Compute BM25 scores for the query against all documents
        scores = bm25.get_scores(tokenized_query)

    else:
        # If an unsupported method is provided, raise an error
        raise ValueError(f"Unsupported sparse embedding method: {embedding_sparse_method}. Use 'tfidf' or 'bm25'.")

    # Get indices of top-k scores (sorted descending)
    top_indices = np.argsort(scores)[::-1][:top_k]
    # Return the top-k most relevant chunks
    results = [sparse_chunks[i] for i in top_indices]
    return results

def rerank_cross_encoder(
    query: str,
    documents: list,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5
) -> list:
    """
    Rerank documents using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders because they jointly encode
    the query and document together, allowing for more nuanced understanding of
    the relationship between them.
    
    Args:
        model_name (str): Name of the cross-encoder model to use
        query (str): The search query
        documents (list): List of documents to rerank (from session state)
        top_k (int, optional): Number of top documents to return. If None, returns all reranked docs
        
    Returns:
        list: Reranked documents sorted by relevance score
    """
    from sentence_transformers import CrossEncoder
    
    # Load the cross-encoder model
    # Cross-encoders are more computationally expensive but more accurate than bi-encoders
    model = CrossEncoder(model_name)
    
    # Extract document content from the documents list
    # Documents from session state are typically Document objects with page_content attribute
    passages = []
    for doc in documents:
        if hasattr(doc, 'page_content'):
            passages.append(doc.page_content)
        elif isinstance(doc, dict) and 'content' in doc:
            passages.append(doc['content'])
        else:
            # Fallback: try to convert to string
            passages.append(str(doc))
    
    # Rerank passages using the cross-encoder
    # The rank method returns a list of dictionaries with 'corpus_id' and 'score'
    ranks = model.rank(query, passages)
    
    # Create reranked documents list
    reranked_docs = []
    for rank in ranks:
        # Get the original document using the corpus_id
        original_doc = documents[rank['corpus_id']]
        
        # Create a new document with the reranking score added to metadata
        if hasattr(original_doc, 'metadata'):
            # If it's a Document object, update its metadata
            original_doc.metadata['rerank_score'] = rank['score']
            reranked_docs.append(original_doc)
        elif isinstance(original_doc, dict):
            # If it's a dictionary, add the score to metadata
            if 'metadata' not in original_doc:
                original_doc['metadata'] = {}
            original_doc['metadata']['rerank_score'] = rank['score']
            reranked_docs.append(original_doc)
        else:
            # Fallback: create a simple dict with content and score
            reranked_docs.append({
                'content': passages[rank['corpus_id']],
                'metadata': {'rerank_score': rank['score']}
            })
    
    # Return top_k documents if specified, otherwise return all
    if top_k is not None:
        return reranked_docs[:top_k]
    
    return reranked_docs

def rerank_google_vertexai(
    documents: list,
    ) -> list:

    """
    Rerank documents using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders because they jointly encode
    the query and document together, allowing for more nuanced understanding of
    the relationship between them.
    
    Args:
        model_name (str): Name of the cross-encoder model to use
        query (str): The search query
        documents (list): List of documents to rerank (from session state)
        top_k (int, optional): Number of top documents to return. If None, returns all reranked docs
        
    Returns:
        list: Reranked documents sorted by relevance score
    """

    # Instantiate the VertexAIReranker with the SDK manager
    reranker = VertexAIRank(
        project_id=PROJECT_ID,
        location_id=RANKING_LOCATION_ID,
        ranking_config="default_ranking_config",
        title_field="source",
        top_n=5,
    )

    basic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # fetch top 5 documents

    # Create the ContextualCompressionRetriever with the VertexAIRanker as a Reranker
    retriever_with_reranker = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=documents
)

def rerank_llm_judge(
    query: str,
    documents: list,
    llm_provider: str = "google",  # 'openai', 'huggingface', 'google'
    llm_model_name: str = "gemini-2.0-flash",  # model name for the provider
    top_k: int = 5,
    scoring_prompt: str = None
) -> list:
    """
    Rerank documents using an LLM as a judge to evaluate relevance.
    
    This approach uses an LLM to directly evaluate how well each document answers
    the user's query, providing more nuanced scoring than similarity-based methods.
    The LLM acts as a human judge would, considering relevance, completeness, and accuracy.
    
    Args:
        query (str): The user's search query
        documents (list): List of documents to rerank (from session state)
        llm_provider (str): LLM provider ('openai', 'huggingface', 'google')
        llm_model_name (str): Specific model name for the provider
        top_k (int): Number of top documents to return
        scoring_prompt (str, optional): Custom prompt for scoring. If None, uses default
        
    Returns:
        list: Reranked documents sorted by LLM relevance score
    """
    # Get API keys from Streamlit secrets
    api_keys = st.secrets["api_keys"]
    
    # Initialize the appropriate LLM based on provider
    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=llm_model_name,
            openai_api_key=api_keys.get("openai_api_key"),
            temperature=0.1  # Low temperature for consistent scoring
        )
    elif llm_provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub
        llm = HuggingFaceHub(
            repo_id=llm_model_name,
            huggingfacehub_api_token=api_keys.get("huggingface_api_key"),
            model_kwargs={"temperature": 0.1}
        )
    elif llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            google_api_key=api_keys.get("google_api_key"),
            temperature=0.1
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    # Default scoring prompt if none provided
    if scoring_prompt is None:
        scoring_prompt = """
You are an expert judge evaluating how well a document answers a user's question.

User Question: {query}

Document Content:
{document_content}

Please evaluate this document on a scale of 1-10, where:
- 10: Perfect answer - directly addresses the question with complete, accurate information
- 8-9: Excellent answer - very relevant and helpful, minor gaps
- 6-7: Good answer - relevant but may have some gaps or be somewhat general
- 4-5: Fair answer - somewhat relevant but missing key information or unclear
- 2-3: Poor answer - barely relevant or contains mostly irrelevant information
- 1: Irrelevant - does not address the question at all

Consider:
1. Relevance: How directly does this document address the question?
2. Completeness: Does it provide a full answer or just partial information?
3. Accuracy: Is the information likely to be correct and up-to-date?
4. Clarity: Is the information presented clearly and understandably?

Respond with ONLY a number between 1-10, followed by a brief explanation (max 2 sentences).

Score: """
    
    # Create the scoring prompt template
    prompt_template = PromptTemplate(
        input_variables=["query", "document_content"],
        template=scoring_prompt
    )
    
    # Process each document and get LLM scores
    scored_documents = []
    
    for i, doc in enumerate(documents):
        # Extract document content
        if hasattr(doc, 'page_content'):
            doc_content = doc.page_content
        elif isinstance(doc, dict) and 'content' in doc:
            doc_content = doc['content']
        else:
            doc_content = str(doc)
        
        # Truncate very long documents to avoid token limits
        # Most LLMs have context limits, so we need to be careful
        max_chars = 4000  # Conservative limit for most models
        if len(doc_content) > max_chars:
            doc_content = doc_content[:max_chars] + "... [truncated]"
        
        # Create the prompt for this document
        prompt = prompt_template.format(
            query=query,
            document_content=doc_content
        )
        
        try:
            # Get LLM response
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            
            # Parse the score from the response
            # Look for a number at the beginning of the response
            import re
            score_match = re.search(r'^(\d+(?:\.\d+)?)', response_text)
            
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is within valid range
                score = max(1.0, min(10.0, score))
            else:
                # Fallback: try to extract any number from the response
                numbers = re.findall(r'\d+(?:\.\d+)?', response_text)
                if numbers:
                    score = float(numbers[0])
                    score = max(1.0, min(10.0, score))
                else:
                    # Default score if parsing fails
                    score = 5.0
            
            # Create scored document
            scored_doc = {
                'document': doc,
                'score': score,
                'explanation': response_text,
                'original_index': i
            }
            
            scored_documents.append(scored_doc)
            
        except Exception as e:
            # If LLM call fails, assign a default score
            print(f"Error scoring document {i}: {str(e)}")
            scored_doc = {
                'document': doc,
                'score': 5.0,  # Neutral score
                'explanation': f"Scoring failed: {str(e)}",
                'original_index': i
            }
            scored_documents.append(scored_doc)
    
    # Sort documents by score (highest first)
    scored_documents.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top_k documents with their scores
    top_documents = []
    for scored_doc in scored_documents[:top_k]:
        # Add score to document metadata
        if hasattr(scored_doc['document'], 'metadata'):
            scored_doc['document'].metadata['llm_judge_score'] = scored_doc['score']
            scored_doc['document'].metadata['llm_judge_explanation'] = scored_doc['explanation']
        elif isinstance(scored_doc['document'], dict):
            if 'metadata' not in scored_doc['document']:
                scored_doc['document']['metadata'] = {}
            scored_doc['document']['metadata']['llm_judge_score'] = scored_doc['score']
            scored_doc['document']['metadata']['llm_judge_explanation'] = scored_doc['explanation']
        
        top_documents.append(scored_doc['document'])
    
    return top_documents

def generate_rag_response(
    query: str,
    context: str,
    llm_provider: str = "google",  # 'openai', 'huggingface', 'google'
    llm_model_name: str = "gemini-2.0-flash",  # model name for the provider
    prompt_template: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 0.9
) -> str:
    """
    Generate a RAG response using the selected LLM provider and model.
    
    This function takes a user query, context from retrieved documents, and generates
    a comprehensive answer using the specified LLM. It supports multiple providers
    and allows customization of generation parameters.
    
    Args:
        query (str): The user's question
        context (str): Retrieved document context to use for answering
        llm_provider (str): LLM provider ('openai', 'huggingface', 'google')
        llm_model_name (str): Specific model name for the provider
        prompt_template (str): Custom prompt template with {context} and {question} placeholders
        temperature (float): Controls randomness in generation (0.0-1.0)
        max_tokens (int): Maximum number of tokens to generate
        top_p (float): Controls diversity via nucleus sampling (0.0-1.0)
        
    Returns:
        str: Generated RAG response
    """
    # Get API keys from Streamlit secrets
    api_keys = st.secrets["api_keys"]
    
    # Default prompt template if none provided
    if prompt_template is None:
        prompt_template = """Context: {context}

                            Question: {question}

                            Please provide a comprehensive answer based on the context provided above. If the context doesn't contain enough information to answer the question, please say so.

                            Answer:"""
    
    # Create the prompt using the template
    prompt = prompt_template.format(context=context, question=query)
    
    
    # Initialize the appropriate LLM based on provider
    llm_provider = llm_provider.lower()

    if llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=llm_model_name,
            openai_api_key=api_keys.get("openai_api_key"),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
    elif llm_provider == "huggingface":
        from langchain_community.llms import HuggingFaceHub
        llm = HuggingFaceHub(
            repo_id=llm_model_name,
            huggingfacehub_api_token=api_keys.get("huggingface_api_key"),
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p
            }
        )
    elif llm_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=llm_model_name,
            google_api_key=api_keys.get("google_api_key"),
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    try:
        # Generate the response
        response = llm.invoke(prompt)
        
        # Extract the response content
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
            
    except Exception as e:
        raise Exception(f"Error generating RAG response: {str(e)}")

def make_ragas_evaluationset(
    documents: List[Dict],
    questions_number: int,
) -> str:
    """
    Create evaluation set for RAGAS by first extracting context and ground truth n times per document, then generating a question and ground truth for each pair.
    This function is designed to help you understand how to build a robust evaluation set for RAG systems.
    
    Args:
        documents: List of document dictionaries with 'content' and optionally 'title' keys
        num_samples: Number of context/ground truth pairs to extract per document
    Returns:
        List of evaluation items (dicts with question, answer, contexts, ground_truths)
    """
    from langchain.schema import HumanMessage

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=st.secrets["api_keys"].get('google_api_key'),
        temperature=0.3
    )

    # Prompt to extract a relevant context and ground truth from the document
    eval_template = PromptTemplate(
        input_variables=["context","num_samples"],
        template="""From the following document, extract self-contained passage (context) and its ground truth (the key fact or answer it contains). Then generate question relevant for the context and ground truth and relevant answer.\n\n
                    Document:\n
                    {context}\n
                    Return in this format, no additional text:\n
                    Context: <the context based on which we create groiund truth, question and answer>\n
                    Ground truth: <the key fact or answer>\n
                    Question: <question created based on context>\n
                    Answer: <answer created based on question and context and ground truth>\n"""
    )

    # evaulation dataset
    evaluation_set = []

    # unique questions list to avoid duplicated questions
    unique_questions = set()

    # count how many questions already generated
    question_counter = 0

    # check if docs provided
    num_docs = len(documents)
    if num_docs == 0:
        print("No documents provided.")
        return []

    # We keep trying until we have enough unique questions
    while question_counter < questions_number:

        # Randomly select a document from the list
        doc_idx = random.randint(0, num_docs - 1)
        doc = documents[doc_idx]
        title = doc.get('title', f'Document {doc_idx + 1}')
        content = doc['content']
        print(f"Randomly selected document {doc_idx + 1}/{num_docs}: {title}")

        try:
            # 1. Extract context and ground truth from the document
            eval_prompt = eval_template.format(context=content)
            response = llm.invoke([HumanMessage(content=eval_prompt)])
            response_text = response.content.strip()


            # Parse context and ground truth from the response
            # Expecting lines: Context: ...\nGround truth: ...
            context = ""
            ground_truth = ""
            question = ""
            answer = ""
            for line in response_text.split('\n'):

                if line.lower().startswith("context:"):
                    context = line[len("Context:"):].strip()
                    #print("CONTEXT:", context,"\n\n")
                elif line.lower().startswith("ground truth:"):
                    ground_truth =   line[len("Ground truth:"):].strip()
                    #print("TRUTH:",ground_truth,"\n\n")
                elif line.lower().startswith("question:"):
                    question = line[len("question:"):].strip()
                    #print("QUESTION:",question,"\n\n")
                elif line.lower().startswith("answer:"):
                    answer = line[len("Answer:"):].strip()
                    #print("ANSWER:",answer,"\n\n")

            if not context:
                print(f"Failed to parse context for document {title}")     
            elif not ground_truth:
                print(f"Failed to parse ground truth for document {title}")      
            elif not question: 
                print(f"Failed to parse question for document {title}")
            elif not answer:
                print(f"Failed to parse answer for document {title}")
            
            # Check for empty question
            if not question:
                print("Empty question found, retrying...")
                continue

            # Check for duplicated question
            if question in unique_questions:
                print("Duplicate question found, retrying...")
                continue

            # 4. Add to evaluation set
            eval_item = {
                "question": question,
                "answer": answer,
                "contexts": [context],
                "ground_truth": ground_truth
            }

            evaluation_set.append(eval_item)
            unique_questions.add(question)
            question_counter += 1
            print(f"Generated sample {question_counter} (from document: {title})")

        except Exception as e:
            print("Error: ",e)

    # Save the evaluation set to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"data/evaluation/synthetics_sets/ragas_eval_dataset_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_set, f, ensure_ascii=False, indent=2)
    print(f"Evaluation SYNTHETIC DATASET saved to: {output_path}")

    return evaluation_set

def evaluate_rag_pipeline(evaluation_dataset):
    """
    Evaluate a RAG pipeline using RAGAS framework.
    
    This function takes an evaluation dataset with RAG pipeline input and runs comprehensive evaluation metrics
    including faithfulness, answer relevancy, context precision, answer correctness, and context recall.
    
    Args:
        evaluation_dataset: List of dicts with 'question', 'answer', 'contexts', 'ground_truth'
        
    Returns:
        dict: Evaluation results from RAGAS
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from datasets import Dataset
    from ragas.llms import LangchainLLMWrapper
    import pandas as pd
    import json
    from datetime import datetime
    import os
    
    # Convert evaluation dataset to pandas DataFrame
    df = pd.DataFrame(evaluation_dataset)
    dataset_for_evaluation = Dataset.from_pandas(df)
    
    try:
        print("Starting RAG evaluation on given evaluation dataset.")

        # Evaluate using RAGAS
        evaluation_result = evaluate(

            dataset=dataset_for_evaluation,
            
            metrics=[faithfulness, answer_relevancy, context_precision, answer_correctness, context_recall],
            
            llm=LangchainLLMWrapper(ChatGoogleGenerativeAI(
                model='gemini-2.0-flash',
                google_api_key=st.secrets["api_keys"].get("google_api_key"),
                temperature=0.3,
                max_output_tokens=1000,
                top_p=0.90,
                timeout=120
            )),
            
            embeddings=GoogleGenerativeAIEmbeddings(
                model='models/embedding-001',
                google_api_key=st.secrets["api_keys"].get("google_api_key")
            )
        )
        
        # Save the evaluation result to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"data/evaluation/results/ragas_evaluation_results_{timestamp}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert result to pandas DataFrame and save as JSON
        result_df = evaluation_result.to_pandas()
        result_dict = result_df.to_dict('records')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
        print(f"RAG evaluation RESULT saved to: {output_path}")
        
        return evaluation_result
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise Exception(f"Evaluation failed: {str(e)}")

def make_eval_dataset_and_results(
    evaluation_set,
    retrieval_type,
    embedding_provider=None, 
    embedding_model_name=None,  
    vector_store=None,
    embedded_documents_sparse=None,
    embedded_documents=None,
    retrieval_dense_top_k=None,
    retrieval_dense_mmr_fetch_k=None,
    retrieval_dense_mmr_lambda_mult=None,
    retrieval_dense_keywords=None,
    retrieval_dense_search_type=None,
    retrieval_sparse_top_k=None,
    embedding_sparse_method=None,
    embedding_sparse_kparam=None,
    embedding_sparse_bparam=None,
    retrieval_hybrid_alpha=None,
    retrieval_hybrid_top_k=None,
    reranked_type=None,
    reranked_type_model=None,
    reranked_type_provider=None,
    reranked_top_k_rerank=None,
    rag_response_llm_provider=None,
    rag_response_llm_model_name=None,
    rag_response_prompt_template=None,
    rag_response_model_temperature=None,
    rag_response_model_max_tokens=None,
    rag_response_model_top_p=None
):
    """
    Create an evaluation dataset with RAG answers for each item in the evaluation set.
    This function encapsulates the logic from the Streamlit button block, making it reusable and testable.
    Args:
        evaluation_set: List of dicts with 'question', 'answer', 'contexts', 'ground_truth'.
        All other arguments: session state variables needed for retrieval, reranking, and generation.
    Returns:
        eval_dataset: List of dicts in RAGAS-compatible format.
        evaluation_results: List of dicts from evluation set and RAGAS metrics associated.
    """
    eval_dataset = []  # This will store the final evaluation dataset

    # Loop through each item in the evaluation set
    for item in evaluation_set:

        # extract query from evaluation dataset - "question" key
        evaluation_query = item['question']
        print("QUERY: ",evaluation_query)

        # creating RAG response with all steps - retrieval -> reranking (if used) -> context -> rag response
        try:
            # --- Retrieval step ---
            if retrieval_type == 'dense':
                print("Retrieval type: DENSE")
                evaluation_retrieved_docs = retrieve_dense(
                    query=evaluation_query,
                    vector_store=vector_store,
                    top_k=retrieval_dense_top_k,
                    fetch_k=retrieval_dense_mmr_fetch_k,
                    lambda_mult=retrieval_dense_mmr_lambda_mult,
                    keywords=retrieval_dense_keywords,
                    search_type=retrieval_dense_search_type
                )
                
            elif retrieval_type == 'sparse':
                print("Retrieval type: SPARSE")
                evaluation_retrieved_docs = retrieve_sparse(
                    query=evaluation_query,
                    sparse_chunks=embedded_documents_sparse or [],
                    top_k=retrieval_sparse_top_k,
                    embedding_sparse_method=embedding_sparse_method,
                    bm25_k1=embedding_sparse_kparam,
                    bm25_b=embedding_sparse_bparam
                )

            elif retrieval_type == 'hybrid':
                print("Retrieval type: HYBRID")
                evaluation_retrieved_docs = retrieve_hybrid(
                    query=evaluation_query,
                    embedding_provider=embedding_provider,  # You can add as needed
                    embedding_model_name=embedding_model_name,  # You can add as needed
                    embedding_sparse_method=embedding_sparse_method,
                    embedding_sparse_kparam=embedding_sparse_kparam,
                    embedding_sparse_bparam=embedding_sparse_bparam,
                    dense_chunks=embedded_documents or [],
                    sparse_chunks=embedded_documents_sparse or [],
                    alpha=retrieval_hybrid_alpha,
                    top_k=retrieval_hybrid_top_k
                )
                
            else:
                print('Retrieval type not chosen')

            # --- Reranking step ---
            if reranked_type == 'cross_encoder':
                print("reranked_type: cross_encoder")
                evaluation_reranked_docs = rerank_cross_encoder(
                    evaluation_query,
                    evaluation_retrieved_docs,
                    model_name=reranked_type_model,
                    top_k=reranked_top_k_rerank
                )
            elif reranked_type == 'llm':
                print("reranked_type: llm")
                evaluation_reranked_docs = rerank_llm_judge(
                    query=evaluation_query,
                    documents=evaluation_retrieved_docs,
                    llm_provider=reranked_type_provider,
                    llm_model_name=reranked_type_model,
                    top_k=reranked_top_k_rerank
                )
                
            else:
                evaluation_reranked_docs = evaluation_retrieved_docs  # No reranking

            # --- Prepare context from reranked documents ---
            context_parts = []
            for i, doc in enumerate(evaluation_reranked_docs):
                # Handle different document formats (LangChain Document, dict, or str)
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif isinstance(doc, dict) and 'content' in doc:
                    content = doc['content']
                else:
                    content = str(doc)
                context_parts.append(f"Document {i+1}:\n{content}\n")
            evaluation_context = "\n".join(context_parts)
            print("Building context: DONE")

            # --- Generate RAG answer ---
            evaluation_rag_response = generate_rag_response(
                query=evaluation_query,
                context=evaluation_context,
                llm_provider=rag_response_llm_provider,
                llm_model_name=rag_response_llm_model_name,
                prompt_template=rag_response_prompt_template,
                temperature=rag_response_model_temperature,
                max_tokens=rag_response_model_max_tokens,
                top_p=rag_response_model_top_p
            )
            print("Generation RAG response: DONE")

            # --- Build the evaluation item in RAGAS-compatible format ---
            question = item["question"]
            ground_truth = item["ground_truth"]
            context = evaluation_context
            answer = evaluation_rag_response
            eval_dataset.append({
                "question": question if isinstance(question, list) else question,
                "answer": answer if isinstance(answer, list) else answer,
                "contexts": context if isinstance(context, list) else [context],
                "ground_truth": ground_truth if isinstance(ground_truth, list) else ground_truth
            })
            print("Evaluation dataset with RAG answers: DONE")

        except Exception as e:
            # If any error occurs, skip this item and continue
            print(f"Error generating question: {str(e)}")
    
    # Save the evaluation set to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"data/evaluation/evaluation_sets/ragas_eval_dataset_with_RAG_{timestamp}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)
    print(f"Evaluation DATASET WITH RAG saved to: {output_path}")

    # Run RAGAS evaluation
    evaluation_results = evaluate_rag_pipeline(eval_dataset)

    return [eval_dataset,evaluation_results]