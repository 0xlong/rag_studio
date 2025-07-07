import pandas as pd

# Dictionary containing chunking strategies and their characteristics
CHUNKING_STRATEGIES = {
    'Strategy': [
        'Fixed-Size Chunking',
        'Recursive Character Text Splitting',
        'Semantic Chunking',
        'Sentence Window',
        'Doc Type',
        'Propositions'
    ],
    'Complexity': [
        'Small',
        'Medium',
        'High',
        'Small',
        'Small',
        'High'
    ],
    'Pros': [
        'Simple to implement, consistent chunk sizes',
        'Respects natural text boundaries, flexible',
        'Creates semantically meaningful chunks',
        'Maintains sentence context, good for QA',
        'Specialized for specific document types',
        'Extracts discrete facts, good for indexing'
    ],
    'Cons': [
        'May split content mid-sentence',
        'More complex to configure',
        'Computationally expensive',
        'May create overlapping chunks',
        'Limited to specific document types',
        'May create very small chunks, requires NLP'
    ]
}

# Create a function to get the chunking strategies dataframe
def chunking_strategies_comparison_df():
    """
    Returns a pandas DataFrame containing information about different chunking strategies.
    
    Returns:
        pd.DataFrame: DataFrame with columns: Strategy, Complexity, Pros, Cons
    """
    return pd.DataFrame(CHUNKING_STRATEGIES)

# List of available chunking strategies
AVAILABLE_CHUNKING_STRATEGIES = [
    "Fixed-Size Chunking",
    "Recursive Character Text Splitting",
    "Semantic Chunking",
    "Sentence Window",
    "Hierarchical",
    "Doc Type",
    "Propositions"

]

def get_proposition_prompt_text():
    """
    Reads the content of the proposition_prompt.txt file and returns it as a string.
    This function can be used in other parts of the app to access the prompt template for LLMs.
    
    Returns:
        str: The content of the proposition_prompt.txt file as a string.
    """
    # Define the path to the prompt file
    prompt_path = "prompts/proposition_prompt.txt"
    
    # Open the file and read its content
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    
    return prompt_text

VECTOR_STORES_COMPARISON = {
    'Vector Store': [
        'Chroma',
        'FAISS',
        'Qdrant',
        'Milvus',
        'Weaviate',
        'PGVector',
        'LanceDB'
    ],
    'OSource': [
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes'
    ],
    'Type': [ # New field added
        'Database',
        'Library',
        'Database',
        'Database',
        'Database',
        'Database',
        'Database'
    ],
    'Pros': [
        'Easy, local dev, persistent.',
        'Extremely fast, in-memory.',
        'High perf, scalable, filter.',
        'Massive scale, distributed.',
        'AI-native, semantic, hybrid.',
        'Uses PostgreSQL, unified data.',
        'Embedded, serverless, local.'
    ],
    'Cons': [
        'Not for extreme scale.',
        'Library only, memory-heavy.',
        'Complex setup, separate service.',
        'Very complex, resource-heavy.',
        'Resource-heavy, setup complex.',
        'Slower than dedicated DBs.',
        'New, evolving features.'
    ],
    'Best for RAG': [
        'Prototyping, small local RAG.',
        'Offline, in-memory RAG.',
        'Production, scalable, filtered RAG.',
        'Enterprise, huge RAG needs.',
        'Neural search, complex RAG.',
        'PostgreSQL-based RAG integration.',
        'Edge, local-first RAG.'
    ]
}

RERANKING_TECHNIQUES_COMPARISON = {
    'Technique': [
        'Hybrid Search',
        'Cross-encoder',
        'LLM as a Judge',
        'MMR (Maximal Marginal Relevance)'
    ],
    'Pros': [
        'Combines keyword and semantic strengths, handles abbreviations/exact matches, improves recall.',
        'High precision, deep semantic understanding of query-document relevance, strong for complex queries.',
        'Leverages LLM reasoning for nuanced relevance judgments, adaptable with prompting, can provide detailed rationales.',
        'Balances relevance with diversity, reduces redundancy, provides more comprehensive context to LLM, mitigates missing nuances.'
    ],
    'Cons': [
        'More complex to implement and tune, slower than pure semantic search due to multiple methods, not all vector DBs support it natively.',
        'Computationally expensive, higher latency due to processing query-document pairs, can be slower for large numbers of documents.',
        'High computational cost (multiple LLM inferences), sensitive to prompt wording, can be slow, potentially over-complex for simple tasks.',
        'Requires defining a diversity metric (often cosine similarity between documents), effectiveness depends on the lambda parameter tuning.'
    ],
    'Best for RAG': [
        'When both exact keyword matches and semantic understanding are crucial (e.g., code search, specific entity retrieval, highly structured data).',
        'Production RAG systems where high accuracy and deep understanding of relevance are paramount, even at the cost of some latency.',
        'Advanced RAG pipelines requiring highly nuanced and context-aware reranking, especially for complex or ambiguous queries, and when interpretability of reranking decisions is desired.',
        'When the LLM context window is limited and you need to ensure the retrieved documents cover a broad range of relevant subtopics and perspectives, avoiding repetitive information.'
    ]
}

RETRIEVAL_TECHNIQUES_COMPARISON = {
    'Technique': [
        'Dense Retrieval',
        'Sparse Retrieval',
        'Hybrid Retrieval'
    ],
    'Pros': [
        'Captures semantic meaning, effective for complex queries, robust to synonyms and paraphrasing, excels in contextual understanding.',
        'Fast and efficient, excels at exact keyword matching, handles structured data well, widely supported by traditional search systems.',
        'Combines strengths of dense and sparse methods, improves recall, balances semantic and keyword-based search, adaptable to diverse query types.'
    ],
    'Cons': [
        'Computationally intensive, requires large-scale pre-trained models, may miss exact keyword matches, sensitive to embedding quality.',
        'Limited to keyword-based matching, struggles with semantic nuances, less effective for complex or ambiguous queries.',
        'More complex to implement and tune, may increase latency due to combining methods, requires balancing weights of dense and sparse components.'
    ],
    'Best for RAG': [
        'RAG systems handling complex, natural language queries where semantic understanding is critical (e.g., conversational AI, contextual Q&A).',
        'RAG pipelines requiring fast retrieval with precise keyword matches (e.g., code search, entity retrieval, structured data queries).',
        'RAG applications needing both semantic and keyword precision, such as mixed query types or when dealing with structured and unstructured data.'
    ]
}