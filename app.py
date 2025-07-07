import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
import numpy as np
import tempfile
import os
from backend import load_document, chunking_fixed_size, chunking_recursive, chunking_by_doc_type, chunking_sentence_window, chunking_semantic, chunking_propositions, embed_dense, embed_sparse, add_to_vector_store, query_transform, retrieve_dense, retrieve_hybrid, retrieve_sparse, rerank_cross_encoder, rerank_llm_judge, generate_rag_response, make_ragas_evaluationset
from data_info import chunking_strategies_comparison_df, get_proposition_prompt_text, VECTOR_STORES_COMPARISON, RERANKING_TECHNIQUES_COMPARISON, RETRIEVAL_TECHNIQUES_COMPARISON
import json
from langchain_community.vectorstores import Chroma, FAISS

# Set page configuration
st.set_page_config(
    page_title="RAG Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme and menu styling
st.markdown("""
    <style>
    /* Main app background */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #252526;
    }
    
    /* Menu items styling */
    .st-emotion-cache-1cypcdb {
        background-color: #252526;
    }
    
    /* Menu item hover effect */
    .st-emotion-cache-1cypcdb:hover {
        background-color: #2A2D2E;
    }
    
    /* Active menu item */
    .st-emotion-cache-1cypcdb[data-selected="true"] {
        background-color: #37373D;
        border-left: 3px solid #007ACC;
    }
    
    /* Menu icons */
    .st-emotion-cache-1cypcdb .menu-icon {
        color: #858585;
    }
    
    /* Menu text */
    .st-emotion-cache-1cypcdb .menu-text {
        color: #CCCCCC;
        font-family: 'Segoe UI', sans-serif;
        font-size: 13px;
    }
    
    /* Section headers */
    h1, h2, h3 {
        color: #E0E0E0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Custom CSS to disable typing in selectbox */
    .stSelectbox input {
        pointer-events: none;
    }
    
    /* Make Chunk button full width */
    .stButton > button {
        width: 100%;
        background-color: #007ACC;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #005999;
    }
     
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation with modern styling
with st.sidebar:
    st.markdown("#### RAG Studio", unsafe_allow_html=True)
    selected = option_menu(
        menu_title=None,
        options=["Data", "Chunking", "Embeddings", "Vector Stores", "Retrieval", 
                "Reranking", "Prompting", "Generation", "Evaluation"],
        icons=['📂', '✂️', '🔤', '🗂️', '🔍', '📊', '💭', '🤖', '📈'],
        menu_icon="",
        default_index=0,
        styles={
            "container": {
                "padding": "0",
                "background-color": "#252526",
            },
            "icon": {
                "color": "#858585",
                "font-size": "14px",
            },
            "nav-link": {
                "color": "#CCCCCC",
                "font-size": "13px",
                "font-family": "'Segoe UI', sans-serif",
                "padding": "8px 16px",
                "margin": "0",
                "border-radius": "0",
            },
            "nav-link-selected": {
                "background-color": "#37373D",
                "color": "#FFFFFF",
                "border-left": "3px solid #007ACC",
            },
        }
    )

    with st.expander("Templates"):
        st.write("MultiQuery")
        st.write("HyDE")
        st.write("IRCoT")

# Main content area
def render_data_section():
    
    # Data Source Configuration
    st.subheader("Data Source Configuration")
    data_source = st.selectbox(
        " ",
        ["Local Files", "Web Content", "Databases", "APIs & Notions"],
        placeholder="Select Data Source",  
        index=0
    )
    
    if data_source == "Local Files":
        uploaded_files = st.file_uploader(
            "Upload your files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'json']
        )
        
        if uploaded_files:
            # Store uploaded files in session state
            if 'documents' not in st.session_state:
                st.session_state.documents = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load document using backend function
                    docs = load_document(tmp_file_path)
                    st.session_state.documents.extend(docs)
                    st.success(f"Successfully loaded {uploaded_file.name} file")
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)} file")
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
            
            # Display loaded documents count
            if st.session_state.documents:
                st.info(f"Total documents loaded: {len(st.session_state.documents)}")

            with st.expander("Example documents:"):
                st.write(st.session_state.documents[0]['content'])
                
    elif data_source == "Web Content":
        urls = st.text_area("Enter URLs (one per line)")
        scrape_depth = st.slider("Scrape Depth", 1, 5, 1)
    elif data_source == "Databases":
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "MongoDB"])
        connection_string = st.text_input("Connection String")
    elif data_source == "APIs & Notions":
        api_type = st.selectbox("API Type", ["Notion", "Custom API"])
        api_key = st.text_input("API Key", type="password")

def render_chunking_section():
    # Text Splitting & Chunking
    st.subheader("Chunking Strategy")
    #st.markdown("Chunking is the process of breaking down large documents into smaller, manageable pieces.")
    
    # Display the table in an expander using data from data_info.py
    with st.expander("Chunking is the process of breaking down large documents into smaller pieces. Here are some strategies to choose from:"):
        st.dataframe(
            chunking_strategies_comparison_df(),
            use_container_width=True, 
            hide_index=True
        )

    # Add expander for chunking strategy selection
    with st.expander("Select Chunking Strategy", expanded=False):
        # Replace selectbox with segmented control for better UI
        chunking_strategy = st.segmented_control(
            "Chunking Strategy",
            options=["Fixed-Size Chunking", "Recursive Character Text Splitting", "Doc Type", "Sentence Window", "Propositions", "Semantic Chunking"],
            help="Choose the strategy that best fits your document type and requirements",
            selection_mode = "single", # only one option can be chosen

        )
    
    if chunking_strategy == "Fixed-Size Chunking":
        with st.expander("Select parameters", expanded=True):
            # Create two columns for better organization
            col1, col2 = st.columns(2, border=True)
            
            with col1:
                # Add splitter type selection
                splitter_type = st.segmented_control(
                    "Splitter type",
                    options=["Character Splitter", "Token Splitter"],
                    help="Token Splitter split by tokens - might be more accurate for LLMs.",
                )
            
            with col2:
                # Chunking parameters in second column
                chunk_size = st.slider("Chunk Size", 100, 2000, 500)
                chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)
        
        if st.button("Chunk", type='primary'):
            if 'documents' in st.session_state and st.session_state.documents:
                try:
                    # Apply chunking using backend function with splitter type
                    chunked_docs = chunking_fixed_size(
                        st.session_state.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        splitter_type=splitter_type
                    )
                    
                    # Store chunked documents in session state
                    st.session_state.chunked_documents = chunked_docs
                    
                    with st.expander("Example chunks:"):
                        st.info(f"Chunks number: {len(chunked_docs):.0f}")
                        st.info(f"Average chunk size: {sum(len(chunk['content']) for chunk in chunked_docs) / len(chunked_docs):.0f} characters")
                        st.write(chunked_docs[0]['content'])
                        st.write(chunked_docs[1]['content'])

                except Exception as e:
                    st.error(f"Error during chunking: {str(e)}")
            else:
                st.warning("Please load documents first in the Data section")

    elif chunking_strategy == "Recursive Character Text Splitting":
        with st.expander("Select parameters", expanded=True):
            # Create two columns for better organization
            col1, col2 = st.columns(2, border=True)
            
            with col1:
                # Add segmented control for separator selection
                separator_mode = st.segmented_control(
                    "Separator Mode", 
                    ["Default", "Custom"],
                    help="Choose between default separators or define your own custom separators",
                )
                
                if separator_mode == "Default":
                    st.info("""Using default separators: split on paragraph, new line, sentence (stop), word space, character""")
                    separators = ["\n\n", ". ", " ", ""]
                else:  # Custom mode
                    custom_separators = st.text_area(
                        "Enter custom separators (one per line)",
                        help="""Enter your custom separators, one per line. The order matters - it will try each separator in sequence."""
                    )
                    separators = [s.strip() for s in custom_separators.split('\n') if s.strip()]
            
            with col2:
                # Chunking parameters in second column
                chunk_size = st.slider("Chunk Size", 100, 2000, 500)
                chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)

        if st.button("Chunk", type='primary'):
            if separator_mode == "Custom" and not separators:
                st.error("Please enter at least one custom separator")
            elif 'documents' in st.session_state and st.session_state.documents:
                try:
                    # Apply chunking using backend function
                    chunked_docs = chunking_recursive(
                        st.session_state.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=separators  # Pass the separators list directly
                    )
                    
                    # Store chunked documents in session state
                    st.session_state.chunked_documents = chunked_docs
                    
                    with st.expander("Example chunks:"):
                        print(type(chunked_docs[0]))
                        json.dumps(chunked_docs, indent=2, ensure_ascii=False)
                        st.info(f"Chunks number: {len(chunked_docs):.0f}")
                        st.info(f"Average chunk size: {sum(len(chunk['content']) for chunk in chunked_docs) / len(chunked_docs):.0f} characters")
                        st.write("Example chunk 1: \n\n", chunked_docs[0]['content'])
                        st.write("Example chunk 2: \n\n", chunked_docs[1]['content'])
                except Exception as e:
                    st.error(f"Error during chunking: {str(e)}")
            else:
                st.warning("Please load documents first in the Data section")

    elif chunking_strategy == "Sentence Window":
        with st.expander("Select parameters", expanded=True):
            # Window size parameter
            window_size = st.slider(
                "Window Size",
                min_value=1,
                max_value=5,
                value=2,
                help="Number of sentences to include before and after the target sentence"
            )
        
        if st.button("Chunk", type='primary'):
            if 'documents' in st.session_state and st.session_state.documents:
                try:
                    # Apply chunking using backend function
                    chunked_docs = chunking_sentence_window(
                        st.session_state.documents,
                        window_size=window_size
                    )
                    
                    # Store chunked documents in session state
                    st.session_state.chunked_documents = chunked_docs
                    
                    with st.expander("Example chunks:"):
                        st.info(f"Chunks number: {len(chunked_docs):.0f}")
                        st.info(f"Average chunk size: {sum(len(chunk['content']) for chunk in chunked_docs) / len(chunked_docs):.0f} characters")
                        
                        # Display example chunks with their target sentences
                        for i in range(min(2, len(chunked_docs))):
                            st.write(f"Example chunk {i+1}:")
                            st.write("Target sentence:", chunked_docs[i]['metadata']['target_sentence'])
                            st.write("Full context:", chunked_docs[i]['content'])
                            st.write("---")

                except Exception as e:
                    st.error(f"Error during chunking: {str(e)}")
            else:
                st.warning("Please load documents first in the Data section")

    elif chunking_strategy == "Semantic Chunking":
        with st.expander("Select parameters", expanded=True):
            # Create two columns for better organization
            col1, col2 = st.columns(2, border=True)
            
            with col1:
                # Embedding model selection - using exact values from backend
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["huggingface", "openai", "google"],
                    help="Select the embedding provider to use"
                )
                
                # Model selection based on provider
                if embedding_model == "huggingface":
                    embedding_model_name = st.selectbox(
                        "Hugging Face Model",
                        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
                        help="Select a Hugging Face model for embeddings"
                    )
                elif embedding_model == "openai":
                    embedding_model_name = st.selectbox(
                        "OpenAI Model",
                        ["text-embedding-3-small", "text-embedding-3-large"],
                        help="Select an OpenAI model for embeddings"
                    )
                else:  # Google
                    embedding_model_name = st.selectbox(
                        "Google Model",
                        ["embedding-001", "textembedding-gecko"],
                        help="Select a Google model for embeddings"
                    )

                 # Buffer size
                buffer_size = st.slider(
                    "Buffer Size",
                    min_value=1,
                    max_value=5,
                    value=1,  # Default value
                    help="Number of sentences to include before and after breakpoints"
                )
            
            with col2:
                # Breakpoint threshold type
                breakpoint_threshold_type = st.segmented_control(
                    "Breakpoint Threshold Type",
                    options=["percentile", "standard_deviation", "interquartile", "gradient"],
                    help="Method to determine semantic breakpoints"
                )

                # Dynamic breakpoint threshold amount based on type
                if breakpoint_threshold_type == "percentile":
                    st.info("Any difference greater than the X percentile is split.")
                    breakpoint_threshold_amount = st.slider(
                        "Percentile Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.95,
                        step=0.01,
                        help="Threshold percentile (0-1) for determining breakpoints"
                    )
                elif breakpoint_threshold_type == "standard_deviation":
                    st.info("Any difference greater than X standard deviations is split.")
                    breakpoint_threshold_amount = st.slider(
                        "Standard Deviation Threshold",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Number of standard deviations from mean for breakpoints"
                    )
                    
                elif breakpoint_threshold_type == "interquartile":
                    st.info("Interquartile distance is used to split chunks.")
                    breakpoint_threshold_amount = st.slider(
                        "IQR Multiplier",
                        min_value=0.5,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        help="Multiplier for interquartile range to determine breakpoints"
                    )
                else:  # gradient
                    st.info("Used when chunks are highly correlated with each other e.g. legal")
                    breakpoint_threshold_amount = st.slider(
                        "Gradient Threshold",
                        min_value=0,
                        max_value=100,
                        value=95,
                        step=1,
                        help="Minimum gradient change to consider as breakpoint"
                    )
                
        
        if st.button("Chunk", type='primary'):
            if 'documents' in st.session_state and st.session_state.documents:
                try:
                    # Apply chunking using backend function
                    chunked_docs = chunking_semantic(
                        st.session_state.documents,
                        embedding_model=embedding_model,  # Pass the provider directly
                        embedding_model_name=embedding_model_name,  # Pass the specific model
                        breakpoint_threshold_type=breakpoint_threshold_type,
                        breakpoint_threshold_amount=breakpoint_threshold_amount,
                        buffer_size=buffer_size
                    )
                    
                    # Store chunked documents in session state
                    st.session_state.chunked_documents = chunked_docs
                    
                    with st.expander("Example chunks:"):

                        # Display chunking metadata
                        st.info(f"""Chunking Configuration:
                            \n - Embedding Model: {embedding_model}
                            \n - Model Name: {embedding_model_name}
                            \n - Breakpoint Threshold Type: {breakpoint_threshold_type}
                            \n - Threshold Amount: {breakpoint_threshold_amount}
                            \n - Buffer Size: {buffer_size}
                        """)

                        st.info(f"Chunks number: {len(chunked_docs):.0f}")
                        st.info(f"Average chunk size: {sum(len(chunk['content']) for chunk in chunked_docs) / len(chunked_docs):.0f} characters")
                        st.write("Example chunk 1: \n\n", chunked_docs[0]['content'])
                        st.write("Example chunk 2: \n\n", chunked_docs[1]['content'])
                        
                except Exception as e:
                    st.error(f"Error during chunking: {str(e)}")
            else:
                st.warning("Please load documents first in the Data section")

    elif chunking_strategy == "Doc Type":
        with st.expander("Select parameters", expanded=True):
            # Create two columns for better organization
            col1, col2 = st.columns(2, border=True)
            
            with col1:
                pass
            
            with col2:
                chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50)
                chunk_size = st.slider("Chunk Size", 100, 2000, 500)
                
        if st.button("Chunk", type='primary'):
            if 'documents' in st.session_state and st.session_state.documents:
                try:
                    # Apply chunking using backend function
                    chunked_docs = chunking_by_doc_type(
                        st.session_state.documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Store chunked documents in session state
                    st.session_state.chunked_documents = chunked_docs
                    
                    with st.expander("Example chunks:"):

                        # Display recognized document types
                        unique_types = list(set(chunk['metadata'].get('chunk_type', 'unknown') for chunk in chunked_docs))
                        st.info(f"Recognized document types: {', '.join(unique_types)}")
                        
                        st.info(f"Chunks number: {len(chunked_docs):.0f}")
                        st.info(f"Average chunk size: {sum(len(chunk['content']) for chunk in chunked_docs) / len(chunked_docs):.0f} characters")
                        st.write(chunked_docs[0]['content'])
                        st.write(chunked_docs[1]['content'])

                except Exception as e:
                    st.error(f"Error during chunking: {str(e)}")
            else:
                st.warning("Please load documents first in the Data section")

    elif chunking_strategy == "Propositions":
        with st.expander("Select parameters", expanded=True):
            # Let user pick the LLM provider (OpenAI, HuggingFace, Google)
            llm_provider = st.selectbox(
                "LLM Provider",
                ["google", "openai", "huggingface"],
                help="Select the LLM provider to use"
            )
            # Let user pick the model name based on provider
            if llm_provider == "huggingface":
                llm_model_name = st.selectbox(
                    "Hugging Face Model",
                    ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
                    help="Select a Hugging Face model for LLM"
                )
            elif llm_provider == "openai":
                llm_model_name = st.selectbox(
                    "OpenAI Model",
                    ["gpt-3.5-turbo", "gpt-4"],
                    help="Select an OpenAI model for LLM"
                )
            else:  # Google
                llm_model_name = st.selectbox(
                    "Google Model",
                    ["gemini-2.0-flash-lite", "gemini-2.0-flash"],
                    help="Select a Google model for LLM"
                )

            default_proposition_template = get_proposition_prompt_text()
            proposition_template = st.text_area(
                "Prompt Template",
                value=default_proposition_template,
                height=150,
                help="Prompt template for extracting propositions (factoids) according to the research paper. Default is recommended."
            )

        # When the user clicks the Chunk button
        if st.button("Chunk", type='primary'):

            if 'documents' in st.session_state and st.session_state.documents:

                # Prepare a list to collect all factoid chunks
                chunked_docs = []

                # Show spinner while extracting propositions
                with st.spinner("Extracting factoids from documents..."):
                        factoids = chunking_propositions(
                            st.session_state.documents,
                            llm_provider,
                            llm_model_name,
                            proposition_template
                        )

                        # Add all factoids to the chunked_docs list
                        chunked_docs.extend(factoids)

                # Display the extracted factoids in a clear, structured way
                with st.expander("Extracted Factoids", expanded=True):
                    if chunked_docs:
                        st.info(f"Total factoids extracted: {len(chunked_docs)}")
                        st.write(factoids)
                    else:
                        st.warning("No factoids extracted.")
            else:
                st.warning("Please load documents first in the Data section")

def render_embeddings_section():
    # Embedding Model Selection UI
    st.subheader("Embedding Model Selection")
    
    # Simple explanation for users and junior devs:
    st.markdown(
        """Embeddings turn text into numbers (vectors) so we can compare meaning quantitatively. There dense and sparse types.<br><br>""",
        unsafe_allow_html=True
    )

    # Choose chunking file
    with st.expander("Select chunking source", expanded=False):

        st.info("Chunking documents exist. Change it below if You wanna change data.")

        # 1. List all chunking files in the data/chunking directory (JSON only)
        chunking_dir = os.path.join("data", "chunking")
        chunking_files = [f for f in os.listdir(chunking_dir) if f.endswith(".json")]

        # 2. Let the user select a file
        selected_file = st.segmented_control(
            "Select chunking file",
            chunking_files,
            label_visibility="collapsed"
        )

        # 3. Load the selected file when chosen
        if selected_file:
            file_path = os.path.join(chunking_dir, selected_file)
            try:
                # Load the JSON file as a list of dicts (each dict is a chunk)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunked_data = json.load(f)
                # Store in session state for use in embedding, etc.
                st.session_state.chunked_documents = chunked_data
                st.success(f"Loaded {len(chunked_data)} chunks from {selected_file}")
 
            except Exception as e:
                st.error(f"Error loading file: {e}")
                chunked_data = []
            

    # Create two columns: one for Dense Embeddings, one for Sparse Embeddings
    col_dense, col_sparse = st.columns(2, border=True)

    # --- DENSE EMBEDDINGS COLUMN ---
    with col_dense:
        st.markdown("#### Dense Embeddings")
        st.markdown("Use neural networks to capture context (good for semantic search).")
        # Dense embeddings are continuous vector representations, typically from neural models (OpenAI, Google, Hugging Face)
        # These are good for capturing semantic similarity
        with st.expander("Select Provider and Model", expanded=False):
            # Let user choose the provider (OpenAI, Google, Hugging Face)
            embedding_provider = st.segmented_control(
                "Select Provider",
                ["OpenAI", "Google", "Hugging Face"]
            )
            # Let user choose the model based on provider
            if embedding_provider == "OpenAI":
                model = st.segmented_control(
                    "Select Model",
                    ["text-embedding-3-small", "text-embedding-3-large"]
                )
            elif embedding_provider == "Google":
                model = st.segmented_control(
                    "Select Model",
                    ["embedding-001", "textembedding-gecko"]
                )
            else:  # Hugging Face
                model = st.segmented_control(
                    "Select Model",
                    ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
                )
            # Comment: This expander groups all model/provider selection logic for clarity and better UX


        #st.success(f"Provider - {embedding_provider} with Model - {model}")

        # Expander for embedding parameters
        if 'model' in locals() and model:
            with st.expander("Select Parameters", expanded=False):
                # Let user set batch size for embedding computation
                if embedding_provider in ["OpenAI", "Google"]:
                    batch_size = st.slider(
                        "Batch Size",
                        1,
                        250 if embedding_provider == "Google" else 2048,
                        32,
                        help=(
                            "Batch size controls how many texts are sent to the embedding API at once. "
                            "Higher values can make embedding much faster by processing more texts in parallel, "
                            "but if set too high, you may hit API rate limits, get errors, or run out of memory. "
                            "Lower values are safer but slower. "
                            "For most users, the default (32) is a good balance. "
                            "If you see errors, try lowering this value. "
                            "If you want faster processing and have a high API quota, you can try increasing it."
                        )
                    )
                # For OpenAI and Google, let user set embedding dimensions (vector size)
                if embedding_provider == "OpenAI":
                    default_dim = 1536 if model == "text-embedding-3-small" else 3072
                    min_dim = 256
                    max_dim = default_dim
                    dimensions = st.slider(
                        "Embedding Dimensions",
                        min_value=min_dim,
                        max_value=max_dim,
                        value=default_dim,
                        help=f"Controls the size of the embedding vector. Default for {model} is {default_dim}. Lower values use less memory but may reduce quality."
                    )
                elif embedding_provider == "Google":
                    if model == "gemini-embedding-001":
                        default_dim = 3072
                    else:
                        default_dim = 768
                    min_dim = 128
                    max_dim = default_dim
                    dimensions = st.slider(
                        "Embedding Dimensions",
                        min_value=min_dim,
                        max_value=max_dim,
                        value=default_dim,
                        help=f"Controls the size of the embedding vector. Default for {model} is {default_dim}. Lower values use less memory but may reduce quality."
                    )
            # save all params in session state for later use
            st.session_state['embedding_provider'] = embedding_provider
            st.session_state['embedding_model_name'] = model
            st.session_state['embedding_dimensions'] = dimensions

            if st.button("Embed (Dense)", type='primary', key="embed_dense"):

                # 1. Check if chunked documents exist in session state
                if 'chunked_documents' not in st.session_state or not st.session_state.chunked_documents:
                    st.warning("Please chunk your documents first (see the Chunking section).")
                else:
                    
                    # 3. Get the model name from UI
                    model_name = model

                    # 4. Get the chunked docs
                    chunked_docs = st.session_state.chunked_documents

                    # 5. Call embed_dense from backend.py
                    with st.spinner("Computing dense embeddings (may take a while for large data)..."):
                        try:
                            embedded_docs = embed_dense(chunked_docs, embedding_provider, model_name, dimensions)
                            st.write("Lenght of embeddings", len(embedded_docs[0]['embedding']))
                            # 6. Store in session state for later use
                            st.session_state.embedded_documents = embedded_docs
                            
                            # 7. Show summary and example
                            st.success(f"Dense embedding complete! Total chunks embedded: {len(embedded_docs)}")
                            with st.expander("Example embedded chunk"):
                                # Show the first chunk's content and a preview of its embedding
                                st.write("**Chunk content:**", embedded_docs[0]['content'])
                                st.write("**Embedding (first 10 dims):**", embedded_docs[0]['embedding'][:10])
                        except Exception as e:
                            st.error(f"Error during dense embedding: {str(e)}")
                # --- DENSE EMBEDDING LOGIC ENDS HERE ---

    # --- SPARSE EMBEDDINGS COLUMN ---
    with col_sparse:
        st.markdown("#### Sparse Embeddings")
        st.markdown("Use classic methods to focus on keywords (good for keyword search).")
        # Sparse embeddings are high-dimensional, mostly-zero vectors, often from traditional IR models (BM25, SPLADE, etc.)
        # These are good for keyword-based retrieval and can complement dense embeddings
        with st.expander("Select Sparse Model", expanded=False):
            # Use a key so the value persists in session_state
            sparse_model = st.segmented_control(
                "Sparse Model",
                ["BM25", "TF-IDF"],
                key="sparse_model_selection",
                help="Sparse models are useful for keyword-based retrieval. BM25 is classic, SPLADE is neural, TF-IDF is simple."
            )

        # Always get the value from session_state
        sparse_model = st.session_state.get("sparse_model_selection")

        if sparse_model:
            with st.expander("Sparse Model Parameters", expanded=False):
                if sparse_model == "BM25":
                    k1 = st.slider(
                        "k1 (BM25 parameter)", 0.5, 3.0, 1.5, step=0.1,
                        help="k1 controls how much term frequency (word count) boosts the score. ..."
                    )
                    b = st.slider(
                        "b (BM25 parameter)", 0.0, 1.0, 0.75, step=0.01,
                        help="b controls how much to normalize for document length. ..."
                    )
                    st.session_state['embedding_sparse_kparam'] = k1
                    st.session_state['embedding_sparse_bparam'] = b

            # Save the method in session_state for later use
            st.session_state['embedding_sparse_method'] = sparse_model.lower().replace("-", "_")

            # --- SPARSE EMBEDDING BUTTON AND LOGIC ---
            if st.button("Embed (Sparse)", type='primary', key="embed_sparse"):
                # 1. Check if chunked documents exist in session state
                if 'chunked_documents' not in st.session_state or not st.session_state.chunked_documents:
                    st.warning("Please chunk your documents first (see the Chunking section) or use chunked files.")
                else:
                    # 2. Get the chunked docs
                    chunked_docs = st.session_state.chunked_documents
                    # 3. Determine method for embed_sparse
                    method = 'bm25' if sparse_model == 'BM25' else 'tfidf'
                    # 4. Call embed_sparse from backend.py
                    # For BM25, pass k1 and b from session state (set in the UI above)
                    with st.spinner("Computing sparse embeddings (this is fast)..."):
                        try:
                            if method == 'bm25':
                                # Pass k1 and b from UI to backend for BM25
                                embedded_docs = embed_sparse(
                                    chunked_docs,
                                    method=method,
                                    k1=st.session_state.get('embedding_sparse_kparam', 1.5),
                                    b=st.session_state.get('embedding_sparse_bparam', 0.75)
                                )
                            else:
                                # For TF-IDF, just call with method
                                embedded_docs = embed_sparse(chunked_docs, method=method)
                            # 5. Store in session state for later use
                            st.session_state.embedded_documents_sparse = embedded_docs
                            st.write(embedded_docs)
                            
                            # 6. Show summary and example
                            st.success(f"Sparse embedding complete! Total chunks embedded: {len(embedded_docs)}")
                            with st.expander("Example embedded chunk (Sparse)"):
                                # Show the first chunk's content and a preview of its embedding
                                st.write("**Chunk content:**", embedded_docs[0]['content'])
                                st.write("**Embedding (first 10 dims):**", embedded_docs[0]['embedding_sparse'][:10])
                        except Exception as e:
                            st.error(f"Error during sparse embedding: {str(e)}")

def render_vector_stores_section():

    # Vector Store Explanation
    st.subheader("Vector Stores")
    st.markdown("A vector store is a system storing embeddings and enabling efficient search.")
    
    # Vector Stores dtaframe comparison
    with st.expander("Vector Stores Comparison"):
        
        df = pd.DataFrame(VECTOR_STORES_COMPARISON)
        st.dataframe(df, hide_index=True)

    # Choose chunking file
    with st.expander("Select embeddings source", expanded=False):
        if 'embedded_documents' in st.session_state:
            st.info("Embeddings exist. Change it below if You wanna change data.")
        else:
            st.info("Embeddings does not exist. Create it or choose from available.")

        st.write(
            f"Provider: {st.session_state.get('embedding_provider')}\n"
            f"Model: {st.session_state.get('embedding_model_name')}\n"
            f"Dimensions: {st.session_state.get('embedding_dimensions')}"
        )

        # list embedding files
        embedding_dir = os.path.join("data", "embeddings","dense")
        embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith(".json")]

        # 2. Let the user select a file
        embedding_file = st.segmented_control(
            "Select embedding file",
            embedding_files,
            label_visibility="collapsed"
        )

        # 3. Load the selected file when chosen
        if embedding_file:
            file_path = os.path.join(embedding_dir, embedding_file)
            try:
                # Load the JSON file as a list of dicts (each dict is a chunk)
                with open(file_path, 'r', encoding='utf-8') as f:
                    embedded_docs = json.load(f)
                # Store in session state for use in embedding, etc.
                st.session_state.embedded_documents = embedded_docs
                st.success(f"Loaded {len(embedded_docs)} chunks from {embedding_file}")

            except Exception as e:
                st.error(f"Error loading file: {e}")
                embedded_docs = []

    # Select vector store
    with st.expander("Vector Stores Selection", expanded=False):

        
        vector_store = st.segmented_control(

            "Select vector store",
            ['Chroma','FAISS','Qdrant','Milvus','Weaviate','PGVector','LanceDB'],
            label_visibility="collapsed"
        )
        
        if vector_store == "Chroma":
            distance_metric = st.segmented_control(
                "Distance Metric",
                ["l2", "ip", "cosine"],
                help="Choose the distance metric for FAISS Flat index. l2=Euclidean, ip=Inner Product, cosine=Cosine Similarity."
            )

        # --- FAISS parameter tuning controls ---
        if vector_store == "FAISS":

            # Let user pick the FAISS index type
            faiss_params = st.segmented_control(

                "FAISS indexing options",
                ["Flat", "IVF", "HNSW"],
                help="Flat: brute-force search (accurate, slow, low memory); IVF: Inverted File index (faster, uses clusters, less memory, may lose some accuracy); HNSW: Hierarchical Navigable Small World graph (very fast, good for large data, uses more memory, approximate results). Choose based on your speed/accuracy/memory needs."
            )

            # --- Distance metric selection based on FAISS index type ---
            # For Flat: l2, ip, cosine; For IVF/HNSW: l2, ip
            if faiss_params:
                if faiss_params.lower() in ['flat','ivf', 'hnsw']:
                    distance_metric = st.segmented_control(
                        "Distance Metric",
                        ["l2", "ip", "cosine"],
                        help="Choose the distance metric for FAISS Flat index. l2=Euclidean, ip=Inner Product, cosine=Cosine Similarity."
                    )
                else:
                    distance_metric = "l2"  # fallback default

            # This segmented control allows the user to select the document store (docstore) type for FAISS.
            docstore_type = st.segmented_control(
                "Docstore Type",
                ["In-memory", "Local Disk"],
                help="Choose where to store documents. 'In-memory' is fast but not persistent. 'Local Disk' saves to disk."
            )
            # Show parameter tuning controls based on selected FAISS index type
            if faiss_params == "IVF":
                # IVF index needs 'nlist' parameter (number of clusters)
                nlist = st.slider(
                    "nlist (number of clusters)",
                    min_value=10,
                    max_value=1024,
                    value=100,
                    step=1,
                    help="Number of clusters for IVF. More clusters = faster but less accurate. Default is 100."
                )
            elif faiss_params == "HNSW":
                # HNSW index needs 'M' parameter (number of neighbors)
                hnsw_m = st.slider(
                    "M (number of neighbors)",
                    min_value=4,
                    max_value=64,
                    value=32,
                    step=1,
                    help="Number of neighbors for HNSW. Higher = more accurate, more memory. Default is 32."
                )

        st.info(f"Provider ({st.session_state.get('embedding_provider')})"
                 f"\n Model ({st.session_state.get('embedding_model_name')}) "
                 f"\n Dimensions ({st.session_state.get('embedding_dimensions')})")
        

    # --- VECTOR STORE CREATION BUTTON ---
    if st.button("Create Vector Store", type='primary', key="create_vector_store") and vector_store:
        
        # It prepares a configuration dictionary for the selected vector store and calls the backend function.
        vector_store_config = {}
        vector_store_config['distance_metric'] = distance_metric

        if vector_store == "FAISS":
            # If FAISS is chosen, we specify the index type and docstore type selected by the user.
            vector_store_config['index_type'] = faiss_params.lower()
            vector_store_config['docstore_type'] = docstore_type.replace(" ", "_").lower()
            
            if faiss_params == "IVF":
                # For IVF, we pass the number of clusters ('nlist').
                vector_store_config['nlist'] = nlist
            elif faiss_params == "HNSW":
                # For HNSW, we pass the number of neighbors ('m').
                vector_store_config['hnsw_m'] = hnsw_m


        # 2. Check if there are embedded documents available in the session state.
        #    This is crucial because we can only create a vector store if we have embeddings.
        if 'embedded_documents' in st.session_state and st.session_state.embedded_documents:

            # A spinner is shown to the user to indicate that a process is running.
            with st.spinner(f"Creating {vector_store} vector store..."):
               
                # 3. Retrieve the embedded documents from the session state.
                embedded_docs = st.session_state.embedded_documents
                
                # 4. Call the backend function 'add_to_vector_store' with the necessary parameters:
                #    - The documents with their embeddings.
                #    - The type of vector store to create (e.g., 'faiss', 'chroma').
                #    - The configuration dictionary with specific parameters.
                vector_store_handle = add_to_vector_store(
                    docs=embedded_docs,
                    vector_store_type=vector_store.lower(),
                    vector_store_config=vector_store_config,
                    embedding_provider=st.session_state.get('embedding_provider'),
                    embedding_model_name=st.session_state.get('embedding_model_name')
                )
                
                # 5. Store the returned vector store object (handle) in the session state.
                #    This allows us to reuse the vector store in other parts of the app, like the Retrieval section.
                st.session_state.vector_store = vector_store_handle
                
                # 6. Display a success message to the user.
                st.success(f"Successfully created and loaded {vector_store} vector store! - {vector_store_handle}")
                with st.expander("Retrieved docs", expanded = True):
                    retriever = st.session_state.vector_store.as_retriever(search_type="similarity_score_threshold",  # Use similarity search (default for most vector stores)
                                                                            search_kwargs={
                                                                                "k": 1,  # Number of top documents to retrieve
                                                                                "score_threshold": 0.0,  # Minimum similarity score to return a result
                                                                                "where_document": {"$or": [{"$contains": "kpathsea"}, # keywords to filter on
                                                                                                           {"$contains": "xxxsandra"}]
                                                                                                    } 
                                                                            })
                    st.write(retriever.invoke("which car?"))
        else:
            # If no embedded documents are found, a warning is shown to the user.
            st.warning("Please embed your documents first in the Embeddings section.")

def render_retrieval_section():
    st.markdown("### Retrieval")
    
    with st.expander("State vars:", expanded=False):
        st.write(dict(st.session_state))

    # --- Vector Store Type Check ---
    vector_store = st.session_state.get('vector_store', None)
    if vector_store is not None:
        if isinstance(vector_store, Chroma):
            st.info("Current vector store: Chroma ")
        elif isinstance(vector_store, FAISS):
            st.info("Current vector store: FAISS")
        else:
            st.info(f"Current vector store: {type(vector_store).__name__} (unknown type)")
    else:
        st.info("No vector store loaded in session state.")

    # Query input field for user to type their search query
    query = st.text_input(
        "Search in document",
        help="This is the text used to find the most relevant chunks."
    )
    
    # Store query in session state for use in other sections
    if query:
        st.session_state['query'] = query

    with st.expander("Retrieval Type Comparison"):

        df = pd.DataFrame(RETRIEVAL_TECHNIQUES_COMPARISON)
        st.dataframe(df, hide_index=True)

    with st.expander("Retrieval Type Selection"):

        # Always initialize these variables to False at the start
        menu_dense_retrieval = False
        menu_sparse_retrieval = False
        menu_hybrid = False

        retrieval_type = st.segmented_control(
            None,
            ["Dense", "Sparse", "Hybrid"],
            help="Choose how to retrieve documents: Dense (semantic), Sparse (keyword), or Hybrid (both).")

        if retrieval_type == "Sparse":
            alpha = 1
            menu_sparse_retrieval = True

        elif retrieval_type == "Dense":
            menu_dense_retrieval = True
            st.info(f"Dense method embedding provider: {st.session_state.get('embedding_provider')}, model: {st.session_state.get('embedding_model_name')}, dimensions: {st.session_state.get('embedding_dimensions')}")
                    
        elif retrieval_type == "Hybrid":
            menu_hybrid = True
        

    if menu_dense_retrieval:

        # Query Transformation
        with st.expander("Query Transformation"):

            # create columns - for transformation type (column 1) and transformed query display (column 2)
            col1, col2 = st.columns(2, border=True)

            with col1:
                query_transform_method = st.segmented_control(
                    "Query transformation method",
                    ["No Transformation", "Multi-Query", "HyDE", "Step-Back Prompting"],
                    label_visibility="collapsed",
                    help=(
                        "Choose how to transform your query before searching:\n"
                        "- No Transformation: Use your query as-is.\n"
                        "- Multi-Query: Generate several alternative phrasings of your query to improve recall.\n"
                        "- HyDE: Create a hypothetical answer to your query, then search for documents similar to that answer.\n"
                        "- Step-Back Prompting: Make your query more general to find broader context, then use that context to answer your specific question.\n"
                        "\n"
                        "Tip: These methods help the system find more relevant information, especially when your original query is too specific or phrased differently than the documents."
                    )
                )  # This segmented control lets the user pick how to transform their query before retrieval. Each method is explained above for clarity.
                
                if st.button("Transform query"):
                    if not query:
                        st.warning("Please enter a query to transform.")
                    else:
                        # Prepare the mode for the backend function (transforms Multi-Query to multi_query)
                        query_transformation_type = query_transform_method.replace(" ", "_").replace("-", "_").lower()

                        # Call the backend function to perform the query transformation
                        transformed_query = query_transform(query, mode=query_transformation_type)

                        # Store the transformed query in session state so it can be accessed later
                        st.session_state['query'] = transformed_query
            
            with col2:
                if 'query' in st.session_state:
                    if query:
                        st.markdown("##### Transformed Query")
                        st.info(st.session_state['query'][0])
        
        with st.expander("Search type"):
            dense_search_type = st.segmented_control(
                    "Search Type for dense retrieval",
                    ["Similarity score","Maximum Marginal Relevance (MMR)"],
                    label_visibility="collapsed",
                    help=("(MMR (Maximum Marginal Relevance): Balances relevance and diversity by selecting documents similar to the query but dissimilar to each other, reducing redundancy. Ideal for varied, relevant results. Similarity Score: Ranks documents by cosine similarity to the query, prioritizing the most similar items for precise, focused searches.)"
                          )
            )

        if dense_search_type == 'Similarity score':
            
            search_type_chosen = 'similarity_score_threshold'

            # similarity_score_threshold
            with st.expander("similarity score threshold"):

                similarity_score_threshold = st.slider(
                "",
                min_value=0.0,
                max_value=1.0,
                value=0.7, # A common starting point, adjust based on your data
                step=0.01,
                format="%.2f",
                help="Documents with a similarity score below this threshold will be filtered out. Higher values mean stricter filtering.",
                label_visibility='visible'
                )

        elif dense_search_type == 'Maximum Marginal Relevance (MMR)':

            search_type_chosen = 'mmr'

            # Number of candidates to fetch before reranking
            with st.expander("Fetch-K (candidates for MMR)"):
                fetch_k = st.slider("Number of candidates to fetch", 5, 100, 20)
                st.session_state['fetch_k'] = fetch_k

            # Lambda multiplier for diversity vs relevance
            with st.expander("Lambda (diversity vs relevance)"):
                lambda_mult = st.slider(
                    "Lambda (0 = more diverse, 1 = more relevant)",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.01
                )
                st.session_state['lambda_mult'] = lambda_mult

        # Top k documents to retrieve
        with st.expander("Top-K documents"):
            top_k = st.slider("", 1, 20, 3)
            st.session_state['top_k'] = top_k

        with st.expander("keywords to filter documents (comma-separated), provide at least 2"):
            # Let the user enter keywords separated by commas
            keyword_input = st.text_input(
                "",
                help="Only documents containing at least one of these keywords will be retrieved. To retrieve docs similarity score has to be set low."
            )
            # Convert the input string to a list of keywords, removing whitespace
            keywords = [kw.strip() for kw in keyword_input.split(",") if kw.strip()]

        if st.button("Retrieve (dense)", type='primary'):

            # Check if a vector store exists in session state
            if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
                st.warning("Please create or load a vector store first in the Vector Stores section.")

            # check if query exist, if not info
            elif not query:
                st.warning("Please enter a query to retrieve documents.")

            # if vector store and query exist
            else:

                # Use the transformed query if it exists in session state, otherwise use the original query.
                query_for_retrieval = st.session_state.get("query")[0]

                with st.spinner("Retrieving relevant documents..."):
                    try:
                        
                        # we have to assign none values for params for mmr (cause we choose similarity score), otherwise we cannot run retrieve_dense
                        if search_type_chosen =='similarity_score_threshold':
                            fetch_k = 0
                            lambda_mult = 0

                        # retrieve documents from vector store related to query with given params
                        retrieved_docs = retrieve_dense(
                            query=query_for_retrieval,
                            vector_store=st.session_state.vector_store,
                            top_k=top_k,
                            fetch_k=fetch_k,
                            lambda_mult=lambda_mult,
                            keywords=keywords,
                            search_type=search_type_chosen
                        )
                        
                        # Show results in the expander
                        with st.expander("Retrieved Documents", expanded=True):
                            
                            # display docs if exist
                            if retrieved_docs:
                                st.write("Retrieved results:", retrieved_docs)
                                st.session_state['retrieved_docs'] = retrieved_docs
                            else:
                                st.info("No documents found above the threshold. (TIP: Lower the threshold to include more results.)")
                            
                    except Exception as e:
                        st.error(f"Error during retrieval: {e}")


    elif menu_hybrid:

        with st.expander("Dense vs Sparse weights"):
            alpha = st.slider("Choose weights - 1-dense scores more important; 0-sparse scores more important", 0.0, 1.0, 0.5)

        with st.expander("How many docs to retrieve"):
            top_k = st.slider("", 0, 20, 1)
        
        # get documents with dense and sparse embeddings (if exist)
        dense_chunks = st.session_state.get('embedded_documents', [])
        sparse_chunks = st.session_state.get('embedded_documents_sparse', [])

        # 3. Get embedding model info
        embedding_provider = st.session_state.get('embedding_provider')
        embedding_model_name = st.session_state.get('embedding_model_name')

        # 4. Get sparse embedding info
        embedding_sparse_method = st.session_state.get('embedding_sparse_method')
        embedding_sparse_kparam = st.session_state.get('embedding_sparse_kparam', 1.5)
        embedding_sparse_bparam = st.session_state.get('embedding_sparse_bparam', 0.75)
        
        if st.button("Hybrid"):
            with st.spinner("Retrieving relevant documents (hybrid)..."):
                try:
                    retrieved_docs_hybrid = retrieve_hybrid(
                        query=query,
                        embedding_provider=embedding_provider,
                        embedding_model_name=embedding_model_name,
                        embedding_sparse_method=embedding_sparse_method,
                        embedding_sparse_kparam=embedding_sparse_kparam,
                        embedding_sparse_bparam=embedding_sparse_bparam,
                        dense_chunks=dense_chunks,
                        sparse_chunks=sparse_chunks,
                        alpha=alpha,
                        top_k=top_k
                    )
                    st.session_state['retrieved_docs_hybrid'] = retrieved_docs_hybrid
                    if st.session_state['retrieved_docs_hybrid']:
                        st.write("Retrieved results:", st.session_state['retrieved_docs_hybrid'])
                    else:
                        st.info("No documents found above the threshold.")

                except Exception as e:
                    st.error(f"Error during hybrid retrieval: {e}")
 
    elif menu_sparse_retrieval:
        
        # display info about sparse params chosen in embedding section if any
        if st.session_state.get('embedding_sparse_method') == 'bm25':
            st.info(f"Sparse method chosen in Embeddings section: {st.session_state.get('embedding_sparse_method')} - (kparam: {st.session_state.get('embedding_sparse_kparam')}, bparam: {st.session_state.get('embedding_sparse_bparam')})")
        elif st.session_state.get('embedding_sparse_method') == 'tf_idf':
            st.info(f"Sparse method chosen in Embeddings section: {st.session_state.get('embedding_sparse_method')}")
        else:
            st.info(f"Sparse params NOT chosen in Embeddings section. Choose below.")
            
        # documents with sparse embeddings from Embedding section which are taken from streamlit session_state
        sparse_chunks = st.session_state.get('embedded_documents_sparse', [])
        if not sparse_chunks:
            st.warning("No sparse embeddings found and no chunked documents available. Please embed your documents first or choose embeddings from below.")
            
            # 1. List all chunking files in the data/chunking directory (JSON only)
            sparse_embeddings_dir = os.path.join("data", "embeddings","sparse")
            sparse_embeddings_files = [f for f in os.listdir(sparse_embeddings_dir) if f.endswith(".json")]

            # 2. Let the user select a file
            selected_file = st.segmented_control(
                "Select embeddding file",
                sparse_embeddings_files,
                label_visibility="collapsed"
            )

            # 3. Load the selected file when chosen
            if selected_file:
                file_path = os.path.join(sparse_embeddings_dir, selected_file)
                try:
                    # Load the JSON file as a list of dicts (each dict is a chunk)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunked_data = json.load(f)
                    # Store in session state for use in embedding, etc.
                    st.session_state.embedded_documents_sparse = chunked_data
                    st.success(f"Loaded {len(chunked_data)} chunks from {selected_file}")
    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    chunked_data = []

        # check if query provided
        if not query:
            st.warning("Please enter a query.")

        # if there is no sparse method then choose params
        if not st.session_state.get('embedding_sparse_method'):
            # Parameters selection for sparse model
            with st.expander("Select Sparse Model", expanded=False):
            
                # Model selection is separated for clarity
                sparse_model = st.segmented_control(
                    "Sparse Model",
                    ["BM25", "TF-IDF"],
                    help="Sparse models are useful for keyword-based retrieval. BM25 is classic, SPLADE is neural, TF-IDF is simple."
                )
            
            # save sparse_model selection in streamlit session state
            if sparse_model:
                st.session_state["embedding_sparse_method"] = sparse_model.lower().replace("-", "")
                
                with st.expander("Sparse Model Parameters", expanded=False):

                    # Show only the parameters relevant to the selected model
                    if sparse_model == "BM25":

                        # BM25 parameters explained
                        k1 = st.slider(
                            "k1 (BM25 parameter)", 0.5, 3.0, 1.5, step=0.1,
                            help=(
                                "k1 controls how much term frequency (word count) boosts the score. "
                                "Low k1: extra occurrences of a word add little. High k1: more boost, but with diminishing returns. "
                                "Typical values: 1.2-2.0. "
                                "If unsure, use the default (1.5)."
                            )
                        )
                        b = st.slider(
                            "b (BM25 parameter)", 0.0, 1.0, 0.75, step=0.01,
                            help=(
                                "b controls how much to normalize for document length. "
                                "b=0: no normalization (long docs not penalized). b=1: full normalization (long docs penalized). "
                                "Typical value: 0.75. "
                                "If your docs are similar in length, b doesn't matter much."
                            )
                        )

                        # save sparse model param in streamlit session state
                        st.session_state['embedding_sparse_kparam'] = k1
                        st.session_state['embedding_sparse_bparam'] = b
            
        # no of docs to retrieve
        with st.expander("Top-K documents (sparse)"):
            sparse_top_k = st.slider("", 1, 20, 3)
            st.write(sparse_top_k)
            st.session_state['sparse_top_k'] = sparse_top_k

        # Button to trigger sparse retrieval
        if st.button("Retrieve (Sparse)", key="retrieve_sparse"):
            # If no sparse embeddings exist, create them from chunked documents

                from backend import retrieve_sparse
                results = retrieve_sparse(
                    query,
                    sparse_chunks,
                    top_k=sparse_top_k,
                    embedding_sparse_method=st.session_state.get('embedding_sparse_method')
                )
                if results:
                    with st.expander("Retrieved Documents", expanded=True):
                        for i, doc in enumerate(results):
                            st.write(f"Result {i+1}:")
                            st.write(doc['content'])
                else:
                    st.info("No documents found for your query.")
    else:
        st.info("No Retrieval Type method chosen.")
    
def render_reranking_section():

    # Reranking Model Selection
    st.subheader("Reranking Model")
    st.markdown("reorders retrieved documents using different techniques")

    with st.expander("Re-ranking techniques Comparison"):
        # Vector Stores dtaframe comparison
        df = pd.DataFrame(RERANKING_TECHNIQUES_COMPARISON)
        st.dataframe(df, hide_index=True)

    with st.expander("Select Reranker"):
        reranker = st.segmented_control(
            "Select reranker",
            ["Cross-Encoder", "LLM-as-a-judge"],
            label_visibility="collapsed"
        )
        
        if reranker == "Cross-Encoder":
            reranker_model = st.selectbox(
                "Select Model",
                ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"]
            )
            
            # Add slider for top_k parameter
            top_k_rerank = st.slider(
                "Top-K documents to rerank",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of top documents to return after reranking. Higher values may include less relevant documents."
            )
        elif reranker == "LLM-as-a-judge":
            # LLM-as-a-judge reranking parameters
            llm_provider = st.selectbox(
                "Select LLM Provider",
                ["google","openai", "huggingface"],
                help="Choose the LLM provider for judging document relevance"
            )
            
            # Model selection based on provider
            if llm_provider == "google":
                llm_model_name = st.selectbox(
                    "Google Model",
                    ["gemini-2.0-flash-lite", "gemini-2.0-flash"],
                    help="Select a Google model for LLM-as-a-judge"
                )
            
            # Add slider for top_k parameter
            top_k_rerank = st.slider(
                "Top-K documents to rerank",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of top documents to return after reranking. Higher values may include less relevant documents."
            )
        
    if st.button("Rerank", key="rerank_crossencoder"):

        if reranker == "Cross-Encoder":
            if 'retrieved_docs' in st.session_state:
                reranked_docs_crossencoder = rerank_cross_encoder(st.session_state['query'],
                                                                    st.session_state['retrieved_docs'],
                                                                    model_name = reranker_model,
                                                                    top_k=top_k_rerank
                                                                    )
                st.session_state['reranked_retrieved_docs'] = reranked_docs_crossencoder
                st.success("Reranking completed successfully!")
            else:
                st.warning("Please perform a retrieval first to get documents to rerank.")
                reranked_docs_crossencoder = None

        elif reranker == "LLM-as-a-judge":

            if 'retrieved_docs' in st.session_state:

                reranked_docs_llm_judge = rerank_llm_judge(
                    query=st.session_state['query'],  # The user's query
                    documents=st.session_state['retrieved_docs'],  # The docs to rerank
                    llm_provider=llm_provider,  # Provider selected in UI
                    llm_model_name=llm_model_name,  # Model selected in UI
                    top_k=top_k_rerank  # How many docs to return
                )

                # Save reranked docs in session state for display
                st.session_state['reranked_retrieved_docs'] = reranked_docs_llm_judge
                st.success("LLM-as-a-judge reranking completed successfully!")
            else:
                st.warning("Please perform a retrieval first to get documents to rerank.")
                reranked_docs_llm_judge = None

        # show reranked docs and original docs side by side
        with st.expander("Comparison: Original vs Reranked Documents", expanded=True):
            col_1, col_2 = st.columns(2, border=True)
            
            with col_1:
                st.markdown("#### Original Documents")
                if 'retrieved_docs' in st.session_state and st.session_state['retrieved_docs']:
                    for i, doc in enumerate(st.session_state['retrieved_docs'][:top_k_rerank]):
                        st.write("---")
                        st.markdown(f"**Document {i+1}:**")
                        # Handle LangChain Document objects (what retrieve_dense returns)
                        if hasattr(doc, 'page_content'):
                            # LangChain Document object
                            content = doc.page_content
                        else:
                            # Fallback: convert to string
                            content = str(doc)
                        
                        # Display content with truncation
                        st.write(content[:200] + "..." if len(content) > 200 else content)
                        
                else:
                    st.info("No original documents to display")
            
            with col_2:
                st.markdown("#### Reranked Documents")
                if 'reranked_retrieved_docs' in st.session_state and st.session_state['reranked_retrieved_docs']:
                    for i, doc in enumerate(st.session_state['reranked_retrieved_docs']):
                        st.write("---")
                        st.markdown(f"**Document {i+1}:**")
                        # Handle LangChain Document objects (what retrieve_dense returns)
                        if hasattr(doc, 'page_content'):
                            # LangChain Document object
                            content = doc.page_content
                        else:
                            # Fallback: convert to string
                            content = str(doc)
                        
                        # Display content with truncation
                        st.write(content[:200] + "..." if len(content) > 200 else content)

                else:
                    st.info("No reranked documents to display")       

def render_prompting_section():
    st.header("💭 Prompting")
    
    # Prompt Template
    st.subheader("Prompt Template")
    prompt_template = st.text_area(
        "Enter your prompt template",
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    # Prompt Engineering
    st.subheader("Prompt Engineering")
    prompt_technique = st.selectbox(
        "Select Prompting Technique",
        ["Standard", "Chain of Thought", "IRCoT", "Self-Correction"]
    )

def render_generation_section():
    
    # LLM Provider Selection
    st.subheader("LLM Provider & Model")
    
    with st.expander("LLM provider and model"):
        llm_provider = st.segmented_control(
            "Select LLM Provider",
            ["OpenAI", "Google", "HuggingFace"]
        )
        
        # Model selection based on provider
        if llm_provider == "OpenAI":
            llm_model_name = st.segmented_control(
                "OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
                help="Select an OpenAI model for text generation"
            )
        elif llm_provider == "Google":
            llm_model_name = st.segmented_control(
                "Google Model",
                ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.0-pro"],
                help="Select a Google model for text generation"
            )
        elif llm_provider == "HuggingFace":
            llm_model_name = st.segmented_control(
                "Hugging Face",
                ["hg1", "hg2", "hg3"],
                help="Select a Hugging Face model for text generation"
            )
        else:
            st.info("Choose LLM provider.")
    
    # Generation Parameters
    with st.expander("Generation Parameters"):
        model_temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        model_max_tokens = st.slider("Max Tokens", 100, 4000, 1000)
        model_top_p = st.slider("Top P", 0.0, 1.0, 0.9)

    # Prompt Template Configuration
    with st.expander("Prompt Template"):
        default_prompt_template = """Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context provided above. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        prompt_template = st.text_area(
            "Prompt Template",
            value=default_prompt_template,
            height=200,
            help="Use {context} for retrieved documents and {question} for user query"
        )

    # Display current session state for debugging
    with st.expander("Current Session State", expanded=False):
        st.write("Query:", st.session_state.get('query', 'No query found'))
        st.write("Reranked docs count:", len(st.session_state.get('reranked_retrieved_docs', [])))
        st.write("Model provider:", llm_provider if 'llm_provider' in locals() else 'Not selected')
        st.write("Model:", llm_model_name if 'llm_model_name' in locals() else 'Not selected')

    # Display reranked docs for reference
    with st.expander("Reranked docs (context)", expanded=False):
        if 'reranked_retrieved_docs' in st.session_state and st.session_state['reranked_retrieved_docs']:
            for i, doc in enumerate(st.session_state['reranked_retrieved_docs']):
                st.markdown(f"**Document {i+1}:**")
                if hasattr(doc, 'page_content'):
                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                elif isinstance(doc, dict) and 'content' in doc:
                    st.write(doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'])
                else:
                    st.write(str(doc)[:300] + "..." if len(str(doc)) > 300 else str(doc))
                st.write("---")
        else:
            st.info("No reranked documents available. Please perform retrieval and reranking first.")   

    # Generate RAG Response Button
    if st.button("Generate RAG Response", type='primary'):
        # Check if we have the necessary data
        if 'query' not in st.session_state or not st.session_state['query']:
            st.error("Please enter a query in the Retrieval section first.")
        elif 'reranked_retrieved_docs' not in st.session_state or not st.session_state['reranked_retrieved_docs']:
            st.error("Please perform retrieval and reranking first to get documents.")
        elif 'llm_provider' not in locals() or not llm_provider:
            st.error("Please select an LLM provider.")
        else:
            try:
                
                # Get the reranked documents
                reranked_docs = st.session_state['reranked_retrieved_docs']
                
                # Prepare context from reranked documents
                context_parts = []
                for i, doc in enumerate(reranked_docs):
                    # Extract content from different document formats
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content
                    elif isinstance(doc, dict) and 'content' in doc:
                        content = doc['content']
                    else:
                        content = str(doc)
                    
                    # Add document number and content to context
                    context_parts.append(f"Document {i+1}:\n{content}\n")
                
                # Combine all context
                context = "\n".join(context_parts)
                
                # Generate response using backend function
                with st.spinner("Generating RAG response..."):
                    rag_response = generate_rag_response(
                        query=st.session_state['query'],
                        context=context,
                        llm_provider=llm_provider.lower(),
                        llm_model_name=llm_model_name,
                        prompt_template=prompt_template,
                        temperature=model_temperature,
                        max_tokens=model_max_tokens,
                        top_p=model_top_p
                    )
                
                st.session_state['rag_response'] = rag_response
                st.success("RAG Response Generated Successfully!")
                
            except Exception as e:
                st.error(f"Error generating RAG response: {str(e)}")

    if 'rag_response' in st.session_state:
        with st.expander("Generated Response", expanded=True):
            st.write(st.session_state['rag_response'])

def render_evaluation_section():
    
    st.subheader("RAG Evaluation")
    st.markdown("Generate evaluation questions from your documents using RAGAS TestsetGenerator.")
    
    # Check if documents are available in session state
    if 'documents' not in st.session_state or not st.session_state.documents:
        st.warning("Please load documents first in the Data section.")
        return
    
    # Evaluation parameters
    with st.expander("Evaluation set", expanded=True):

        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of evaluation questions to generate per document"
        )

        #st.info(st.session_state.documents[6:8])
    
        # Generate evaluation set button
        if st.button("Generate Evaluation Set", type='primary'):
            if not num_questions:
                st.error("Please select at least one question type.")
            else:
                try:
                    with st.spinner("Generating evaluation set with RAGAS..."):
                        
                        # Call the backend function with documents from session state
                        eval_set = make_ragas_evaluationset(
                            documents=st.session_state.documents,
                            questions_number=num_questions
                        )
                    
                    # Display results
                    st.success(f"Successfully generated evaluation set with {len(eval_set)} questions!")
                    
                    # Show sample questions
                    with st.expander("Sample Generated Questions", expanded=True):
                        if eval_set and len(eval_set) > 0:
                            # Display first few questions as examples
                            for i, question_data in enumerate(eval_set[:3]):  # Show first 3 questions
                                st.markdown(f"**Question {i+1}:**")
                                st.write(question_data.get('question', 'No question found'))
                                st.markdown(f"**Answer:**")
                                st.write(question_data.get('answer', 'No answer found'))
                                st.markdown(f"**Context:**")
                                st.write(question_data.get('context', 'No context found')[:200] + "..." if len(question_data.get('context', '')) > 200 else question_data.get('context', ''))
                                st.write("---")
                        else:
                            st.info("No questions were generated.")
                    
                    # Store evaluation set in session state for potential use
                    st.session_state.evaluation_set = eval_set
                    
                except Exception as e:
                    st.error(f"Error generating evaluation set: {str(e)}")
    
    # Display existing evaluation sets if any
    evaluation_dir = "data/evaluation"
    if os.path.exists(evaluation_dir):
        eval_files = [f for f in os.listdir(evaluation_dir) if f.endswith('.json')]
        if eval_files:
            with st.expander("Existing Evaluation Sets", expanded=False):
                st.write("Previously generated evaluation sets:")
                for file in eval_files:
                    st.write(f"- {file}")
    
    # Additional evaluation metrics section
    st.subheader("Evaluation Metrics")
    metrics = st.multiselect(
        "Select Metrics to Track",
        ["Context Precision", "Context Recall", "Answer Faithfulness",
         "Answer Relevancy", "Answer Correctness", "Latency", "Cost per Query"]
    )
    
    # Sample visualization
    if metrics:
        data = pd.DataFrame({
            'Metric': metrics,
            'Score': np.random.uniform(0.7, 1.0, len(metrics))
        })
        fig = px.bar(data, x='Metric', y='Score', title='Evaluation Metrics')
        st.plotly_chart(fig)

# Render the selected section
if selected == "Data":
    render_data_section()
elif selected == "Chunking":
    render_chunking_section()
elif selected == "Embeddings":
    render_embeddings_section()
elif selected == "Vector Stores":
    render_vector_stores_section()
elif selected == "Retrieval":
    render_retrieval_section()
elif selected == "Reranking":
    render_reranking_section()
elif selected == "Prompting":
    render_prompting_section()
elif selected == "Generation":
    render_generation_section()
elif selected == "Evaluation":
    render_evaluation_section()