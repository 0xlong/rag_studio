# RAG Studio

A powerful web application for designing and implementing RAG (Retrieval-Augmented Generation) solutions with state-of-the-art techniques.

<!--
====================
 TABLE OF CONTENTS
====================
A table of contents helps users quickly navigate large READMEs.
-->
## Table of Contents
- [Features](#features)
- [Screenshots](#screenshots)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [Testing](#testing)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [References](#references)
- [License](#license)
- [Contact & Acknowledgements](#contact--acknowledgements)

<!--
====================
 RAG STUDIO PIPELINE OVERVIEW
====================
A high-level, actionable summary of the app's main logic and flow for junior developers.
-->
## RAG Studio Pipeline Overview

The application consists of two main components:
1. **Frontend (UI)**: Built with Streamlit, this is the user-facing application where you configure your RAG pipeline.
   - **Data Section**: Define your data sources (e.g., documents, PDFs, text files).
   - **Indexing & Storage**: Set up your vector store (e.g., FAISS, Pinecone, ChromaDB) to efficiently store and retrieve embeddings.
   - **Retrieval & Reranking**: Configure how to retrieve relevant documents and re-rank them for better quality.
   - **Generation & Prompting**: Define your prompt templates and generation strategy.
   - **Model Management**: Choose and configure your language models (e.g., GPT-4, Claude, LLaMA) for generation.
   - **Evaluation**: Measure the performance of your RAG system using metrics like MRR, MAP, and ROUGE.

2. **Backend (RAG Pipeline)**: The logic that runs in the background, handling data ingestion, embedding generation, retrieval, and generation.
   - **Data Ingestion**: Load and process your documents (e.g., chunking, cleaning, embedding).
   - **Indexing**: Store embeddings in your vector store.
   - **Retrieval**: Use your vector store to find relevant documents.
   - **Reranking**: Re-rank retrieved documents to improve the quality of the final answer.
   - **Generation**: Use your language model to generate the final answer based on the retrieved and re-ranked documents.

The app provides a user-friendly interface to configure these components and monitor their performance.

<!--
====================
 FEATURES
====================
List the main features of your project. Use emojis for clarity and visual appeal.
-->
## Features
- 📂 Data Ingestion & Processing
- 🗂️ Indexing & Storage
- 🔍 Retrieval & Reranking
- 💭 Generation & Prompting
- 🤖 Model Management
- 📊 Evaluation & Analytics

<!--
====================
 SCREENSHOTS / DEMO
====================
Show what your app looks like. Replace the placeholder with your own images or GIFs.
-->
## Screenshots

[Screenshot1](/app_screenshot.PNG)


<!--
====================
 GETTING STARTED
====================
Explain prerequisites, installation, and quickstart. This helps new users get up and running fast.
-->
## Getting Started

### Prerequisites
- Python 3.8+
- OS: Windows, MacOS, or Linux
- [Streamlit](https://streamlit.io/) for UI

### Installation
```bash
pip install -r requirements.txt
```

### Quickstart
```bash
streamlit run app.py
```
<!--
====================
 USAGE
====================
Step-by-step guide for using the app. Add more details as needed.
-->
## Usage
1. **Data** – Configure and upload your data sources (e.g., files, web, databases, APIs).  # This is where you tell the app what information to work with.
2. **Chunking** – Choose how to split your documents into smaller, manageable pieces for processing.  # Chunking helps break big documents into parts that are easier for the model to handle.
3. **Embeddings** – Select and generate vector representations (embeddings) for your document chunks.  # Embeddings turn text into numbers so the computer can compare meanings.
4. **Vector Stores** – Set up where and how your embeddings are stored for fast retrieval (e.g., Chroma, FAISS).  # Vector stores are like databases for your embeddings, making search fast.
5. **Retrieval** – Configure how the system finds relevant chunks for a user’s query.  # Retrieval finds the most useful pieces of information for a question.
6. **Reranking** – Improve the quality of results by reordering retrieved chunks using advanced models.  # Reranking makes sure the best answers are at the top.
7. **Generation** – Set up your language model and prompt strategy to generate answers from retrieved information.  # This is where the app creates a final answer using the found information.
8. **Evaluation** – Measure and analyze the performance of your RAG pipeline.  # Evaluation helps you see how well your setup is working.
## User Journey: Step by Step

This section walks you through the typical flow a user will follow in RAG Studio, from start to finish:

1. **Launch the App**
   - Run `streamlit run app.py` in your terminal to start the web interface.
   - The app opens in your browser, ready for configuration.

2. **Add Your Data**
   - Go to the **Data** section.
   - Upload or connect your data sources (documents, PDFs, text files, etc.).
   - The app will list your uploaded or connected files.

3. **Choose Chunking Strategy**
   - Select how to split your documents into smaller pieces (chunks).
   - Options include fixed size, recursive, or by document type.
   - Chunking helps the model process and retrieve information more efficiently.

4. **Generate Embeddings**
   - Pick an embedding model (e.g., OpenAI, Google, or local models).
   - Click to generate embeddings for your document chunks.
   - Embeddings are stored as vectors for fast searching later.

5. **Set Up Vector Store**
   - Choose where to store your embeddings (e.g., Chroma, FAISS).
   - The app will handle saving and indexing your vectors for retrieval.

6. **Configure Retrieval**
   - Decide how the app will search for relevant chunks when you ask a question.
   - You can adjust retrieval settings for accuracy or speed.

7. **Enable Reranking (Optional)**
   - Turn on reranking to improve the order of search results using advanced models.
   - This step helps ensure the best information is used for answers.

8. **Set Up Generation**
   - Choose your language model (e.g., GPT-4, Claude, LLaMA).
   - Define prompt templates or use the defaults.
   - The app will use the retrieved and reranked chunks to generate answers.

9. **Ask Questions & Get Answers**
   - Enter your questions in the app interface.
   - The app retrieves, reranks, and generates answers using your configured pipeline.

10. **Evaluate Performance**
    - Go to the **Evaluation** section.
    - Run evaluations to see how well your setup is working (metrics like MRR, MAP, ROUGE).
    - Results are saved in `data/evaluation/results/` for review.

11. **Iterate & Improve**
    - Adjust chunking, embeddings, retrieval, or generation settings as needed.
    - Re-run evaluations to track improvements.

---

This journey helps you build, test, and refine a complete RAG pipeline, all from a simple web interface.

<!--
====================
 PROJECT STRUCTURE
====================
Briefly explain the main files and folders. This helps users understand where to look for things.
-->
## Project Structure
```
rag_studio/
  app.py                # Main Streamlit app
  backend.py            # Backend logic for RAG pipeline
  data/                 # Data, embeddings, evaluation sets, vector stores
  prompts/              # Prompt templates
  requirements.txt      # Python dependencies
  README.md             # This file
```

<!--
====================
 CONFIGURATION
====================
Explain how to configure models, data sources, and environment variables.
-->
## Configuration
- Edit `app.py` and `backend.py` to set model parameters, data paths, and other options.
- Place your data and embeddings in the `data/` directory.
- Store variables for API keys and secrets in .streamlit/secrets.toml file.

<!--
====================
 ARCHITECTURE
====================
Describe the system design. Add a diagram if possible.
-->
## Architecture
The application follows a modular design with separate sections for each major component of the RAG pipeline. The UI is built with Streamlit for a clean, responsive interface.


<!--
====================
 EVALUATION & BENCHMARKS
====================
Show how to evaluate the system and any benchmark results.
-->
## Evaluation & Benchmarks
- Use the Evaluation section in the app to measure performance.
- Results are saved in `data/evaluation/results/`.

<!--
====================
 REFERENCES
====================
Cite papers, tutorials, or external resources.
-->
## References
- [Chunking knowledge tutorial](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [RAG from Sratch with Langchain](https://github.com/langchain-ai/rag-from-scratch)
- [RAG tutorial Langchain](https://python.langchain.com/docs/tutorials/rag/)
<!--
====================
 LICENSE
====================
State the license for your project.
-->
## License
This project is licensed under the MIT License.

<!--
====================
 CONTACT & ACKNOWLEDGEMENTS
====================
How to reach the authors and credits for inspiration or dependencies.
-->
## Contact & Acknowledgements
- Author: 0xlong
- Thanks to the open-source community and contributors.