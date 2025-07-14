# RAG Chatbot for Aetna Member Handbook

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on the Aetna Better Health of Illinois Member Handbook PDF. It uses LangChain for orchestration, PyMuPDF for PDF text extraction, OpenAI for embeddings and language model capabilities, and ChromaDB as an in-memory vector store.

Here's the project structure:

rag-chatbot/
├── .env                  # Stores your OpenAI API key (NOT committed to Git)
├── .gitignore            # Specifies files/folders to ignore in Git
├── README.md             # Project documentation and instructions
├── requirements.txt      # List of Python dependencies
├── app.py                # Main chatbot application
├── config.py             # Configuration parameters
├── pdf_processor.py      # Handles PDF download and text extraction
└── rag_pipeline.py       # Manages chunking, embeddings, vector store, and RAG chain
└── chroma_db/            # Directory for ChromaDB persistence (ignored by Git)
## Features

* **PDF Ingestion:** Downloads and extracts text from the specified PDF document.
* **Text Chunking:** Splits the document into manageable chunks for efficient retrieval.
* **Vector Embeddings:** Converts text chunks into numerical vectors using OpenAI embeddings.
* **Vector Store:** Stores and retrieves relevant document chunks using ChromaDB.
* **Conversational AI:** Utilizes OpenAI's LLM to generate contextual answers, maintaining chat history.
* **Source Attribution:** Provides page numbers for retrieved information.
* **Modular Design:** Organized into separate Python files for maintainability and scalability.

## Project Structure

## Setup Instructions

### 1. Clone the Repository (if applicable)

If you're setting this up from a Git repository, clone it first:

```bash
git clone <your-repo-url>
cd rag-chatbot


1. config.py
This file will hold all your configurable parameters, making it easy to adjust settings without touching the core logic.

2. pdf_processor.py
This file will contain functions related to handling the PDF.

3. rag_pipeline.py
This file will encapsulate the core RAG logic.
-    """Splits a list of Langchain Document objects into smaller chunks."""
-    """
    Creates and persists a ChromaDB vector store from text chunks.
    If the directory exists, it loads the existing store.
    """
-   """Builds a LangChain ConversationalRetrievalChain for RAG with chat history, optionally including a re-ranking step."""
-    # Base retriever: retrieves a larger set of candidates for re-ranking

4. app.py
This will be your main application file.

    # --- 1. Download PDF ---
    # --- 2. Extract Text ---
    # --- 3. Chunk Text ---
    # --- 4. Create/Load Vector Store ---
    # --- 5. Build Conversational RAG Chain ---
    # --- 6. Simple CLI Chat Interface ---
            # Invoke the RAG chain with the user's query
            # Get the answer from the result
            # Optionally, print the source documents

5. requirements.txt
This file lists all the Python packages your project depends on.

6. .gitignore
This file tells Git which files and directories to ignore when committing to your repository. This is crucial for security (not committing API keys) and cleanliness (not committing generated data).

7. .env (Create manually, do NOT commit to Git)
Created a file named .env in the root of rag-chatbot directory.
Added  OpenAI API key and Cohere API key


8. README.md
This file will serve as your project's documentation.


## Setup Instructions

### 1. Clone the Repository (if applicable)

If you're setting this up from a Git repository, clone it first:

```bash
git clone <your-repo-url>
cd rag-chatbot
2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

Bash

python3 -m venv venv
3. Activate the Virtual Environment
macOS/Linux:
Bash : venv\Scripts\activate.bat
Windows (PowerShell): .\venv\Scripts\Activate.ps1

4. Install Dependencies
Once the virtual environment is active, install the required packages:
Bash : 
pip install -r requirements.txt

5. Configure API Keys
Create a file named .env in the root directory of the rag-chatbot project (next to app.py). Add your OpenAI API key to this file:
OPENAI_API_KEY="your_openai_api_key_here"
To enable reranking in config.py, also add:
# COHERE_API_KEY="your_cohere_api_key_here"

Replace your_openai_api_key_here (and your_cohere_api_key_here if applicable) with your actual API keys. Do not commit this file to your Git repository. 
The .gitignore file is configured to prevent this.

6. How to Run the Chatbot
Activate your virtual environment (if not already active).

Run the main application script:
Bash: python app.py
The script will first download the PDF (if not already present), process it, build the vector store (this might take a few minutes the first time), and then start the conversational interface. Type your questions and press Enter. Type exit to quit.
