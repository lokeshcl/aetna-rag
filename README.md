# RAG Chatbot for Aetna Member Handbook

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on the Aetna Better Health of Illinois Member Handbook PDF. It uses LangChain for orchestration, PyMuPDF for PDF text extraction, OpenAI for embeddings and language model capabilities, and ChromaDB as an in-memory vector store.

Here's the project structure:
```
rag-chatbot/
├── .env                  # Stores the OpenAI API key (NOT committed to Git)
├── .gitignore            # Specifies files/folders to ignore in Git
├── README.md             # Project documentation and instructions
├── requirements.txt      # List of Python dependencies
├── app.py                # Main chatbot application
├── config.py             # Configuration parameters
├── pdf_processor.py      # Handles PDF download and text extraction
└── rag_pipeline.py       # Manages chunking, embeddings, vector store, and RAG chain
└── chroma_db/            # Directory for ChromaDB persistence (ignored by Git) 
```

## Features
------------
* **PDF Ingestion:** Downloads and extracts text from the specified PDF document.
* **Text Chunking:** Splits the document into manageable chunks for efficient retrieval.
* **Vector Embeddings:** Converts text chunks into numerical vectors using OpenAI embeddings.
* **Vector Store:** Stores and retrieves relevant document chunks using ChromaDB.
* **Conversational AI:** Utilizes OpenAI's LLM to generate contextual answers, maintaining chat history.
* **Source Attribution:** Provides page numbers for retrieved information.
* **Modular Design:** Organized into separate Python files for maintainability and scalability.

## Project Structure

### Setup Instructions- bash
git clone git@github.com:lokeshcl/aetna-rag.git
cd rag-chatbot

#### 1. config.py
This file will hold all the configurable parameters, making it easy to adjust settings without touching the core logic.

#### 2. pdf_processor.py
This file will contain functions related to handling the PDF.

#### 3. rag_pipeline.py
This file will encapsulate the core RAG logic.
-    Splits a list of Langchain Document objects into smaller chunks.
-    Creates and persists a ChromaDB vector store from text chunks. 
-    Builds a LangChain ConversationalRetrievalChain for RAG with chat history, optionally including a re-ranking step.

#### 4. app.py
```This will be the main application file.
    - 1. Download PDF 
    - 2. Extract Text 
    - 3. Chunk Text 
    - 4. Create/Load Vector Store 
    - 5. Build Conversational RAG Chain 
    - 6. Simple CLI Chat Interface 
            ** Invoke the RAG chain with the user's query
            ** Get the answer from the result
            ** print the source documents for reasoning
```

##### Explanation of Layers Added to improve the RAG performance:
###### Query Rephrasing (Implicit in ConversationalRetrievalChain):
Where it happens: This is handled internally by LangChain's ConversationalRetrievalChain (specifically by its question_generator component).

Why it's important: When you ask a follow-up question like "How often she should receive health checks?" after saying "My child is 4 months old," the chain needs to understand that "she" refers to "my 4-month-old child." It rephrases the conversational query (e.g., "How often should my 4-month-old child receive health checks?") into a standalone query that is more effective for searching the vector database. This prevents the retriever from missing context that was only present in previous turns of the conversation.

In the code: I've added a print statement in app.py to inform the user about this step.

###### Reranking Layer (Optional, using Cohere Rerank):
Where it happens: This is configured within the build_conversational_rag_chain function in rag_pipeline.py. If COHERE_API_KEY is provided and the Cohere library is installed, a ContextualCompressionRetriever with CohereRerank is used.

Why it's important: After the initial retrieval from the vector store, you might get a set of documents that are semantically similar but not all equally relevant to the specific question. Reranking models (like Cohere's) take the original query and the retrieved documents and re-score them based on their true relevance, pushing the most pertinent documents to the top. This significantly improves the quality of the context provided to the LLM, leading to more accurate answers and reducing "hallucinations."

In the code:
In rag_pipeline.py it explicitly checks for cohere_api_key and attempts to set up the ContextualCompressionRetriever with CohereRerank. It includes error handling if Cohere setup fails.

app.py includes a conditional print statement to indicate this step only when Cohere reranking is active.

#### 5. requirements.txt
This file lists all the Python packages the project depends on.

#### 6. .gitignore
This file tells Git which files and directories to ignore when committing to the repository. This is crucial for security (not committing API keys) and cleanliness (not committing generated data).

#### 7. .env (Create manually, do NOT commit to Git)
Created a file named .env in the root of rag-chatbot directory.
Added  OpenAI API key and Cohere API key

#### 8. README.md
This file will serve as the project's documentation.

### Setup Instructions
----------------------
#### 1. Clone the Repository:
**bash**
    git clone git@github.com:lokeshcl/aetna-rag.git
    cd rag-chatbot

#### 2. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.
python3 -m venv venv

#### 3. Activate the Virtual Environment
Bash : venv\Scripts\activate.bat
Windows (PowerShell): .\venv\Scripts\Activate.ps1

#### 4. Install Dependencies
Once the virtual environment is active, install the required packages:
pip install -r requirements.txt

#### 5. Configure API Keys
Create a file named .env in the root directory of the rag-chatbot project (next to app.py). Add  OpenAI API key to this file:
OPENAI_API_KEY="openai_api_key_here"
To enable reranking in config.py, also add:
COHERE_API_KEY="cohere_api_key_here"

Replace openai_api_key_here and cohere_api_key_here with actual API keys. Do not commit this file to the Git repository. 
The .gitignore file is configured to prevent this.

#### 6. How to Run the Chatbot
Activate the virtual environment (if not already active).

Run the main application script:
Bash: python app.py
-- The script will first download the PDF (if not already present), process it, build the vector store (this might take a few minutes the first time), and then start the conversational interface. Type the questions and press Enter. Type exit to quit.

## Logging and Error Handling
•	Error handling is well-implemented with try-except blocks and conditional checks to gracefully manage common failure points (network issues, missing API keys, failed processing steps).
•	Logging is present and informative through the use of print() statements, providing visibility into the application flow and status. To improve logging we can structured Logging by replacing print() statements with Python's built-in logging module. This will allow different log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL) and Configurable log outputs (console, file, etc.).
