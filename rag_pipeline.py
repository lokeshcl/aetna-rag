# rag_pipeline.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME, 
    CHAT_MODEL_NAME, TEMPERATURE, CHROMA_DB_DIR
)

# Optional: Import RERANK_MODEL_NAME and RERANK_TOP_N if using reranking
# from config import RERANK_MODEL_NAME, RERANK_TOP_N 

def get_text_chunks(documents: list) -> list:
    """Splits a list of Langchain Document objects into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len, # Use character length
        add_start_index=True, # Adds start index to metadata
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the PDF.")
    return chunks

def create_vector_store(chunks: list, openai_api_key: str) -> Chroma:
    """
    Creates and persists a ChromaDB vector store from text chunks.
    If the directory exists, it loads the existing store.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=openai_api_key)
    
    # Check if the ChromaDB directory already exists and contains data
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print(f"Loading existing ChromaDB from {CHROMA_DB_DIR}...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        print("ChromaDB loaded successfully.")
    else:
        print(f"Creating new ChromaDB and persisting to {CHROMA_DB_DIR}...")
        print("This may take a few minutes depending on the PDF size and internet speed for embeddings...")

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIR
            )
            vectorstore.persist()
            print("ChromaDB created and persisted successfully.")
        except Exception as e:
            print(f"Error creating/persisting ChromaDB: {e}")
            print("Please ensure your OpenAI API key is valid and you have an active internet connection.")
            # Re-raise or handle appropriately, possibly exiting here
            raise # Re-raise the exception so app.py can catch it and exit cleanly

    return vectorstore

def build_conversational_rag_chain(vectorstore: Chroma, openai_api_key: str, cohere_api_key: str = None):
    """
    Builds a LangChain ConversationalRetrievalChain for RAG with chat history,
    optionally including a re-ranking step.
    """
    # Initialize the LLM for chat
    llm = ChatOpenAI(model_name=CHAT_MODEL_NAME, temperature=TEMPERATURE, openai_api_key=openai_api_key)
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, api_key=openai_api_key)


    # Initialize memory for chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

    # Custom prompt for the conversational chain
    qa_prompt_template = """
    You are a helpful AI assistant specializing in the Aetna Better Health of Illinois Member Handbook.
    Answer the user's question ONLY based on the provided context. 
    If the answer is not found in the context, clearly state that you don't have enough information 
    from the handbook to answer the question. Do not make up answers.

    First, provide a concise answer.
    Then, in a separate "Reasoning" section, elaborate on your answer, directly quoting or paraphrasing key details from the context to support your response.

    Context:
    {context}

    Question: {question}

    Concise Answer:
    Reasoning:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_prompt_template)

    # --- Retriever Setup (with optional Re-ranking) ---
    # Base retriever: retrieves a larger set of candidates for re-ranking
    # If not using re-ranking, this will be the final retriever.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve more for reranking

    retriever = base_retriever # Default to base retriever

    # Check if Cohere API key is available for re-ranking
    cohere_api_key = os.getenv("COHERE_API_KEY")
    # If using reranking, uncomment the following block
    if cohere_api_key:
        try:
            from config import RERANK_MODEL_NAME, RERANK_TOP_N
            compressor = CohereRerank(model=RERANK_MODEL_NAME, cohere_api_key=cohere_api_key)
            retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=compressor
            )
            print("Using Cohere Re-ranker for improved retrieval.")
        except Exception as e:
            print(f"Cohere Rerank setup failed ({e}). Falling back to standard vector search.")
            print("Ensure 'cohere' is installed and COHERE_API_KEY is valid.")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Fallback k
    else:
        print("COHERE_API_KEY not found. Skipping re-ranking and using standard vector search.")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Default k if no reranker


    # Build the ConversationalRetrievalChain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, # Use the configured retriever (with or without re-ranker)
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True 
    )
    print("Conversational RAG chain built successfully.")
    return conversation_chain

if __name__ == "__main__":
    # This block demonstrates how the functions could be used together
    # In a real app, this would be orchestrated by app.py
    from dotenv import load_dotenv
    load_dotenv()
    
    # Dummy setup for testing
    from langchain_core.documents import Document
    dummy_docs = [
        Document(page_content="This is a test document about health benefits.", metadata={"page": 1}),
        Document(page_content="Another document discussing insurance policies.", metadata={"page": 2})
    ]
    
    chunks = get_text_chunks(dummy_docs)
    
    # Ensure OPENAI_API_KEY is set for this test
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        vectorstore = create_vector_store(chunks, openai_key)
        qa_chain = build_conversational_rag_chain(vectorstore, openai_key)
        print("RAG pipeline components initialized for testing.")
    else:
        print("OPENAI_API_KEY not set. Cannot run RAG pipeline test.")