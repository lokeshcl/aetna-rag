# app.py
import os
from dotenv import load_dotenv
from pdf_processor import download_pdf, extract_text_pypdf
from rag_pipeline import get_text_chunks, create_vector_store, build_conversational_rag_chain
import openai # Import openai for specific error types

def main():
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY") 

    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found in .env file. Please set it.")
        return

    # PDF Configuration
    PDF_URL = "https://www.aetnabetterhealth.com/content/dam/aetna/medicaid/illinois/pdf/ABHIL_Member_Handbook.pdf"
    LOCAL_PDF_PATH = "ABHIL_Member_Handbook.pdf"

    # --- 1. Download PDF ---
    if not download_pdf(PDF_URL, LOCAL_PDF_PATH):
        print("Failed to download PDF. Exiting.")
        return

    # --- 2. Extract Text ---
    docs = extract_text_pypdf(LOCAL_PDF_PATH)
    if not docs:
        print("Failed to extract text from PDF. Exiting.")
        return
    print(f"Extracted {len(docs)} documents (pages) from PDF.")

    # --- 3. Get Text Chunks ---
    chunks = get_text_chunks(docs)
    if not chunks:
        print("Failed to create text chunks. Exiting.")
        return
    print(f"Created {len(chunks)} text chunks.")

    # --- 4. Create/Load Vector Store ---
    vectorstore = None
    try:
        vectorstore = create_vector_store(chunks, openai_api_key)
    except ValueError as ve: # Catch the specific ValueError for invalid API key
        print(f"Critical Error during vector store creation: {ve}. Please fix your API key.")
        return
    except Exception as e: # Catch any other general errors during vector store creation
        print(f"An unexpected error occurred during vector store creation: {e}")
        print("Please ensure your internet connection is stable and try again.")
        return

    if not vectorstore:
        print("Vector store could not be initialized. Exiting.")
        return

    # --- 5. Build Conversational RAG Chain ---
    qa_chain = None
    try:
        qa_chain = build_conversational_rag_chain(vectorstore, openai_api_key, cohere_api_key)
    except Exception as e:
        print(f"Error building conversational RAG chain: {e}")
        print("Please check your OpenAI API key and Cohere API key (if used) and ensure all dependencies are installed.")
        return
        
    if not qa_chain:
        print("RAG chain could not be initialized. Exiting.")
        return

    print("\n--- Chatbot Ready! Type 'exit' to quit ---")
    chat_history = []

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break

        try:
            # The actual call to OpenAI for answering the question
            result = qa_chain.invoke({"question": user_query, "chat_history": chat_history})
            full_answer_text = result["answer"]

            # --- Parsing the Answer and Reasoning (as discussed previously) ---
            concise_answer = ""
            reasoning = ""

            if "Concise Answer:" in full_answer_text:
                parts = full_answer_text.split("Reasoning:", 1)
                concise_answer_part = parts[0].replace("Concise Answer:", "").strip()
                concise_answer = concise_answer_part
                if len(parts) > 1:
                    reasoning = parts[1].strip()
            else:
                concise_answer = full_answer_text # Fallback

            print(f"Chatbot: {concise_answer}")
            if reasoning:
                print(f"\nReasoning: {reasoning}")

            # --- Processing and Displaying Source Documents ---
            source_documents = result.get("source_documents", [])
            if source_documents:
                print("\nSources:")
                displayed_sources = []
                seen_sources = set()

                for i, doc in enumerate(source_documents):
                    page_info = doc.metadata.get('page', 'N/A')
                    source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                    source_identifier = f"Page {page_info} of {source_file}"

                    if source_identifier not in seen_sources:
                        clean_content = doc.page_content[:150].replace('\n', ' ')
                        displayed_sources.append(f"- Source {len(displayed_sources)+1} ({source_identifier}): {clean_content}")
                        seen_sources.add(source_identifier)
                    
                    if len(displayed_sources) >= 3: # Display top 3 unique sources
                        break

                for src_text in displayed_sources:
                    print(src_text)
            
            chat_history.append((user_query, concise_answer))

        except (openai.APIError, requests.exceptions.RequestException) as e:
            print(f"Chatbot Error (API/Network): {e}")
            if "Incorrect API key" in str(e) or "authentication error" in str(e).lower():
                print("It looks like your OpenAI API key is invalid or has issues. Please verify it.")
            elif "rate limit" in str(e).lower():
                print("You might be hitting OpenAI's rate limits. Please wait a moment and try again.")
            else:
                print("A network or API communication error occurred. Please check your internet connection and try again.")
            # Do not add to chat history if there was an error
        except Exception as e:
            print(f"An unexpected error occurred during chat interaction: {e}")
            print("Please try your query again or restart the application.")

if __name__ == "__main__":
    main()