# main.py
import sys
import os

# Path fix to ensure 'src' is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from src.ingest import partition_document, create_chunks_by_title
from src.processing import summarise_chunks
from src.database import create_vector_store, generate_final_answer
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.config import DB_DIR, EMBEDDING_MODEL

load_dotenv()

def ingest_data():
    """Runs the full ingestion pipeline and saves to disk"""
    print(" Starting Ingestion Pipeline...")
    elements = partition_document()
    chunks = create_chunks_by_title(elements)
    summarized_docs = summarise_chunks(chunks)
    db = create_vector_store(summarized_docs)
    print(f"✅ Ingestion complete. Database saved at: {DB_DIR}")
    return db

def query_mode():
    """Loads existing database and allows for querying"""
    if not os.path.exists(DB_DIR):
        print(f" Error: Database folder '{DB_DIR}' not found. Please run --ingest first.")
        return

    print(" Loading existing Vector Store...")
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    while True:
        query = input("\n🔎 Enter your question (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
        
        print(" Searching and generating answer...")
        results = retriever.invoke(query)
        answer = generate_final_answer(results, query)
        
        print(f"\n ANSWER:\n{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    # Simple CLI logic
    if len(sys.argv) > 1:
        if sys.argv[1] == "--ingest":
            ingest_data()
        elif sys.argv[1] == "--query":
            query_mode()
    else:
        # Default behavior: Ask user what to do
        choice = input("Do you want to (1) Ingest new data or (2) Query existing data? Enter 1 or 2: ")
        if choice == '1':
            ingest_data()
        elif choice == '2':
            query_mode()
        else:
            print("Invalid choice. Use --ingest or --query as arguments.")