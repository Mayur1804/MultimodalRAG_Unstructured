# src/database.py
import json
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from src.config import EMBEDDING_MODEL, LLM_MODEL, DB_DIR

def create_vector_store(documents):
    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_DIR, 
        collection_metadata={"hnsw:space": "cosine"}
    )

def generate_final_answer(chunks, query):
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        prompt_text = f"Based on the following documents, please answer this question: {query}\n\nCONTENT TO ANALYZE:\n"
        
        # We need to build the content carefully
        for i, chunk in enumerate(chunks):
            prompt_text += f"\n--- Document {i+1} ---\n"
            data = json.loads(chunk.metadata.get("original_content", "{}"))
            
            if data.get("raw_text"):
                prompt_text += f"TEXT:\n{data['raw_text']}\n"
            
            if data.get("tables_html"):
                prompt_text += "TABLES:\n"
                for table in data["tables_html"]:
                    prompt_text += f"{table}\n"
        
        prompt_text += "\nPlease provide a clear, comprehensive answer using the text and tables provided. If you can't find it, say you don't know.\n\nANSWER:"
        
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images
        for chunk in chunks:
            data = json.loads(chunk.metadata.get("original_content", "{}"))
            for img_b64 in data.get("images_base64", []):
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
                
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as e:
        return f"Error in answer generation: {str(e)}"