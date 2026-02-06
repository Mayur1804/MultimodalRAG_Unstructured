# src/processing.py
import json
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from src.config import LLM_MODEL

def separate_content_types(chunk):
    content_data = {'text': chunk.text, 'tables': [], 'images': [], 'types': ['text']}
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            elif element_type == 'Image' and hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                content_data['types'].append('image')
                content_data['images'].append(element.metadata.image_base64)
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        prompt_text = f"""You are creating a searchable description for document content retrieval.
        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}
        """
        
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += """
        YOUR TASK:
        Generate a comprehensive, searchable description that covers:
        1. Key facts, numbers, and data points from text and tables
        2. Main topics and concepts discussed  
        3. Questions this content could answer
        4. Visual content analysis (charts, diagrams, patterns in images)
        5. Alternative search terms users might use
        Make it detailed and searchable - prioritize findability over brevity.
        SEARCHABLE DESCRIPTION:"""

        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as e:
        print(f"Summary Error: {e}")
        return text

def summarise_chunks(chunks):
    print("🧠 Processing chunks with AI Summaries...")
    langchain_documents = []
    for i, chunk in enumerate(chunks):
        data = separate_content_types(chunk)
        # Match your original logic: Only summarize if there are tables/images
        if data['tables'] or data['images']:
            print(f"   → Creating AI summary for mixed content (Chunk {i+1})...")
            enhanced_content = create_ai_enhanced_summary(data['text'], data['tables'], data['images'])
        else:
            enhanced_content = data['text']
        
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": data['text'],
                    "tables_html": data['tables'],
                    "images_base64": data['images']
                })
            }
        )
        langchain_documents.append(doc)
    return langchain_documents