# Multimodal RAG Pipeline: Visual & Tabular Intelligence

A sophisticated, local-first Retrieval-Augmented Generation (RAG) pipeline designed to handle complex documents. Unlike standard RAG systems that only process raw text, this pipeline extracts, understands, and retrieves information from **tables** and **images** using Vision LLMs.



## 🚀 Key Features

* **Intelligent PDF Partitioning:** Uses `unstructured` with a High-Res strategy to identify and extract text, tables, and image blocks.
* **Vision-Augmented Indexing:** Automatically generates searchable AI descriptions for images/charts using **Gemma 3** via Ollama.
* **Structured Table Recovery:** Extracts tables as HTML to preserve data relationships that standard text-chunking loses.
* **Local-Only Stack:** Runs entirely on your machine using **Ollama** (Gemma 3 & EmbeddingGemma) and **ChromaDB** for maximum data privacy.
* **Multimodal Reasoning:** The final answer generation "looks" at retrieved images (base64) and HTML tables simultaneously to provide factually grounded answers.

## 🛠️ Tech Stack

* **Orchestration:** LangChain
* **LLMs:** Gemma 3 (4b) & EmbeddingGemma (via Ollama)
* **Vector Database:** ChromaDB
* **Document Processing:** Unstructured.io

## 📖 How It Works

### 1. Ingestion & Analysis
The pipeline partitions the PDF into atomic elements. When an image or table is identified, it is processed by a Vision LLM to create an **AI-Enhanced Summary**. This summary captures key facts, visual patterns, and data points, making non-text content fully searchable.

### 2. Contextual Chunking
Instead of fixed character counts, chunks are created using a `title` based strategy. This ensures that tables and images stay attached to their relevant sections, preserving document logic.

### 3. Multimodal Retrieval & Generation
Upon a user query:
1.  **Retrieval:** Relevant chunks are pulled from ChromaDB based on the AI-enhanced summaries.
2.  **Reconstruction:** The system fetches the original Base64 images and HTML tables stored in the metadata.
3.  **Synthesis:** A multimodal prompt is sent to the LLM, containing the raw text, rendered tables, and actual images to generate a comprehensive answer.

## 💻 Quick Start

### Prerequisites
* Install [Ollama](https://ollama.com/)
* Pull models: `ollama pull gemma3:4b` and `ollama pull embeddinggemma`

### Installation
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your-username/multimodal-rag-pipeline.git](https://github.com/your-username/multimodal-rag-pipeline.git)
    cd multimodal-rag-pipeline
    ```
2.  **Install dependencies:**
    ```bash
    pip install unstructured[pdf] langchain-ollama langchain-groq chromadb python-dotenv
    ```
3.  **Run the pipeline:**
    Place your PDF in `./docs/` and run the script to build your local vector store and query your data.

## 📝 Example Query
**Query:** *"What are the two main components of the Transformer architecture?"* **Output:** The system retrieves the architecture diagram and the description text to explain the Encoder and Decoder blocks in detail.
