# Question Answering RAG Chatbot: OpenRouter Embeddings + Groq Inference
An intelligent Retrieval-Augmented Generation (RAG) chatbot web application built using Streamlit and LangChain (v0.3+).

This project features a unique hybrid API architecture optimized for efficiency: it uses OpenRouter's free-tier multimodal embedding model to convert documents into vector data, and routes high-speed conversation synthesis through Groq's LPU infrastructure using the openai/gpt-oss-120b open-weight model.

## Key Features
Dual Ingestion Channels: * Document Upload: Supports extraction and context chunking from local .pdf, .docx, and .txt files.

Wikipedia Integration: Dynamically fetches context updates from real-time Wikipedia search queries directly into the knowledge graph.

Context-Aware Dialog Handling: Uses modern LangChain LCEL structures (create_history_aware_retriever) to rewrite user prompts into standalone search vectors based on conversational history.

Smarter Vector Processing: Employs a custom asynchronous implementation wrapper mapping text chunks into OpenRouter's specific nested formatting expectations.

Production Security Ready: No exposed API inputs on the UI canvas. Sensitive tokens are fully isolated utilizing Streamlit's structural native secrets.toml layer.

## Tech Stack & Architecture
- Frontend UI: Streamlit

- Orchestration Framework: LangChain (Modern LCEL Architecture)

- Text Embedding Generation: OpenRouter API (nvidia/llama-nemotron-embed-vl-1b-v2:free)

- LLM Core Engine: Groq Cloud API (openai/gpt-oss-120b)

- Vector Storage Database: ChromaDB

## Data Pipeline Overview
Ingestion: Files or Wikipedia pages are loaded and chunked via RecursiveCharacterTextSplitter.

Vectorization: Chunked text is transformed into high-dimensional vectors via OpenRouter's free embedding engine and initialized into an in-memory Chroma database.

Retrieval Optimization: When a user submits an execution query, a Groq-powered parsing pass uses conversation history to clarify ambiguous user questions into clean, searchable terms.

Synthesis: Relevant vector fragments are retrieved from Chroma, structured alongside the conversational history, and synthesized into a precise response by the Groq core language model.

## Web App SS
<img width="1884" height="865" alt="image" src="https://github.com/user-attachments/assets/5c2e6a14-858f-4a18-8d86-abba4172ddf4" />
<img width="1890" height="855" alt="image" src="https://github.com/user-attachments/assets/b77f8d6f-286e-41cf-b74a-74d9d067c0b6" />


