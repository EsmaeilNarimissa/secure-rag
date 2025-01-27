# Secure RAG

## Objective
This repository implements advanced Retrieval-Augmented Generation (RAG) pipelines optimized for scientific document retrieval and analysis. The codebase provides secure, cost-effective, and scalable solutions tailored for tasks requiring high precision and nuanced content retrieval. With support for multiple pipelines, it ensures flexibility and ease of use for diverse use cases.

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a `.env` file in your project directory with the following contents:
```plaintext
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_index_name
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_langchain_project_name
```

### 3. Install and Pull Ollama Models
1. Install Ollama from [Ollama website](https://ollama.com/docs/install).
2. Pull required models:
   ```bash
   ollama pull deepseek-r1:14b
   ollama pull llama3.2-vision
   ```

### 4. Run the Application
```bash
streamlit run app.py
```

---
## Ranked RAG Pipelines

The following pipelines are ranked based on their **security** and **cost-effectiveness**, ensuring users can choose configurations tailored to their needs.

### 1. Chroma DB-Based Pipelines with Ollama Models (Most Secure and Cost-Effective)
1. **Chroma DB + BGE Embedding + Ollama**  
   - **Description**: Optimal configuration leveraging advanced embedding techniques with secure, open-source models.
   - **Ollama Models**: `deepseek-r1:14b`, `llama3.2-vision`.

2. **Chroma DB + OAI text-embedding-3-small + Ollama**  
   - **Description**: Cost-free option with less advanced embeddings, maintaining robust splitting techniques.

### 2. Chroma DB-Based Pipelines with OpenAI Models
1. **Chroma DB + BGE Embedding + OpenAI**  
   - **Description**: Combines robust embedding techniques with OpenAI's versatile LLMs.
   - **OpenAI Models**: `GPT-4o-mini`, `GPT-4o`, `GPT-4 Turbo`.

2. **Chroma DB + OAI text-embedding-3-small + OpenAI**  
   - **Description**: Effective and affordable combination of OpenAI LLMs with simpler embeddings.

### 3. Pinecone-Based Pipelines (Cloud-Based and Less Secure)
1. **Pinecone + OAI text-embedding-3-small + OpenAI**  
   - **Description**: Relies on cloud infrastructure, making it less secure and slightly less cost-effective than Chroma DB-based pipelines.

---

## Pipeline Selection Guide

| **Criteria**               | **Recommended Pipeline**                        |
|-----------------------------|------------------------------------------------|
| High Security               | Chroma DB + BGE Embedding + Ollama             |
| Cost-Effective              | Chroma DB + OAI text-embedding-3-small + Ollama|
| High Precision              | Chroma DB + BGE Embedding + OpenAI             |
| Broad Context and Flexibility| Chroma DB + OpenAI text-embedding-3-small      |
| Cloud-Based Solution        | Pinecone + OpenAI text-embedding-3-small       |

### Pipeline Creation Instructions
The following examples demonstrate how to create and customize different RAG pipelines using the provided Jupyter notebooks:

1. **Chroma DB + BGE Embedding + Ollama Pipeline** (Recommended):
   - Best performance and cost-effective option
   - To create this pipeline, navigate to the `Ingestion.ipynb` notebook and run the following sections:
     - **"## 1.1 PDF Loading and Text Splitting with CharacterTextSplitter"**
     - **"#### 1.2.2.2 ChromaDB + BAAI's `bge-large-en-v1.5` + CTS"**

2. **Chroma DB + OpenAI Embedding + Ollama Pipeline** (Alternative):
   - Strong performance with OpenAI's embedding quality
   - Same process as above, but in `backend/main_cdb_Final.py`, uncomment the OpenAI embeddings option:
     ```python
     # Option 2: OpenAI Embeddings (Uncomment to use)
     # embeddings = OpenAIEmbeddings()
     ```
   - Note: Make sure to use the same embedding model during both ingestion and retrieval
   - Requires OpenAI API key but maintains local document storage

3. **Pinecone-Based Pipelines**:
   - To create this pipeline, go to the `Ingestion.ipynb` notebook and run:
     - **"## 1.1 PDF Loading and Text Splitting with CharacterTextSplitter"**
     - **"## 1.2.1 Pinecone + OpenAI Embedding"**

---

## Adding Your PDFs
Simply add your PDFs to the `./PDFs` directory. The pipeline will automatically process all files in this folder, ensuring seamless integration without additional setup.

---

## Research Findings

### 1. RecursiveCharacterTextSplitter (RCTS) vs. CharacterTextSplitter (CTS)
- **Key Finding**: CharacterTextSplitter (CTS) demonstrates superior performance in our specific use case, particularly in reducing hallucinations

- **CTS** (Current Implementation):
  - **Advantages**:
    - Significantly lower hallucination rates in responses
    - Better preservation of context within chunks
    - Improved retrieval accuracy due to cleaner text boundaries
  - **Technical Details**:
    - Produces **38 chunks** with larger sizes (average: **4255.6 characters**)
    - More natural text boundaries maintain document coherence
    - Ideal for maintaining complete contextual information

- **RCTS** (Previous Implementation):
  - **Technical Details**:
    - Generated **213 chunks** with smaller sizes (average: **902.9 characters**)
    - More granular splitting pattern
  - **Trade-offs**:
    - Better at capturing individual equations and figures
    - Higher risk of context fragmentation
    - Can lead to increased hallucination due to lost context

- **Implementation Decision**:
  - Switched from RCTS to CTS for better response quality
  - Benefits of complete context outweigh the advantages of granular splitting
  - Particularly effective for scientific document retrieval where context preservation is crucial

### 2. OpenAI vs. BGE Embeddings
- **OpenAI**:
  - Average similarity: **0.6096**; captures semantic diversity and nuance.
  - Suitable for exploratory queries and flexible contextual understanding.
- **BGE**:
  - Average similarity: **0.8041**; provides tight clustering and consistency.
  - Best for precision-driven retrieval in structured data (e.g., equations, tables).

### Summary Recommendations
1. Use **CTS** with **BGE** for high-precision scientific content retrieval.
2. Combine **CTS** with **OpenAI** for broader, exploratory workflows.

---

## System Requirements and Ollama Models
- For the `deepseek-r1:14b` model, ensure the following hardware configuration:
  - **GPU**: Minimum 16 GB VRAM.
  - **System RAM**: At least 32 GB.
  - **CPU**: Modern 8-core processor.
- Ollama integrates seamlessly with open-source models, providing high efficiency for on-device inference. Follow installation instructions [here](https://ollama.com/docs/install).