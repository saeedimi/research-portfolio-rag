# ResearchGPT: Retrieval-Augmented Portfolio Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot that enables natural language interaction with my research portfolio, including my PhD thesis, one peer-reviewed publication, CV, and machine learning GitHub projects.

The system combines **hybrid retrieval (FAISS + BM25)** with **cross-encoder reranking** and a **Large Language Model (LLM)** to generate grounded, citation-supported responses.

> 🚀 **Live Demo:**[Open ResearchGPT Live Demo](https://huggingface.co/spaces/Miladsaeedi70/research-portfolio-rag)

> 💻 **Source Code:** https://github.com/saeedimi/research-portfolio-rag

---

# Overview

ResearchGPT is an end-to-end Retrieval-Augmented Generation (RAG) system built to showcase my research and technical portfolio through a conversational interface.

Rather than relying solely on an LLM's internal knowledge, the application retrieves relevant information from a curated knowledge base before generating responses. This helps improve factual grounding while allowing users to explore my research, publications, technical projects, and experience using natural language.

The knowledge base currently includes:

- PhD thesis
- One peer-reviewed publication
- Academic CV
- Machine learning GitHub repositories
- Project documentation
- Jupyter notebooks

---

# Features

- Hybrid retrieval using **FAISS** and **BM25**
- Cross-encoder reranking for improved retrieval quality
- Conversational memory for follow-up questions
- Query rewriting for conversational search
- Technical abbreviation expansion (GAN, CNN, LLM, LoRA, GRPO, etc.)
- Grounded responses with inline citations
- Automatic source attribution
- Interactive Gradio web interface
- Ready for deployment on Hugging Face Spaces

---

# Knowledge Base

| Source | Description |
|---------|-------------|
| PhD Thesis | Doctoral research on mobile air pollution mapping and machine learning |
| Peer-reviewed Publication | *Urban Air Pollution Data Collection, Mapping, and Prediction Using Mobile Sensors Installed on Courier Trucks* |
| Academic CV | Education, research experience, technical skills, publications |
| GitHub Projects | Machine learning, computer vision, deep learning, generative AI and LLM projects |
| Jupyter Notebooks | Educational implementations and experiments |
| README Files | Project documentation and technical descriptions |

---

# System Architecture

```text
                Documents
                     │
                     ▼
            Document Loaders
                     │
                     ▼
      Cleaning & Metadata Extraction
                     │
                     ▼
                Chunking
                     │
                     ▼
             BGE Embeddings
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
      FAISS Index          BM25 Index
          └──────────┬──────────┘
                     ▼
            Hybrid Retrieval
                     │
                     ▼
      Cross-Encoder Reranker
                     │
                     ▼
            Prompt Construction
                     │
                     ▼
              Gemini / LLM
                     │
                     ▼
          Grounded AI Response
```

---

# Retrieval Pipeline

The system follows these stages:

1. Load portfolio documents
2. Extract text and metadata
3. Remove irrelevant pages
4. Chunk documents
5. Generate dense embeddings
6. Build a FAISS vector index
7. Build a BM25 sparse index
8. Perform hybrid retrieval
9. Rerank retrieved candidates with a cross-encoder
10. Construct a grounded prompt
11. Generate a citation-supported response

---

# Repository Structure

```text
research-portfolio-rag/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── artifacts/
│   ├── research_index.faiss
│   ├── chunks.json
│   ├── retrieval_texts.json
│   └── index_config.json
│
├── notebooks/
│
├── assets/
│
└── evaluation/
```

---

# Running Locally

```bash
git clone https://github.com/saeedimi/research-portfolio-rag.git

cd research-portfolio-rag

pip install -r requirements.txt

export GEMINI_API_KEY=YOUR_API_KEY

python app.py
```

Open:

```
http://localhost:7860
```

---

# Example Questions

- Summarize Milad Saeedi's research.
- What are the main contributions of his PhD thesis?
- Summarize his GitHub GAN projects.
- Which computer vision projects has he completed?
- What experience does he have with LoRA and GRPO?
- Explain his geospatial machine learning research.
- What publications are included in the knowledge base?

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Dense Retrieval | Sentence Transformers (BGE) |
| Vector Database | FAISS |
| Sparse Retrieval | BM25 |
| Reranker | BGE Cross-Encoder |
| LLM | Gemini |
| User Interface | Gradio |
| Deployment | Hugging Face Spaces |

---

# Future Improvements

- Reciprocal Rank Fusion (RRF)
- Multi-query retrieval
- Improved notebook parsing
- Image retrieval
- Automatic evaluation pipeline
- Streaming responses
- Persistent conversation history

---

# Acknowledgements

This project was developed as part of my exploration of modern Retrieval-Augmented Generation (RAG), semantic search, and LLM applications for scientific research and technical portfolios.

---

# License

This project is released under the MIT License.