# Task 2: Real Estate & Hotel Recommendation RAG System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-darkgreen.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Groq](https://img.shields.io/badge/LLM%20Inference-Groq-orange.svg)](https://groq.com/)

An enterprise-scale **Retrieval-Augmented Generation (RAG)** pipeline designed for intelligent property and hotel discovery. The system ingests a catalog of over 1,000,000 hotel listings, indexes dense semantic vector representations in **ChromaDB**, retrieves top-$K$ candidates via cosine similarity, and synthesizes grounded, hallucination-free recommendations using **Groq** high-speed LLM inference.

Developed as part of the **Growfinix Technology Data Science Internship (Month 3: Advanced AI, Deep Learning & LLMs)**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [RAG Architecture](#-rag-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset Specifications](#-dataset-specifications)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Execution & Workflow](#-execution--workflow)
- [Prompt Engineering & Guardrails](#-prompt-engineering--guardrails)
- [Sample Queries & Evaluation](#-sample-queries--evaluation)
- [Limitations & Scalability](#-limitations--scalability)

---

## 📖 Overview
Standard keyword search fails when users describe hotel requirements conceptually (e.g., *"quiet boutique stays with mountain views, infinity pool, and breakfast near the city center"*).

This project addresses this by:
1. Converting structured and unstructured property records (hotel descriptions, facilities, locations, ratings, and attractions) into a unified narrative representation.
2. Embedding textual chunks using `sentence-transformers/all-MiniLM-L6-v2` into a 384-dimensional vector space.
3. Persisting vectors in a local vector database (**ChromaDB**).
4. Retrieving relevant properties based on semantic similarity rather than exact keywords.
5. Augmenting a strictly conditioned LLM prompt (**ChatGroq** with Qwen 2.5/3.6 models) to eliminate hallucinations and produce structured, verifiable recommendations.

---

## ⚡ Key Features
* **Large-Scale Data Handling:** Data processing pipeline tested on an international hotel dataset of **1,010,033 rows**, featuring deduplication and text normalization.
* **Deterministic Embeddings:** Normalized 384-dimensional embeddings for fast cosine similarity search.
* **Persistent Vector Store:** ChromaDB indexing with batch ingestion and disk persistence.
* **Grounded Retrieval Prompts:** Strict system constraints instructing the model to rely exclusively on retrieved context and report missing attributes explicitly.
* **Interactive CLI Terminal:** Real-time query-response console with document source attribution and location mapping.

---

## 🏗 RAG Architecture
```text
[Raw Dataset: hotels.csv (1M+ Records)]
│
▼
[Data Cleaning & Normalization (Regex)]
│
▼
[Structured Property Representation (Document Builder)]
│
▼
[Chunking: RecursiveCharacterTextSplitter (1000/150)]
│
▼
[Dense Vectors: sentence-transformers/all-MiniLM-L6-v2]
│
▼
[Local Vector Store: ChromaDB]
│
──────────────────────────────────┼──────────────────────────────────
                                  │
[User Query]                      │
│                                 │
▼                                 ▼
[Query Vector] ───► [Top-K Cosine Similarity Retrieval]
                                  │
                                  ▼
                      [Formatted Context Block]
                                  │
                                  ▼
              [Grounded Prompt Template + Groq LLM]
                                  │
                                  ▼
        [Structured AI Recommendation + Source URLs]
