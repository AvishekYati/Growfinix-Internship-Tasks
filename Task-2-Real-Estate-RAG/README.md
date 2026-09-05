# Growfinix Real Estate RAG System

## Growfinix Technology Internship - Task 2

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG)
system for intelligent hotel property search.

The system allows users to describe their hotel requirements
using natural language. Relevant hotel properties are retrieved
from a vector database using semantic similarity search and
summarized using a Large Language Model.

## Problem Statement

Traditional keyword-based hotel search may fail when the user
expresses a requirement using different wording from the dataset.

For example:

"Show me modern hotels with large balconies."

The RAG system understands the semantic meaning of the request
and retrieves relevant hotel properties.

## Objectives

- Process a large hotel dataset
- Clean hotel descriptions and facilities
- Generate semantic embeddings
- Store embeddings in ChromaDB
- Retrieve relevant properties
- Use an LLM to generate grounded recommendations
- Support natural-language hotel queries

## Dataset

The project uses a large hotel dataset containing more than
one million hotel records.

The dataset contains fields such as:

- HotelCode
- HotelName
- HotelRating
- Address
- Attractions
- Description
- HotelFacilities
- Map
- cityName
- countyName

## Technologies

- Python
- Pandas
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Groq API
- Qwen LLM
- Jupyter Notebook

## Architecture

Dataset
↓
Data Cleaning
↓
Hotel Document Creation
↓
Hugging Face Embeddings
↓
ChromaDB
↓
Semantic Retrieval
↓
Retrieved Hotel Context
↓
Groq LLM
↓
Final Recommendation

## How RAG Works

1. The hotel dataset is loaded.
2. Hotel descriptions and facilities are cleaned.
3. Hotel information is converted into searchable documents.
4. Documents are converted into vector embeddings.
5. Embeddings are stored in ChromaDB.
6. A user submits a natural-language query.
7. The query is compared against the stored vectors.
8. The most relevant hotel properties are retrieved.
9. Retrieved information is provided to the LLM.
10. The LLM generates a grounded recommendation.

## Example Query

Show me modern hotels with large balconies,
beautiful views and swimming pools.

## Example Use Cases

- Family hotel search
- Luxury hotel search
- Hotels with swimming pools
- Hotels with balconies
- Hotels with spa facilities
- Hotels near attractions
- Nature-focused hotel search

## Project Structure

```text
Task-2-Real-Estate-RAG/
├── data/
├── notebooks/
├── screenshots/
├── chroma_db/
├── README.md
├── requirements.txt
├── .gitignore
└── .env