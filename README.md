# Growfinix Technology Data Science Internship

## Advanced AI, Deep Learning and LLM Projects

This repository contains projects completed as part of the **Growfinix Technology Data Science Internship — Month 3: Advanced AI, Deep Learning and LLMs**.

The projects demonstrate practical implementation of:

* Data Science
* Machine Learning
* Deep Learning
* Computer Vision
* Natural Language Processing
* Generative AI
* Retrieval-Augmented Generation
* Large Language Models
* MLOps

---

## Internship Information

**Organization:** Growfinix Technology

**Program:** Data Science Internship

**Module:** Month 3 - Advanced AI, Deep Learning and LLMs

**Repository:** Growfinix-Internship-Tasks

---

# Completed Tasks

## Task 1 - Automated Image/Video Metadata Extraction

A computer vision pipeline that analyzes images and video frames and automatically extracts useful metadata.

### Key Capabilities

* Image analysis
* Video frame analysis
* Object detection integration
* Object confidence scoring
* Bounding box detection
* Dominant color extraction
* RGB color analysis
* HEX color representation
* Brightness analysis
* Lighting analysis
* Contrast analysis
* Image quality analysis
* Automatic metadata generation
* Annotated output generation

### Technologies

* Python
* OpenCV
* PyTorch
* YOLO / object detection
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Pillow
* Jupyter Notebook

### Project

[Task 1 - Automated Image/Video Metadata Extraction](./Task-1-Automated-Media-Metadata/)

---

# Task 2 - Real Estate RAG System

An intelligent property-search system based on **Retrieval-Augmented Generation (RAG)**.

The system accepts natural-language property queries such as:

> "Show me modern properties with large balconies"

It converts property descriptions into vector embeddings, stores them in a vector database, retrieves the most relevant properties using semantic similarity, and generates a grounded response using an LLM.

### Key Capabilities

* Property dataset processing
* Property description processing
* LangChain Document creation
* Hugging Face embeddings
* Semantic similarity search
* Vector database storage
* ChromaDB integration
* Top-K property retrieval
* Natural-language property search
* Retrieved-context generation
* Grounded LLM responses
* Property source identification

### RAG Pipeline

```text
Property Dataset
       ↓
Property Descriptions
       ↓
LangChain Documents
       ↓
Hugging Face Embeddings
       ↓
ChromaDB Vector Database
       ↓
Natural-Language Query
       ↓
Query Embedding
       ↓
Similarity Search
       ↓
Top-K Relevant Properties
       ↓
Retrieved Context
       ↓
LangChain Prompt
       ↓
LLM
       ↓
Grounded Property Recommendation
```

### Technologies

* Python
* Pandas
* LangChain
* Hugging Face
* Sentence Transformers
* ChromaDB
* OpenAI / LLM API
* Jupyter Notebook

### Project

[Task 2 - Real Estate RAG](./Task-2-Real-Estate-RAG/)

---

# Repository Structure

```text
Growfinix-Internship-Tasks/
│
├── README.md
│
├── Task-1-Automated-Media-Metadata/
│   ├── README.md
│   ├── .gitignore
│   ├── requirements.txt
│   ├── Task1_Automated_Media_Metadata.ipynb
│   ├── input/
│   ├── output/
│   └── screenshots/
│
└── Task-2-Real-Estate-RAG/
    ├── README.md
    ├── .gitignore
    ├── requirements.txt
    ├── Growfinix_Task_2_Real_Estate_RAG.ipynb
    ├── app.py
    ├── data/
    ├── chroma_db/
    └── screenshots/
```

---

# Technologies Used

| Area             | Technologies            |
| ---------------- | ----------------------- |
| Programming      | Python                  |
| Data Processing  | Pandas, NumPy           |
| Machine Learning | Scikit-learn            |
| Computer Vision  | OpenCV, PyTorch         |
| NLP / LLM        | LangChain, Hugging Face |
| RAG              | LangChain, ChromaDB     |
| Embeddings       | Sentence Transformers   |
| Visualization    | Matplotlib              |
| Development      | Jupyter Notebook        |
| Version Control  | Git, GitHub             |

---

# Project Documentation

Each completed task contains its own README with:

* Project description
* Objectives
* Technologies used
* Implementation details
* Project structure
* Installation instructions
* Execution instructions
* Results
* Screenshots
* Limitations
* Future improvements

---

# Internship Submission

The projects are organized to support the internship requirements for:

* GitHub project submission
* Technical documentation
* Screenshots and execution evidence
* Project demonstration
* LinkedIn project presentation

Growfinix requires completed tasks to include GitHub documentation and project evidence, along with a demonstration video and LinkedIn post for completed tasks.

---

# Author

**Avishek Kumar Yati**

Data Science | Machine Learning | Generative AI | Computer Vision | MLOps
