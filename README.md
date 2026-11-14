# Growfinix Technology – Data Science Internship
## Month 3: Advanced AI, Deep Learning & Large Language Models

Welcome to my **Growfinix Technology Data Science Internship – Month 3 Project Portfolio**.

This repository contains five projects developed as part of the Advanced AI, Deep Learning and LLM internship tasks. The projects cover Computer Vision, Natural Language Processing, Retrieval-Augmented Generation (RAG), Unsupervised Machine Learning, Large Language Model Fine-Tuning and Machine Learning Operations (MLOps).

The objective of this portfolio is to demonstrate practical implementation, experimentation, problem-solving and the application of modern AI and Machine Learning technologies.

---

## 👨‍💻 About Me

**Name:** Avishek Kumar Yati  
**Program:** Data Science Internship – Month 3  
**Organization:** Growfinix Technology  
**Focus Areas:** Data Science | Machine Learning | Artificial Intelligence | NLP | Generative AI | Deep Learning | MLOps

---

# 📌 Project Portfolio

| Task | Project | Domain | Main Technologies |
|---|---|---|---|
| Task 1 | Automated Image/Video Metadata Extraction | Computer Vision | Python, OpenCV, NumPy, PyTorch/Torchvision |
| Task 2 | Real Estate RAG System | Generative AI / NLP | Python, Pandas, Scikit-learn, RAG, LangChain integration |
| Task 3 | Customer Segmentation | Unsupervised Learning | Python, Pandas, Scikit-learn, K-Means, DBSCAN, Plotly |
| Task 4 | Tour Enquiry LLM Fine-Tuning | Deep Learning / LLM | Python, PyTorch, Hugging Face, Transformers, PEFT, LoRA |
| Task 5 | IPL MLOps Pipeline | Machine Learning Operations | Python, Scikit-learn, FastAPI, MLflow integration, Docker, GitHub Actions |

---

# 🚀 Task 1 – Automated Image/Video Metadata Extraction

**Project Folder:** `Task-1-Automated-Media-Metadata`

### Objective

Develop a Computer Vision pipeline that analyzes images and video frames, extracts useful visual information and automatically generates structured media metadata.

### Key Features

- Image and video input validation.
- Video frame extraction and sampling.
- Image resolution and dimension analysis.
- Dominant colour palette extraction.
- RGB and HEX colour information.
- Brightness and contrast analysis.
- Lighting condition classification.
- Automatic metadata and tag generation.
- JSON metadata export.
- Annotated image and video-frame outputs.
- Optional PyTorch/Torchvision object-detection integration.

### Technologies

- Python
- OpenCV
- NumPy
- JSON
- PyTorch/Torchvision integration

### Project Workflow

```text
Input Image / Video
        |
        v
Media Validation
        |
        v
Image Processing / Video Frame Extraction
        |
        v
Colour Palette Extraction
        |
        v
Brightness & Contrast Analysis
        |
        v
Lighting Classification
        |
        v
Metadata & Tag Generation
        |
        v
JSON Output + Annotated Media
```

### Demonstrated Results

The executed OpenCV-based demonstration processed a synthetic image and video.

Recorded image output:
- Resolution: 960 × 540
- Brightness: 101.582
- Contrast: 34.463
- Lighting classification: Balanced

Recorded video output:
- Resolution: 960 × 540
- Frame rate: 15 FPS
- Total frames: 45
- Duration: 3 seconds

The demonstrated execution used the OpenCV heuristic fallback. Pretrained object-detection weights were not available in that environment, so no object-detection accuracy claim is made.

### Run the Project

```bash
cd Task-1-Automated-Media-Metadata

python run.py

pytest -q tests
```

---

# 🏡 Task 2 – Real Estate RAG (Retrieval-Augmented Generation) System

**Project Folder:** `Task-2-Real-Estate-RAG`

### Objective

Develop an intelligent real-estate search backend that accepts natural-language queries, retrieves relevant property listings and generates context-based answers with identifiable property sources.

The system is designed to make property discovery easier by allowing users to search using everyday language instead of relying only on traditional keyword-based searches.

### Example Query

> Show me modern properties with large balconies.

### Key Features

- Synthetic real-estate property dataset generation.
- Property description processing.
- Natural-language query handling.
- Similarity-based property retrieval.
- Top-K property ranking.
- Retrieval of relevant property descriptions.
- Grounded answer generation.
- Property ID and source identification.
- Multiple query demonstrations.
- Optional ChromaDB vector-store integration.
- Optional Hugging Face embedding integration.
- Optional LangChain and OpenAI LLM integration.

### Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF retrieval
- NLP
- Retrieval-Augmented Generation
- LangChain integration
- ChromaDB integration
- Hugging Face / OpenAI integration

### Project Workflow

```text
Property Dataset
       |
       v
Property Description Processing
       |
       v
Document Preparation
       |
       v
Similarity-Based Retrieval
       |
       v
Natural-Language Query
       |
       v
Top-K Relevant Properties
       |
       v
Retrieved Context
       |
       v
Grounded Answer Generation
       |
       v
Property IDs and Sources
```

### Demonstrated Results

The offline retrieval demonstration successfully executed four natural-language queries.

For the example query:

> Show me modern properties with large balconies.

The recorded top-three retrieved property IDs were:

- P001
- P005
- P003

The system generated a grounded response referencing the retrieved property IDs.

**Implementation note:** The offline TF-IDF retrieval and deterministic answer-generation fallback were demonstrated. ChromaDB, external embedding-model and external LLM integrations are provided as optional extensions and were not confirmed as executed in the recorded environment.

### Run the Project

```bash
cd Task-2-Real-Estate-RAG

python app.py

pytest -q tests
```

The Jupyter notebook is also available for interactive execution and demonstration.

---

# 📊 Task 3 – Customer Segmentation Using Unsupervised Learning

**Project Folder:** `Task-3-Customer-Segmentation`

### Objective

Discover hidden customer segments from unlabeled customer behaviour data using unsupervised machine learning algorithms.

The project explores customer patterns and groups customers with similar behavioural characteristics to support data-driven marketing strategies.

### Key Features

- Synthetic customer dataset generation.
- Exploratory Data Analysis.
- Data cleaning and preprocessing.
- Feature selection and standardization.
- K-Means clustering.
- DBSCAN clustering.
- Elbow Method analysis.
- Silhouette Score evaluation.
- PCA-based dimensionality reduction.
- Interactive 3D Plotly visualization.
- Customer segment profiling.
- Marketing strategy recommendations.

### Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- K-Means
- DBSCAN
- PCA
- Matplotlib
- Plotly
- Jupyter Notebook

### Project Workflow

```text
Customer Behaviour Dataset
          |
          v
Exploratory Data Analysis
          |
          v
Data Preprocessing
          |
          v
Feature Standardization
          |
          v
K-Means Clustering
          |
          v
DBSCAN Clustering
          |
          v
Model Evaluation
          |
          v
PCA Dimensionality Reduction
          |
          v
Interactive 3D Visualization
          |
          v
Customer Segment Profiling
          |
          v
Marketing Recommendations
```

### Demonstrated Results

The executed experiment used 360 synthetic customer records generated with a fixed random seed.

Recorded K-Means results:
- Selected K: 2
- Silhouette Score: 0.4602

Recorded DBSCAN results:
- Non-noise clusters: 4
- Noise points: 310

### Generated Outputs

- Customer cluster assignments.
- Segment profile reports.
- Elbow Method visualization.
- Silhouette analysis visualization.
- DBSCAN clustering visualization.
- Interactive Plotly 3D visualization.
- Marketing strategy recommendations.

### Run the Project

```bash
cd Task-3-Customer-Segmentation

python src/generate_data.py

python -c "from src.segment import run; print(run('data/customers.csv','outputs'))"

pytest -q tests
```

Open the generated interactive visualization:

```text
outputs/customer_segments_3d.html
```

---

# 🤖 Task 4 – Fine-Tuning an Open-Source LLM Using LoRA

**Project Folder:** `Task-4-Tour-Enquiry-LLM-FineTuning`

### Objective

Prepare a parameter-efficient fine-tuning pipeline for an open-source language model to generate customer-service and tour-enquiry responses using a specific company-oriented communication style.

The project explores how a lightweight Large Language Model can be adapted to a specialized domain using Low-Rank Adaptation (LoRA).

### Key Features

- Synthetic tour-enquiry instruction dataset.
- Customer-service response preparation.
- Open-source language model configuration.
- Hugging Face Transformers integration.
- PyTorch training pipeline.
- Parameter-Efficient Fine-Tuning using PEFT.
- LoRA configuration.
- Training configuration and reproducibility settings.
- Dataset and configuration validation.
- Model training entry point.

### Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- LoRA
- TinyLlama 1.1B Chat

### Project Workflow

```text
Tour Enquiry Dataset
         |
         v
Instruction Data Preparation
         |
         v
Tokenization
         |
         v
Base Language Model
         |
         v
LoRA Configuration
         |
         v
Parameter-Efficient Fine-Tuning
         |
         v
Model Evaluation
         |
         v
Tour Enquiry Response Generation
```

### Model Configuration

| Parameter | Configuration |
|---|---|
| Base Model | TinyLlama 1.1B Chat |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.05 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Training Epochs | 1 |
| Maximum Sequence Length | 512 |
| Random Seed | 42 |

### Validation Status

The project includes 23 synthetic customer-service and tour-enquiry instruction examples.

The dataset and configuration smoke test was successfully completed.

Full model fine-tuning was not executed in the recorded environment because the required Transformers/PEFT dependencies and model download were unavailable.

Therefore, no completed training loss or before-and-after model performance is claimed.

### Run the Validation

```bash
cd Task-4-Tour-Enquiry-LLM-FineTuning

python src/smoke_test.py

pytest -q tests
```

Full LoRA training requires the relevant dependencies, model access and a suitable training environment.

---

# 🏏 Task 5 – IPL End-to-End MLOps Pipeline

**Project Folder:** `Task-5-IPL-MLOps-Pipeline`

### Objective

Develop an automated Machine Learning Operations workflow for IPL prediction that combines data processing, model training, evaluation, API development, experiment-tracking integration, containerization and CI/CD automation.

The intended workflow supports model retraining when new data is pushed to the repository, records model metrics and prepares an updated prediction API for containerized deployment.

### Key Features

- IPL match-state dataset preparation.
- Data validation and preprocessing.
- Feature engineering.
- Machine learning model training.
- Model evaluation.
- Accuracy and classification metrics.
- Model serialization using Joblib.
- FastAPI prediction service.
- Health-check endpoint.
- Model-information endpoint.
- Prediction endpoint.
- MLflow experiment-tracking integration.
- Docker containerization.
- GitHub Actions automation workflow.

### Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- MLflow integration
- Docker
- GitHub Actions

### Project Workflow

```text
IPL Match-State Data
         |
         v
Data Validation
         |
         v
Feature Engineering
         |
         v
Model Training
         |
         v
Model Evaluation
         |
         v
MLflow Tracking Integration
         |
         v
Model Serialization
         |
         v
FastAPI Prediction Service
         |
         v
Docker Containerization
         |
         v
GitHub Actions Automation
```

### Demonstrated Results

The recorded experiment used 2,500 synthetic IPL match-state records.

Recorded model evaluation:

| Metric | Result |
|---|---:|
| Accuracy | 0.8420 |
| Precision | 0.7784 |
| Recall | 0.7558 |
| F1 Score | 0.7670 |
| ROC-AUC | 0.9153 |
| Training Records | 2,000 |
| Testing Records | 500 |

### API Endpoints

| Endpoint | Purpose |
|---|---|
| `/health` | Check API health |
| `/model-info` | Retrieve model information |
| `/predict` | Generate an IPL prediction |

The FastAPI endpoints were exercised locally using TestClient.

### MLOps Execution Status

The model-training and API demonstration were completed in the recorded environment.

The MLflow integration, Docker configuration and GitHub Actions workflow are included in the project. However, actual MLflow run creation, Docker image building and remote GitHub Actions execution were not confirmed in that environment.

### Run the Project

Refer to the Task 5 README for the exact environment setup, training and API commands.

The planned local execution sequence is:

```bash
# Activate the project virtual environment

# Install the project dependencies
python -m pip install -r requirements.txt

# Run tests
pytest -v

# Train the model
python -m ipl_mlops.train

# Start the API
uvicorn ipl_mlops.api:app --host 127.0.0.1 --port 8000
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

**Note:** The above commands assume the `src/ipl_mlops/` project structure. Follow the Task 5 README if the local implementation has a different entry point.

---

# 🛠️ Repository Structure

```text
Growfinix-Internship-Tasks/
│
├── Task-1-Automated-Media-Metadata/
│   ├── data/
│   ├── outputs/
│   ├── src/
│   ├── tests/
│   └── README.md
│
├── Task-2-Real-Estate-RAG/
│   ├── data/
│   ├── outputs/
│   ├── tests/
│   ├── app.py
│   ├── Growfinix_Task_2_Real_Estate_RAG.ipynb
│   └── README.md
│
├── Task-3-Customer-Segmentation/
│   ├── data/
│   ├── outputs/
│   ├── src/
│   ├── tests/
│   └── README.md
│
├── Task-4-Tour-Enquiry-LLM-FineTuning/
│   ├── data/
│   ├── src/
│   ├── tests/
│   └── README.md
│
├── Task-5-IPL-MLOps-Pipeline/
│   ├── data/
│   ├── artifacts/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   └── README.md
│
├── .github/
│   └── workflows/
│
├── SUBMISSION_REPORT.md
├── SUBMISSION_CHECKLIST.md
├── DEMO_SCRIPTS.md
├── LINKEDIN_POSTS.md
│
└── README.md
```

*This is the intended high-level structure. Individual project folders may contain additional files and subdirectories.*

---

# 🧪 Implementation and Validation Summary

| Task | Main Implementation | Recorded Validation |
|---|---|---|
| Task 1 | Image/video metadata extraction | OpenCV fallback executed; 3 tests passed |
| Task 2 | Natural-language property retrieval | Offline retrieval executed; source-grounding test passed |
| Task 3 | K-Means and DBSCAN clustering | Clustering and visualization executed |
| Task 4 | Transformers and LoRA fine-tuning pipeline | Dataset/configuration smoke test passed |
| Task 5 | IPL prediction and API pipeline | Model training and API testing executed |

The recorded aggregate automated test run completed with **8 passing tests**.

Some optional integrations require additional dependencies, pretrained model downloads, external API credentials or a suitable runtime environment. Their execution status is documented rather than assumed.

---

# 📹 Project Demonstrations

Each completed task is intended to have an individual LinkedIn demonstration video.

The demonstrations cover:

1. Project objective and architecture.
2. Important implementation components.
3. Actual execution and outputs.
4. Technologies and tools used.
5. GitHub source-code repository.

LinkedIn posts should mention **@GrowfinixTechnology** and include relevant project hashtags.

---

# 📚 Learning Outcomes

Through these five projects, I gained practical exposure to:

- Computer Vision and image/video processing.
- Automated media metadata extraction.
- Natural Language Processing.
- Retrieval-Augmented Generation.
- Similarity-based document retrieval.
- Vector database concepts.
- Unsupervised Machine Learning.
- K-Means and DBSCAN clustering.
- Interactive data visualization.
- Large Language Models.
- Parameter-Efficient Fine-Tuning.
- LoRA and Hugging Face.
- Machine Learning model development and evaluation.
- REST API development using FastAPI.
- MLflow experiment-tracking concepts.
- Docker containerization.
- GitHub Actions and CI/CD automation.

---

# ⚙️ General Development Environment

The projects use Python and their respective task-specific dependencies.

Recommended general setup:

```bash
python --version

git --version
```

Create a virtual environment inside the individual task folder:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies using the requirements file provided in that task folder:

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

Some projects have additional requirements for optional integrations.

Refer to the individual project README files for complete setup instructions.

---

# 📌 Internship Submission

This repository is maintained as part of the **Growfinix Technology – Data Science Internship, Month 3**.

The project submission includes:

- Public GitHub repository.
- Individual task source code.
- Project documentation and README files.
- Demonstration outputs and screenshots.
- LinkedIn demonstration videos.
- LinkedIn posts mentioning @GrowfinixTechnology.

Official submission portal:

https://www.growfinix.in/task/submit

---

# 🙏 Acknowledgement

I would like to thank **Growfinix Technology** for providing this internship opportunity and allowing me to work on practical projects involving Artificial Intelligence, Machine Learning, Deep Learning, Generative AI and MLOps.

These projects have helped me strengthen my technical understanding and gain hands-on experience in developing and documenting end-to-end AI and Machine Learning applications.

---

## Author

**Avishek Kumar Yati**

Data Science Intern | MCA – Data Science

GitHub: https://github.com/AvishekYati

**Repository:** Growfinix-Internship-Tasks

---

⭐ If you find these projects useful, feel free to explore the individual task folders and their documentation.
