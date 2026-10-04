# NewsNER-AI — News Article Named Entity Recognition

A Transformer-based Named Entity Recognition system for extracting **People, Organizations, Locations, and Miscellaneous entities** from news articles using a fine-tuned DistilBERT model.

<p align="center">

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](https://news-ner-entity-recognition.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/AbdelrahmanAkl/NewsNER-DistilBERT)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?logo=huggingface\&logoColor=black)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io/)

</p>

---

## Live Demo

Try the deployed application:

**https://news-ner-entity-recognition.streamlit.app/**

The application loads the fine-tuned model from Hugging Face and performs Named Entity Recognition on news article text.

---

## Overview

**NewsNER-AI** is an end-to-end NLP project that demonstrates how a Transformer model can be fine-tuned and deployed for Named Entity Recognition.

The project uses **DistilBERT** fine-tuned on the **CoNLL-2003** news dataset to identify four entity categories:

| Label  | Entity        |
| ------ | ------------- |
| `PER`  | Person        |
| `ORG`  | Organization  |
| `LOC`  | Location      |
| `MISC` | Miscellaneous |

The project covers the complete workflow:

**Dataset → Fine-Tuning → Entity-Level Evaluation → Benchmarking → Model Publishing → Streamlit Deployment**

---

## Problem Statement

News articles contain large amounts of unstructured information involving people, companies, institutions, countries, cities, and other named entities.

Automatically identifying these entities is an important NLP capability for applications such as:

* News intelligence
* Information extraction
* Search and indexing
* Document understanding
* Knowledge graph construction
* Content analytics
* Entity-based recommendation systems

The goal of NewsNER-AI is to build a practical Transformer-based NER system that can recognize named entities from news-style text and expose the trained model through an interactive web application.

---

## Key Features

* Fine-tuned **DistilBERT** for Named Entity Recognition
* Trained on the **CoNLL-2003** dataset
* Four entity categories: `PER`, `ORG`, `LOC`, `MISC`
* Entity-level Precision, Recall, and F1 evaluation
* Comparison against Rule-Based and spaCy baselines
* Confidence scores for detected entities
* Entity highlighting inside the submitted article
* Public model hosted on Hugging Face
* Streamlit-based interactive inference application
* CPU-compatible deployment
* Clean portfolio-oriented project structure

---

## Architecture

```text
                         News Article
                              |
                              v
                    +-------------------+
                    |   Streamlit App   |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | DistilBERT Model  |
                    | Token Classification |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Entity Prediction |
                    +-------------------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
            PER              ORG              LOC
             |                |                |
             +----------------+----------------+
                              |
                              v
                             MISC
                              |
                              v
                    Entity Aggregation
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
           Entity Statistics       Highlighted Article
                  |
                  v
           Confidence Scores
```

---

## NLP Workflow

```text
CoNLL-2003
    |
    v
Data Preparation
    |
    v
DistilBERT Tokenization
    |
    v
Label Alignment
    |
    v
Transformer Fine-Tuning
    |
    v
Entity-Level Evaluation
    |
    v
Baseline Benchmarking
    |
    v
Hugging Face Model
    |
    v
Streamlit Deployment
```

---

## Technologies

| Technology                | Role                        |
| ------------------------- | --------------------------- |
| Python                    | Core development            |
| PyTorch                   | Deep learning framework     |
| Hugging Face Transformers | Transformer implementation  |
| DistilBERT                | Fine-tuned NER model        |
| Hugging Face Hub          | Model hosting               |
| Streamlit                 | Interactive web application |
| spaCy                     | Baseline comparison         |
| CoNLL-2003                | NER dataset                 |
| Jupyter Notebook          | Experiments and analysis    |

---

# Dataset

The model was trained and evaluated using the **CoNLL-2003 Named Entity Recognition dataset**.

### Dataset Splits

| Split      | Samples |
| ---------- | ------: |
| Train      |  14,041 |
| Validation |   3,250 |
| Test       |   3,453 |

The dataset contains news articles annotated with four entity categories:

* `PER` — Person
* `ORG` — Organization
* `LOC` — Location
* `MISC` — Miscellaneous

---

# Model

## Fine-Tuned DistilBERT

The project uses **DistilBERT** as the Transformer backbone and fine-tunes it for token classification.

The training workflow includes:

1. Loading the CoNLL-2003 dataset
2. Preparing token-level labels
3. Tokenizing the input text
4. Aligning labels with the tokenized representation
5. Fine-tuning DistilBERT
6. Evaluating on the test set
7. Saving the trained model
8. Publishing the model to Hugging Face

### Hugging Face Model

**NewsNER-DistilBERT**

https://huggingface.co/AbdelrahmanAkl/NewsNER-DistilBERT

The Streamlit application loads the published model directly from Hugging Face rather than storing the model weights in the GitHub repository.

---

# Evaluation

## Official Entity-Level Results

The final model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Precision | **87.92%** |
| Recall    | **89.68%** |
| F1        | **88.79%** |

### Headline Metric

> **Entity-Level F1: 88.79%**

This is the official evaluation result used for the project.

The evaluation uses an entity-level methodology rather than the earlier subword-aligned evaluation setup.

---

## Per-Entity Performance

| Entity | Precision | Recall |         F1 |
| ------ | --------: | -----: | ---------: |
| PER    |    95.02% | 94.50% | **94.76%** |
| ORG    |    84.28% | 87.78% | **85.99%** |
| LOC    |    91.84% | 91.07% | **91.45%** |
| MISC   |    72.82% | 79.77% | **76.14%** |

The model performs particularly strongly on **Person** and **Location** entities, while **Miscellaneous** entities remain the most challenging category.

---

# Benchmark Comparison

The fine-tuned DistilBERT model was compared against multiple approaches.

| Approach                  | Entity-Level F1 |
| ------------------------- | --------------: |
| Rule-Based                |          10.15% |
| spaCy Small               |          56.22% |
| spaCy Medium              |          57.19% |
| **Fine-tuned DistilBERT** |      **88.79%** |

The fine-tuned Transformer substantially outperformed the evaluated rule-based and spaCy baselines.

---

# Example Inference

### Input

```text
Apple CEO Tim Cook announced a new investment in India during a meeting in New Delhi. Microsoft also plans to expand its operations in London.
```

### Detected Entities

| Entity    | Type | Confidence |
| --------- | ---- | ---------: |
| Apple     | ORG  |     97.72% |
| Tim Cook  | PER  |     99.88% |
| India     | LOC  |     99.91% |
| New Delhi | LOC  |     98.89% |
| Microsoft | ORG  |     99.49% |
| London    | LOC  |     99.88% |

### Entity-Highlighted Output

```text
Apple [ORG] CEO Tim Cook [PER] announced a new investment
in India [LOC] during a meeting in New Delhi [LOC].
Microsoft [ORG] also plans to expand its operations
in London [LOC].
```

---

# Streamlit Application

The application provides an interactive interface for real-time NER inference.

### Features

* News article text input
* Named Entity Recognition
* Entity counts
* Confidence scores
* Entity highlighting
* CPU inference
* Analyze and Clear controls

### Live Demo

**https://news-ner-entity-recognition.streamlit.app/**

The deployed application has been tested with real inference using the published Hugging Face model.

---

# Project Structure

```text
News-NER-Entity-Recognition/
│
├── app.py
├── requirements.txt
├── .gitignore
│
└── notebooks/
    └── News-Article-NER-with-SpaCy.ipynb
```

### Main Components

#### `app.py`

Streamlit inference application responsible for:

* Loading the Hugging Face model
* Running NER inference
* Calculating entity statistics
* Displaying confidence scores
* Highlighting detected entities

#### `notebooks/News-Article-NER-with-SpaCy.ipynb`

Notebook containing the experimentation and NER workflow.

#### `requirements.txt`

Application dependencies required for local and Streamlit deployment.

---

# Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/News-NER-Entity-Recognition.git
cd News-NER-Entity-Recognition
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# Deployment

The project separates application code and model hosting:

```text
GitHub
   |
   v
Streamlit Cloud
   |
   v
app.py
   |
   v
Hugging Face Hub
   |
   v
NewsNER-DistilBERT
   |
   v
NER Inference
```

This approach keeps the GitHub repository lightweight while allowing the deployed application to retrieve the trained model when needed.

---

# Why DistilBERT?

DistilBERT provides a practical balance between:

* Contextual language understanding
* Model size
* Inference efficiency
* Fine-tuning flexibility

It is well suited for building practical Transformer-based NLP applications where model quality and deployment efficiency both matter.

---

# Future Improvements

Potential extensions include:

* Fine-tuning on larger news-domain datasets
* Improving MISC entity recognition
* Detailed error analysis
* Confidence calibration
* Batch document processing
* PDF and document ingestion
* Entity relationship extraction
* Knowledge graph generation
* Multilingual NER
* API deployment
* Production monitoring
* Model optimization and quantization

---

# Portfolio Highlights

This project demonstrates practical experience with:

* Natural Language Processing
* Named Entity Recognition
* Transformer fine-tuning
* DistilBERT
* Token classification
* PyTorch
* Hugging Face Transformers
* Entity-level evaluation
* Model benchmarking
* Model hosting
* Streamlit deployment
* Production-oriented ML workflows

---

# Author

**Abdelrahman Akl**

AI Engineer | NLP | LLMs | RAG | Agentic AI

**GitHub:**
https://github.com/AbdelrhmanAkl

**LinkedIn:**
https://linkedin.com/in/abdelrahmanakl/

---

# Project Links

| Resource           | Link                                                         |
| ------------------ | ------------------------------------------------------------ |
| GitHub Repository  | https://github.com/AbdelrhmanAkl/News-NER-Entity-Recognition |
| Live Demo          | https://news-ner-entity-recognition.streamlit.app/           |
| Hugging Face Model | https://huggingface.co/AbdelrahmanAkl/NewsNER-DistilBERT     |

---

# License

This project is licensed under the **Apache License 2.0**.
