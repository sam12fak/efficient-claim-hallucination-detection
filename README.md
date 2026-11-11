# Efficient Claim-Level Hallucination Detection in Large Language Models Using Semantic and Linguistic Features

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/sam12fak/efficient-claim-hallucination-detection)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C.svg)](https://pytorch.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-2.7-yellow.svg)](https://sbert.net/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red.svg)](https://xgboost.readthedocs.io/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-09A3D5.svg)](https://spacy.io/)

---

## 1. Problem Statement & Motivation
Large Language Models (LLMs) frequently hallucinate factual inaccuracies—generating fluent, confident statements that are unsupported or contradicted by reality. The prevailing verification paradigm relies on another massive 7B–70B+ parameter model as an evaluation judge (e.g., *LLM-as-a-judge*). However, this incurs severe bottlenecks:
- **Prohibitive Compute & Latency**: Running 7B+ LLM evaluations per claim is too slow for interactive real-time applications.
- **High Operational Costs**: Commercial LLM APIs cost substantial amounts when verifying thousands of sentences at scale.
- **Judge Hallucination**: The evaluator LLM can itself hallucinate or suffer from sycophancy bias.

### Core Research Question
> **Can a lightweight hybrid NLP model combining compact semantic embeddings and interpretable linguistic features detect hallucinated claims effectively while requiring orders of magnitude less computation than deploying a large language model as a judge?**

This repository provides an end-to-end framework designed to operate on **consumer hardware (4–8 GB GPU VRAM or standard CPU)** at **> 980,000 claims/sec throughput** (< 0.002 ms latency per claim).

---

## 2. High-Level Architecture

```text
                           USER INPUT
                               │
                       ┌───────┴────────┐
                       │                │
                    Question         LLM Answer
                                        │
                                        ▼
                               Sentence Splitting
                               (spaCy Sentencizer)
                                        │
                                        ▼
                                Claim Extraction
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
             Semantic Features                     Linguistic Features
                    │                                       │
                    ▼                                       ▼
            Compact Transformer                        spaCy NLP
            (all-MiniLM-L6-v2)                     (en_core_web_sm)
         Cosine & Distance Metrics              Entities, POS, Ratios,
         (LRU Embedding Caching)                Numerics & Stance Lexicons
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                                 Feature Fusion
                               (39-D Vector Table)
                                        │
                                        ▼
                             Lightweight Classifier
                          (Calibrated Hybrid XGBoost /
                                 Random Forest)
                                        │
                                        ▼
                             Claim-Level Prediction
                            (Supported vs. Hallucinated)
                                        │
                                        ▼
                            Calibrated Risk Percentage
                             + Model Explainability
                                        │
                                        ▼
                             Interactive Streamlit UI
```

---

## 3. Empirical Results & Benchmarks

All models were evaluated on a held-out test split of **4,228 claims** partitioned at the question level to guarantee **zero data leakage**.

### Classification Performance & Efficiency Comparison

| Model Architecture | Feature Set | F1-Score | Precision | Recall | Accuracy | ROC-AUC | Latency (ms/claim) | Throughput (claims/s) | Model Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Linguistic)** | Linguistic Baseline | 0.7370 | 0.6990 | 0.7793 | 0.7067 | 0.7812 | 0.007 ms | 135,284 | 11.15 MB |
| **Logistic Regression (Semantic)** | Semantic Baseline | 0.6564 | 0.6263 | 0.6895 | 0.6194 | 0.6694 | < 0.001 ms | 4,305,324 | < 0.01 MB |
| **Random Forest (Semantic)** | Semantic Baseline | 0.7120 | 0.6986 | 0.7259 | 0.6904 | 0.7766 | 0.007 ms | 134,955 | 6.33 MB |
| **Random Forest (Hybrid)** | **Hybrid (Semantic + Linguistic)** | **0.7838** | **0.7680** | **0.8004** | **0.7673** | **0.8550** | 0.011 ms | 90,752 | 21.95 MB |
| **XGBoost (Hybrid)** | **Hybrid (Semantic + Linguistic)** | 0.7798 | 0.7620 | 0.7986 | 0.7623 | **0.8549** | **0.001 ms** | **980,833** | **0.36 MB** |

> **Key Takeaway**: Combining compact semantic embeddings with linguistic features yields a **+6.8% F1** and **+0.078 ROC-AUC** gain over semantic features alone, and a **+4.1% F1** and **+0.071 ROC-AUC** gain over linguistic features alone. The hybrid XGBoost classifier operates at **under 0.002 ms latency per claim** with a microscopic **0.36 MB** disk footprint.

---

### Feature Ablation Study

| Configuration | Features Included | F1-Score | Accuracy | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Hybrid** | 39 | **0.7831** | **0.7585** | 0.7437 | **0.8268** | **0.8509** |
| **Semantic Only** | 4 | 0.7147 | 0.6902 | 0.6945 | 0.7362 | 0.7727 |
| **Linguistic Only** | 35 | 0.7425 | 0.7065 | 0.6907 | 0.8026 | 0.7799 |
| **No Entities** | 29 | 0.7848 | 0.7625 | **0.7513** | 0.8214 | 0.8533 |
| **No Numerical** | 33 | 0.7831 | 0.7614 | 0.7519 | 0.8170 | 0.8516 |
| **No POS Ratios** | 29 | 0.7748 | 0.7453 | 0.7255 | 0.8313 | 0.8398 |
| **No Certainty Lexicons** | 35 | 0.7855 | 0.7611 | 0.7459 | 0.8295 | 0.8514 |

---

## 4. Feature Engineering

The pipeline extracts **39 structured features** per claim:

### 1. Semantic Features (4 features)
- `question_claim_similarity`: Cosine similarity between claim and question embedding.
- `question_claim_distance`: $1 - \text{cosine\_similarity}$.
- `claim_answer_similarity`: Coherence of claim with the broader answer context.
- `context_claim_similarity`: Similarity against reference context (if provided, default 0.0).
- *Optimization*: In-memory LRU vector cache to avoid redundant neural forward passes.

### 2. Linguistic Features (35 features)
- **Surface Statistics**: `char_count`, `word_count`, `token_count`, `sentence_length`, `avg_word_length`.
- **Named Entity Extraction**: Entity frequencies for `PERSON`, `ORG`, `GPE`, `LOC`, `DATE`, `CARDINAL`, `MONEY`, `PERCENT`, plus `total_entity_count` and `entity_density`.
- **Numerical & Chronological**: `year_count` (regex 1000–2099), `number_count`, `percent_count`, `money_count`, `has_year`, `has_numeric`.
- **Part-of-Speech**: Counts and relative ratios for `NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`.
- **Epistemic Stance**: Frequencies and ratios of certainty markers (*definitely, certainly, clearly, undeniably*) and uncertainty markers (*probably, possibly, maybe, reportedly, alleged*). Treated strictly as statistical signals.

---

## 5. Dataset & Partitioning
- **Primary Source**: [HaluEval Benchmark](https://github.com/RUCAIBox/HaluEval) (EMNLP 2023).
- **Subsets**: Multi-domain Question Answering (HotpotQA base) and Conversational Dialogue.
- **Scale**: 10,000 contexts yielding **28,251 structured claims**.
- **Labels**:
  - `0`: Supported / Non-hallucinated (Truthful ground truth & verified context assertions).
  - `1`: Hallucinated (Plausible, fabricated assertions generated by ChatGPT).
- **Split Strategy**: Stratified question-level partition (70% Train: 19,814 claims, 15% Val: 4,209 claims, 15% Test: 4,228 claims) preventing cross-split topic leakage.

---

## 6. Repository Structure

```text
efficient-claim-hallucination-detection/
├── app/
│   └── streamlit_app.py          # Interactive web UI with demo scenarios
├── configs/
│   └── config.yaml               # Pipeline configurations & thresholds
├── data/
│   ├── raw/                      # Downloaded HaluEval datasets (.gitkeep)
│   ├── processed/                # Processed claim splits & full tables (.gitkeep)
│   ├── train/                    # Training claims and feature matrices (.gitkeep)
│   └── test/                     # Held-out test claims and matrices (.gitkeep)
├── models/
│   ├── best_model.joblib         # Calibrated hybrid detector
│   ├── hybrid_xgboost.joblib     # Calibrated XGBoost model
│   ├── hybrid_random_forest.joblib # Calibrated Random Forest model
│   ├── feature_scaler.joblib     # StandardScaler fit on training features
│   ├── feature_names.json        # Feature definitions & ablation subsets
│   └── training_summary.json     # Hyperparameters and validation metrics
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baselines.ipynb
│   └── 04_model_comparison.ipynb
├── results/
│   ├── metrics_comparison.png    # F1, accuracy, and ROC-AUC bar plot
│   ├── confusion_matrices.png    # Heatmaps for all models
│   ├── roc_curves.png            # ROC Curves
│   ├── feature_importance.png    # Top 20 model features
│   ├── ablation_study.png        # Feature group ablation bar chart
│   ├── evaluation_report.json    # Full test evaluation metrics
│   └── experiment_results.md     # Formatted markdown benchmark tables
├── scripts/
│   └── make_notebooks.py         # Utility to generate starter notebooks
├── src/
│   ├── __init__.py
│   ├── claim_extraction.py       # Sentence segmentation and sanitization
│   ├── semantic_features.py      # MiniLM embedding calculation & LRU cache
│   ├── linguistic_features.py    # spaCy surface, NER, POS & stance parsing
│   ├── feature_fusion.py         # Feature concatenation, scaling & persistence
│   ├── preprocessing.py          # HaluEval downloader, parser & split generator
│   ├── train.py                  # Baseline & hybrid training with Platt scaling
│   ├── evaluate.py               # Benchmarking, ablation & visualization
│   └── predict.py                # End-to-end inference & explainability engine
├── tests/
│   └── test_pipeline.py          # Pytest suite (13/13 passing tests)
├── .gitignore
├── PROJECT_PLAN.md
├── PROJECT_STATUS.md
├── requirements.txt
└── README.md
```

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.9+ (tested on Python 3.9.0)
- Windows / Linux / macOS
- Minimum 4 GB RAM (runs on CPU or CUDA GPU)

### 1. Clone the Repository
```bash
git clone https://github.com/sam12fak/efficient-claim-hallucination-detection.git
cd efficient-claim-hallucination-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download spaCy English Language Model
```bash
python -m spacy download en_core_web_sm
```

---

## 8. Usage

### Launch Interactive Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
*The app includes 4 preconfigured demo scenarios (Factual, Clearly Hallucinated, Mixed Multi-Claim, and Long Narrative) as well as free-form testing with optional reference context.*

### Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```

### One-Click Launchers (Windows)
- **`run_all.bat`**: Launches Ollama, Streamlit Interactive RAG (port 8501), and Netflix Technical Showcase (port 8080) simultaneously.
- **`run_project.bat`**: Launches the Streamlit Application on `http://localhost:8501`.
- **`run_webapp.bat`**: Launches the Netflix Technical Showcase on `http://localhost:8080`.
- **`stop_all.bat`**: Safely shuts down all running web servers on ports 8501 and 8080.

---

## 9. Interactive RAG with Local Ollama (`qwen2.5:3b`)

NeuralFact includes an end-to-end Retrieval-Augmented Generation (RAG) agent that parses course lecture materials (PDF syllabi, PPTX presentations), queries a local Ollama LLM, and immediately applies sub-millisecond claim verification guardrails:

```python
from src.rag_pipeline import get_rag_pipeline

# Initialize RAG pipeline (loads dense index over PPTX/PDF course materials)
pipeline = get_rag_pipeline()

# Run interactive query
result = pipeline.run_query(
    query="What are the grading percentages and assignments in the course syllabus?",
    top_k=3,
    model_name="qwen2.5:3b"
)

print(f"Model Answer:\n{result.llm_answer}\n")
print(f"Faithfulness Score: {result.faithfulness_score}% (Risk: {result.overall_risk_percent}%)")
print(f"Latency: Retrieval={result.timing['retrieval_ms']}ms | LLM={result.timing['llm_ms']}ms | Verification={result.timing['detection_ms']}ms")

# Inspect claim-level verification
for claim in result.hallucination_analysis["claims"]:
    print(f"[{claim['prediction']}] (Risk {claim['hallucination_risk']}%) {claim['claim_text']}")
```

---

## 10. Python API Example (Direct Claim Detection)

```python
from src.predict import predict_hallucinations

question = "Who discovered penicillin?"
answer = "Marie Curie discovered penicillin in 1928 at St. Mary's Hospital in London."

result = predict_hallucinations(question, answer)

print(f"Overall Prediction: {result['overall_prediction']}")
print(f"Overall Risk: {result['overall_risk_percent']}%")

for claim in result["claims"]:
    print(f"\nClaim: {claim['claim_text']}")
    print(f"Prediction: {claim['prediction']} (Risk: {claim['hallucination_risk']}%)")
    print("Signals:")
    for signal in claim["model_signals"]:
        print(f" - {signal}")
```

---

## 10. Limitations & Responsible AI Use
- **Statistical Risk Score, Not Universal Factuality**: Predictions reflect statistical alignment and linguistic patterns learned from benchmark data. They do not constitute formal logical or metaphysical proof of falsehood.
- **Novel / Unseen Real-World Facts**: Facts absent from general knowledge representations require reference context to be reliably verified.
- **Domain Specialization**: Highly specialized technical domains (e.g., clinical pathology, proprietary legal documents) may display distinct linguistic distributions and require domain-adapted training data.

---

## 11. License
This project is licensed under the [MIT License](LICENSE).
