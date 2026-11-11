# How to Run NeuralFact: Local Ollama RAG & Hallucination Guardrail System

This guide provides complete instructions for launching and interacting with the **NeuralFact** platform, which combines **Local Ollama LLM inference**, **dense vector retrieval** over course lecture materials (PDF & PPTX), and **sub-millisecond claim-level hallucination verification**.

---

## ⚡ Quick Start: 1-Click Launch (Recommended)

Simply double-click the Windows batch file in the project root:

```bash
run_all.bat
```

This single command will automatically:
1. Detect and launch the **local Ollama server** in the background (if not already running).
2. Start the **Netflix-Style Technical Showcase** on **[http://localhost:8080](http://localhost:8080)**.
3. Launch the **Streamlit Interactive RAG Application** on **[http://localhost:8501](http://localhost:8501)**.
4. Open the interactive dashboard in your default browser.

---

## 🛑 How to Stop All Services

To cleanly terminate all running servers (Streamlit on port `8501` and Showcase on port `8080`):

```bash
stop_all.bat
```

---

## 🛠️ Individual Component Launchers

If you prefer to run services in separate terminal windows:

### 1. Launch Streamlit Project Only (Port 8501)
```bash
run_project.bat
```
*Or via command line:*
```bash
venv\Scripts\streamlit run app/streamlit_app.py --server.port 8501
```

### 2. Launch Technical Showcase Only (Port 8080)
```bash
run_webapp.bat
```
*Or via command line:*
```bash
venv\Scripts\python -m http.server 8080 --directory showcase
```

### 3. Ensure Local Ollama is Running
The local LLM runs on port `11434`. If it is not running:
```bash
ollama serve
```
To verify the model (`qwen2.5:3b`) is available:
```bash
ollama list
```
*(If you ever need to download it again: `ollama pull qwen2.5:3b`)*

---

## 🎯 How to Use the Interactive Dashboard (`http://localhost:8501`)

### Tab 1: ⚡ Interactive RAG with Ollama
1. **Choose a Pre-Configured Course Query** or type your own question:
   - 📋 *Grading & Syllabus*: "What are the grading percentages, assignments, and late policy in the course syllabus?"
   - 📦 *Products of CL (Unit 3)*: "What are the main products and applications of Computational Linguistics in Unit 3?"
   - 🌳 *CKY & Parsing (Unit 4)*: "Explain how linguistic models and syntax parsing transform text to meaning."
   - 🔤 *Morphology & Tokenizers*: "What is the difference between Porter stemming, lemmatization, and subword tokenization?"
2. **Configure Settings**:
   - **Retrieved Chunks (Top-K)**: Slider from `1` to `6` context chunks.
   - **⚠️ Hallucination Stress-Test**: Check this box to instruct the LLM to speculate beyond notes, allowing you to observe the detector catch hallucinations live!
3. **Click `🚀 Ask RAG & Verify Claims`**:
   - **Real-Time Latency Dashboard**: View Vector Retrieval (ms), Ollama Inference (ms), Tokens/sec, and Fact Verification (ms).
   - **Faithfulness Scorecard**: Overall Faithfulness %, Hallucination Risk %, and Supported Claims count.
   - **LLM Generated Answer**: The local model's response.
   - **Claim-by-Claim Verification**: Color-coded cards highlighting supported claims in green and hallucinated claims in red, with detected entity alignments and 39-D feature inspections.
   - **Retrieved Context**: Expandable list showing exact source files (`Unit III CL.pdf`, `Unit_IV_ppt ref.pptx`, etc.) and similarity scores.

### Tab 2: 🔬 Standalone Claim Detector
Test arbitrary Question + Answer pairs (with optional custom context) against the calibrated XGBoost or Random Forest classifiers.

### Tab 3: 📚 Course Knowledge Base & Document Uploader
- View all 7 indexed slide decks and PDF syllabi.
- Drag and drop any new `.pdf`, `.pptx`, `.txt`, or `.md` files and click **"Save and Re-Index Documents"** to update the RAG vector store instantly.

---

## 🧪 Running the Automated Test Suite

To verify system integrity, run:
```bash
venv\Scripts\pytest tests/ -v
```
All 18 unit tests in `tests/test_pipeline.py` and `tests/test_rag_pipeline.py` should pass with green checks.

---

## 💻 Python API Usage

To query the RAG and hallucination detection pipeline programmatically in Python:

```python
from src.rag_pipeline import get_rag_pipeline

# Initialize pipeline (loads cached 769 vector chunks in <0.5s)
pipeline = get_rag_pipeline()

# Query the course assistant
result = pipeline.run_query(
    query="What is the late submission policy in the course syllabus?",
    top_k=2,
    model_name="qwen2.5:3b"
)

print("Answer:", result.llm_answer)
print(f"Faithfulness: {result.faithfulness_score}% | Latency: {result.timing['total_ms']}ms")

for claim in result.hallucination_analysis["claims"]:
    print(f"[{claim['prediction']}] (Risk {claim['hallucination_risk']}%) {claim['claim_text']}")
```

