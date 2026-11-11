# Course Syllabus: CS 6320 / LING 4300 - Computational Linguistics & NLP
**Semester**: Fall 2026  
**Instructor**: Prof. Eleanor Vance (Department of Computer Science & Linguistics)  
**Lecture Times**: Tuesday & Thursday 10:00 AM - 11:30 AM  
**Location**: Turing Hall 301  
**Course Website**: https://cl-nlp-course.edu/fall2026  

---

## Course Overview
Computational Linguistics (CL) investigates computational models of human language. This course provides a rigorous foundation in statistical, neural, and symbolic natural language processing (NLP). Topics span morphological analysis, syntax and parsing, distributional semantics, modern transformer-based language models, and hallucination detection in generative AI.

---

## Required Textbooks & Resources
1. **Primary Textbook**: Daniel Jurafsky and James H. Martin, *Speech and Language Processing*, 3rd Edition Draft (2024).
2. **Supplementary Reference**: Christopher D. Manning and Hinrich Schütze, *Foundations of Statistical Natural Language Processing*, MIT Press.
3. **Required Python Libraries**: PyTorch, Hugging Face Transformers, Sentence-Transformers, SpaCy, Scikit-Learn, and XGBoost.

---

## Grading Scheme & Evaluation Policy
* **Homework & Programming Assignments (4 assignments)**: 35% total (8.75% each)
  - Assignment 1: Byte-Pair Encoding (BPE) & Morphological Analyzers
  - Assignment 2: CKY Syntactic Parser & Dependency Grammar Analysis
  - Assignment 3: Fine-tuning BERT & Dense Vector Embeddings for Semantic Similarity
  - Assignment 4: Retrieval-Augmented Generation (RAG) & Claim Verification
* **Midterm Examination**: 20% (In-class, closed book, Week 8)
* **Final Term Project (Factuality & Hallucination Mitigation)**: 30%
  - Deliverable: Calibrated claim-level hallucination detection pipeline and paper report.
* **Class Attendance, Quizzes & Active Participation**: 15%

**Late Submission Policy**: Students receive 3 free slip days for the semester. Subsequent late submissions incur a 10% penalty per 24-hour period. Submissions delayed over 72 hours receive zero credit.

---

## Weekly Schedule & Lecture Plan
* **Week 1-2: Foundations of Language & Tokenization**
  - Morphology, stemmers (Porter algorithm), WordNet lemmatizers.
  - Subword tokenization: Byte-Pair Encoding (BPE), WordPiece, SentencePiece.
* **Week 3-4: Syntactic Structures & Parsing**
  - Formal grammars, Context-Free Grammars (CFGs), Chomsky Normal Form (CNF).
  - CKY Dynamic Programming Parser, Shift-Reduce and Arc-Eager Dependency Parsers.
* **Week 5-6: Vector Semantics & Word Representations**
  - Distributional semantics: TF-IDF, PPMI matrices.
  - Distributed representations: Word2Vec (Skip-Gram with Negative Sampling), GloVe, FastText.
* **Week 7-8: Sequential Models & Pre-trained Encoders**
  - Recurrent neural networks (RNN), LSTM, GRU architectures.
  - Bidirectional Encoder Representations from Transformers (BERT), RoBERTa.
* **Week 9-10: Transformer Architecture & Generative LLMs**
  - Scaled dot-product multi-head attention mechanisms, positional encodings.
  - Autoregressive causal models (GPT-4, Llama 3, Qwen 2.5).
* **Week 11-12: Retrieval-Augmented Generation (RAG)**
  - Dense passage retrieval, vector databases, bi-encoder and cross-encoder re-ranking.
  - Retrieval latency vs contextual precision tradeoffs.
* **Week 13-14: Hallucination Detection & Factuality Calibration**
  - Intrinsic vs extrinsic hallucination typology in generative models.
  - Claim extraction, semantic grounding, linguistic certainty cues, and hybrid calibrated tree classifiers.
* **Week 15: Final Project Demonstrations & Poster Session**

---

## Academic Integrity
All submitted code and write-ups must be individual work unless explicitly designated as group collaboration. Generative AI tools (ChatGPT, Claude, Gemini) are permitted for debugging, but all claims and derivations must be attributed.

