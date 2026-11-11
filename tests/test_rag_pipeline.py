"""
Test Suite for RAG Pipeline with Local Ollama & Hallucination Guardrails
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from src.rag_pipeline import (
    DocumentChunk,
    RAGKnowledgeBase,
    OllamaClient,
    RAGHallucinationPipeline,
    get_rag_pipeline
)


class TestRAGKnowledgeBase:
    def test_chunk_text_sliding_window(self):
        kb = RAGKnowledgeBase(docs_dir="data/rag_docs")
        sample_text = "Word " * 200
        chunks = kb._chunk_text(sample_text, max_chars=100, overlap=20)
        assert len(chunks) > 1
        for ch in chunks:
            assert len(ch) <= 150

    def test_retrieval_returns_scored_chunks(self):
        pipeline = get_rag_pipeline()
        results = pipeline.kb.retrieve(
            query="grading percentages and assignments",
            top_k=3
        )
        assert len(results) == 3
        for chunk, score in results:
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            assert len(chunk.text) > 0


class TestOllamaClient:
    def test_client_init(self):
        client = OllamaClient()
        assert client.base_url == "http://127.0.0.1:11434"
        assert client.default_model == "qwen2.5:3b"

    def test_list_models_returns_list(self):
        client = OllamaClient()
        models = client.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1


class TestEndToEndRAGPipeline:
    def test_run_query_structure(self):
        pipeline = get_rag_pipeline()
        res = pipeline.run_query(
            query="What is the late submission policy in the course syllabus?",
            top_k=2
        )
        assert res.query == "What is the late submission policy in the course syllabus?"
        assert len(res.retrieved_chunks) == 2
        assert len(res.llm_answer) > 0
        assert "retrieval_ms" in res.timing
        assert "llm_ms" in res.timing
        assert "detection_ms" in res.timing
        assert "total_ms" in res.timing
        assert 0.0 <= res.faithfulness_score <= 100.0
        assert 0.0 <= res.overall_risk_percent <= 100.0
        assert res.status in ["Supported", "Partially Supported", "Hallucinated", "No Claims"]

