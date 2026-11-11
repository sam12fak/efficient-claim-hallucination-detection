"""
RAG (Retrieval-Augmented Generation) Pipeline with Ollama & Calibrated Hallucination Detection
Integrates dense vector retrieval over Computational Linguistics lecture materials (PDF, PPTX, MD),
small local LLM inference (Ollama qwen2.5:3b), and real-time claim-level hallucination scoring.
"""

import os
import sys
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from src.semantic_features import get_embedding_model
from src.predict import predict_hallucinations, load_detector_model

logger = logging.getLogger("RAGPipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    source_type: str  # pdf, pptx, md, txt
    page_or_slide: int
    text: str


@dataclass
class RAGResult:
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    combined_context: str
    llm_answer: str
    llm_model: str
    timing: Dict[str, float]  # retrieval_ms, llm_ms, detection_ms, total_ms
    hallucination_analysis: Dict[str, Any]
    faithfulness_score: float  # 0 to 100
    overall_risk_percent: float  # 0 to 100
    supported_claims_count: int
    hallucinated_claims_count: int
    total_claims: int
    status: str  # "Supported", "Partially Supported", "Hallucinated", "No Claims"


class RAGKnowledgeBase:
    """Manages document ingestion, chunking, dense embedding, and cosine retrieval."""

    def __init__(self, docs_dir: str = "data/rag_docs", cache_path: str = "models/rag_kb_cache.joblib"):
        self.docs_dir = docs_dir
        self.cache_path = cache_path
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._embedding_model = None

    def _get_embedder(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    def extract_text_from_file(self, file_path: str) -> List[Tuple[int, str]]:
        """Extract pages or slides as (index, text) tuples."""
        ext = os.path.splitext(file_path)[1].lower()
        items = []

        try:
            if ext == ".pdf":
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for page_num, page in enumerate(reader.pages, start=1):
                    txt = page.extract_text() or ""
                    txt = " ".join(txt.split())
                    if len(txt) > 30:
                        items.append((page_num, txt))

            elif ext in (".pptx", ".ppt"):
                import pptx
                prs = pptx.Presentation(file_path)
                for slide_num, slide in enumerate(prs.slides, start=1):
                    slide_texts = []
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for paragraph in shape.text_frame.paragraphs:
                                line = paragraph.text.strip()
                                if line:
                                    slide_texts.append(line)
                    txt = " ".join(slide_texts)
                    txt = " ".join(txt.split())
                    if len(txt) > 25:
                        items.append((slide_num, txt))

            elif ext in (".md", ".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Split markdown into sections
                sections = content.split("## ")
                for sec_idx, sec in enumerate(sections, start=1):
                    sec_clean = " ".join(sec.split())
                    if len(sec_clean) > 30:
                        items.append((sec_idx, sec_clean))
        except Exception as e:
            logger.warning(f"Error extracting from {file_path}: {e}")

        return items

    def _chunk_text(self, text: str, max_chars: int = 600, overlap: int = 100) -> List[str]:
        """Sliding window chunking with word boundary respect."""
        words = text.split()
        chunks = []
        current_words = []
        current_len = 0

        for w in words:
            current_words.append(w)
            current_len += len(w) + 1
            if current_len >= max_chars:
                chunks.append(" ".join(current_words))
                # keep last few words for overlap
                overlap_words = current_words[-max(2, int(len(current_words) * 0.2)):]
                current_words = list(overlap_words)
                current_len = sum(len(x) + 1 for x in current_words)

        if current_words:
            remainder = " ".join(current_words)
            if len(remainder) > 30:
                chunks.append(remainder)
        return chunks or [text]

    def build_or_load_index(self, force_rebuild: bool = False) -> int:
        """Loads cached index or builds it from docs_dir."""
        if not force_rebuild and os.path.exists(self.cache_path):
            try:
                cached = joblib.load(self.cache_path)
                raw_chunks = cached["chunks"]
                self.chunks = [
                    DocumentChunk(**c) if isinstance(c, dict) else c
                    for c in raw_chunks
                ]
                self.embeddings = cached["embeddings"]
                logger.info(f"Loaded {len(self.chunks)} RAG chunks from cache {self.cache_path}")
                return len(self.chunks)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding...")

        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)

        all_chunks: List[DocumentChunk] = []
        doc_files = [
            os.path.join(self.docs_dir, f)
            for f in os.listdir(self.docs_dir)
            if os.path.splitext(f)[1].lower() in [".pdf", ".pptx", ".md", ".txt"]
        ]

        logger.info(f"Indexing {len(doc_files)} files from {self.docs_dir}...")
        for file_path in doc_files:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower().replace(".", "")
            items = self.extract_text_from_file(file_path)

            for item_idx, item_text in items:
                sub_chunks = self._chunk_text(item_text, max_chars=700, overlap=120)
                for chunk_num, c_text in enumerate(sub_chunks):
                    chunk_id = f"{file_name}_p{item_idx}_c{chunk_num}"
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_file=file_name,
                            source_type=file_ext,
                            page_or_slide=item_idx,
                            text=c_text,
                        )
                    )

        if not all_chunks:
            all_chunks.append(
                DocumentChunk(
                    chunk_id="seed_cl_01",
                    source_file="CL_Course_Syllabus.md",
                    source_type="md",
                    page_or_slide=1,
                    text="CS 6320 / LING 4300 Computational Linguistics covers syntax, semantics, and hallucination detection.",
                )
            )

        self.chunks = all_chunks
        texts_to_embed = [c.text for c in self.chunks]
        logger.info(f"Encoding {len(texts_to_embed)} chunks with SentenceTransformer...")

        embedder = self._get_embedder()
        embs = embedder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = embs.astype(np.float32)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        chunk_dicts = [asdict(c) for c in self.chunks]
        joblib.dump({"chunks": chunk_dicts, "embeddings": self.embeddings}, self.cache_path)
        logger.info(f"Index built and cached to {self.cache_path} ({len(self.chunks)} chunks).")
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top_k chunks based on cosine similarity."""
        if self.embeddings is None or len(self.chunks) == 0:
            self.build_or_load_index()

        embedder = self._get_embedder()
        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        # Cosine similarity is dot product because vectors are normalized
        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results


class OllamaClient:
    """Interface to local Ollama server with health checking and fallbacks."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", default_model: str = "qwen2.5:3b"):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    def is_available(self) -> bool:
        """Check if Ollama server responds."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List locally available models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3.0)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return [self.default_model]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Generate response via Ollama /api/generate."""
        chosen_model = model or self.default_model

        payload = {
            "model": chosen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_t = time.perf_counter()
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=timeout)
            duration_ms = (time.perf_counter() - start_t) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("response", "").strip()
                eval_count = data.get("eval_count", len(answer.split()))
                tokens_per_sec = (eval_count / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0

                return {
                    "text": answer,
                    "model": chosen_model,
                    "duration_ms": duration_ms,
                    "eval_count": eval_count,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "success": True,
                }
            else:
                return {
                    "text": f"Ollama HTTP {resp.status_code}: {resp.text}",
                    "model": chosen_model,
                    "duration_ms": duration_ms,
                    "eval_count": 0,
                    "tokens_per_sec": 0.0,
                    "success": False,
                }
        except Exception as e:
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            logger.warning(f"Ollama call failed ({e}). Returning fallback response.")
            return {
                "text": f"Local inference fallback (Ollama unavailable or timed out: {e}).",
                "model": "fallback",
                "duration_ms": duration_ms,
                "eval_count": 0,
                "tokens_per_sec": 0.0,
                "success": False,
            }


class RAGHallucinationPipeline:
    """
    End-to-End RAG + Hallucination Verification Engine:
    1. Dense semantic search over course materials (PDF, PPTX, MD).
    2. Ollama LLM prompt formulation & generation.
    3. Claim-level calibrated hallucination detection.
    4. Granular metric & latency profiling.
    """

    def __init__(
        self,
        docs_dir: str = "data/rag_docs",
        cache_path: str = "models/rag_kb_cache.joblib",
        ollama_model: str = "qwen2.5:3b",
    ):
        self.kb = RAGKnowledgeBase(docs_dir=docs_dir, cache_path=cache_path)
        self.ollama = OllamaClient(default_model=ollama_model)

    def initialize(self):
        """Preload the knowledge base index and model weights."""
        self.kb.build_or_load_index()
        load_detector_model()

    def run_query(
        self,
        query: str,
        top_k: int = 3,
        model_name: Optional[str] = None,
        stress_test: bool = False,
        detector_model_type: str = "hybrid_xgboost",
    ) -> RAGResult:
        """
        Executes query retrieval, LLM response generation, and claim-level hallucination scoring.
        """
        total_start = time.perf_counter()

        # Step 1: Retrieval
        t_ret_start = time.perf_counter()
        retrieved_items = self.kb.retrieve(query=query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0

        retrieved_chunks = []
        context_snippets = []
        for chunk, score in retrieved_items:
            retrieved_chunks.append({
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "source_type": chunk.source_type,
                "page_or_slide": chunk.page_or_slide,
                "similarity_score": round(score, 4),
                "text": chunk.text,
            })
            context_snippets.append(
                f"[Source: {chunk.source_file} (Slide/Page {chunk.page_or_slide})]: {chunk.text}"
            )

        combined_context = "\n\n".join(context_snippets)

        # Step 2: Generation via Ollama
        system_prompt = (
            "You are an academic Computational Linguistics assistant. Answer the user query "
            "based strictly and accurately on the provided course lecture context. "
            "Keep the answer concise, factually grounded, and precise in 2 to 4 sentences."
        )

        if stress_test:
            # Instruct model to speculate or introduce unverified details to test the detector
            generation_prompt = (
                f"Context from Computational Linguistics lecture materials:\n{combined_context}\n\n"
                f"Question: {query}\n\n"
                f"Instructions: Answer the question, but feel free to extrapolate, invent related historical trivia, "
                f"and speculate on future dates or external claims not found in the context."
            )
            temp = 0.8
        else:
            generation_prompt = (
                f"Context from Computational Linguistics lecture materials:\n{combined_context}\n\n"
                f"Question: {query}\n\n"
                f"Instructions: Answer the question directly using ONLY facts corroborated by the context above."
            )
            temp = 0.15

        gen_res = self.ollama.generate(
            prompt=generation_prompt,
            system_prompt=system_prompt,
            model=model_name,
            temperature=temp,
        )
        llm_answer = gen_res["text"]
        llm_ms = gen_res["duration_ms"]
        used_model = gen_res["model"]

        # Step 3: Hallucination Detection
        t_det_start = time.perf_counter()
        if detector_model_type and not detector_model_type.endswith(".joblib"):
            detector_path = f"models/{detector_model_type}.joblib"
        else:
            detector_path = detector_model_type
            
        detector = None
        if detector_path and os.path.exists(detector_path):
            detector_obj, _ = load_detector_model(detector_path)
            detector = detector_obj

        detection_result = predict_hallucinations(
            question=query,
            answer=llm_answer,
            context=combined_context,
            model=detector,
        )
        detection_ms = (time.perf_counter() - t_det_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        # Step 4: Extract Composite Scores
        total_claims = detection_result.get("total_claims", 0)
        overall_risk_percent = detection_result.get("overall_risk_percent", 0.0)
        faithfulness_score = max(0.0, round(100.0 - overall_risk_percent, 1))

        claims = detection_result.get("claims", [])
        hallucinated_count = sum(1 for c in claims if c.get("prediction") == "Hallucination")
        supported_count = sum(1 for c in claims if c.get("prediction") == "Supported")

        if total_claims == 0:
            status = "No Claims"
        elif hallucinated_count == 0:
            status = "Supported"
        elif supported_count == 0:
            status = "Hallucinated"
        else:
            status = "Partially Supported"

        return RAGResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            combined_context=combined_context,
            llm_answer=llm_answer,
            llm_model=used_model,
            timing={
                "retrieval_ms": round(retrieval_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "detection_ms": round(detection_ms, 1),
                "total_ms": round(total_ms, 1),
                "tokens_per_sec": gen_res.get("tokens_per_sec", 0.0),
            },
            hallucination_analysis=detection_result,
            faithfulness_score=faithfulness_score,
            overall_risk_percent=round(overall_risk_percent, 1),
            supported_claims_count=supported_count,
            hallucinated_claims_count=hallucinated_count,
            total_claims=total_claims,
            status=status,
        )


# Global singleton instance for easy import in Streamlit / APIs
_pipeline_instance: Optional[RAGHallucinationPipeline] = None


def get_rag_pipeline(force_reload: bool = False) -> RAGHallucinationPipeline:
    global _pipeline_instance
    if _pipeline_instance is None or force_reload:
        _pipeline_instance = RAGHallucinationPipeline()
        _pipeline_instance.initialize()
    return _pipeline_instance


if __name__ == "__main__":
    print("Testing RAG Hallucination Pipeline...")
    pipeline = get_rag_pipeline()
    sample_query = "What topics and grading weights are in the Computational Linguistics course?"
    result = pipeline.run_query(sample_query, top_k=2)
    print("\n--- QUERY RESULT ---")
    print(f"Query: {result.query}")
    print(f"Model: {result.llm_model}")
    print(f"Timing: {result.timing}")
    print(f"Faithfulness Score: {result.faithfulness_score}% (Risk: {result.overall_risk_percent}%)")
    print(f"Status: {result.status} (Supported: {result.supported_claims_count}, Hallucinated: {result.hallucinated_claims_count})")
    print(f"\nAnswer:\n{result.llm_answer}\n")
    for c in result.hallucination_analysis.get("claims", []):
        print(f"  [{c['prediction']}] (Risk {c['hallucination_risk']}%) {c['claim_text']}")

