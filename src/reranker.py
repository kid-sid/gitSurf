from typing import List, Dict
import os
import contextlib
from sentence_transformers import CrossEncoder

# Global cache for the Cross-Encoder model so it's only loaded once per session
_MODEL_CACHE: Dict[str, CrossEncoder] = {}

class CrossEncoderReranker:
    """Reranks candidate search results using a local BERT-based cross-encoder.
    Provides a more accurate relevance score than raw vector similarity
    and works entirely locally after the initial model download.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Args:
            model_name: The name of the cross-encoder model to use.
            Default is ms-marco-MiniLM-L-6-v2 (fast and effective)."""

        global _MODEL_CACHE

        if model_name in _MODEL_CACHE:
            self.model = _MODEL_CACHE[model_name]
            return

        print(f"   [Reranker] Loading local model: {model_name}...")
        try:
            # Suppress tqdm and other loading logs
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    self.model = CrossEncoder(model_name)
                    _MODEL_CACHE[model_name] = self.model
        except Exception as e:
            print(f"   [Reranker] Warning: Failed to load local model '{model_name}': {e}")
            print("              Reranking will be skipped.")
            self.model = None

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Scores each chunk for relevance to the query and returns the top_k.
        """
        if not chunks or not self.model:
            return chunks[:top_k]

        # Preparing pairs for the cross-encoder: [(query, chunk1), (query, chunk2), ...]
        pairs = [(query, chunk['content'][:1000]) for chunk in chunks] # Truncating content slightly to ensure it fits the context window

        try:
            scores = self.model.predict(pairs)
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
                scored_chunks.append(chunk)
            scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_chunks[:top_k]

        except Exception as e:
            print(f"Error during reranking: {e}")
            return chunks[:top_k]
