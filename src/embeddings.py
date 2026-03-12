"""
Embedding client for generating text embeddings.
Supports both a FREE local model (sentence-transformers) and the OpenAI API.

Default: Local model (all-MiniLM-L6-v2) — fast, free, no API key needed.
Set EMBEDDING_PROVIDER=openai in .env to use OpenAI instead.
"""

import os
import numpy as np
from typing import List


class EmbeddingClient:
    """Generates text embeddings using a local model (default) or OpenAI API."""

    MAX_TEXT_LENGTH = 8000  # Approx safe char limit per text

    def __init__(self, provider: str = None, api_key: str = None):
        """
        Args:
            provider: 'local' (default, free) or 'openai' (paid API).
            api_key: Required only if provider is 'openai'.
        """
        self.provider = provider or os.getenv("EMBEDDING_PROVIDER", "local")
        self._model = None
        self._dimensions = None

        if self.provider == "openai":
            from openai import OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when using OpenAI embeddings. "
                    "Set it in your .env file or use EMBEDDING_PROVIDER=local."
                )
            self.client = OpenAI(api_key=self.api_key)
            self.openai_model = "text-embedding-3-small"
            self._dimensions = 1536
        else:
            # Local model — loaded lazily on first use
            self._local_model_name = "all-MiniLM-L6-v2"
            self._dimensions = 384

    @property
    def DIMENSIONS(self):
        return self._dimensions

    def _get_local_model(self):
        """Lazily load the local SentenceTransformer model."""
        if self._model is None:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            from sentence_transformers import SentenceTransformer
            print(f"   [Embeddings] Loading local model: {self._local_model_name}...")
            self._model = SentenceTransformer(self._local_model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Returns:
            NumPy array of shape (len(texts), DIMENSIONS) with L2-normalized vectors.
        """
        if not texts:
            return np.empty((0, self.DIMENSIONS), dtype=np.float32)

        # Truncate overly long texts
        texts = [t[:self.MAX_TEXT_LENGTH] if len(t) > self.MAX_TEXT_LENGTH else t for t in texts]
        texts = [t if t.strip() else " " for t in texts]

        if self.provider == "openai":
            vectors = self._embed_openai(texts)
        else:
            vectors = self._embed_local(texts)

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        return vectors

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Embed using the local SentenceTransformer model (FREE)."""
        model = self._get_local_model()
        print(f"   [Embeddings] Encoding {len(texts)} chunks locally...")
        vectors = model.encode(
            texts,
            show_progress_bar=len(texts) > 500,
            batch_size=256,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed using the OpenAI API (paid)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        MAX_BATCH = 2048
        batches = []
        for i in range(0, len(texts), MAX_BATCH):
            batches.append((i, texts[i : i + MAX_BATCH]))

        all_embeddings = [None] * len(texts)
        total_batches = len(batches)

        print(f"   [Embeddings] Embedding {len(texts)} chunks via OpenAI ({total_batches} batches)...")

        def _call_api(batch):
            response = self.client.embeddings.create(
                model=self.openai_model,
                input=batch,
            )
            return [item.embedding for item in response.data]

        if total_batches == 1:
            result = _call_api(batches[0][1])
            for j, emb in enumerate(result):
                all_embeddings[j] = emb
        else:
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_map = {
                    executor.submit(_call_api, batch): (idx, batch)
                    for idx, batch in batches
                }
                done = 0
                for future in as_completed(future_map):
                    idx, batch = future_map[future]
                    result = future.result()
                    for j, emb in enumerate(result):
                        all_embeddings[idx + j] = emb
                    done += 1
                    print(f"   [Embeddings] Progress: {done}/{total_batches}")

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            NumPy array of shape (1, DIMENSIONS), L2-normalized.
        """
        return self.embed([query])
