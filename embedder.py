import os
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name=MODEL_NAME, cache_path=None):
        self.model = SentenceTransformer(model_name)
        self.cache_path = cache_path
        self.vectors = None
        self.docs = None

    def encode(self, docs):
        vectors = self.model.encode(docs)
        return np.array(vectors)

    def encode_with_cache(self, docs):
        if self.cache_path and os.path.exists(self.cache_path):
            data = joblib.load(self.cache_path)
            self.vectors, self.docs = data['vectors'], data['docs']
            if self.docs == docs:
                return self.vectors
        vectors = self.encode(docs)
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            joblib.dump({'vectors': vectors, 'docs': docs}, self.cache_path)
        return vectors
