import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, vectors, docs):
        self.index.add(np.array(vectors))
        self.docs.extend(docs)

    def search(self, query_vec, k=5):
        D, I = self.index.search(np.array(query_vec), k)
        return [(self.docs[i], D[0][idx]) for idx, i in enumerate(I[0])]
