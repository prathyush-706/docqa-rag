import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class DocumentIndexer:
    def __init__(self, model_name="all-MiniLM-L6-v2", dim: int = 384):
        self.embedder = SentenceTransformer(model_name)
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)     # inner-product for cosine
        self.texts: List[str] = []

    def add_chunks(self, chunks: List[str]):
        embs = self.embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embs)
        self.texts.extend(chunks)

    def save(self, index_path="faiss.index", txt_path="chunks.txt"):
        faiss.write_index(self.index, index_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n<<CHUNK>>\n".join(self.texts))

    def load(self, index_path="faiss.index", txt_path="chunks.txt"):
        self.index = faiss.read_index(index_path)
        with open(txt_path, "r", encoding="utf-8") as f:
            self.texts = f.read().split("\n<<CHUNK>>\n")
