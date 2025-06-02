from sentence_transformers import CrossEncoder
from typing import List, Tuple
class ReRanker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)

    def rerank(self, question: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        inputs = [[question, c] for c in candidates]
        scores = self.model.predict(inputs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
