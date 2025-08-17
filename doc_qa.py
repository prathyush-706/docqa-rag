import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from embedding import DocumentIndexer
from generate_answer import generate_answer
from reranker import ReRanker
from doc_ingest import load_text, chunk_text
from typing import List, Dict

class DocumentQA:
    def __init__(self,
                 embedder_name="all-MiniLM-L6-v2",
                 reranker_model=None):
        # 1. Indexer
        self.indexer = DocumentIndexer(model_name=embedder_name)
        # 2. Optional reranker
        self.reranker = ReRanker(reranker_model) if reranker_model else None
        # 3. Conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def ingest(self, path: str):
        text = load_text(path)
        chunks = chunk_text(text)
        self.indexer.add_chunks(chunks)

    def query(self, question: str,
              k: int = 10,
              use_rerank: bool = True,
              max_history: int = 5) -> str:
        # 1. retrieve top-k by vector similarity
        q_embed = self.indexer.embedder.encode([question],
                                                convert_to_numpy=True,
                                                normalize_embeddings=True)
        D, I = self.indexer.index.search(q_embed, k)
        candidates = [ self.indexer.texts[i] for i in I[0] ]
        # 2. optional rerank
        if use_rerank and self.reranker:
            candidates, _ = zip(*self.reranker.rerank(question, candidates, top_k=k))
        # 3. generate with conversation history
        answer = generate_answer(
            question, 
            list(candidates[:k]), 
            conversation_history=self.conversation_history[-max_history:]  # Only use last max_history conversations
        )
        # 4. update conversation history
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })
        return answer
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
if __name__ == "__main__":
    qa_system = DocumentQA()
    qa_system.ingest("Prathyush_Resume_TA_Position.pdf")
    print("Ready to answer questions about the document.")
    print("Type 'quit' to exit, 'clear' to clear conversation history.")
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == 'quit':
            break
        elif question.lower() == 'clear':
            qa_system.clear_history()
            print("Conversation history cleared.")
            continue
        answer = qa_system.query(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")
