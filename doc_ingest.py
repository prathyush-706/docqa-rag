from typing import List, Tuple
import os
from transformers import AutoTokenizer

def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from PyPDF2 import PdfReader
        txt = ""
        for p in PdfReader(path).pages:
            txt += p.extract_text() + "\n"
        return txt
    else:
        return open(path, encoding="utf-8").read()

def chunk_text(text: str,
               tokenizer_name: str = "UW/OLMo2-8B-SuperBPE-t180k",
               max_tokens: int = 500,
               overlap: int = 100
              ) -> List[str]:
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Encode the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Create chunks
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        # Decode the chunk of tokens back to text
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        start += max_tokens - overlap
    
    return chunks
