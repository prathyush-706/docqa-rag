from google import genai
import os
from typing import List, Dict
from dotenv import load_dotenv
from google.genai import types
import time

load_dotenv()

# Initialize the Gemini client
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
print("GOOGLE_API_KEY: ", os.getenv("GOOGLE_API_KEY"))


def generate_answer(question: str,
                    context_chunks: List[str],
                    conversation_history: List[Dict[str, str]] = None,
                    model="gemini-2.5-flash-preview-05-20",
                    max_tokens: int = 256
                   ) -> str:
    # Build conversation context
    conversation_context = ""
    if conversation_history:
        conversation_context = "\n\nPrevious conversation:\n"
        for entry in conversation_history:
            conversation_context += f"Question: {entry['question']}\nAnswer: {entry['answer']}\n\n"
    
    prompt = (
        "You are an expert assistant. Answer the question based on the context and previous conversation history provided below. For questions whose answers is not directly provided in the context, use your reasoning skills based on the content in the document to answer the question.\n\n"
        "CONTEXT:\n" + "\n\n---\n\n".join(context_chunks) +
        conversation_context +
        "\n\nCURRENT QUESTION:\n" + question +
        "\n\nAnswer:"
    )
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens
            )
        )
        return response.text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise  # This will trigger the retry
