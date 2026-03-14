from app.core.config import GROQ_API_KEY
from groq import Groq

import os

client = Groq(api_key=GROQ_API_KEY)


def get_answer(question:str, chunks:list):
    context = "\n".join([chunk.payload["text"] for chunk in chunks])
    final_prompt = f"Context:\n{context}\n\nQuestion:{question}"
    
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's question based only on the provided context. If the answer is not found in the context, say 'I don't know based on the provided document'."        },
        {
            "role": "user",
            "content": final_prompt,
        }
    ],
    model="llama-3.3-70b-versatile",
    )
    
    return chat_completion.choices[0].message.content