import os
import pickle
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
API_KEY = os.getenv("OPENROUTER_API_KEY")

embedder = SentenceTransformer(EMBEDDING_MODEL)
index = faiss.read_index('faiss_index.index')
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

def retrieve(query, top_k=3, min_sim=0.5):
    query_emb = embedder.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        # Normalize distance to similarity and filter
        sim = 1 / (1 + dist)
        if sim > min_sim:
            results.append(documents[idx])
    return results

@lru_cache(maxsize=128)
def generate_answer(query, context, model="openai/gpt-3.5-turbo"):
    prompt = f"""
    You are a helpful medical FAQ assistant. Use the following context to answer the user's query accurately.
    Context:
    {'\n\n'.join(context)}
    Query: {query}
    Answer:
    """
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return chat_completion.choices[0].message.content
    except Exception:
        # Fallback model
        chat_completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
        return chat_completion.choices[0].message.content

def rag_query(query, model="openai/gpt-3.5-turbo"):
    context = retrieve(query)
    if not context:
        return "Sorry, I couldn't find relevant information."
    return generate_answer(query, tuple(context), model)
