import os
from openai import OpenAI   # ✅ use OpenAI client instead of Groq
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

# =========================
# 🔑 Manually set OpenRouter API Key
# =========================
api_key = "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"   # ⬅️ put your key here

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set. Please add your API key in the code.")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# =========================
# Load embeddings model, FAISS index, and documents
# =========================
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('faiss_index.index')

with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f)

# =========================
# Retrieval function
# =========================
def retrieve(query, top_k=3):
    query_emb = embedder.encode([query])
    _, indices = index.search(np.array(query_emb).astype('float32'), top_k)
    return [documents[i] for i in indices[0]]

# =========================
# Answer generation function
# =========================
def generate_answer(query, context, model="openai/gpt-3.5-turbo"):
    prompt = f"""
    You are a helpful medical FAQ assistant. Use the following context to answer the user's query accurately and naturally. 
    Stick to the provided information; do not add external knowledge or speculate.

    Context:
    {'\n\n'.join(context)}

    Query: {query}

    Answer:
    """
    chat_completion = client.chat.completions.create(
        model=model,   # ✅ e.g., "openai/gpt-3.5-turbo" or "openai/gpt-5"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return chat_completion.choices[0].message.content

# =========================
# RAG pipeline
# =========================
def rag_query(query, model="openai/gpt-3.5-turbo"):
    context = retrieve(query)
    if not context:
        return "Sorry, I couldn't find relevant information."
    return generate_answer(query, context, model)


