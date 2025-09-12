import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Load dataset
df = pd.read_csv('train.csv')
# Slice to ~100 for testing (optional)
df = df.head(100)
print(df)
documents = [f"Question: {row['Question']}\nAnswer: {row['Answer']}" for _, row in df.iterrows()]

# Embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(documents)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# Save
faiss.write_index(index, 'faiss_index.index')
with open('documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print("Index built!")