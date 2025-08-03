import fitz
import re
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import streamlit as st
import os
import google.generativeai as genai

# ✅ Use Streamlit secrets when deployed
api_key = st.secrets.get("GOOGLE_API_KEY", None)
if api_key is None:
    api_key = os.getenv("GOOGLE_API_KEY")  # Local fallback

# ✅ PDF path
pdf_path = "modules/Reminiscences.pdf"
doc = fitz.open(pdf_path)

# ✅ Extract text
pages_text = []
for i in range(len(doc)):
    text = doc[i].get_text()
    text = re.sub(r'\n+', ' ', text).strip()
    if text:
        pages_text.append(text)

# ✅ Chunking
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = []
metadata = []

for page_num, page_text in enumerate(pages_text):
    for chunk_id, chunk in enumerate(chunk_text(page_text, chunk_size=400)):
        chunks.append(chunk)
        metadata.append({"page": page_num+1, "chunk_id": chunk_id})

print(f"Total chunks created: {len(chunks)}")

# ✅ Generate embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_numpy=True)

# ✅ Store FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "livermore_index.faiss")
with open("livermore_metadata.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "metadata": metadata}, f)

# ✅ Load FAISS index and metadata
index = faiss.read_index("livermore_index.faiss")
with open("livermore_metadata.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
metadata = data["metadata"]

# ✅ Embedding model
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ✅ RAG Function
def get_livermore_answer(question, k=3):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are Jesse Livermore, the legendary trader.
    Use ONLY the context from *Reminiscences of a Stock Operator* to answer the question.
    Answer in a practical, trading-focused tone like Livermore would.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    return response.text
