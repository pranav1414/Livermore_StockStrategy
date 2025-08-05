import fitz
import re
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import streamlit as st
import os

# ✅ Use Streamlit secrets when deployed
api_key = st.secrets.get("GOOGLE_API_KEY", None)
if api_key is None:
    api_key = os.getenv("GOOGLE_API_KEY")  # Local fallback
genai.configure(api_key=api_key)

# ✅ PDF path
pdf_path = "modules/Reminiscences.pdf"
doc = fitz.open(pdf_path)

# ✅ Extract book text
pages_text = []
for i in range(len(doc)):
    text = doc[i].get_text()
    text = re.sub(r'\n+', ' ', text).strip()
    if text:
        pages_text.append(text)

# ✅ Chunking helper
def chunk_text(text, chunk_size=500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

chunks = []
metadata = []

# ✅ Book chunks
for page_num, page_text in enumerate(pages_text):
    for chunk_id, chunk in enumerate(chunk_text(page_text, chunk_size=400)):
        chunks.append(chunk)
        metadata.append({"source": "book", "page": page_num+1, "chunk_id": chunk_id})

# ✅ Add Q&A chunks from CSV
qa_df = pd.read_csv("modules/Livermore_QA.csv")
for i, row in qa_df.iterrows():
    q, a = row["Question"], row["Answer"]
    qa_chunk = f"Q: {q}\nA: {a}"
    chunks.append(qa_chunk)
    metadata.append({"source": "qa", "qa_id": i+1})

print(f"✅ Total chunks (book + Q&A): {len(chunks)}")

# ✅ Generate embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks, convert_to_numpy=True)

# ✅ Save FAISS index + metadata
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "livermore_index.faiss")
with open("livermore_metadata.pkl", "wb") as f:
    pickle.dump({"chunks": chunks, "metadata": metadata}, f)

# ✅ Reload for RAG
index = faiss.read_index("livermore_index.faiss")
with open("livermore_metadata.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
metadata = data["metadata"]
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ✅ RAG answer function
def get_livermore_answer(question, k=3):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are Jesse Livermore, the legendary trader.
    Use ONLY the context below to answer the question.
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


