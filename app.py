import streamlit as st
import os
from pdf_extractor import extract_texts_from_folder
from embedder import Embedder
from vector_store import VectorStore
from llm import LLM
import numpy as np

DATA_DIR = "data/pdfs/"
CACHE_PATH = "cache/embeddings.pkl"
MODEL_DIM = 384  # For MiniLM

st.title("🤖 Youri's Chatbot (RAG PDF Assistant)")
st.markdown("Ask a question about your private PDF documents.")

# Upload PDF
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)
    for uploaded in uploaded_files:
        with open(os.path.join(DATA_DIR, uploaded.name), "wb") as f:
            f.write(uploaded.getbuffer())
    st.success("PDFs uploaded!")

# Extraction
if st.button("Index PDFs"):
    pdf_texts = extract_texts_from_folder(DATA_DIR)
    docs = list(pdf_texts.values())
    embedder = Embedder(cache_path=CACHE_PATH)
    vectors = embedder.encode_with_cache(docs)
    vector_store = VectorStore(MODEL_DIM)
    vector_store.add(vectors, docs)
    st.session_state["vector_store"] = vector_store
    st.session_state["docs"] = docs
    st.success("Indexing complete!")

# Question
question = st.text_input("Ask your question")
if st.button("Send") and "vector_store" in st.session_state:
    embedder = Embedder()
    query_vec = embedder.encode([question])
    results = st.session_state["vector_store"].search(query_vec, k=5)
    context = "\n".join([doc for doc, _ in results])
    prompt = f"Here is a document:\n{context}\n\nAnswer this question: {question}"
    llm = LLM()
    response = llm.generate(prompt)
    st.markdown("**Answer:** " + response)
