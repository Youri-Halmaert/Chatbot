# 📚 Youri's Chatbot (RAG PDF Assistant)

This project creates an assistant that answers natural language questions based on your private PDF documents, using Retrieval-Augmented Generation (RAG).


## Access the app here: [PDF-QA Chatbot](https://pdf-app-chatbot-4ovbnutopxatdm8vevdtpk.streamlit.app/)


## ⚙️ Pipeline

1. **PDF Text Extraction**: PyMuPDF
2. **Embeddings**: SentenceTransformers (local) or OpenAI (optional)
3. **Vector Search**: FAISS
4. **Answer Generation**: GPT (API) or local LLM
5. **Interface**: Streamlit

---

## 📂 Project Structure

```
youri_chatbot/
│
├── app.py                  # Streamlit interface
├── pdf_extractor.py        # PDF text extraction
├── embedder.py             # Embedding generation
├── vector_store.py         # Vector indexing/search
├── llm.py                  # Answer generation (API/local)
├── requirements.txt        # Python dependencies
├── Dockerfile              # (optional) Deployment
├── README.md               # Documentation + diagram
├── data/
│   └── pdfs/               # PDFs to index
└── cache/                  # (optional) Embedding cache
```

---

## 🗂️ Architecture Diagram

```
[PDFs] → [Text Extraction] → [Embeddings] → [FAISS] ← [User Question]
                                               ↓
                                    [Relevant Context]
                                               ↓
                                   [LLM (local/API)]
                                               ↓
                                   [Displayed Answer]
```

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Put your PDFs in `data/pdfs/`
3. Launch the interface:
   ```bash
   streamlit run app.py
   ```

---

## 💡 Tips
- Built-in logging and error handling
- Local and API mode supported
- Multilingual
- Embedding cache enabled
- README with architecture diagram


