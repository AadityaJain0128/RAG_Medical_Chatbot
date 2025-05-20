# 🩺 RAG-Based Health Assistant Chatbot

This project is a Retrieval-Augmented Generation (RAG) based chatbot built using **LangChain**, **Groq (LLaMA3)**, and **HuggingFace MiniLM** embeddings. It answers medical-related queries by retrieving contextually relevant information from a pre-embedded knowledge base.

---

## 🚀 Features

- Ask medical questions and get AI-generated answers.
- Uses **FAISS vector store** for efficient semantic search.
- Powered by **LLaMA3-8B via Groq API** for fast, accurate answers.
- Retrieves supporting context from uploaded documents.
- Streamlit-based simple user interface.

---

## 🧠 How It Works

1. Input question via UI.
2. The question is embedded using `all-MiniLM-L6-v2`.
3. FAISS retrieves top relevant chunks from vector store.
4. Groq’s LLaMA3-8B model generates a response using the retrieved context.
5. The app returns the final answer and the source chunks used.

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API (LLaMA3)](https://groq.com/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [HuggingFace Transformers](https://huggingface.co/)
- Python 🐍

---

## 🧪 Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AadityaJain0128/RAG_Health_Bot.git