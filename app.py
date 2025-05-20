import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from os import environ


load_dotenv()
API_KEY = environ.get("API_KEY")


model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = FAISS.load_local("vectors", model, allow_dangerous_deserialization=True)
retriever = vectors.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    api_key=API_KEY,
    model="llama3-8b-8192",
    temperature=0
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


st.title("Medical Health Assistant")
query = st.text_input("Ask a Question !")

if query:
    with st.spinner("Thinking ..."):
        result = chain.invoke({"query": query})
        st.markdown("## Answer:")
        st.write(result["result"])

        st.markdown("### Sources:")
        for doc in result["source_documents"]:
            st.markdown(f"- {doc.page_content[:250]}...")
        st.markdown("> These sources may or may not be related to the answer.")
