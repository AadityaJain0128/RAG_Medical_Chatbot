import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from os import environ
from sentence_transformers.util import cos_sim


load_dotenv()
API_KEY = environ.get("API_KEY")


model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
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
query = st.text_area("Ask a Question !")

if query:
    with st.spinner("Thinking ..."):
        result = chain.invoke({"query": query})

        st.markdown("## Answer:")
        st.write(result["result"])

        st.markdown("### Sources:")
        for doc in result["source_documents"]:
            context = doc.page_content

            # Computing cosine similarity between query and context vectors
            similarity = cos_sim([model.embed_query(query)], [model.embed_query(context)]).item()
            mp = round(similarity * 100, 2)

            st.markdown(f"- {context[:250]} ...  \n**{mp}% match**")
        st.markdown("> These sources may or may not be related to the answer.")
