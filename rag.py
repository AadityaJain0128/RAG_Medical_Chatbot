import pandas as pd
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


df = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/pqa_labeled/train-00000-of-00001.parquet")

docs = []
for _, row in df.iterrows():
  content = "Contexts:\n" + ", ".join(row["context"]["contexts"]) + "\n\nLabels:\n" + ", ".join(row["context"]["labels"]) + "\n\nMeshes:\n" + ", ".join(row["context"]["meshes"]) + "\n\nAnswer:\n" + row["long_answer"] + "\n\nFinal Decision:\n" + row["final_decision"]
  metadata = {
      "pubid" : row["pubid"],
      "question" : row["question"]
  }

  docs.append(Document(page_content=content, metadata=metadata))


model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = FAISS.from_documents(docs, model)
vectors.save_local("vectors")
print("Vectors are stored in '/vectors'.")