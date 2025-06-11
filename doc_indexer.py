import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Docs directories
DOCUMENTS_DIR = "docs"
INDEX_DIR = "faiss_index"

def load_and_index_documents():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load Index (if exist)
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("✅ FAISS index found. Loading...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("📁 📁 FAISS index not found. Creating index from documents...")

    all_docs = []
    for file in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, file)
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file  # Save the doc name
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)  # 💾 Save de index

    return db
