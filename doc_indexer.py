import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 📁 Diretórios
DOCUMENTS_DIR = "docs"
INDEX_DIR = "faiss_index"

def load_and_index_documents():
    # 🔤 Carregar embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 🔍 Tentar carregar índice FAISS existente
    index_file = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        print("✅ FAISS index found. Loading...")
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    print("📁 FAISS index not found. Creating index from documents...")

    # 📄 Carregar todos os documentos da pasta 'docs'
    all_docs = []
    for file in os.listdir(DOCUMENTS_DIR):
        if not file.lower().endswith((".pdf", ".txt", ".docx")):
            continue  # carregar apenas arquivos suportados
        file_path = os.path.join(DOCUMENTS_DIR, file)
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file  # Rastrear nome do arquivo
        all_docs.extend(docs)

    # ✂️ Melhor splitter com tamanho e overlap maiores para mais contexto
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,   # aumentar o chunk para pegar parágrafos completos
        chunk_overlap=300  # overlap maior para manter contexto em bordas
    )
    chunks = splitter.split_documents(all_docs)

    print(f"🔢 Total chunks criados: {len(chunks)}")

    # 🧠 Construir e salvar índice FAISS
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)
    print("💾 FAISS index created and saved successfully.")

    return db
