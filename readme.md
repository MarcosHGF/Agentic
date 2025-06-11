# 🧠 Whisper RAG Assistant

A local Retrieval-Augmented Generation (RAG) assistant that reads and answers questions about your documents using natural language. This project combines **LangChain**, **FAISS**, and **Ollama** running the `qwen:0.6b` model to generate context-aware responses based on your local files.

---

## 📚 Features

- 📄 Load and process local documents (PDF, TXT, DOCX)
- 🧠 Embed content using HuggingFace transformers
- 🗂️ Index and search vectors with FAISS
- 🤖 Answer natural language questions using `qwen:0.6b` via Ollama
- 🛠️ Modular tool integration with LangChain's agent system

---

## ⚙️ Technologies Used

- [Python 3.10+](https://www.python.org)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com) — running `qwen:0.6b`
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## 🚀 Getting Started

### 1. Install Ollama

Download and install [Ollama](https://ollama.com/download) for your platform.

Then pull the model:

```bash
ollama pull qwen:0.6b
```

### 2. Clone the Repository
git clone https://github.com/MarcosHGF/Agentic.git
cd Agentic

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Your Documents
Place your .pdf, .txt, or .docx files inside the docs/ folder.

Example:
```bash
whisper-rag-assistant/
├── docs/
│   ├── my_notes.pdf
│   └── project_spec.txt
```

### 5. Run ollama and the Assistant
```bash
ollama run qwen:0.6b
python main.py
```

The system will:
-Check for an existing FAISS index
-If not found, index all documents
-Use qwen:0.6b via Ollama to respond to your queries

### Example Usage
```bash
> What is the main objective of the project?

✅ FAISS index found. Loading...

Answer:
The project aims to create an AI-powered chatbot that helps users choose items based on preferences, occasion, and mood. It combines computer vision with natural language processing for personalized recommendations.
```
### Project Structure
```bash
Agentic/
├── main.py                # Entry point script
├── doc_indexer.py         # Handles loading, splitting, and indexing documents
├── tools/                 # LangChain tools (e.g., RAG tool)
│   ├── tools.py           
│   └──mathtools.py
├── docs/                  # Directory to store documents
├── faiss_index/           # FAISS index directory (auto-generated)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

### Model Used
```bash
The assistant runs entirely locally using the Qwen 0.6b language model via Ollama. This ensures:

-Privacy — your documents never leave your machine
-Speed — no external API calls
-Flexibility — use your own tools, data and logic
```