from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from datetime import datetime
import logging

# === Save tool ===

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_structured_text_file",
    func=save_to_txt,
    description=(
        "Use this tool to **save research summaries, structured data, or results** into a local `.txt` file. "
        "Pass the content as a plain text string. The tool will store it persistently for later use or reference."
    )
)

# === Web search tool ===

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search_web_information",
    func=search.run,
    description=(
        "Use this tool to **search the internet** for up-to-date information on any topic. "
        "It is ideal for retrieving facts, news, statistics, and external references that are not already known."
        "Use **ONLY** when is needed and required by the user to **search in the internet**"
    )
)

# === RAG tool ===
from doc_indexer import load_and_index_documents

def get_query_local_docs(vectorstore):
    def query_local_docs(query: str) -> str:
        logging.info(f"Querying vectorstore with query: {query}")
        results = vectorstore.similarity_search(query, k=5)  # aumento para 5 para maior recall
        if not results:
            return "Nothing relevant found in the document."
        
        # Concatenar resultados com fonte e limite de tamanho
        response_chunks = []
        total_length = 0
        max_length = 3000  # Limitar para evitar excesso de texto
        
        for doc in results:
            text = doc.page_content.strip()
            source = doc.metadata.get("source", "unknown source")
            snippet = f"[Source: {source}]\n{text}"
            
            if total_length + len(snippet) > max_length:
                break
            response_chunks.append(snippet)
            total_length += len(snippet)
        
        response = "\n\n---\n\n".join(response_chunks)
        return response

    return query_local_docs


def build_rag_tool():
    # Carregar índice apenas uma vez ao construir a tool
    vectorstore = load_and_index_documents()
    return Tool(
        name="query_documents",
        func=get_query_local_docs(vectorstore),
        description=(
            "Use this to find and analyze content from uploaded documents like PDFs or text files. "
            "Ideal for answering detailed questions based on document content with source references."
        )
    )

# == analyze tool ==

from langchain_core.language_models import BaseLanguageModel

def make_analyze_documents_tool(llm: BaseLanguageModel) -> Tool:
    # Pré-carregar índice para evitar carregamentos repetidos
    vectorstore = load_and_index_documents()
    
    def analyze_documents(_: str) -> str:
        logging.info("Analyzing top documents from the vectorstore.")
        # Buscar um conjunto mais amplo e mais relevante para análise geral
        docs = vectorstore.similarity_search("summary overview main themes", k=10)
        
        combined_text = ""
        total_length = 0
        max_length = 3500  # Pode ajustar conforme contexto do LLM
        
        for doc in docs:
            text = doc.page_content.strip()
            # Acrescentar fonte para rastreabilidade
            source = doc.metadata.get("source", "unknown source")
            snippet = f"[Source: {source}]\n{text}\n\n"
            
            if total_length + len(snippet) > max_length:
                break
            combined_text += snippet
            total_length += len(snippet)

        # Prompt para análise detalhada e estruturada
        prompt = (
            "You are a professional analyst. Read the following content extracted from documents "
            "and provide a detailed summary. Include the main themes, recurring ideas, and any "
            "interesting patterns or conclusions you can draw:\n\n"
            f"{combined_text}\n\n"
            "Please structure the summary in bullet points and include references to sources when applicable."
        )

        # Chamada ao LLM — usar método correto para chamar o modelo (invoke, generate ou outro)
        response = llm.invoke(prompt)
        return response

    return Tool(
        name="analyze_documents",
        func=analyze_documents,
        description="Use this to generate a high-level, detailed analysis of the content in the uploaded documents."
    )