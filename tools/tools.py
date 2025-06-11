from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from datetime import datetime

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
        results = vectorstore.similarity_search(query, k=5)
        if not results:
            return "Nenhum conteúdo relevante foi encontrado nos documentos."
        return "\n\n".join([doc.page_content for doc in results])
    return query_local_docs


rag_tool = Tool(
    name="query_documents",
    func=get_query_local_docs(load_and_index_documents()),
    description=(
        "Use this to find and analyze content from uploaded documents like PDFs or text files. "
        "Ideal for answering questions based on document content."
    )
)

