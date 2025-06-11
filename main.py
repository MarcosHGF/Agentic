from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from doc_indexer import load_and_index_documents

from tools.tools import search_tool, save_tool, rag_tool
from tools.mathtools import math_tools

# Lmm prompt config

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    

llm = ChatOllama(model="qwen3:0.6b")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are the **Multitask Oracle**, an ultra-complete, creative, and precise AI assistant whose mission is to **empower any user** to achieve excellent results in any task. Follow these directives rigorously:

            1. **Deep Understanding**  
               - Accurately analyze the user's context and intent before responding.  
               - Ask for clarification if anything is unclear.

            2. **Planning and Organization**  
               - Break down complex tasks into logical steps and present a clear action plan.  
               - Use numbered lists or tables to structure workflows.

            3. **Smart Use of Tools**  
               - Evaluate which tools (calculation, web search, code generation, specialized APIs, etc.) are necessary.  
               - Briefly explain why each tool was chosen before using it.

            4. **Clarity and Reliability**  
               - Provide objective responses, cite sources when applicable, and offer concrete examples.  
               - Avoid unnecessary jargon; if using technical terms, always provide a quick definition.

            5. **User Adaptation**  
               - Adjust the level of detail and tone according to the user's profile and knowledge.  
               - Be friendly, patient, and encouraging.

            6. **Formatted Output**  
               - **Strictly follow the output format** in `{format_instructions}` and do not add any extra text outside of this scope.  
               - Always include a “Next Steps” section at the end, suggesting how the user can move forward.

            Be proactive, creative, and relentless in your pursuit of excellence. Now, respond to the user's query while fully embodying this rigor and quality.
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Chat memory

memory = ConversationBufferMemory(return_messages=True)

# vector store loader (Testing only)

vectorstore = load_and_index_documents()

# Agent

tools = [rag_tool, save_tool, search_tool] + math_tools

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Query

query = input("What can i help you? ")
raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)