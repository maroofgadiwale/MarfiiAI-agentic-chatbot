# Program to implement chat-bot:
# Doc Loading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
# ----------------
from langgraph.graph import StateGraph,START
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,SystemMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import os
import requests
import sqlite3
from typing import TypedDict,Annotated,Dict,Any,Optional
from langgraph.graph.message import add_messages
# Tools libraries:
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# LLM:
model = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API")
)

# form embeddings:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Tools:

# -------------------
# PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# Search Tool:
search_tool = DuckDuckGoSearchRun(region="us-en")

# Calci Tool:
@tool
def calculator(first_num:float, second_num:float, operation:str)->dict:
    """
    Perform basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation in ["mul","multiply","product"]:
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error":"Can't divide by zero"}
            else:
                result = first_num / second_num
        else:
            result = "Unsupported operation"
        return {"first_num":first_num,"second_num":second_num,"result":result}
    except Exception as e:
        return {"error":f"{e}"}

# Current Weather:
@tool
def fetch_current_weather(location:str)->dict:
    """
    Provide the weather details to a user
    on basis of provided location
    """
    params = {
        "appid": os.environ.get("WEATHER_API"),
        "q": location,
    }
    response = requests.get(url="https://api.openweathermap.org/data/2.5/weather", params=params)
    return response.json()

# RAG Tool:
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool,calculator,fetch_current_weather,rag_tool]
model_with_tools = model.bind_tools(tools)

# State:
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_message(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful, reliable, and tool-aware AI assistant.\n\n"

            "Document Handling:\n"
            "- If the user's question is related to the uploaded PDF, you MUST call the `rag_tool`.\n"
            f"- Always include the thread_id: `{thread_id}` when calling the `rag_tool`.\n"
            "- Use only the information retrieved from the document to answer such questions.\n"
            "- If no PDF is available and the user asks document-related questions, politely ask them to upload a PDF.\n\n"

            "Tool Usage:\n"
            "- Use `web_search` for real-time or external information.\n"
            "- Use `fetch_current_weather` only for weather-related queries.\n"
            "- Use the `calculator` tool for mathematical calculations.\n\n"

            "General Rules:\n"
            "- Do not hallucinate document content.\n"
            "- Do not use tools unnecessarily.\n"
            "- Choose the most appropriate tool based on user intent."
        )
    )

    messages = [system_message, *state["messages"]]
    response = model_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

# Define tool node:
tool_node = ToolNode(tools)

# DB Connectivity:
conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)

# For persistency:
memory = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat",chat_message)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat")
graph.add_conditional_edges("chat",tools_condition)
graph.add_edge("tools","chat")

chatbot = graph.compile(checkpointer=memory)

# Display threads from db:
def retrieve_all_threads():
    all_threads = set()
    for thread in memory.list(None):
        all_threads.add(thread.config["configurable"]["thread_id"])
    return list(all_threads)

# Doc related threads:
def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})