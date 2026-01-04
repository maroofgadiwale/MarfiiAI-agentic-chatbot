# Program to implement chat bot:
import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from lang_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ======================= CONSTANTS =======================
USER_AVATAR = "images/user.png"
BOT_AVATAR = "images/bot.png"

# ======================= THEME =======================
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] button {
    background-color: #6a00ff;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px;
    font-weight: 600;
}
[data-testid="stSidebar"] button:hover {
    background-color: #8f2bff;
}
</style>
""", unsafe_allow_html=True)

# ======================= UTILITIES =======================
def generate_thread_id():
    return uuid.uuid4()

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)
    st.session_state["message_history"] = []
    st.session_state["ingested_docs"].setdefault(str(new_id), {})

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])

# ======================= SESSION INIT =======================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ======================= SIDEBAR =======================
st.sidebar.title("MarfiiAI ❤")
# st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")
st.sidebar.caption("How can I assist you today?")
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ---- PDF STATUS ----
if thread_docs:
    latest = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"📄 `{latest['filename']}`\n\n"
        f"Chunks: {latest['chunks']} | Pages: {latest['documents']}"
    )
else:
    st.sidebar.info("No PDF indexed yet")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", type=["pdf"]
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info("PDF already indexed for this thread")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status.update(label="✅ PDF indexed", state="complete")

# ---- PAST THREADS ----
st.sidebar.subheader("Past Conversations")
for tid in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(tid), key=f"thread-{tid}"):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in load_conversation(tid)
        ]
        st.session_state["ingested_docs"].setdefault(str(tid), {})
        st.rerun()

# ======================= MAIN UI =======================
st.title("MarfiiAI ❤")
st.caption("Smart decisions, powered by AI")

# ---- CHAT HISTORY ----
for msg in st.session_state["message_history"]:
    avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.text(msg["content"])

# ---- USER INPUT ----
user_input = st.chat_input("Ask anything...")

if user_input:
    # Store user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user", avatar=USER_AVATAR):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # ======================= ASSISTANT STREAM =======================
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        status_box = st.status("Thinking...", expanded=True)
        placeholder = st.empty()

        full_response = ""
        tool_used = False

        for chunk, meta in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            time.sleep(0.07)  # smooth typing feel

            # TOOL MESSAGE
            if isinstance(chunk, ToolMessage):
                tool_used = True
                tool_name = getattr(chunk, "name", "tool")
                status_box.update(
                    label=f"🔧 Using `{tool_name}`",
                    state="running",
                )
                continue

            # AI MESSAGE STREAM
            if isinstance(chunk, AIMessage) and chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▍")

        # FINALIZE
        placeholder.markdown(full_response)

        if tool_used:
            status_box.update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )
        else:
            status_box.update(
                label="💬 Response",
                state="complete",
                expanded=False,
            )

    # Store assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": full_response}
    )

    # ---- DOCUMENT METADATA ----
    meta = thread_document_metadata(thread_key)
    if meta:
        st.caption(
            f"📄 {meta['filename']} | "
            f"Chunks: {meta['chunks']} | Pages: {meta['documents']}"
        )
