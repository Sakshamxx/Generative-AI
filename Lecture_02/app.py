import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind • Chat with your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --accent-1: #7c5cff;
            --accent-2: #22d3ee;
            --bg-deep: #0b0d17;
            --bg-panel: #12141f;
            --bg-card: #161927;
            --border-soft: rgba(255,255,255,0.08);
            --text-dim: #9ca3af;
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(124,92,255,0.16), transparent 40%),
                radial-gradient(circle at 85% 0%, rgba(34,211,238,0.14), transparent 40%),
                var(--bg-deep);
        }

        /* Header banner */
        .hero {
            padding: 1.6rem 2rem;
            border-radius: 20px;
            background: linear-gradient(135deg, rgba(124,92,255,0.18), rgba(34,211,238,0.10));
            border: 1px solid var(--border-soft);
            margin-bottom: 1.4rem;
            backdrop-filter: blur(6px);
        }
        .hero h1 {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg, #ffffff, #c9c3ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: var(--text-dim);
            margin: 0.35rem 0 0 0;
            font-size: 0.95rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: var(--bg-panel);
            border-right: 1px solid var(--border-soft);
        }
        section[data-testid="stSidebar"] .stMarkdown h3 {
            color: #e5e7eb;
            font-weight: 700;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            background: var(--bg-card);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 0.4rem 0.6rem;
            margin-bottom: 0.6rem;
        }

        /* Badge / pill */
        .pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid var(--border-soft);
        }
        .pill-green { background: rgba(34,197,94,0.15); color: #4ade80; }
        .pill-gray  { background: rgba(156,163,175,0.15); color: #d1d5db; }

        /* Source snippet card */
        .source-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border-soft);
            border-left: 3px solid var(--accent-1);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: #cbd5e1;
        }

        .stButton>button {
            border-radius: 10px;
            border: 1px solid var(--border-soft);
            background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
            color: white;
            font-weight: 600;
            transition: transform 0.15s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            filter: brightness(1.08);
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: var(--bg-card);
            border: 2px dashed rgba(124,92,255,0.45);
            border-radius: 14px;
            padding: 0.6rem;
            transition: border-color 0.15s ease, background 0.15s ease;
        }
        div[data-testid="stFileUploaderDropzone"]:hover {
            border-color: var(--accent-2);
            background: rgba(34,211,238,0.06);
        }
        div[data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
            color: white !important;
            border: none !important;
        }

        footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
defaults = {
    "messages": [],
    "vector_store": None,
    "retriever": None,
    "pdf_name": None,
    "chunk_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return MistralAIEmbeddings()


def get_llm(model_name: str, temperature: float):
    return ChatMistralAI(model=model_name, temperature=temperature)


def build_vector_store(pdf_files, chunk_size: int, chunk_overlap: int):
    """pdf_files: list of uploaded file objects (from st.file_uploader, multi=True)."""
    all_pages = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        for p in pages:
            p.metadata["source_file"] = pdf_file.name
        all_pages.extend(pages)

        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(all_pages)

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"session_{int(time.time())}",
    )

    return vector_store, len(chunks)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key_input = st.text_input(
        "Mistral API Key",
        type="password",
        value=os.getenv("MISTRAL_API_KEY", ""),
        help="Overrides the MISTRAL_API_KEY in your .env for this session.",
    )
    if api_key_input:
        os.environ["MISTRAL_API_KEY"] = api_key_input

    st.markdown("---")
    st.markdown("### 📄 Document")

    st.caption("Drag & drop one or more PDFs below, or click to browse.")
    uploaded_pdfs = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    with st.expander("Chunking & retrieval settings"):
        chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
        top_k = st.slider("Chunks retrieved (k)", 1, 10, 4)
        fetch_k = st.slider("MMR fetch_k", top_k, 30, max(10, top_k * 2))
        lambda_mult = st.slider("MMR diversity (lambda)", 0.0, 1.0, 0.5)

    with st.expander("Model settings"):
        model_name = st.selectbox(
            "Chat model", ["ministral-3b-2512", "mistral-large-latest", "mistral-small-latest"]
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2)

    process_clicked = st.button("🚀 Process PDF", use_container_width=True)

    if process_clicked:
        if not os.getenv("MISTRAL_API_KEY"):
            st.error("Please provide a Mistral API key first.")
        elif not uploaded_pdfs:
            st.error("Please drag & drop or browse for at least one PDF.")
        else:
            with st.spinner(f"Reading, chunking, and embedding {len(uploaded_pdfs)} document(s)..."):
                vs, n_chunks = build_vector_store(
                    uploaded_pdfs, chunk_size, chunk_overlap
                )
                st.session_state.vector_store = vs
                st.session_state.chunk_count = n_chunks
                st.session_state.pdf_name = ", ".join(f.name for f in uploaded_pdfs)
                st.session_state.messages = []
            st.success(f"Indexed {n_chunks} chunks from **{len(uploaded_pdfs)} file(s)**")

    st.markdown("---")
    if st.session_state.pdf_name:
        st.markdown(
            f'<span class="pill pill-green">● {st.session_state.pdf_name}</span> '
            f'<span class="pill pill-gray">{st.session_state.chunk_count} chunks</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="pill pill-gray">No document indexed</span>', unsafe_allow_html=True)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📄 DocMind — Chat with your PDF</h1>
        <p>Upload a document, index it, and ask grounded questions answered strictly from its content.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    page = src.metadata.get("page", "?")
                    fname = src.metadata.get("source_file", "")
                    st.markdown(
                        f'<div class="source-card"><b>Chunk {i} · {fname} · page {page}</b><br>{src.page_content[:400]}...</div>',
                        unsafe_allow_html=True,
                    )

# --------------------------------------------------------------------------
# CHAT INPUT
# --------------------------------------------------------------------------
query = st.chat_input("Ask something about your document...")

if query:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a PDF first (see sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)

                final_prompt = PROMPT.invoke({"context": context, "question": query})
                llm = get_llm(model_name, temperature)
                response = llm.invoke(final_prompt)

            st.markdown(response.content)
            with st.expander("📚 View source chunks"):
                for i, src in enumerate(docs, 1):
                    page = src.metadata.get("page", "?")
                    fname = src.metadata.get("source_file", "")
                    st.markdown(
                        f'<div class="source-card"><b>Chunk {i} · {fname} · page {page}</b><br>{src.page_content[:400]}...</div>',
                        unsafe_allow_html=True,
                    )

        st.session_state.messages.append(
            {"role": "assistant", "content": response.content, "sources": docs}
        )