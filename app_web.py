import streamlit as st
import os
import re
import uuid
import hashlib
import tempfile

# --- 1. PYSQLITE3 BYPASS (MUST RUN FIRST BEFORE ANY OTHER IMPORTS) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- 2. THEME & HEADER CONFIGURATION ---
st.set_page_config(
    page_title="JurisDocs AI - Legal RAG Engine",
    page_icon="⚖️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #121212, #080808) !important;
        color: #E2E8F0 !important;
        font-family: 'Helvetica Neue', Inter, sans-serif;
    }
    h1 {
        background: linear-gradient(135deg, #F3E5AB 0%, #D4AF37 50%, #AA7C11 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3, h4, h5 {
        color: #E5C158 !important;
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0E0E0E !important;
        border-right: 1px solid #1C1C1C;
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #C5A059 !important;
        background-color: #141414;
        border-radius: 12px;
        padding: 18px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
    }
    div[data-testid="stChatInput"] input {
        background-color: #141414 !important;
        border: 1px solid #2A2A2A !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    div[data-testid="stChatInput"] input:focus {
        border-color: #D4AF37 !important;
    }
    </style>
""", unsafe_allow_html=True)


# =====================================================================
# RETRIEVAL LAYER
# =====================================================================

# Matches "Sec. 24.008", "Section 24.008", "§ 24.008", "SUBCHAPTER 24.008",
# and bare citations like "24.008" or "91.001(a)".
SECTION_PATTERN = re.compile(
    r"(?:sec(?:tion)?s?\.?|§+|art(?:icle)?\.?)?\s*"
    r"(\d{1,4}\.\d{1,4}[A-Za-z]?)",
    re.IGNORECASE,
)

# Tokenizer for BM25. Critically, it keeps "24.008" as a SINGLE token instead of
# letting Python's default .split() produce "24.008." (with trailing period),
# which would never match the user's query token.
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)*")


def bm25_preprocess(text: str):
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def extract_section_numbers(text: str):
    """Pull every statutory citation out of a string, in order, de-duplicated."""
    found = []
    for match in SECTION_PATTERN.finditer(text):
        num = match.group(1)
        if num not in found:
            found.append(num)
    return found


@st.cache_resource(show_spinner=False)
def load_embeddings():
    """Loaded once per process instead of on every single Streamlit rerun."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def chunk_documents(documents):
    """
    Statute-aware splitting.

    Two things matter for legal text that generic splitting gets wrong:
      1. Split on section boundaries first, so a section's heading and its body
         stay in the same chunk wherever possible.
      2. Stamp every chunk with the section it belongs to, and prepend that
         section label to the chunk text. A chunk that reads "...the landlord is
         liable for actual damages..." is invisible to a query about "Section
         24.008" unless the section number is physically present in the chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=[
            "\nSec. ", "\nSEC. ", "\nSection ", "\nSECTION ",
            "\n§ ", "\nArticle ", "\nARTICLE ",
            "\n\n", "\n", ". ", " ", "",
        ],
    )
    raw_chunks = splitter.split_documents(documents)

    chunks = []
    current_section = None
    for chunk in raw_chunks:
        sections = extract_section_numbers(chunk.page_content)
        if sections:
            current_section = sections[0]

        chunk.metadata["sections"] = ", ".join(sections) if sections else ""
        chunk.metadata["section"] = current_section or ""

        page = chunk.metadata.get("page")
        chunk.metadata["page_label"] = page + 1 if isinstance(page, int) else "?"

        if current_section and current_section not in chunk.page_content:
            chunk.page_content = f"[Sec. {current_section}]\n{chunk.page_content}"

        chunks.append(chunk)

    return chunks


@st.cache_resource(show_spinner=False)
def build_index(doc_id: str, _file_bytes: bytes):
    """
    Build the vector store + keyword index exactly once per uploaded document.

    The original code rebuilt Chroma on every rerun and re-inserted the same
    chunks into the same named collection, so after five chat messages the
    collection held five duplicate copies of every chunk. That crowds the top-k
    window with repeats of whatever chunk happens to embed closest, which is a
    large part of why the correct section never surfaced.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_file_bytes)
        tmp_path = tmp.name

    try:
        documents = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    total_chars = sum(len(d.page_content) for d in documents)
    chunks = chunk_documents(documents)

    embeddings = load_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"doc_{doc_id}_{uuid.uuid4().hex[:6]}",
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    keyword_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=bm25_preprocess,
    )
    keyword_retriever.k = 8

    return {
        "chunks": chunks,
        "dense": dense_retriever,
        "keyword": keyword_retriever,
        "pages": len(documents),
        "total_chars": total_chars,
    }


def hybrid_retrieve(index, query: str, top_k: int = 6):
    """
    Three retrieval passes, merged and de-duplicated:

      1. EXACT  - literal string match on any statutory number in the query.
                  Deterministic and always ranked first. This is the pass that
                  fixes the "Section 24.008" failure outright.
      2. BM25   - keyword/lexical match. Catches legal terms of art
                  ("constructive eviction", "writ of possession") that dense
                  embeddings blur together.
      3. DENSE  - semantic match. Catches paraphrased questions where the user
                  never uses the document's own vocabulary.

    Dense-only retrieval is the wrong tool for citation lookup: MiniLM maps
    "24.008" and "24.005" to nearly identical vectors, so the nearest neighbour
    is essentially arbitrary among sections.
    """
    ordered = []
    seen = set()

    def add(doc):
        key = doc.page_content[:300]
        if key not in seen:
            seen.add(key)
            ordered.append(doc)

    # Pass 1 - exact citation match
    query_sections = extract_section_numbers(query)
    if query_sections:
        for num in query_sections:
            for chunk in index["chunks"]:
                if num in chunk.page_content or num == chunk.metadata.get("section"):
                    add(chunk)

    # Pass 2 - lexical. Expand the query with every way a statute gets written,
    # so BM25 can hit whichever convention this particular PDF uses.
    expanded = query
    for num in query_sections:
        expanded += f" Sec. {num} Section {num} § {num} {num}"

    try:
        for doc in index["keyword"].invoke(expanded):
            add(doc)
    except Exception:
        pass

    # Pass 3 - semantic
    try:
        for doc in index["dense"].invoke(query):
            add(doc)
    except Exception:
        pass

    return ordered[:top_k], query_sections


def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page_label", "?")
        section = doc.metadata.get("section") or "n/a"
        blocks.append(
            f"[Source {i} | page {page} | section {section}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/D4AF37/gavel.png", width=70)
    st.title("JurisDocs AI")
    st.subheader("Legal RAG Engine")
    st.write("---")

    if "HF_TOKEN" in st.secrets:
        HF_TOKEN = st.secrets["HF_TOKEN"]
        st.success("HuggingFace API Token Connected")
    else:
        st.warning("HF_TOKEN missing in Secrets! Please update advanced settings.")
        st.stop()

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

    st.write("---")
    show_debug = st.toggle("Show retrieval diagnostics", value=False)

    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

    st.write("---")
    st.markdown("""
        <div style="text-align: center; margin-top: 30px;">
            <p style="font-size: 10px; color: #666666; margin-bottom: 2px; letter-spacing: 1.5px;">SYSTEM ARCHITECTURE BY</p>
            <p style="font-size: 14px; font-weight: bold; color: #D4AF37; margin-bottom: 12px; font-family: 'Georgia', serif; letter-spacing: 2px;">ISHAN MISHRA</p>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <a href="https://github.com/ishanmishra0827" target="_blank">
                    <img src="https://img.icons8.com/ios-glyphs/24/C5A059/github.png" width="18"/>
                </a>
                <a href="https://linkedin.com/in/ishanmishra0827" target="_blank">
                    <img src="https://img.icons8.com/ios-glyphs/24/C5A059/linkedin.png" width="18"/>
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =====================================================================
# MAIN LAYOUT
# =====================================================================
st.title("⚖️ JurisDocs AI")
st.markdown("##### *Empowering legal document parsing through local Retrieval-Augmented Generation (RAG)*")

with st.expander("📖 How to Use JurisDocs AI", expanded=False):
    st.markdown("""
    1. **Upload a Legal Document:** Choose a statutory code, contract, or lease agreement in **PDF** format below.
    2. **Wait for DB Compilation:** The RAG pipeline chunks, embeds, and indexes your document into a secure RAM vector store.
    3. **Query the Engine:** Type a targeted legal question (e.g., notice requirements, specific section clauses, or statutory penalties).
    4. **Inspect Source Citations:** Review the AI analysis along with exact page-level text citations retrieved directly from your file.
    """)

with st.expander("🔒 Privacy & Compliance Notice", expanded=False):
    st.markdown("""
    * **Zero Storage:** Uploaded PDFs are processed temporarily in memory. Files are instantly deleted from disk after text extraction.
    * **RAM-Only Indexing:** The document search index exists strictly in RAM and vanishes when your session ends.
    * **No AI Model Training:** Data processed through your HuggingFace API token is used solely to generate your immediate response.
    * **Not Legal Advice:** This is an informational research tool. Output may be incomplete or wrong and is not a substitute for a licensed attorney.
    """)

st.write("---")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_file = st.file_uploader("Upload Statutory Document / Case File (PDF)", type=["pdf"])

if uploaded_file is None:
    st.info("Please upload a PDF document in the dashboard above to initiate analysis.")
    st.stop()

file_bytes = uploaded_file.getvalue()
doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

with st.spinner("Compiling statutory index (runs once per document)..."):
    index = build_index(doc_id, file_bytes)

# Scanned/image-only PDFs extract almost no text. Without this check the app
# silently indexes empty pages and blames the model for the resulting answers.
if index["total_chars"] < 500:
    st.error(
        f"Only {index['total_chars']} characters were extracted from {index['pages']} pages. "
        "This PDF is most likely a scan with no text layer — the retriever has nothing to search. "
        "Run it through OCR (e.g. `ocrmypdf input.pdf output.pdf`) and re-upload."
    )
    st.stop()

st.success(
    f"Legal database active — {index['pages']} pages indexed into {len(index['chunks'])} chunks. "
    "Ask questions below!"
)
st.write("---")

base_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.1,
    max_new_tokens=1024,
    repetition_penalty=1.05,
)
llm = ChatHuggingFace(llm=base_llm)

SYSTEM_PROMPT = (
    "You are an expert legal research assistant analyzing a specific statutory or "
    "contractual document. Answer strictly from the excerpts provided below.\n\n"
    "Rules:\n"
    "1. Cite the governing section number and page for every substantive claim, "
    "e.g. '(Sec. 24.0061, p. 12)'.\n"
    "2. If the excerpts contain the relevant provision, answer it fully — set out the "
    "elements, the remedies, the damages formula, and any notice or timing requirements "
    "spelled out in the text.\n"
    "3. If the excerpts address the topic but under a different section number than the "
    "one asked about, say so explicitly and answer from the section that does govern it. "
    "Do not refuse merely because the exact number the user typed does not appear.\n"
    "4. Only say the document does not cover the question when the excerpts genuinely "
    "contain nothing on point. Never invent statutory text, section numbers, or figures.\n"
    "5. Quote the operative language verbatim where the precise wording carries legal weight.\n\n"
    "Prior conversation:\n{chat_history}\n\n"
    "Document excerpts:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])
chain = prompt | llm


def render_citations(citations):
    with st.expander("📌 View Statutory Citations"):
        for idx, citation in enumerate(citations, start=1):
            st.markdown(
                f"**Citation {idx} — Page {citation['page']}"
                + (f" · Sec. {citation['section']}" if citation['section'] else "")
                + "**"
            )
            st.info(citation["content"])


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])

if user_query := st.chat_input("Ask a legal question based on this document..."):
    st.chat_message("user").markdown(user_query)
    st.session_state["messages"].append({"role": "user", "content": user_query})

    with st.spinner("Analyzing document structure & statutes..."):
        docs, query_sections = hybrid_retrieve(index, user_query, top_k=6)

        history_turns = st.session_state["messages"][-7:-1]
        chat_history = "\n".join(
            f"{m['role'].upper()}: {m['content'][:400]}" for m in history_turns
        ) or "(none)"

        try:
            response = chain.invoke({
                "input": user_query,
                "context": format_context(docs),
                "chat_history": chat_history,
            })
            answer = response.content
        except Exception as exc:
            answer = f"The inference endpoint returned an error: `{exc}`"

        citations = [
            {
                "page": doc.metadata.get("page_label", "?"),
                "section": doc.metadata.get("section", ""),
                "content": doc.page_content,
            }
            for doc in docs
        ]

        with st.chat_message("assistant"):
            st.markdown(f"### 🏛️ Legal Analysis:\n{answer}")
            render_citations(citations)

            if show_debug:
                with st.expander("🔍 Retrieval diagnostics"):
                    st.write(f"Section numbers parsed from query: `{query_sections or 'none'}`")
                    hits = [
                        c.metadata.get("section", "") for c in index["chunks"]
                        if any(n in c.page_content for n in query_sections)
                    ] if query_sections else []
                    st.write(f"Chunks in the corpus containing those numbers: **{len(hits)}**")
                    st.write(f"Chunks passed to the model: **{len(docs)}**")
                    st.write(
                        "Sections retrieved: "
                        f"`{[d.metadata.get('section', '?') for d in docs]}`"
                    )

        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"### 🏛️ Legal Analysis:\n{answer}",
            "citations": citations,
        })