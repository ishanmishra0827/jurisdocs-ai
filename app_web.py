import streamlit as st
import os
import hashlib

import rag_core
from rag_core import BM25_AVAILABLE

# --- THEME & PAGE CONFIG ---
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


# Streamlit-side caching only. All retrieval logic lives in rag_core so the
# evaluation harness exercises the identical code path.
@st.cache_resource(show_spinner=False)
def build_index(doc_id: str, _blobs):
    return rag_core.build_index_from_byte_list(_blobs)


@st.cache_resource(show_spinner=False)
def get_chain(model_key: str):
    return rag_core.build_chain()


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/D4AF37/gavel.png", width=70)
    st.title("JurisDocs AI")
    st.subheader("Legal RAG Engine")
    st.write("---")

    # Any one of these works. Groq and Google have free tiers; Ollama runs local.
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "LLM_PROVIDER", "LLM_MODEL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]

    provider = rag_core.detect_provider()
    if provider is None:
        st.error(
            "No LLM credential in Secrets. Add one of GROQ_API_KEY, "
            "GOOGLE_API_KEY, or HF_TOKEN."
        )
        st.stop()
    active_model = os.environ.get("LLM_MODEL") or rag_core.DEFAULT_MODELS.get(provider, "unknown")
    st.success(f"Provider: {provider}")
    st.caption(f"Model: {active_model}")

    st.write("---")
    show_debug = st.toggle("Show retrieval diagnostics", value=False)

    if not BM25_AVAILABLE:
        st.warning(
            "Keyword retrieval is offline — `rank_bm25` is not installed. "
            "Add it to requirements.txt."
        )

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


# --- MAIN ---
st.title("⚖️ JurisDocs AI")
st.markdown("##### *Empowering legal document parsing through local Retrieval-Augmented Generation (RAG)*")

with st.expander("📖 How to Use JurisDocs AI", expanded=False):
    st.markdown("""
    1. **Upload a Legal Document:** Choose a statutory code, contract, or lease agreement in **PDF** format below.
    2. **Wait for DB Compilation:** The pipeline chunks, embeds, and indexes your document into a RAM vector store.
    3. **Query the Engine:** Type a targeted legal question.
    4. **Inspect Source Citations:** Review the analysis alongside page-level text retrieved from your file.
    """)

with st.expander("⚠️ Accuracy & Verification", expanded=False):
    st.markdown("""
    * **Verify every citation before relying on it.** This tool retrieves and summarizes; it does not validate.
    * **Coverage is limited to the document you upload.** A question governed by a chapter you have not indexed will be declined, correctly, even though an answer exists elsewhere in the code.
    * **Not legal advice.** This is a research aid, not a substitute for a licensed attorney.
    """)

with st.expander("🔒 Privacy Notice", expanded=False):
    st.markdown("""
    * **No persistent storage:** Uploaded PDFs are processed in memory and the temporary file is deleted after text extraction. The search index exists only in RAM and is discarded when the session ends.
    * **Third-party processing:** Your questions and excerpts of the uploaded document are transmitted to a third-party inference provider (currently Groq) to generate each answer. They leave this application.
    * **Confidential material:** Do not upload documents containing client names, addresses, case details, or other identifying information without reviewing that provider's terms against your own confidentiality obligations. This application also supports fully local inference, where no data leaves the machine.
    * **Not legal advice:** This is a document research aid. It retrieves and summarizes statutory text; it does not verify its own output and is not a substitute for a licensed attorney.
    """)

st.write("---")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_files = st.file_uploader(
    "Upload Statutory Documents / Case Files (PDF) — you can select more than one",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(
        "Upload one or more PDFs above to begin. Landlord-tenant questions usually "
        "need both Property Code Chapter 24 (eviction procedure) and Chapter 92 "
        "(residential tenancies) — upload both so questions aren't declined simply "
        "because the governing chapter is missing."
    )
    st.stop()

blobs = [f.getvalue() for f in uploaded_files]
doc_id = hashlib.sha256(b"".join(blobs)).hexdigest()[:16]

with st.spinner(f"Compiling statutory index from {len(blobs)} document(s)..."):
    index = build_index(doc_id, blobs)

if index["total_chars"] < 500:
    st.error(
        f"Only {index['total_chars']} characters were extracted from {index['pages']} pages. "
        "This PDF is most likely a scan with no text layer. Run it through OCR "
        "(e.g. `ocrmypdf input.pdf output.pdf`) and re-upload."
    )
    st.stop()

st.success(
    f"Legal database active — {index['pages']} pages indexed into {len(index['chunks'])} chunks."
)
st.write("---")

chain = get_chain(active_model)


def render_citations(citations):
    with st.expander("📌 View Statutory Citations"):
        for idx, citation in enumerate(citations, start=1):
            header = f"**Citation {idx} — Page {citation['page']}"
            if citation["section"]:
                header += f" · Sec. {citation['section']}"
            st.markdown(header + "**")
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
        history_turns = st.session_state["messages"][-7:-1]
        chat_history = "\n".join(
            f"{m['role'].upper()}: " + rag_core.clean_answer(m["content"])[:400]
            for m in history_turns
        ) or "(none)"

        try:
            result = rag_core.answer_question(index, chain, user_query, chat_history)
            answer = result["answer"]
            docs = result["docs"]
            query_sections = result["query_sections"]
        except Exception as exc:
            answer = f"The inference endpoint returned an error: `{exc}`"
            docs, query_sections = [], []

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
            if citations:
                render_citations(citations)

            if show_debug:
                with st.expander("🔍 Retrieval diagnostics"):
                    st.write(f"Sections parsed from query: `{query_sections or 'none'}`")
                    st.write(f"Chunks passed to the model: **{len(docs)}**")
                    st.write(
                        "Sections retrieved: "
                        f"`{list(dict.fromkeys(d.metadata.get('section', '?') for d in docs))}`"
                    )

        st.session_state["messages"].append({
            "role": "assistant",
            "content": f"### 🏛️ Legal Analysis:\n{answer}",
            "citations": citations,
        })
