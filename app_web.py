import streamlit as st
import os

# --- 1. PYSQLITE3 BYPASS (MUST RUN FIRST BEFORE ANY OTHER IMPORTS) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 2. THEME & HEADER CONFIGURATION ---
st.set_page_config(
    page_title="JurisDocs AI - Legal RAG Engine",
    page_icon="⚖️",
    layout="wide"
)

# Dark Gold & Charcoal Premium Theme Injection
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background: radial-gradient(circle at top left, #121212, #080808) !important;
        color: #E2E8F0 !important;
        font-family: 'Helvetica Neue', Inter, sans-serif;
    }

    /* Elegant Title Typography */
    h1 {
        background: linear-gradient(135deg, #F3E5AB 0%, #D4AF37 50%, #AA7C11 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2, h3, h4, h5 {
        color: #E5C158 !important; /* Soft Champagne Gold */
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 600 !important;
    }

    /* Clean, Borderless Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0E0E0E !important;
        border-right: 1px solid #1C1C1C;
    }

    /* Sleek Border Accents for Uploaders */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #C5A059 !important;
        background-color: #141414;
        border-radius: 12px;
        padding: 18px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);
    }

    /* Beautiful Input Fields */
    div[data-testid="stTextInput"] input {
        background-color: #141414 !important;
        border: 1px solid #2A2A2A !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #D4AF37 !important;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.15) !important;
    }

    /* Premium Button Style (Rich Metallic look) */
    .stButton>button {
        background: linear-gradient(135deg, #D4AF37 0%, #C5A059 100%) !important;
        color: #080808 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 10px 24px !important;
        letter-spacing: 0.5px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
    }

    /* Privacy banner */
    .privacy-banner {
        border: 1px solid #2A2A2A;
        background-color: #141414;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 18px;
        font-size: 13.5px;
        color: #C9C9C9;
    }
    .privacy-banner b { color: #E5C158; }

    /* How-to-use step cards */
    .step-card {
        border: 1px solid #2A2A2A;
        background-color: #121212;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #D4AF37 0%, #C5A059 100%);
        color: #080808;
        font-weight: 800;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        text-align: center;
        line-height: 26px;
        margin-right: 10px;
        font-size: 13px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR & IDENTITY ---
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/D4AF37/gavel.png", width=70)
    st.title("JurisDocs AI")
    st.subheader("Legal RAG Engine")
    st.write("---")

    # Secure API Keys Check
    if "HF_TOKEN" in st.secrets:
        HF_TOKEN = st.secrets["HF_TOKEN"]
        st.success("HuggingFace API Token Connected")
    else:
        st.warning("HF_TOKEN missing in Secrets! Please update advanced settings.")
        st.stop()

    # Set HF Token directly to system environment so client grabs it natively
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    os.environ["HF_TOKEN"] = HF_TOKEN

    st.write("---")
    st.markdown(
        "**🔒 Privacy**\n\n"
        "Documents are processed in memory for this session only and are "
        "never permanently stored. [Details below.](#privacy)"
    )
    st.write("---")

    # Premium Engineering Signature
    st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
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

# --- 4. SESSION STATE INITIALIZATION ---
# messages: what's rendered in the chat UI (includes citation payloads)
# lc_chat_history: LangChain message objects fed to the chain for follow-up
#                  questions ("What about clause 3?" needs to know what came before)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_chat_history" not in st.session_state:
    st.session_state.lc_chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None


def reset_session():
    st.session_state.messages = []
    st.session_state.lc_chat_history = []
    st.session_state.rag_chain = None
    st.session_state.processed_filename = None


def build_rag_chain(pdf_path: str):
    """Loads a PDF, chunks it, embeds it into an in-memory Chroma store, and
    wraps it in a history-aware conversational retrieval chain."""

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Statutes reference their own subsections constantly (e.g. Sec. 24.005(f)
    # answered by 24.005(f-3)), so blind character-count chunking can slice a
    # section apart from the subsection that actually answers a question about
    # it. Splitting preferentially on statutory section headers first keeps
    # whole sections together as single chunks wherever possible, only
    # falling back to paragraph/sentence splits if a section itself is too
    # long. This MUST be a regex (is_separator_regex=True): pypdf's actual
    # extracted spacing around "Sec." varies ("Sec. 24.005", "Sec. A 24.005",
    # etc. depending on the source PDF's internal character spacing), so a
    # literal string separator silently never matches and this step becomes
    # a no-op. chunk_size is raised accordingly since sections vary a lot in
    # length.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=300,
        separators=[r"\nSec\.\s*A?\s*\d+\.\d+", "\n\n", "\n", ". ", " ", ""],
        is_separator_regex=True,
    )
    chunks = text_splitter.split_documents(documents)

    # Embed text and construct local, in-memory ChromaDB (no persist_directory
    # is set, so nothing is written to disk here — it lives only in RAM for
    # the life of this session).
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    # k=5 with MMR (rather than pure similarity) pulls in more chunks while
    # actively avoiding near-duplicates, so a question has a better chance of
    # actually surfacing the specific subsection that answers it.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 15},
    )

    # mistralai/Mistral-7B-Instruct-v0.3 has been pulled entirely from
    # Hugging Face's serverless Inference Providers. Qwen2.5-7B-Instruct is
    # currently live (via the "together" provider) and ungated. Wrapping in
    # ChatHuggingFace makes LangChain call the chat/conversational endpoint.
    base_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.5,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=base_llm)

    # Step 1: given chat history + a new question, rewrite the question as a
    # standalone query so follow-ups like "what about the deposit clause?"
    # can be retrieved correctly even without repeating context.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given a chat history and the latest user question, which might "
         "reference earlier context, rewrite it as a standalone question "
         "that can be understood without the chat history. Do NOT answer "
         "the question, just reformulate it if needed, otherwise return it "
         "as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Step 2: answer using retrieved context, still aware of the conversation.
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a legal research assistant helping analyze a specific "
         "uploaded document. Answer the user's question using ONLY the "
         "context below. If the answer isn't in the context, say so plainly "
         "instead of guessing. Be precise and cite clause/section numbers "
         "when the document provides them. This is general legal "
         "information, not legal advice.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- 5. MAIN LAYOUT ---
st.title("⚖️ JurisDocs AI")
st.markdown("##### *Empowering legal document parsing through local Retrieval-Augmented Generation (RAG)*")
st.write("---")

# ---- LANDING PAGE (shown until a document has been processed) ----
if st.session_state.rag_chain is None:

    st.markdown("### 👋 How to Use JurisDocs AI")
    steps = [
        ("Upload a document", "Drop in a statute, case file, contract, or brief (PDF only) using the uploader below."),
        ("Wait for indexing", "JurisDocs AI reads and chunks the document, building a temporary, session-only search index — usually a few seconds for a typical filing."),
        ("Ask questions", "Ask anything about the document in plain English. Ask follow-ups too — JurisDocs AI remembers the conversation."),
        ("Review citations", "Every answer includes the exact page(s) it was drawn from, so you can verify it against the source."),
    ]
    for i, (t, d) in enumerate(steps, start=1):
        st.markdown(
            f'<div class="step-card"><span class="step-number">{i}</span>'
            f'<b>{t}</b><br><span style="color:#A0A0A0; font-size: 13.5px;">{d}</span></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### 📄 Try It Instantly")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(
            "Don't have a document handy? Download this sample lease agreement "
            "(fictional, for testing only) and upload it below to see JurisDocs AI in action."
        )
    with col2:
        try:
            with open("assets/sample_legal_document.pdf", "rb") as f:
                st.download_button(
                    label="⬇️ Download Sample PDF",
                    data=f.read(),
                    file_name="sample_legal_document.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except FileNotFoundError:
            st.caption("(Sample PDF not found — add assets/sample_legal_document.pdf to the repo.)")

    st.write("")
    st.markdown('<a name="privacy"></a>', unsafe_allow_html=True)
    st.markdown(
        '<div class="privacy-banner">🔒 <b>Privacy Notice:</b> We do not store your documents. '
        "Uploaded PDFs are processed temporarily in memory to answer your questions, the file itself is "
        "deleted from disk immediately after text extraction, and the search index exists only in RAM for "
        "the duration of your session. Nothing is written to a persistent database or shared with third "
        "parties beyond the AI model provider used to generate answers.</div>",
        unsafe_allow_html=True,
    )

    st.write("---")

uploaded_file = st.file_uploader("Upload Statutory Document / Case File (PDF)", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.processed_filename:
    temp_path = "temp_doc.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading document and building a temporary search index..."):
        try:
            st.session_state.rag_chain = build_rag_chain(temp_path)
            st.session_state.processed_filename = uploaded_file.name
            st.session_state.messages = []
            st.session_state.lc_chat_history = []
        finally:
            # Privacy: the uploaded file is deleted from disk as soon as we're
            # done extracting text from it. Nothing persists beyond this point.
            if os.path.exists(temp_path):
                os.remove(temp_path)

    st.success(f"'{uploaded_file.name}' indexed. Ready for questions below.")

# ---- CHAT INTERFACE (shown once a document has been processed) ----
if st.session_state.rag_chain is not None:
    top_col1, top_col2 = st.columns([4, 1])
    with top_col1:
        st.caption(f"📄 Currently analyzing: **{st.session_state.processed_filename}**")
    with top_col2:
        if st.button("🗑️ New Document", use_container_width=True):
            reset_session()
            st.rerun()

    # Render existing conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 View Statutory Citations"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"**Citation {idx + 1} (Page {src['page']}):**")
                        st.info(src["content"])

    # New question input
    user_query = st.chat_input("Ask a legal question based on this document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document structure & statutes..."):
                try:
                    response = st.session_state.rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.lc_chat_history,
                    })
                    answer = response["answer"]
                    sources = [
                        {
                            "page": doc.metadata.get("page", "Unknown"),
                            "content": doc.page_content,
                        }
                        for doc in response.get("context", [])
                    ]

                    st.write(answer)
                    if sources:
                        with st.expander("📌 View Statutory Citations"):
                            for idx, src in enumerate(sources):
                                st.markdown(f"**Citation {idx + 1} (Page {src['page']}):**")
                                st.info(src["content"])

                    # Persist to both the display history and the LangChain
                    # history used to resolve follow-up questions.
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                    st.session_state.lc_chat_history.append(HumanMessage(content=user_query))
                    st.session_state.lc_chat_history.append(AIMessage(content=answer))

                except Exception as e:
                    error_msg = f"Something went wrong generating a response: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif uploaded_file is None:
    st.info("Please upload a PDF document above to begin.")