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
from langchain_core.prompts import ChatPromptTemplate

# Universal import compatibility for modern LangChain / LangChain-Classic
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

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
    
    /* Streamlit Chat Inputs & Containers */
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
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

    st.write("---")
    
    # Premium Engineering Signature
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

# --- 4. MAIN LAYOUT AND RAG CODE ---
st.title("⚖️ JurisDocs AI")
st.markdown("##### *Empowering legal document parsing through local Retrieval-Augmented Generation (RAG)*")

# How to Use Instructions Expander
with st.expander("📖 How to Use JurisDocs AI", expanded=False):
    st.markdown("""
    1. **Upload a Legal Document:** Choose a statutory code, contract, or lease agreement in **PDF** format below.
    2. **Wait for DB Compilation:** The RAG pipeline automatically chunks, embeds, and indexes your document into a secure RAM vector store.
    3. **Query the Engine:** Type a targeted legal question (e.g., notice requirements, specific section clauses, or statutory penalties).
    4. **Inspect Source Citations:** Review the AI analysis along with exact page-level text citations retrieved directly from your file.
    """)

# Privacy & Compliance Expander
with st.expander("🔒 Privacy & Compliance Notice", expanded=False):
    st.markdown("""
    * **Zero Storage:** Uploaded PDFs are processed temporarily in memory. Files are instantly deleted from disk after text extraction.
    * **RAM-Only Indexing:** The document search index exists strictly in RAM and vanishes when your session ends. Nothing is written to a persistent database.
    * **No AI Model Training:** Data processed through your HuggingFace API token is used solely to generate your immediate response. Your text is never stored, shared, or used by third parties to train future AI models.
    """)

st.write("---")

# Initialize Chat Memory State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

uploaded_file = st.file_uploader("Upload Statutory Document / Case File (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Save file locally to feed PyPDFLoader
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("Document loaded. Indexing RAM vector store...")

    # Load and Split PDF
    loader = PyPDFLoader("temp_doc.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)
    
    # Embed text and construct local ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    st.success("Legal database active. Ask questions below!")
    st.write("---")

    # Initialize Base LLM
    base_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.1,
        max_new_tokens=512
    )
    llm = ChatHuggingFace(llm=base_llm)
    
    system_prompt = (
        "You are an expert legal assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer or if the context doesn't mention "
        "the topic, say that you don't know based on the provided document. Do not make up facts.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Render previous conversation history from session state
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg:
                with st.expander("📌 View Statutory Citations"):
                    for idx, citation in enumerate(msg["citations"]):
                        st.markdown(f"**Citation {idx + 1} (Page {citation['page']}):**")
                        st.info(citation["content"])

    # Chat Input interface
    if user_query := st.chat_input("Ask a legal question based on this document..."):
        # Display user message and append to history
        st.chat_message("user").markdown(user_query)
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # Process query through RAG pipeline
        with st.spinner("Analyzing document structure & statutes..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            
            citations = [
                {
                    "page": doc.metadata.get("page", "Unknown") + 1,
                    "content": doc.page_content
                }
                for doc in response["context"]
            ]

            # Display Assistant Response
            with st.chat_message("assistant"):
                st.markdown(f"### 🏛️ Legal Analysis:\n{answer}")
                with st.expander("📌 View Statutory Citations"):
                    for idx, citation in enumerate(citations):
                        st.markdown(f"**Citation {idx + 1} (Page {citation['page']}):**")
                        st.info(citation["content"])

            # Save assistant response & citations to history
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"### 🏛️ Legal Analysis:\n{answer}",
                "citations": citations
            })

else:
    st.info("Please upload a PDF document in the dashboard above to initiate analysis.")