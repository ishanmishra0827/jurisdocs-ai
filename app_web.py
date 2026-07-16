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
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

# --- 2. THEME & HEADER CONFIGURATION ---
st.set_page_config(
    page_title="JurisDocs AI - Texas Legal RAG",
    page_icon="⚖️",
    layout="wide"
)

# Dark Gold & Charcoal Premium Theme Injection
st.markdown("""
    <style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #111111;
        color: #E5E5E5;
    }
    
    /* Main Headers */
    h1, h2, h3 {
        color: #D4AF37 !important; /* Metallic Gold */
        font-family: 'Georgia', serif;
    }
    
    /* Custom Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A !important;
        border-right: 1px solid #2D2D2D;
    }
    
    /* Premium Border Accents for Uploaders & Containers */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #D4AF37 !important;
        background-color: #1A1A1A;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Elegant Button Customization */
    .stButton>button {
        background-color: #D4AF37 !important;
        color: #111111 !important;
        border-radius: 6px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #AA7C11 !important;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.4);
    }
    </style>
""", unsafe_style_with_html=True)

# --- 3. SIDEBAR & IDENTITY ---
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/D4AF37/gavel.png", width=80)
    st.title("JurisDocs AI")
    st.subheader("Texas Legal RAG Engine")
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
    
    # Premium Engineering Signature
    st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <p style="font-size: 11px; color: #888888; margin-bottom: 2px;">SYSTEM ARCHITECTURE BY</p>
            <p style="font-size: 14px; font-weight: bold; color: #D4AF37; margin-bottom: 12px; font-family: 'Georgia', serif; letter-spacing: 1px;">ISHAN MISHRA</p>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <a href="https://github.com/ishanmishra0827" target="_blank">
                    <img src="https://img.icons8.com/ios-glyphs/24/D4AF37/github.png" width="20"/>
                </a>
                <a href="https://linkedin.com/in/ishanmishra0827" target="_blank">
                    <img src="https://img.icons8.com/ios-glyphs/24/D4AF37/linkedin.png" width="20"/>
                </a>
            </div>
        </div>
    """, unsafe_style_with_html=True)

# --- 4. MAIN LAYOUT AND RAG CODE ---
st.title("⚖️ Texas JurisDocs AI")
st.markdown("##### *Empowering legal document parsing through local Retrieval-Augmented Generation (RAG)*")
st.write("---")

uploaded_file = st.file_uploader("Upload Statutory Document / Case File (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Save the file locally to feed PyPDFLoader
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.info("Document uploaded successfully! Compiling semantic database...")

    # Load and Split PDF
    loader = PyPDFLoader("temp_doc.pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    # Embed text and construct local ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    st.success("Legal database initialized. Ready for legal query!")
    
    # Initialize the LLM (Mistral 7B) natively without direct Token parameters to prevent text_generation() type errors
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        max_new_tokens=512
    )
    
    # Establish retrieval QA chain using fallback langchain-classic
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    st.write("---")
    user_query = st.text_input("Ask a legal question based on this document:")
    
    if user_query:
        with st.spinner("Analyzing document structure & statutes..."):
            response = qa_chain.invoke({"query": user_query})
            
            st.markdown("### 🏛️ Legal Analysis:")
            st.write(response["result"])
            
            # Display source citations nicely
            with st.expander("📌 View Statutory Citations"):
                for idx, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Citation {idx + 1} (Page {doc.metadata.get('page', 'Unknown') + 1}):**")
                    st.info(doc.page_content)

else:
    st.info("Please upload a PDF document in the dashboard above to initiate analysis.")