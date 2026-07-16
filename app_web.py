import os
import io
import streamlit as st
import pysqlite3 as sqlite3
import sys
# Hack to make Chromadb/Langchain use the newer SQLite on Streamlit Cloud
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint
# NEW LINE (Line 14)
from langchain_classic.chains import RetrievalQA
from chromadb.config import Settings
import shutil

# --- SECRETS CONFIGURATION ---
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    st.error("Error: HF_TOKEN not found in Streamlit Secrets! Please add your token under Advanced Settings.")
    st.stop()

# --- THEME/STYLING CONFIG ---
st.set_page_config(page_title="JurisDocs AI", page_icon="⚖️", layout="wide")

PRIMARY_COLOR = "#E0A96D" # Warm Gold
BG_COLOR = "#1C1F26"      # Rich Charcoal/Navy

# Advanced CSS injection for complete theme overhaul
st.markdown(f"""
    <style>
        /* Main background and app alignment */
        .stApp {{
            background-color: {BG_COLOR};
            color: #F4F5F6;
        }}
        /* Customize standard headers */
        h1, h2, h3 {{
            color: {PRIMARY_COLOR} !important;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 700;
        }}
        /* Style the chat input box */
        .stChatInput {{
            border-radius: 20px;
            border: 1px solid {PRIMARY_COLOR};
        }}
        /* Style the main upload card */
        .stFileUploader {{
            border: 2px dashed {PRIMARY_COLOR};
            border-radius: 10px;
            padding: 20px;
        }}
        /* Sidebar customization */
        [data-testid="stSidebar"] {{
            background-color: #121419;
            border-right: 1px solid #333;
        }}
    </style>
""", unsafe_allow_html=True)


# --- INITIALIZATION FUNCTIONS ---

@st.cache_resource(show_spinner=False)
def get_ai_chains():
    """Initializes and caches the Embedding model and LLM."""
    
    # 1. Initialize Vector Brain (BAAI/bge-large for elite context capture)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. Initialize LLM (Mistral-7B for advanced legal reasoning)
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.1,  # Low temperature = precision, no hallucination
        max_new_tokens=1024,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"prompt_template": "[INST]Context: {context} Question: {question} [/INST]"}
    )
    
    return embeddings, llm

# --- DOC PROCESSING ENGINE ---

CHROMA_DB_PATH = "./chroma_db_locker"

def process_and_index_document(uploaded_file, embeddings_model):
    """Saves PDF, loads it, splits into chunks, and stores in vector DB."""
    
    # 1. Clean out the old 'Document Locker' to start fresh
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    temp_filename = f"temp_{uploaded_file.name}"
    
    with st.spinner("⏳ Locating document and mapping legal text..."):
        # Save uploaded file locally so PyPDF can read it
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and split the PDF
        loader = PyPDFLoader(temp_filename)
        pages = loader.load()
        
        # Split text (smart chunking preserving paragraphs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, 
            separators=["\n\n", "\n", " ", ""]
        )
        docs = text_splitter.split_documents(pages)
        
        # Add a placeholder 'filename' meta tag
        for doc in docs:
            doc.metadata["source_name"] = uploaded_file.name

        # Create and store the vectors in Chroma
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings_model,
            persist_directory=CHROMA_DB_PATH
        )
        vector_db.persist()

    # Cleanup temp file
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    return vector_db


# --- MAIN UI ---

def main():
    embeddings_model, llm_model = get_ai_chains()
    
    # Persistent app state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    # --- SIDEBAR & CREDITS ---
    with st.sidebar:
        st.title("⚖️ JurisDocs AI")
        st.subheader("Your Intelligent Legal Locker")
        st.markdown("""
            This advanced RAG assistant uses:
            * **Mistral 7B** LLM
            * **BGE-Large** Embeddings
            * **ChromaDB** Vector Storage
            
            1. Upload a Legal PDF
            2. Ask questions below!
        """)
        st.markdown("---")
        
        # --- NEW PROFESSIONAL DEVELOPER FOOTER ---
        st.markdown(f"""
            <div style='text-align: center; color: {PRIMARY_COLOR}; font-size: 0.9em; font-family: sans-serif;'>
                <strong>Engineered by Ishan Mishra</strong><br/>
                <span style='color: #F4F5F6; font-size: 0.85em; opacity: 0.9;'>RAG Legal Assistant</span><br/>
                <a href="https://github.com/ishanmishra0827" target="_blank" style="color: {PRIMARY_COLOR}; text-decoration: none; font-weight: bold;">GitHub</a> | <a href="https://www.linkedin.com/in/ishan-mishra/" target="_blank" style="color: {PRIMARY_COLOR}; text-decoration: none; font-weight: bold;">LinkedIn</a>
            </div>
        """, unsafe_allow_html=True)


    # --- MAIN CONTENT ---
    st.header("1. Step 1: Upload Statutory Document")
    
    uploaded_file = st.file_uploader(
        "Securely drop your legal PDF (Statute, Case, or Contract)", 
        type="pdf",
        help="JurisDocs AI will parse this text and create an intelligent searchable index."
    )

    if uploaded_file:
        # Avoid processing identical file twice
        if st.session_state.vector_store is None:
             st.session_state.vector_store = process_and_index_document(uploaded_file, embeddings_model)
             st.success(f"✅ Success! '**{uploaded_file.name}**' is locked and indexed.")
        
        # Reveal Chat Interface once indexed
        st.markdown("---")
        st.header("2. Step 2: Query the Locker")

        # Define QA Chain (Retrieve context, then synthesize answer)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True # Get page citations
        )

        user_query = st.chat_input("Ex: What is the required notice period for commercial evictions?")

        if user_query:
            with st.spinner("Analyzing legal context..."):
                response = qa_chain.invoke({"query": user_query})
                ans = response["result"]
                source_docs = response["source_documents"]
            
            # Format and Display the AI Response
            st.markdown(f"### JurisDocs AI Response")
            st.write(ans)
            
            # Formatted Citations
            if source_docs:
                with st.expander("📌 View Statutory Citations"):
                    st.markdown("#### Evidence Found in Document:")
                    for idx, doc in enumerate(source_docs):
                        filename = doc.metadata.get("source_name", "pr.24.pdf")
                        page_num = doc.metadata.get("page", 0) + 1 # Convert 0-index
                        st.markdown(f"> **Source**: {filename} (Page {page_num})")
                        st.text(doc.page_content[:500] + "...") # Show snippet
                        st.markdown("---")

if __name__ == "__main__":
    main()