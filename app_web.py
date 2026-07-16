import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Set up page configurations
st.set_page_config(page_title="JurisDocs AI", page_icon="⚖️", layout="wide")

# Custom Midnight Legal Tech Styling
st.markdown("""
    <style>
        /* Main background and font */
        .stApp {
            background-color: #0B132B;
            color: #F4F5F6;
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #1C2541 !important;
        }
        /* Titles & Headers */
        h1 {
            color: #E0A96D !important; /* Elegant Gold */
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-weight: 700;
        }
        h2, h3 {
            color: #5BC0BE !important; /* Sleek Teal */
        }
        /* Custom card container for results */
        .legal-card {
            background-color: #1C2541;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #E0A96D;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Hugging Face API Token (Loaded securely from Streamlit Secrets)
HF_TOKEN = st.secrets["HF_TOKEN"]

# Initialize Semantic Embeddings & HF Client
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    client = InferenceClient(token=HF_TOKEN)
    return embeddings, client

embeddings, client = load_resources()

# ==========================================
# UI Layout
# ==========================================
st.title("⚖️ JurisDocs AI")
st.write("Upload your legal documents in the sidebar, and let AI analyze, summarize, and cite them instantly.")

# Sidebar for PDF Uploads
with st.sidebar:
    st.markdown("<h2 style='color:#E0A96D !important;'>1. Document Locker</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload legal PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.write("🚀 Powered by Qwen-2.5 & LangChain")

# Save uploaded files to a temp directory and process them
documents_to_index = []
TEMP_DIR = "./temp_documents"
os.makedirs(TEMP_DIR, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = PyPDFLoader(temp_path)
            documents_to_index.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            
    st.sidebar.success(f"Successfully loaded {len(uploaded_files)} PDF(s)!")

# ==========================================
# RAG Processing
# ==========================================
if documents_to_index:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents_to_index)
    
    @st.cache_resource(show_spinner="Indexing your legal documents...")
    def build_vector_store(_docs):
        return InMemoryVectorStore.from_documents(_docs, embeddings)
        
    vector_store = build_vector_store(split_docs)
    
    st.markdown("<h2 style='color:#E0A96D !important;'>2. Search & Analysis</h2>", unsafe_allow_html=True)
    user_query = st.text_input("Enter your question here (e.g., 'What are the notice requirements?'):")
    
    if user_query:
        with st.spinner("Analyzing legal contexts and generating answer..."):
            results = vector_store.similarity_search_with_score(user_query, k=3)
            
            valid_contexts = []
            for doc, score in results:
                source_page = doc.metadata.get("page", 0) + 1
                source_file = os.path.basename(doc.metadata.get("source", "Uploaded PDF"))
                valid_contexts.append(f"Source: {source_file} (Page {source_page}):\n{doc.page_content}")
            
            context_str = "\n\n---\n\n".join(valid_contexts)
            
            system_prompt = f"""You are an elite, highly precise legal research assistant. Your sole task is to analyze the provided legal contexts and answer the user's query.

CRITICAL INSTRUCTIONS FOR ACCURACY:
1. EXCLUSIVE RELIANCE: You must formulate your answer ONLY using the facts explicitly stated in the "Retrieved Context" section below. Do not assume, extrapolate, or bring in outside legal knowledge.
2. STRICT CITATION: For every legal rule, timeline, requirement, or right you describe in your answer, you MUST explicitly cite the specific source document and page number.
3. ABSENCE OF INFORMATION: If the provided context does not contain sufficient, direct information to answer the user's query, you must state exactly: "Information not found in public files."

### Retrieved Context:
{context_str}

### User Query:
{user_query}

### Legal Response & Verification:"""

            try:
                completion = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[{"role": "user", "content": system_prompt}],
                    max_tokens=1024,
                    temperature=0.1
                )
                response_text = completion.choices[0].message.content
                
                # Render results in the custom card UI
                st.markdown("<h3 style='color:#E0A96D !important;'>⚖️ AI Legal Analysis</h3>", unsafe_allow_html=True)
                st.markdown(f'<div class="legal-card">{response_text}</div>', unsafe_allow_html=True)
                
                with st.expander("View Raw Retrieved Sources (Citations)"):
                    for ctx in valid_contexts:
                        st.text(ctx)
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error contacting AI: {e}")
else:
    st.info("👈 Please drag and drop or upload one or more legal PDFs in the sidebar to get started!")