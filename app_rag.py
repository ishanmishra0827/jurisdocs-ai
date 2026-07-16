import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Set your Hugging Face API Token directly
HF_TOKEN = "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"

# ==========================================
# 1. Automatically Load and Parse PDFs
# ==========================================
DOCUMENTS_DIR = "./documents"
all_docs = []

print("Scanning 'documents/' folder for PDFs...")

pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith('.pdf')]

if not pdf_files:
    print(f"\n⚠️ WARNING: No PDF files found in '{DOCUMENTS_DIR}'.")
    print("Please drop a legal PDF into that folder and run this script again!")
    exit()

for pdf_file in pdf_files:
    pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file)
    print(f"Reading: {pdf_file}...")
    try:
        loader = PyPDFLoader(pdf_path)
        all_docs.extend(loader.load())
    except Exception as e:
        print(f"Error reading {pdf_file}: {e}")

print(f"Successfully loaded {len(all_docs)} pages from your PDF(s).")

# ==========================================
# 2. Smart Chunking
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
split_documents = text_splitter.split_documents(all_docs)
print(f"Split pages into {len(split_documents)} optimized context chunks.")

# ==========================================
# 3. Native Embedding and Indexing
# ==========================================
print("\nLoading semantic embedding model (BGE-Large)...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

print("Indexing chunks into your local vector store...")
vector_store = InMemoryVectorStore.from_documents(split_documents, embeddings)

# ==========================================
# 4. Initialize Native Hugging Face Client
# ==========================================
print("\nConnecting to Hugging Face Client (using Qwen 2.5)...")
client = InferenceClient(token=HF_TOKEN)

# ==========================================
# 5. Retrieval and LLM Generation
# ==========================================
def run_pdf_query(user_query):
    # Retrieve the top 3 most relevant chunks from the PDF
    results = vector_store.similarity_search_with_score(user_query, k=3)
    
    valid_contexts = []
    for doc, score in results:
        source_page = doc.metadata.get("page", 0) + 1
        source_file = os.path.basename(doc.metadata.get("source", "Unknown PDF"))
        valid_contexts.append(f"Source: {source_file} (Page {source_page}):\n{doc.page_content}")

    if not valid_contexts:
        print(f"\nQUERY: {user_query}\n-> Output: Information not found in public files.\n")
        return
    
    context_str = "\n\n---\n\n".join(valid_contexts)
    
    # Strict prompt rules forcing the LLM to only use the retrieved PDF context
    system_prompt = f"""You are an elite, highly precise legal research assistant working for Legal Aid of NorthWest Texas. Your sole task is to analyze the provided legal contexts and answer the user's query.

CRITICAL INSTRUCTIONS FOR ACCURACY:
1. EXCLUSIVE RELIANCE: You must formulate your answer ONLY using the facts explicitly stated in the "Retrieved Context" section below. Do not assume, extrapolate, or bring in outside legal knowledge.
2. STRICT CITATION: For every legal rule, timeline, requirement, or right you describe in your answer, you MUST explicitly cite the specific source document and page number.
3. ABSENCE OF INFORMATION: If the provided context does not contain sufficient, direct information to answer the user's query, you must state exactly: "Information not found in public files."

### Retrieved Context:
{context_str}

### User Query:
{user_query}

### Legal Response & Verification:"""

    print("\n" + "="*60)
    print(f"USER QUERY: {user_query}")
    print("="*60)
    print("\n[Thinking... Asking LLM to answer using only your PDF...]")
    
    # Run the query through the native Hugging Face client with Qwen
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": system_prompt}],
            max_tokens=512,
            temperature=0.1
        )
        response_text = completion.choices[0].message.content
        
        print("\n--- LLM GENERATED ANSWER ---")
        print(response_text)
        print("\n" + "="*60)
    except Exception as e:
        print(f"\n❌ Error contacting Hugging Face API: {e}")

# ==========================================
# 6. Interactive Terminal Chat Loop
# ==========================================
print("\n🎉 Legal RAG Pipeline is ready!")
print("Type your questions about the loaded documents below.")
print("Type 'exit' or 'quit' to stop.")
print("="*60)

while True:
    user_prompt = input("\nAsk a legal question: ")
    if user_prompt.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    if not user_prompt.strip():
        continue
    run_pdf_query(user_prompt)