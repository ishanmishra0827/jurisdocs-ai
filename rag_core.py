"""
rag_core.py — retrieval pipeline for JurisDocs AI, with no Streamlit dependency.

Kept deliberately free of `import streamlit` so the evaluation harness can drive
the exact same code path the web app uses. If these ever diverge, the eval stops
measuring the deployed system and becomes decorative.
"""

import os
import re
import uuid
import tempfile

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

try:
    import rank_bm25  # noqa: F401
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


SECTION_PATTERN = re.compile(
    r"(?:sec(?:tion)?s?\.?|§+|art(?:icle)?\.?)?\s*"
    r"(\d{1,4}\.\d{1,4}[A-Za-z]?)",
    re.IGNORECASE,
)

# A real section HEADING, as opposed to a cross-reference to another chapter.
# Headings are "Sec. 24.008. EFFECT ON OTHER ACTIONS" — number, period, then an
# ALL-CAPS title. A citation like "see Section 92.017" has no caps title after
# it, so this pattern ignores it. Distinguishing the two is what stops chunks
# from being labelled with whatever chapter they happen to mention.
HEADING_PATTERN = re.compile(
    r"(?:Sec|SEC|Section|SECTION|Art|ART)\.?\s*A*\s*"
    r"(\d{1,4}\.\d{1,4}[A-Za-z]?)\s*\.\s*A*\s*"
    r"[A-Z][A-Z0-9 ,'\-]{3,}"
)

TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)*")


def normalize_statute_text(text: str) -> str:
    """
    Texas statute PDFs encode a special space character that pypdf extracts as a
    literal "A", producing "Sec.A 24.008.AA EFFECT ON OTHER ACTIONS". Left alone
    it breaks heading detection and pollutes the keyword index.
    """
    text = text.replace("\u00a0", " ")
    text = re.sub(r"(Sec|SEC|Section|SECTION|Art|ART)\.\s*A+\s*(\d)", r"\1. \2", text)
    text = re.sub(r"(\d)\.\s*A+\s+([A-Z]{2,})", r"\1. \2", text)
    return text

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def bm25_preprocess(text: str):
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def extract_section_numbers(text: str):
    found = []
    for match in SECTION_PATTERN.finditer(text):
        num = match.group(1)
        if num not in found:
            found.append(num)
    return found


def is_toc_chunk(text: str) -> bool:
    return text.count("...") >= 3 or text.count(". . .") >= 2


def extract_headings(text: str):
    """Section numbers that are actually declared in this text, in order."""
    found = []
    for match in HEADING_PATTERN.finditer(text):
        num = match.group(1)
        if num not in found:
            found.append(num)
    return found


def chunk_documents(documents):
    for doc in documents:
        doc.page_content = normalize_statute_text(doc.page_content)

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
        headings = extract_headings(chunk.page_content)
        if headings:
            # The last heading governs the tail of the chunk, which is what
            # continues into whatever follows.
            current_section = headings[-1]

        chunk.metadata["headings"] = ", ".join(headings)
        chunk.metadata["section"] = current_section or ""
        chunk.metadata["mentions"] = ", ".join(
            extract_section_numbers(chunk.page_content)
        )

        page = chunk.metadata.get("page")
        chunk.metadata["page_label"] = page + 1 if isinstance(page, int) else "?"

        label = ", ".join(headings) if headings else (current_section or "")
        if label and label not in chunk.page_content[:60]:
            chunk.page_content = f"[Sec. {label}]\n{chunk.page_content}"

        chunks.append(chunk)

    return chunks


def _assert_pdf(path: str):
    """
    A downloaded error page saved as .pdf is a common failure. pypdf reports it
    as 'Stream has ended unexpectedly' several frames deep, which looks like
    corruption rather than 'this is HTML'.
    """
    with open(path, "rb") as fh:
        header = fh.read(5)
    if header != b"%PDF-":
        preview = header.decode("utf-8", errors="replace")
        raise ValueError(
            f"{path} is not a PDF (starts with {preview!r}). "
            "If it was downloaded with curl or wget, it is most likely an HTML "
            "error page saved under a .pdf name. Check with: file " + path
        )


def build_index_from_paths(pdf_paths):
    """Index one or more PDFs into a single corpus (e.g. Chapter 24 + Chapter 92)."""
    documents = []
    for path in pdf_paths:
        _assert_pdf(path)
        documents.extend(PyPDFLoader(path).load())

    total_chars = sum(len(d.page_content) for d in documents)
    chunks = chunk_documents(documents)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"corpus_{uuid.uuid4().hex[:8]}",
    )
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    keyword_retriever = None
    if BM25_AVAILABLE:
        keyword_retriever = BM25Retriever.from_documents(
            chunks, preprocess_func=bm25_preprocess
        )
        keyword_retriever.k = 8

    return {
        "chunks": chunks,
        "dense": dense_retriever,
        "keyword": keyword_retriever,
        "pages": len(documents),
        "total_chars": total_chars,
    }


def build_index_from_bytes(file_bytes: bytes):
    return build_index_from_byte_list([file_bytes])


def build_index_from_byte_list(byte_list):
    """Index several uploaded PDFs as one corpus."""
    paths = []
    try:
        for blob in byte_list:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(blob)
                paths.append(tmp.name)
        return build_index_from_paths(paths)
    finally:
        for path in paths:
            try:
                os.unlink(path)
            except OSError:
                pass


def gather_sections(index, nums):
    body, mentions = [], []
    target = set(nums)
    for chunk in index["chunks"]:
        if is_toc_chunk(chunk.page_content):
            continue
        declared = {h.strip() for h in chunk.metadata.get("headings", "").split(",") if h.strip()}
        declared.add(chunk.metadata.get("section", ""))
        if target & declared:
            body.append(chunk)
        elif any(n in chunk.page_content for n in nums):
            mentions.append(chunk)
    return body, mentions


def hybrid_retrieve(index, query: str, top_k: int = 8):
    ordered = []
    seen = set()

    def add(doc):
        key = doc.page_content[:300]
        if key not in seen:
            seen.add(key)
            ordered.append(doc)

    query_sections = extract_section_numbers(query)
    if query_sections:
        body, mentions = gather_sections(index, query_sections)
        for doc in body[:6]:
            add(doc)
        for doc in mentions[:2]:
            add(doc)
    else:
        # No section cited, so there is no anchor and no reserved budget. These
        # queries need MORE breadth, not less — a tenant asking "how long to get
        # my deposit back" is competing against every chunk that mentions
        # "security deposit". Without this, paraphrase queries got a smaller
        # context window than queries that already named the section.
        top_k = max(top_k, 14)

    reserved = len(ordered)

    expanded = query
    for num in query_sections:
        expanded += f" Sec. {num} Section {num} § {num} {num}"

    if index["keyword"] is not None:
        try:
            for doc in index["keyword"].invoke(expanded):
                add(doc)
        except Exception:
            pass

    try:
        for doc in index["dense"].invoke(query):
            add(doc)
    except Exception:
        pass

    limit = max(top_k, reserved + 4)
    return ordered[:limit], query_sections


def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page_label", "?")
        section = doc.metadata.get("section") or "n/a"
        blocks.append(
            f"[Source {i} | page {page} | section {section}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


SYSTEM_PROMPT = (
    "You are a legal research assistant analyzing the statutory excerpts provided "
    "below. Answer ONLY from those excerpts. You have no reliable knowledge of law "
    "beyond them.\n\n"
    "First decide which case applies, then follow it exactly.\n\n"
    "CASE A — The excerpts contain the provision that governs the question.\n"
    "  Answer it. Lead with the direct answer in your first sentence. Cite the "
    "governing section AND subsection with page, e.g. '(Sec. 92.103(a), p. 57)'. "
    "Set out the elements, deadlines, dollar amounts, and notice requirements the "
    "text actually states. Quote operative language where exact wording carries "
    "legal weight.\n\n"
    "CASE B — The excerpts do NOT contain the provision that governs the question.\n"
    "  Your FIRST sentence must say the document does not address the question. If "
    "the excerpts make clear which chapter or subject area would govern, name it. "
    "Then STOP. Do not survey related sections. Do not reason out loud about where "
    "the answer might be. Do not say what the rule 'typically' or 'generally' is. "
    "Two sentences maximum. A reader skimming your answer will take the first "
    "citation as the answer, so a refusal that opens with a tangential section is "
    "worse than no answer at all.\n\n"
    "CASE C — The user is asking what they personally should do ('should I sue', "
    "'should I fight this', 'what would you do').\n"
    "  Do not list statutory sections. Say plainly that you cannot advise on what "
    "someone should do in their own situation, and direct them to a licensed "
    "attorney or a local legal aid organization. Someone asking this is usually in "
    "a difficult position and needs a referral, not a reading list.\n\n"
    "TWO ABSOLUTE RULES, in every case:\n"
    "  Never attach a section citation to a proposition that section does not "
    "state. Before citing, confirm the excerpt actually contains the rule you are "
    "describing. Citing a topically adjacent section makes an unsupported claim "
    "look verified, which is more dangerous than saying nothing.\n"
    "  Never invent section numbers, statutory text, deadlines, or dollar "
    "amounts.\n\n"
    "Prior conversation:\n{chat_history}\n\n"
    "Document excerpts:\n{context}"
)


def detect_provider():
    """Pick a provider from whichever credential is present."""
    if os.environ.get("LLM_PROVIDER"):
        return os.environ["LLM_PROVIDER"].lower()
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("GOOGLE_API_KEY"):
        return "google"
    if os.environ.get("OLLAMA_MODEL"):
        return "ollama"
    if os.environ.get("HF_TOKEN"):
        return "huggingface"
    return None


# Model names change as providers deprecate and release. If a call fails with
# "model not found", check the provider's current model list rather than
# assuming the code is broken.
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "google": "gemini-2.0-flash",
    "ollama": "qwen2.5:7b",
    "huggingface": "Qwen/Qwen2.5-7B-Instruct",
}


def build_llm(provider=None, temperature: float = 0.1, max_new_tokens: int = 1024):
    provider = provider or detect_provider()
    if provider is None:
        raise RuntimeError(
            "No LLM credential found. Set one of: GROQ_API_KEY (free tier), "
            "GOOGLE_API_KEY (free tier), OLLAMA_MODEL (local, offline), or HF_TOKEN."
        )

    model = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(provider))

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=temperature, max_tokens=max_new_tokens)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature, max_output_tokens=max_new_tokens
        )

    if provider == "ollama":
        # Fully local and offline. Nothing leaves the machine, which also makes
        # this the only option that is safe for confidential client documents.
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)

    if provider == "huggingface":
        base_llm = HuggingFaceEndpoint(
            repo_id=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
        )
        return ChatHuggingFace(llm=base_llm)

    raise RuntimeError(f"Unknown provider: {provider}")


def build_chain(temperature: float = 0.1, max_new_tokens: int = 1024, provider=None):
    llm = build_llm(provider, temperature, max_new_tokens)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | llm


def clean_answer(text: str) -> str:
    return re.sub(r"^\s*#*\s*(🏛️)?\s*Legal Analysis:?\s*", "", text).strip()


def answer_question(index, chain, question: str, chat_history: str = "(none)", top_k: int = 8):
    docs, query_sections = hybrid_retrieve(index, question, top_k=top_k)
    context = format_context(docs)
    response = chain.invoke({
        "input": question,
        "context": context,
        "chat_history": chat_history,
    })
    return {
        "answer": clean_answer(response.content),
        "docs": docs,
        "context": context,
        "query_sections": query_sections,
        "retrieved_sections": [d.metadata.get("section", "") for d in docs],
    }