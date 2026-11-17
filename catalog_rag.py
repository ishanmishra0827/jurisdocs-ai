"""
catalog_rag.py — question answering over parsed Prosper ISD course records.

Design note, carried over from the statute version of this project:

There, chunking by character count split Sec. 92.331(a) from 92.331(b), and the
model — shown half a rule — stated confidently that the other half did not
exist. Four separate fixes were needed before that class of failure was closed.

Here it cannot happen. One course is one chunk, so a course's prerequisites are
never separated from the course. The structure of the source document is doing
the work that retrieval heuristics had to do before.
"""

import json
import os
import re

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

import rag_core

try:
    from langchain_community.retrievers import BM25Retriever
    import rank_bm25  # noqa: F401
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

import uuid


COURSE_CODE_IN_QUERY = re.compile(
    r"\b(?:[A-Z][A-Z0-9]{2,8}/[A-Z]|[A-Z]{1,6}\d[A-Z0-9]{0,6}(?:/[A-Z])?)\b"
)


def record_to_text(rec: dict) -> str:
    """One course, rendered so every field is retrievable and quotable."""
    lines = [f"Course: {rec['name']}"]
    if rec.get("codes"):
        lines.append(f"Course code: {', '.join(rec['codes'])}")
    if rec.get("grade"):
        lines.append(f"Grade levels: {rec['grade']}")
    if rec.get("credit"):
        lines.append(f"Credit: {rec['credit']}")
    if rec.get("gpa"):
        lines.append(f"GPA weight: {rec['gpa']}")
    lines.append(f"Prerequisite: {rec.get('prerequisite') or 'not listed in the catalog'}")
    if rec.get("corequisite"):
        lines.append(f"Corequisite: {rec['corequisite']}")
    if rec.get("listed_with") and len(rec["listed_with"]) > 1:
        lines.append(f"Listed in the catalog alongside: {', '.join(rec['listed_with'])}")
    lines.append(f"Catalog page: {rec.get('page', '?')}")
    if rec.get("description"):
        lines.append(f"Description: {rec['description']}")
    return "\n".join(lines)


def build_index(records_path: str = "catalog_records.json"):
    with open(records_path) as fh:
        records = json.load(fh)

    docs = []
    for i, rec in enumerate(records):
        docs.append(Document(
            page_content=record_to_text(rec),
            metadata={
                "name": rec["name"],
                "codes": ", ".join(rec.get("codes", [])),
                "page": rec.get("page", "?"),
                "order": i,
            },
        ))

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=rag_core.get_embeddings(),
        collection_name=f"catalog_{uuid.uuid4().hex[:8]}",
    )
    dense = vector_store.as_retriever(search_kwargs={"k": 8})

    keyword = None
    if BM25_AVAILABLE:
        keyword = BM25Retriever.from_documents(docs, preprocess_func=rag_core.bm25_preprocess)
        keyword.k = 8

    return {"docs": docs, "records": records, "dense": dense, "keyword": keyword}


def find_named_courses(index, query: str):
    """
    Exact matches first: a course code in the query, or a catalog course name
    appearing in it. Embeddings are unreliable for both — "Spanish III" and
    "Spanish IV" sit almost on top of each other in vector space, and picking
    the wrong one silently gives a student the wrong prerequisite.
    """
    hits = []
    lowered = query.lower()

    codes = set(COURSE_CODE_IN_QUERY.findall(query.upper()))
    for doc in index["docs"]:
        doc_codes = {c.strip().upper() for c in doc.metadata["codes"].split(",") if c.strip()}
        if codes & doc_codes:
            hits.append(doc)

    # Longest names first so "AP Spanish IV" wins over "Spanish".
    by_length = sorted(index["docs"], key=lambda d: -len(d.metadata["name"]))
    for doc in by_length:
        name = doc.metadata["name"].lower()
        if len(name) >= 5 and name in lowered and doc not in hits:
            hits.append(doc)

    return hits


def retrieve(index, query: str, top_k: int = 8):
    ordered = []
    seen = set()

    def add(doc):
        key = doc.metadata["name"] + doc.metadata["codes"]
        if key not in seen:
            seen.add(key)
            ordered.append(doc)

    for doc in find_named_courses(index, query)[:6]:
        add(doc)
    reserved = len(ordered)

    if index["keyword"] is not None:
        try:
            for doc in index["keyword"].invoke(query):
                add(doc)
        except Exception:
            pass

    try:
        for doc in index["dense"].invoke(query):
            add(doc)
    except Exception:
        pass

    return ordered[:max(top_k, reserved + 4)]


def format_context(docs):
    return "\n\n---\n\n".join(
        f"[Course {i}]\n{d.page_content}" for i, d in enumerate(docs, start=1)
    )


SYSTEM_PROMPT = (
    "You answer questions about the Prosper ISD high school course catalog using "
    "ONLY the course records provided below. You have no reliable knowledge of "
    "this district's offerings beyond these records.\n\n"
    "Decide which case applies before you begin writing.\n\n"
    "CASE A — The records contain the course or information asked about.\n"
    "  Answer directly in your first sentence. Cite the course name, its code, and "
    "the catalog page, e.g. '(Theatre Arts III, E2013A/B, p. 95)'. State "
    "prerequisites, corequisites, grade levels, credit, and GPA weight exactly as "
    "the record gives them. Never soften or generalize a prerequisite.\n\n"
    "  If a record's course name is contained within the name the student asked "
    "about, or the reverse — 'Cloud Computing' when they asked about 'Advanced "
    "Cloud Computing' — and its course code and prerequisites are consistent, "
    "treat it as the same course and answer from it. Note the exact title as the "
    "catalog gives it. Catalog headings do not always extract cleanly, and "
    "refusing on a near-identical title tells a student a real course does not "
    "exist.\n\n"
    "  If a record's course name is contained within the name the student asked "

    "about, or the reverse — 'Cloud Computing' when they asked about 'Advanced "

    "Cloud Computing' — and its course code and prerequisites are consistent, "

    "treat it as the same course and answer from it. Note the exact title as the "

    "catalog gives it. Catalog headings do not always extract cleanly, and "

    "refusing on a near-identical title tells a student a real course does not "

    "exist.\n\n"

    "CASE B — The records do not contain the course or information asked about.\n"
    "  Say so in your first sentence and stop. Do not guess at a course that "
    "sounds similar, and do not describe what a course by that name 'typically' "
    "involves. A student who is told a course exists when it does not, or given a "
    "prerequisite that was inferred rather than listed, may plan a schedule around "
    "it and lose a semester.\n\n"
    "CASE D — The context begins with MATCHING COURSES. This is a browse "
    "question, and the list is COMPLETE for the catalog.\n"
    "  Your FIRST line states the number of courses found. Then list every "
    "one with its code and "
    "page. Keep each to one line: name, code, page, grade levels, credit, "
    "weight, prerequisite. Do not drop any, do not add any, and do not expand "
    "into full descriptions. End by offering to go into detail on any of "
    "them.\n\n"
    "CASE C — The student asks what they personally should take, whether a course "
    "is right for them, or how to plan their schedule.\n"
    "  You may lay out what the catalog says about the relevant courses, but do "
    "not recommend a schedule or tell a student what to choose. Say that course "
    "selection should go through their counselor, who can see their transcript, "
    "graduation plan, and endorsement requirements. This tool cannot see any of "
    "that.\n\n"
    "In every case: never invent a course, course code, prerequisite, credit "
    "value, or page number. If a record says a prerequisite is 'not listed in the "
    "catalog', say exactly that rather than concluding there is none.\n\n"
    "Course records:\n{context}"
)



# ---------------------------------------------------------------------
# Category listing
#
# Two different questions need two different retrieval strategies:
#   "What are the prerequisites for Animation I?"  -> one course, full detail
#   "What dual credit courses are offered?"        -> every match, compactly
#
# Top-k retrieval cannot answer the second. With 24 Dual Credit courses and 8
# context slots, the model listed four it inferred from the AP/DC/DE weight and
# named none of the 24 — while phrasing the answer as though it were complete.
# An incomplete list that reads as complete is worse than a refusal.
# ---------------------------------------------------------------------

STOPWORDS = {
    "what", "which", "are", "is", "the", "a", "an", "do", "does", "you", "i",
    "offer", "offered", "offers", "have", "has", "there", "any", "all", "list",
    "show", "me", "can", "take", "about", "tell", "for", "of", "in", "at", "to",
    "my", "school", "prosper", "isd", "high", "district", "classes", "class",
    "courses", "course", "available", "options", "option", "please", "get",
    "related", "like", "kind", "kinds", "type", "types", "some", "and", "or",
}

LISTING_SIGNALS = re.compile(
    r"\b(what|which|list|show|any|all|offered?|offers|available|are there|"
    r"classes|courses)\b", re.I
)
LOOKUP_SIGNALS = re.compile(r"\bprerequisite|what do i need|required|before\b", re.I)


def content_terms(query: str):
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'\-]*", query.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def looks_like_listing(query: str) -> bool:
    if not LISTING_SIGNALS.search(query):
        return False
    if LOOKUP_SIGNALS.search(query):
        return False
    return True


MAX_LISTING = 40


def category_matches(records, query):
    """
    Matching courses, strongest tier first.

    Tier 0 is a title match, 1 is the phrase anywhere, 2 is all terms anywhere.
    A broad term needs the tiers: "art" matches 84 records once Language Arts,
    Culinary Arts and every description mentioning art are counted, and a list
    of 84 is no more useful than the truncation this path exists to fix. When
    titles alone give a usable answer, that is what the student meant.
    """
    terms = content_terms(query)
    if not terms:
        return [], []
    phrase = " ".join(terms)
    tiers = {0: [], 1: [], 2: []}
    for r in records:
        name_lo = r.get("name", "").lower()
        hay = " ".join([
            r.get("name", ""), r.get("gpa", ""), (r.get("description") or "")[:400],
        ]).lower()
        # Anchor at a word boundary, or "art" matches "Earth Systems Science".
        # Start-only, so "art" still finds "Theatre Arts" and "Culinary Arts".
        anchored = rf"\b{re.escape(phrase)}"
        if re.search(anchored, name_lo):
            rank = 0
        elif re.search(anchored, hay):
            rank = 1
        elif all(re.search(rf"\b{re.escape(t)}", hay) for t in terms):
            rank = 2
        else:
            continue
        tiers[rank].append((r.get("page", 0), r))

    total = sum(len(v) for v in tiers.values())
    if len(tiers[0]) >= 3 and total > MAX_LISTING:
        chosen = tiers[0]
    else:
        chosen = tiers[0] + tiers[1] + tiers[2]
    chosen.sort(key=lambda x: x[0])

    # The catalog lists some courses twice — once under their department and
    # again in a cross-reference section ("CTE courses for Fine Arts credit").
    # Both parse into records, so a listing would name the same course twice
    # and overstate how many options a student has.
    seen, deduped = set(), []
    for _, r in chosen:
        key = (r.get("name", "").strip().lower(),
               tuple(c.upper() for c in r.get("codes", [])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped[:MAX_LISTING], terms


def format_listing(records, terms):
    header = f"MATCHING COURSES: {len(records)} found for '{' '.join(terms)}'"
    if len(records) >= MAX_LISTING:
        header += (" (list capped; tell the reader there may be more and to"
                   " narrow the search)")
    lines = [header, ""]
    for r in records:
        codes = ", ".join(r.get("codes", [])) or "-"
        parts = [
            r["name"],
            f"{codes} p{r.get('page', '?')}",
            f"gr{r.get('grade', '?')}",
            f"{r.get('credit', '?')}cr",
            r.get("gpa", "") or "weight?",
            f"pre: {r.get('prerequisite') or 'none listed'}",
        ]
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)


def build_chain():
    # A browse answer lists every matching course on its own line. With 28 Dual
    # Credit courses that is ~1,000 tokens of output, and the 1,024 default cut
    # the list off mid-sentence on the third entry. gpt-oss models also spend
    # part of the budget reasoning before any text appears. Only generated
    # tokens are billed, so a higher cap costs nothing on short answers.
    llm = rag_core.build_llm(max_new_tokens=4000, reasoning_effort="low")
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt | llm


def answer(index, chain, question: str, top_k: int = 8):
    listing = None
    if looks_like_listing(question):
        matches, terms = category_matches(index["records"], question)
        if len(matches) > 3:
            listing = (matches, terms)

    if listing:
        matches, terms = listing
        context = format_listing(matches, terms)
        names = [r["name"] for r in matches]
        docs = []
    else:
        docs = retrieve(index, question, top_k=top_k)
        context = format_context(docs)
        names = [d.metadata["name"] for d in docs]

    response = chain.invoke({"input": question, "context": context})
    return {
        "answer": response.content.strip(),
        "docs": docs,
        "courses": names,
        "listing": bool(listing),
    }


if __name__ == "__main__":
    import sys

    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("HF_TOKEN"):
        sys.exit("Set GROQ_API_KEY (or another provider credential) first.")

    idx = build_index()
    print(f"Indexed {len(idx['docs'])} courses.")
    if not BM25_AVAILABLE:
        print("WARNING: rank_bm25 missing — keyword retrieval disabled.")

    chain = build_chain()
    questions = sys.argv[1:] or [
        "What are the prerequisites for AP Precalculus?",
        "What do I need to take before AP Spanish V?",
        "Is there a course on Mexican American Studies?",
        "What is the prerequisite for Underwater Basket Weaving?",
        "Should I take AP Calculus or AP Statistics next year?",
    ]
    for q in questions:
        print("\n" + "=" * 70)
        print("Q:", q)
        result = answer(idx, chain, q)
        print("-" * 70)
        print(result["answer"])
        print("-" * 70)
        print("retrieved:", result["courses"][:6])
