import os
import streamlit as st

import catalog_rag

st.set_page_config(
    page_title="Prosper ISD Course Catalog Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown("""
    <style>
    .stApp {
        background: #0F1117;
        color: #E8EAED;
        font-family: -apple-system, 'Segoe UI', Inter, sans-serif;
    }
    h1 {
        color: #F5C542 !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
        font-size: 2rem !important;
    }
    h2, h3, h4 { color: #E8EAED !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] {
        background-color: #0A0C11 !important;
        border-right: 1px solid #1E2129;
    }
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] input {
        background-color: #171A21 !important;
        border: 1px solid #2A2E38 !important;
        color: #F1F3F5 !important;
    }
    div[data-testid="stChatInput"] textarea:focus { border-color: #F5C542 !important; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_index():
    return catalog_rag.build_index()


@st.cache_resource(show_spinner=False)
def load_chain(model_key: str):
    return catalog_rag.build_chain()


with st.sidebar:
    st.title("🎓 Course Catalog Assistant")
    st.caption("Prosper ISD · 2026–2027 Academic Course Guide")
    st.write("---")

    try:
        for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN", "LLM_PROVIDER", "LLM_MODEL"):
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
    except Exception:
        pass

    import rag_core
    provider = rag_core.detect_provider()
    if provider is None:
        st.error("No model credential configured. Add GROQ_API_KEY to Secrets.")
        st.stop()
    active_model = os.environ.get("LLM_MODEL") or rag_core.DEFAULT_MODELS.get(provider, "unknown")
    st.caption(f"Provider: {provider} · {active_model}")

    st.write("---")
    st.markdown("**What this can answer**")
    st.markdown(
        "- Prerequisites and corequisites\n"
        "- Which courses are offered\n"
        "- Credit values, GPA weight, grade levels\n"
        "- What a course covers"
    )
    st.markdown("**What it can't**")
    st.markdown(
        "- See your transcript or credits\n"
        "- Tell you which courses to pick\n"
        "- Confirm graduation or endorsement status\n\n"
        "Those go to your counselor."
    )

    st.write("---")
    show_sources = st.toggle("Show retrieved courses", value=False)
    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

    st.write("---")
    st.caption("Built by Ishan Mishra · Prosper High School")


st.title("Prosper ISD Course Catalog Assistant")
st.markdown(
    "Ask about prerequisites, course offerings, credits, or GPA weight. "
    "Answers come only from the 2026–2027 course catalog, with the page cited."
)

st.warning(
    "**Verify before you register.** This is a search tool for the catalog, not "
    "an official source. Confirm anything that affects your schedule with your "
    "counselor. It cannot see your transcript, credits, or graduation plan.\n\n"
    "A small number of CTE course titles may display incorrectly due to the "
    "catalog's PDF layout — double-check prerequisites in that section.",
    icon="⚠️",
)

with st.spinner("Loading course catalog…"):
    index = load_index()
chain = load_chain(active_model)

st.caption(f"{len(index['docs'])} courses indexed from the 2026–2027 catalog.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state["messages"]:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    examples = [
        "What are the prerequisites for AP Precalculus?",
        "Is there a course on Mexican American Studies?",
        "What do I need before Dual Credit Composition I?",
        "How can I earn my PE credit?",
    ]
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"ex{i}", use_container_width=True):
            st.session_state["pending"] = example
            st.rerun()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("courses") and show_sources:
            with st.expander("Courses retrieved"):
                for name in msg["courses"]:
                    st.markdown(f"- {name}")

query = st.chat_input("Ask about a course…") or st.session_state.pop("pending", None)

if query:
    st.chat_message("user").markdown(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Searching the catalog…"):
        try:
            result = catalog_rag.answer(index, chain, query)
            reply = result["answer"]
            courses = result["courses"]
        except Exception as exc:
            reply = (
                "The model service returned an error, so I can't answer right now. "
                f"\n\n`{type(exc).__name__}: {exc}`"
            )
            courses = []

    with st.chat_message("assistant"):
        st.markdown(reply)
        if courses and show_sources:
            with st.expander("Courses retrieved"):
                for name in courses:
                    st.markdown(f"- {name}")

    st.session_state["messages"].append(
        {"role": "assistant", "content": reply, "courses": courses}
    )
