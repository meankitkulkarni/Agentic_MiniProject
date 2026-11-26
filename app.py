import ollama
import streamlit as st

from agent import DEPTH, research, sources_md, standalone, synthesize_stream

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
[data-testid="stToolbar"], [data-testid="stDecoration"], footer {display: none;}
.block-container {max-width: 760px; padding-top: 2.5rem; padding-bottom: 7rem;}
.hero {text-align: center; margin: 14vh 0 1.5rem;}
.hero h1 {font-weight: 600; font-size: 2rem;}
.hero p {opacity: .6;}
</style>""", unsafe_allow_html=True)

EXAMPLES = ["Brief me about Konkan Railway",
            "Latest advances in solid-state batteries",
            "How does the UPI payment system work?"]


@st.cache_data(ttl=30)
def local_models():
    try:
        return [m.get("model") or m.get("name") for m in ollama.list()["models"]]
    except Exception:
        return []


with st.sidebar:
    st.subheader("Settings")
    models = local_models()
    if models:
        model = st.selectbox("Model", models,
                             index=models.index("llama3.1:8b") if "llama3.1:8b" in models else 0)
    else:
        st.warning("Ollama not reachable.")
        model = st.text_input("Model", "llama3.1:8b")
    depth = st.radio("Research depth", list(DEPTH), index=1,
                     help="Quick is fastest; Deep reads more sources over more rounds.")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.session_state.setdefault("messages", [])
msgs = st.session_state.messages

if not msgs:
    st.markdown('<div class="hero"><h1>What would you like to research?</h1>'
                '<p>I search the web, read the pages, and answer with citations.</p></div>',
                unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for m in msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"Sources ({m['sources'].count(chr(10)) + 1})"):
                st.markdown(m["sources"])

prompt = st.chat_input("Ask anything…") or st.session_state.pop("pending", None)

if prompt:
    history = [m["content"] for m in msgs if m["role"] == "user"]
    msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        src_text = ""
        try:
            with st.status("Researching…", expanded=False) as status:
                question = standalone(prompt, history, model)
                sources = research(question, model, depth,
                                   log_fn=lambda t: (status.update(label=t), st.write(t)))
                status.update(label=f"Read {len(sources)} relevant sources", state="complete")
            if sources:
                answer = st.write_stream(synthesize_stream(question, sources, model))
                src_text = sources_md(sources)
                with st.expander(f"Sources ({len(sources)})"):
                    st.markdown(src_text)
            else:
                answer = ("I couldn't find relevant sources for that. "
                          "Try rephrasing, or switch to a deeper research setting.")
                st.markdown(answer)
        except Exception as e:
            answer = f"Something went wrong: `{type(e).__name__}: {e}`"
            st.error(answer)
    msgs.append({"role": "assistant", "content": answer, "sources": src_text})