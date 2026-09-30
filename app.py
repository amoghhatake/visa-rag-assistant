import re
import streamlit as st
from rag_pipeline import build_index_cached, retrieve, generate, is_unsafe_llm_judge

st.set_page_config(page_title="Visa Conditions Assistant", page_icon="🛂", layout="centered")

st.markdown("""
<style>
.main .block-container { padding-top: 2rem; max-width: 720px; }
div[data-testid="stChatMessage"] { border-radius: 12px; }
.footer-note { text-align: center; color: #6b7280; font-size: 12px; margin-top: 24px; }
.badge { font-size: 12px; opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


def redact_pii(text):
    text = re.sub(r'\b[A-Z]{1,2}\d{6,9}\b', '[REDACTED-ID]', text)
    text = re.sub(r'\bvisa\s+grant\s+number\s*:?\s*\d{6,12}\b', 'visa grant number [REDACTED-ID]',
                  text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED-DATE]', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[REDACTED-DATE]', text)
    return text


def hours_check(query, limit=48):
    nums = [float(n) for n in
            re.findall(r'(\d+(?:\.\d+)?)\D{0,2}\s*(?:hours|hrs|hr)\b', query.lower())]
    if len(nums) >= 2:
        total = sum(nums)
        status = "MORE than" if total > limit else "within"
        return (f"The hours you mentioned add up to {total:g}, which is {status} the "
                f"{limit}-hour fortnightly limit, if they fall in the same fortnight during semester. "
                f"Please confirm your situation with VEVO or your international student office.")
    return None


EXAMPLE_QUESTIONS = [
    "How many hours can I work per fortnight?",
    "Can I work unlimited hours during scheduled course breaks?",
    "What is the age limit for the Temporary Graduate visa?",
    "Do I need OSHC for my whole stay in Australia?",
]


@st.cache_resource
def load_index():
    return build_index_cached()


st.title("🛂 Visa Conditions Assistant")
st.caption("Grounded answers for Australian student and graduate visa conditions, with cited sources. "
           "General information only, not migration advice.")
st.caption("⚠️ Don't share passport numbers, visa grant numbers, or date of birth here.")

if "index" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.index = load_index()

with st.sidebar:
    st.subheader("Your visa")
    persona = st.selectbox("Select your visa subclass", ["500 (Student)", "485 (Temporary Graduate)"])
    persona_key = persona.split()[0]

    use_filter = st.toggle("Persona-filtered retrieval", value=True,
                           help="ON: only searches your visa's documents plus general docs.")

    doc_count = len({item["source"] for item in st.session_state.index})
    persona_count = len([i for i in st.session_state.index if i["persona"] == persona_key])
    st.caption(f"{doc_count} documents indexed  ·  {persona_count} chunks for {persona_key}")

    st.divider()
    st.caption("🟢 Strong match &nbsp;&nbsp; 🟡 Weaker match &nbsp;&nbsp; 🔴 Not found")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! Select your visa type on the left, then ask a question, or try one below.",
         "meta": None}
    ]

for msg in st.session_state.messages:
    avatar = "🛂" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"]["label"])
            if msg["meta"].get("sources"):
                with st.expander("Show sources"):
                    for s in msg["meta"]["sources"]:
                        st.markdown(f"- `{s}`")

clicked = None
if len(st.session_state.messages) == 1:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True, key=f"ex_{i}"):
            clicked = q

user_input = st.chat_input("Type your question...")
query = user_input or clicked

if query:
    original_query = query
    query = redact_pii(query)

    st.session_state.messages.append({"role": "user", "content": query, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.write(query)
        if query != original_query:
            st.caption("🔒 Personal identifiers were removed before processing.")

    with st.chat_message("assistant", avatar="🛂"):
        with st.spinner("Checking..."):
            unsafe_flag, verdict = is_unsafe_llm_judge(query)

        if unsafe_flag:
            answer = ("I can't help with questions about circumventing visa conditions or immigration law. "
                      "If you're facing a difficult visa situation, please contact your institution's "
                      "international student office or a registered migration agent.")
            meta = {"label": "🔴 Declined, flagged as unsafe", "sources": []}
            st.write(answer)
            st.caption(meta["label"])
        else:
            with st.spinner("Checking sources..."):
                retrieved = retrieve(query, st.session_state.index, persona_key, use_filter=use_filter)
                answer, sources, score = generate(query, retrieved)

            note = hours_check(query) if persona_key == "500" else None
            if note:
                answer = ("Under a Student visa (subclass 500), you can work up to 48 hours per fortnight "
                          "while your course is in session. " + note)

            declined = any(p in answer.lower() for p in
                          ["i don't know", "i do not know", "do not have information", "don't have information"])
            label = ("🔴 Not found in sources" if declined else
                    "🟢 Strong source match" if score > 0.7 else
                    "🟡 Weaker source match")

            meta = {"label": f"{label}  ·  Similarity: {score:.2f}", "sources": sources}
            st.write(answer)
            st.caption(meta["label"])
            if sources:
                with st.expander("Show sources"):
                    for s in sources:
                        st.markdown(f"- `{s}`")

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})

st.markdown('<div class="footer-note">Prototype for a university WIL project. Not migration advice.</div>',
           unsafe_allow_html=True)