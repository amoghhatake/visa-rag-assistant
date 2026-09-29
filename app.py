import re
import streamlit as st
from rag_pipeline import build_index_cached, retrieve, generate, is_unsafe_llm_judge


def redact_pii(text):
    text = re.sub(r'\b[A-Z]{1,2}\d{6,9}\b', '[REDACTED-ID]', text)
    text = re.sub(r'\bvisa\s+grant\s+number\s*:?\s*\d{6,12}\b', 'visa grant number [REDACTED-ID]',
                  text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED-DATE]', text)
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[REDACTED-DATE]', text)
    return text


def hours_check(query, limit=48):
    """If the user gives two or more hour figures, add them up in code (the model can't be trusted to)."""
    nums = [float(n) for n in
            re.findall(r'(\d+(?:\.\d+)?)\D{0,2}\s*(?:hours|hrs|hr)\b', query.lower())]
    if len(nums) >= 2:
        total = sum(nums)
        status = "MORE than" if total > limit else "within"
        return (f"The hours you mentioned add up to {total:g}, which is {status} the "
                f"{limit}-hour fortnightly limit, if they fall in the same fortnight during semester. "
                f"Please confirm your situation with VEVO or your international student office.")
    return None


st.set_page_config(page_title="Visa Conditions Assistant", page_icon="🛂", layout="centered")

st.title("🛂 Visa Conditions Assistant")
st.caption("Ask me about your student or graduate visa conditions. I only answer from verified sources — "
           "I'll say \"I don't know\" if I'm not sure. This is general information, not migration advice.")
st.caption("⚠️ Please don't share personal identifiers (passport numbers, visa grant numbers, date of birth) in your questions.")


@st.cache_resource
def load_index():
    return build_index_cached()


if "index" not in st.session_state:
    with st.spinner("Loading knowledge base (the first start can take a few minutes)..."):
        st.session_state.index = load_index()

with st.sidebar:
    st.subheader("Your visa")
    persona = st.selectbox("Select your visa subclass", ["500 (Student)", "485 (Temporary Graduate)"])
    persona_key = persona.split()[0]

    st.divider()
    use_filter = st.toggle("Persona-filtered retrieval", value=True)
    st.divider()

    st.caption("🟢 Verified — high-confidence match")
    st.caption("🟡 Uncertain — weak match, double-check")
    st.caption("🔴 Not Found / Declined")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi! I'm your visa conditions assistant. Select your visa type on the left, then ask away.",
         "meta": None}
    ]

for msg in st.session_state.messages:
    avatar = "🛂" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])

if query := st.chat_input("Type your question..."):
    original_query = query
    query = redact_pii(query)

    st.session_state.messages.append({"role": "user", "content": query, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.write(query)
        if query != original_query:
            st.caption("🔒 Personal identifiers were removed from your message before processing.")

    with st.chat_message("assistant", avatar="🛂"):
        with st.spinner("Checking..."):
            unsafe_flag, verdict = is_unsafe_llm_judge(query)

        if unsafe_flag:
            answer = ("I can't help with questions about circumventing visa conditions or immigration law. "
                      "If you're facing a difficult visa situation, please contact your institution's "
                      "international student office or a registered migration agent.")
            meta = "🔴 Declined — flagged as unsafe"
            st.write(answer)
            st.caption(meta)
        else:
            with st.spinner("Checking sources..."):
                retrieved = retrieve(query, st.session_state.index, persona_key, use_filter=use_filter)
                answer, sources, score = generate(query, retrieved)

            note = hours_check(query) if persona_key == "500" else None
            if note:
                answer = ("Under a Student visa (subclass 500), you can work up to 48 hours per fortnight "
                          "while your course is in session. " + note)

            badge = "🟢 Verified" if score > 0.7 else "🟡 Uncertain" if score > 0.55 else "🔴 Not Found"
            meta = (f"{badge}  ·  Sources: {', '.join(sources) if sources else 'none'}  ·  "
                    f"Similarity: {score:.2f}")
            st.write(answer)
            st.caption(meta)

    st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})