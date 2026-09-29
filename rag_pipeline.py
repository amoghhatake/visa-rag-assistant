import os, glob, hashlib, pickle, requests, numpy as np
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = os.environ.get("GEN_MODEL", "llama3.2:3b")
CHUNK_SIZE = 500
SIM_THRESHOLD = 0.55
LLM_OPTIONS = {"temperature": 0, "seed": 42}     # repeatable results


def embed(text):
    r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                      json={"model": EMBED_MODEL, "prompt": text})
    return np.array(r.json()["embedding"])


def chunk_text(text, size=CHUNK_SIZE):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > size:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- Building the knowledge base from kb/500, kb/485, kb/general ----------

def read_file(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return open(path, encoding="utf-8", errors="ignore").read()


def build_index(kb_dir="kb"):
    index = []
    for persona_folder in sorted(glob.glob(f"{kb_dir}/*")):
        if not os.path.isdir(persona_folder):
            continue
        persona = os.path.basename(persona_folder).lower()      # "500", "485", "general"
        for path in sorted(glob.glob(f"{persona_folder}/*")):
            if not path.lower().endswith((".txt", ".pdf")):
                continue
            text = read_file(path)
            if len(text.split()) < 30:
                print(f"Skipped (no readable text): {path}")
                continue
            for chunk in chunk_text(text):
                index.append({"text": chunk,
                              "source": os.path.basename(path),
                              "persona": persona,
                              "embedding": embed(chunk)})
    return index


def _kb_signature(kb_dir="kb"):
    parts = [str(CHUNK_SIZE), EMBED_MODEL]
    for p in sorted(glob.glob(f"{kb_dir}/*/*")):
        if p.lower().endswith((".txt", ".pdf")):
            parts.append(f"{p}:{os.path.getsize(p)}:{int(os.path.getmtime(p))}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def build_index_cached(kb_dir="kb", cache_file="index_cache.pkl"):
    """Rebuilds the index only when a file in kb/ has changed."""
    sig = _kb_signature(kb_dir)
    if os.path.exists(cache_file):
        try:
            data = pickle.load(open(cache_file, "rb"))
            if data.get("sig") == sig:
                return data["index"]
        except Exception:
            pass
    index = build_index(kb_dir)
    pickle.dump({"sig": sig, "index": index}, open(cache_file, "wb"))
    return index


# ---------- Retrieval and generation ----------

def retrieve(query, index, persona, top_k=3, use_filter=True):
    q_emb = embed(query)
    if use_filter:
        candidates = [item for item in index if item["persona"] in (persona, "general")]
    else:
        candidates = index
    scored = [(cosine_sim(q_emb, item["embedding"]), item) for item in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def generate(query, retrieved):
    top_score = retrieved[0][0]
    if top_score < SIM_THRESHOLD:
        return "I don't know — this isn't covered in the knowledge base.", [], top_score

    context = "\n\n".join([f"[Source: {item['source']}]\n{item['text']}" for _, item in retrieved])
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say "I don't know."

Rules:
1. Read the context carefully. Do not reverse or negate what it says. Answer in one or two sentences.
2. Do not provide advice that helps someone circumvent visa conditions, immigration law, or engage in fraud.
3. If the user describes their OWN situation with specific facts (for example hours they worked, dates, or what they did) and asks whether they broke a rule or are allowed, do NOT give a yes/no verdict and do NOT do any calculation. Instead: state the relevant rule from the context, explain how it is measured (for example per fortnight, and whether the course is in session or on a break), and tell them to confirm their situation using VEVO or their international student office.
4. General questions about the rules (for example "how many hours can I work?") are NOT personal situations. Answer those normally.
5. Only answer what the user actually asked. If the question does not mention work, do not discuss work hours. If the context does not directly address the situation the user describes (for example taking time off during their course), say you do not have information on that specific situation and tell them to check with their international student office.

Context:
{context}

Question: {query}

Answer:"""

    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": GEN_MODEL, "prompt": prompt, "stream": False,
                            "options": LLM_OPTIONS})
    answer = r.json()["response"]
    sources = list(dict.fromkeys(item["source"] for _, item in retrieved))
    return answer, sources, top_score


# ---------- Safety check: LLM-as-judge ----------

REFUSAL_SIGNALS = ["CAN'T PROVIDE", "CANNOT PROVIDE", "CAN'T HELP", "CANNOT HELP",
                   "CAN'T ASSIST", "CANNOT ASSIST", "I'M NOT ABLE", "I AM NOT ABLE",
                   "ILLEGAL OR HARMFUL", "NOT APPROPRIATE"]


def is_unsafe_llm_judge(query):
    judge_prompt = f"""You are a safety classifier for a visa information assistant.

Classify the following user question into exactly one category:
- UNSAFE: the user is asking for help committing fraud, forging documents, evading visa conditions, lying to immigration officials, or otherwise circumventing immigration law.
- SAFE: the user is asking a legitimate question, even if it mentions a sensitive topic (e.g. asking what a term means, reporting someone else's wrongdoing, asking about consequences of an accident, asking how to stay compliant, or asking whether their own situation is allowed).

Important: asking ABOUT a concept (e.g. "what is X", "is X illegal", "what happens if X") is SAFE.
Only classify as UNSAFE if the user is seeking help TO DO the unsafe action themselves.

Respond with ONLY the single word SAFE or UNSAFE. Do not explain your reasoning. Do not add any other text.

Question: "{query}"

Classification:"""

    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": GEN_MODEL, "prompt": judge_prompt, "stream": False,
                            "options": LLM_OPTIONS})
    verdict = r.json()["response"].strip().upper()
    is_unsafe = ("UNSAFE" in verdict) or any(sig in verdict for sig in REFUSAL_SIGNALS)
    return is_unsafe, verdict


if __name__ == "__main__":
    print("Building index...")
    index = build_index_cached()
    print(f"Indexed {len(index)} chunks across: {sorted(set(item['persona'] for item in index))}")