import os
import time
import fitz
import faiss
import pytesseract
import streamlit as st
import numpy as np

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# ===================== ENV =====================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Context-Aware ChatPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SESSION =====================
defaults = {
    "logged_in": False,
    "role": None,
    "chunks": [],
    "index": None,
    "memory": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== USERS =====================
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"}
}

def authenticate(u, p):
    return USERS[u]["role"] if u in USERS and USERS[u]["password"] == p else None

# ===================== UI / CSS =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0F2027, #203A43, #2C5364);
    background-size: 400% 400%;
    animation: bg 12s ease infinite;
    color: white;
}

@keyframes bg {
    0% {background-position:0% 50%}
    50% {background-position:100% 50%}
    100% {background-position:0% 50%}
}

.glass {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 25px 50px rgba(0,0,0,0.45);
    color: white;
}

input, textarea {
    background: rgba(0,0,0,0.35) !important;
    color: white !important;
}

.stButton>button {
    background: linear-gradient(135deg,#FF4ECD,#7F7FD5);
    color: white;
    border-radius: 14px;
    padding: 10px 26px;
}

.answer-box {
    background: linear-gradient(135deg,#00E676,#00C853);
    color: black;
    padding: 20px;
    border-radius: 14px;
}

.typing::after {
    content: "▌";
    animation: blink 1s infinite;
}

@keyframes blink {
    50% {opacity:0}
}

.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: rgba(0,0,0,0.6);
    color: #00F5FF;
    padding: 8px;
    font-weight: 600;
    overflow: hidden;
}

.footer span {
    display: inline-block;
    animation: move 18s linear infinite;
    white-space: nowrap;
}

@keyframes move {
    0% {transform: translateX(100%)}
    100% {transform: translateX(-100%)}
}
</style>
""", unsafe_allow_html=True)

# ===================== LOGIN =====================
if not st.session_state.logged_in:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    if os.path.exists("assets/university_logo.png"):
        st.image("assets/university_logo.png", width=120)

    st.markdown("## Context-Aware ChatPDF")
    st.markdown("### LLM-Based Document Intelligence System")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        role = authenticate(u, p)
        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ===================== MODELS =====================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = genai.GenerativeModel("gemini-1.5-flash")

# ===================== OCR =====================
def ocr_pdf(path):
    images = convert_from_path(path)
    return " ".join(pytesseract.image_to_string(img) for img in images)

# ===================== SECTION-AWARE CHUNKING =====================
SECTION_KEYWORDS = {
    "education": ["education", "qualification", "degree"],
    "skills": ["skills", "technologies", "tools"],
    "experience": ["experience", "internship", "work"],
    "projects": ["project", "projects"],
    "summary": ["summary", "profile"]
}

def detect_section(line):
    for sec, keys in SECTION_KEYWORDS.items():
        if any(k in line.lower() for k in keys):
            return sec
    return "general"

def extract_chunks(path, name):
    doc = fitz.open(path)
    chunks = []
    current = "general"

    for page_no, page in enumerate(doc, start=1):
        text = page.get_text().strip() or ocr_pdf(path)
        for line in text.split("\n"):
            sec = detect_section(line)
            if sec != "general":
                current = sec
            if line.strip():
                chunks.append({
                    "text": line.strip(),
                    "section": current,
                    "pdf": name,
                    "page": page_no
                })
    return chunks

def build_index(chunks):
    emb = embedder.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return index

# ===================== QUESTION NORMALIZER =====================
def normalize_question(q):
    q = q.lower().strip()
    mapping = {
        "name": "What is the name mentioned in the document?",
        "qualification": "What are the educational qualifications?",
        "qualifications": "What are the educational qualifications?",
        "education": "What is the educational background?",
        "skills": "What skills are mentioned?",
        "projects": "What projects are mentioned?",
        "experience": "What experience is mentioned?"
    }
    if q in mapping:
        return mapping[q]
    if len(q.split()) <= 2:
        return f"What information does the document provide about {q}?"
    return q

# ===================== SECTION-AWARE FALLBACK =====================
def extractive_fallback(question, chunks):
    q = question.lower()

    section_map = {
        "project": "projects",
        "projects": "projects",
        "skill": "skills",
        "skills": "skills",
        "experience": "experience",
        "education": "education",
        "qualification": "education",
        "qualifications": "education"
    }

    for key, sec in section_map.items():
        if key in q:
            lines = [
                c["text"] for c in chunks
                if c["section"] == sec and len(c["text"]) > 3
            ]
            if lines:
                return f"{sec.capitalize()} found in document:\n" + "\n".join(lines)

    # General fallback
    q_words = set(q.split())
    scored = []
    for c in chunks:
        score = sum(1 for w in q_words if w in c["text"].lower())
        if score > 0:
            scored.append((score, c["text"]))

    if scored:
        scored.sort(reverse=True)
        return "Relevant information found in document:\n" + \
               "\n".join(t for _, t in scored[:7])

    return "The document does not contain sufficient information to answer this question."

# ===================== ANSWER =====================
def generate_answer(q, context_text, context_chunks):
    prompt = f"""
Answer using ONLY the document context.
Summarize if required.
If multiple items exist, list them clearly.
If missing, say so honestly.

Context:
{context_text}

Question:
{q}
"""
    try:
        ans = llm.generate_content(prompt).text.strip()
        if ans and "not found" not in ans.lower():
            return ans
        return extractive_fallback(q, context_chunks)
    except:
        return extractive_fallback(q, context_chunks)

# ===================== TYPEWRITER =====================
def typewriter(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"<div class='typing'>{out}</div>", unsafe_allow_html=True)
        time.sleep(0.015)
    box.markdown(out)

# ===================== ADMIN UPLOAD =====================
if st.session_state.role == "admin":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("## Upload PDFs")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files:
        os.makedirs("uploaded_pdfs", exist_ok=True)
        all_chunks = []
        for f in files:
            path = f"uploaded_pdfs/{f.name}"
            with open(path, "wb") as fp:
                fp.write(f.read())
            all_chunks.extend(extract_chunks(path, f.name))

        st.session_state.chunks = all_chunks
        st.session_state.index = build_index(all_chunks)
        st.success("PDFs indexed successfully")

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== CHAT =====================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("## Ask Your Documents")

raw_q = st.text_input("Ask a question")

if raw_q and st.button("Ask"):
    q = normalize_question(raw_q)

    with st.spinner("Thinking..."):
        q_emb = embedder.encode([q])
        _, idx = st.session_state.index.search(q_emb, 10)

        retrieved = [st.session_state.chunks[i] for i in idx[0]]
        context_text = "\n".join(c["text"] for c in retrieved)

        answer = generate_answer(q, context_text, retrieved)

    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.markdown("### Answer")
    typewriter(answer)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<div class="footer">
<span>
👥 Project Team: Shubhadeep Patra | Soumyashree Priyadarshi Kar | Pratik Kumar Kar | Pratyush Kumar Pani | Siksha O Anusandhan University (ITER)
</span>
</div>
""", unsafe_allow_html=True)
