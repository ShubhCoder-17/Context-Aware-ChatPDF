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

# ===================== LOAD ENV =====================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Context-Aware ChatPDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE =====================
defaults = {
    "logged_in": False,
    "role": None,
    "theme": "dark",
    "memory": [],
    "chunks": [],
    "index": None,
    "pdf_count": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== USERS =====================
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}

def authenticate(u, p):
    if u in USERS and USERS[u]["password"] == p:
        return USERS[u]["role"]
    return None

# ===================== CSS + 3D UI =====================
st.markdown("""
<style>
/* ===== TYPING CURSOR ===== */
.typing-cursor::after {
    content: "|";
    margin-left: 4px;
    animation: blink 1s infinite;
    font-weight: 600;
}

@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(-45deg, #0F2027, #203A43, #2C5364, #1A2980);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: #FFFFFF;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ===== GLASS / BUBBLE FIX ===== */
.glass {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 20px 45px rgba(0,0,0,0.45);
    color: #FFFFFF;               /* 🔑 FIX */
}

/* ===== FIX INPUT TEXT VISIBILITY ===== */
.stTextInput input,
.stTextArea textarea {
    color: #FFFFFF !important;
    background-color: rgba(0,0,0,0.35) !important;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: rgba(255,255,255,0.6);
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #FF4ECD, #7F7FD5);
    color: #FFFFFF;
    border-radius: 14px;
    padding: 12px 28px;
    font-size: 16px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.4);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* ===== TITLES ===== */
.login-title {
    font-size: 42px;
    font-weight: 700;
    color: #00F5FF;
    text-shadow: 0 0 20px rgba(0,245,255,0.6);
}

.login-subtitle {
    color: #E0E0E0;
    margin-bottom: 30px;
}

/* ===== ANSWER BOX ===== */
.answer-box {
    background: linear-gradient(135deg, #00E676, #00C853);
    color: #000000;
    padding: 20px;
    border-radius: 14px;
}

/* ===== SOURCES ===== */
.source {
    color: #FFD54F;
}

</style>
""", unsafe_allow_html=True)

# ===================== LOGIN PAGE =====================
if not st.session_state.logged_in:
    st.markdown("<div class='login-wrapper'>", unsafe_allow_html=True)
    st.image("assets/university_logo.png", width=120)
    st.markdown("<h1 class='login-title'>Context-Aware ChatPDF</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='login-subtitle'>LLM-Based Document Intelligence System</h3>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
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

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()

# ===================== MODELS =====================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
llm = genai.GenerativeModel("gemini-1.5-flash")

# ===================== OCR =====================
def ocr_pdf(path):
    images = convert_from_path(path)
    return " ".join(pytesseract.image_to_string(img) for img in images)

# ===================== PDF PROCESSING =====================
def extract_chunks(path, name):
    doc = fitz.open(path)
    chunks = []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text().strip() or ocr_pdf(path)
        words = text.split()
        for i in range(0, len(words), 400):
            chunks.append({
                "text": " ".join(words[i:i+400]),
                "pdf": name,
                "page": page_no
            })
    return chunks

def build_index(chunks):
    emb = embedder.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(np.array(emb))
    return index

# ===================== GEMINI ANSWER =====================
def generate_answer(q, context):
    prompt = f"""
Answer strictly using the context.
If not found, say "Not found in document".

Context:
{context}

Question:
{q}
"""
    try:
        return llm.generate_content(prompt).text
    except:
        return "Answer not found in document."

# ===================== TYPEWRITER =====================
def typewriter(text):
    placeholder = st.empty()
    rendered_text = ""

    for char in text:
        rendered_text += char
        placeholder.markdown(
            f"<div class='typing-cursor'>{rendered_text}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.015)

    # Remove cursor after typing completes
    placeholder.markdown(rendered_text)


# ===================== ADMIN UPLOAD =====================
if st.session_state.role == "admin":
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("## 📤 Upload PDFs")

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
        st.session_state.pdf_count = len(files)
        st.success("PDFs indexed successfully")

    st.markdown("</div>", unsafe_allow_html=True)

# ===================== CHAT =====================
st.markdown("<div class='glass'>", unsafe_allow_html=True)
st.markdown("## 💬 Ask Your Documents")

q = st.text_input("Ask a question")

if st.button("Ask") and q:
    with st.spinner("Analyzing documents..."):
        q_emb = embedder.encode([q])
        _, idx = st.session_state.index.search(q_emb, 5)

        context, sources = "", set()
        for i in idx[0]:
            c = st.session_state.chunks[i]
            context += c["text"] + "\n"
            sources.add(f"{c['pdf']} - Page {c['page']}")

        answer = generate_answer(q, context)
        st.session_state.memory.append((q, answer))
        st.session_state.memory = st.session_state.memory[-5:]

    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.markdown("### ✅ Answer")
    typewriter(answer)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 📚 Sources")
    for s in sources:
        st.markdown(f"<div class='source'>• {s}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("""
<style>
.footer-marquee {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(8px);
    color: #00F5FF;
    padding: 8px 0;
    font-weight: 600;
    z-index: 999;
    overflow: hidden;
}

.footer-marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: scrollText 18s linear infinite;
    white-space: nowrap;
}

@keyframes scrollText {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
</style>

<div class="footer-marquee">
    <span>
        👥 Project Team: Shubhadeep Patra | Soumyashree Priyadarshi Kar | Pratik Kumar Kar | Pratyush Kumar Pani | Siksha O Anusandhan University (ITER)
    </span>
</div>
""", unsafe_allow_html=True)
