import streamlit as st
import os
import re
import threading
import queue

# ===============================
# API KEY
# ===============================
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("❌ Please add GOOGLE_API_KEY in secrets.toml")
    st.stop()

# ===============================
# IMPORTS
# ===============================
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

@st.cache_data(show_spinner=False)
def cached_llm_response(query, chain_type):

    if chain_type == "subject":
        chain = get_subject_chain()
    else:
        chain = get_project_chain()

    result = chain.invoke({"input": query})

    return result["answer"]
# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Statistical Data Mining-2 Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===============================
# SESSION STATE
# ===============================
defaults = {
    "messages": [],
    "context_state": None,
    "zoom_state": None,
    "chat_history": [],
    "response_cache": {},
    "cancel_flag": threading.Event(),
    "is_generating": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
.main .block-container {
    background: #030712 !important;
    color: white;
}
.stApp {
    background:
        radial-gradient(circle at top left, rgba(120,119,198,.15), transparent 25%),
        radial-gradient(circle at bottom right, rgba(91,134,229,.12), transparent 25%),
        #030712 !important;
    overflow-x: hidden;
}
.stApp::before {
    content:""; position:fixed;
    width:700px; height:700px;
    background:radial-gradient(circle,rgba(139,92,246,.18),transparent 70%);
    top:-250px; left:-200px; z-index:-1; filter:blur(80px);
}
#MainMenu,footer,header { visibility:hidden; }
section[data-testid="stSidebar"] { width:0 !important; min-width:0 !important; }
[data-testid="collapsedControl"] { display:none; }
.block-container { max-width:860px; padding-top:1rem; padding-bottom:9rem; }

/* ---- HERO ---- */
.hero-wrap { text-align:center; margin-top:-60px; margin-bottom:28px; }
.hero-icon { font-size:62px; display:block; margin-bottom:6px;
             filter:drop-shadow(0 0 18px rgba(139,92,246,.4)); }
.hero-title {
    font-size:52px; font-weight:800; letter-spacing:-2px; line-height:1.1;
    background:linear-gradient(90deg,#d8b4fe,#818cf8,#60a5fa);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:12px;
}
.hero-sub  { font-size:17px; color:#a78bfa; font-weight:500; margin-bottom:5px; }
.hero-desc { font-size:14px; color:#9ca3af; }

/* ---- CAPS ROW ---- */
.caps-row {
    display:flex; flex-direction:row; flex-wrap:nowrap;
    gap:10px; justify-content:center; align-items:stretch;
    margin:22px 0 28px; overflow-x:auto;
}
.cap-card {
    display:flex; align-items:center; gap:10px;
    background:rgba(255,255,255,.04);
    border:1px solid rgba(139,92,246,.22);
    border-radius:16px; padding:12px 16px;
    white-space:nowrap; flex-shrink:0;
    transition:all .22s ease; cursor:default;
}
.cap-card:hover {
    background:rgba(139,92,246,.10);
    border-color:rgba(139,92,246,.45);
    transform:translateY(-2px);
}
.cap-icon {
    width:36px; height:36px; border-radius:10px;
    display:flex; align-items:center; justify-content:center;
    font-size:18px; flex-shrink:0;
}
.cap-icon.purple { background:rgba(139,92,246,.22); }
.cap-icon.blue   { background:rgba(59,130,246,.22); }
.cap-icon.teal   { background:rgba(20,184,166,.22); }
.cap-icon.indigo { background:rgba(99,102,241,.22); }
.cap-label { font-size:13px; font-weight:600; color:#e2e8f0; line-height:1.2; }
.cap-sub   { font-size:11px; color:#9ca3af; margin-top:2px; }

/* ---- TOP-RIGHT BUTTONS ---- */
div[data-testid="stHorizontalBlock"]:first-of-type {
    position:fixed !important;
    top:14px !important; right:20px !important;
    z-index:9999 !important;
    width:auto !important;
    background:transparent !important;
    gap:8px !important;
    align-items:center !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type
[data-testid="stVerticalBlockBorderWrapper"] {
    background:transparent !important; border:none !important; padding:0 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    border-radius:20px !important;
    height:36px !important;
    font-size:12px !important;
    font-weight:600 !important;
    backdrop-filter:blur(14px) !important;
    padding:0 14px !important;
    white-space:nowrap !important;
    transition:all .22s ease !important;
    width:auto !important; min-width:0 !important; line-height:1 !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type
.stButton:first-child > button {
    background:rgba(10,14,35,.85) !important;
    color:#fca5a5 !important;
    border:1px solid rgba(239,68,68,.45) !important;
    box-shadow:0 0 12px rgba(239,68,68,.12) !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type
.stButton:first-child > button:hover {
    background:rgba(50,10,10,.92) !important;
    border-color:rgba(239,68,68,.75) !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type
.stButton:last-child > button {
    background:rgba(10,14,35,.85) !important;
    color:#c4b5fd !important;
    border:1px solid rgba(139,92,246,.45) !important;
    box-shadow:0 0 12px rgba(139,92,246,.12) !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type
.stButton:last-child > button:hover:not(:disabled) {
    background:rgba(20,10,45,.92) !important;
    border-color:rgba(139,92,246,.78) !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type
.stButton:last-child > button:disabled {
    opacity:0.35 !important; cursor:not-allowed !important;
}

/* ---- CHAT BUBBLES ---- */
[data-testid="stChatMessageAvatar"] { width:28px !important; height:28px !important; opacity:.75; }
[data-testid="stChatMessage"] { background:transparent !important; border:none !important; margin-bottom:16px; }
[data-testid="stChatMessage"] > div { background:transparent !important; }

/* USER bubble — injected via st.markdown with custom HTML */
.user-bubble-row {
    display: flex;
    justify-content: flex-end;
    align-items: flex-end;
    gap: 8px;
    margin-bottom: 16px;
    padding-right: 4px;
}
.user-bubble {
    background: linear-gradient(135deg, rgba(139,92,246,.95), rgba(99,102,241,.90));
    color: #fff;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 24px rgba(139,92,246,.35);
    padding: 10px 14px;
    max-width: 58%;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
}
.user-avatar {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: rgba(139,92,246,.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}

/* ASSISTANT bubble */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    padding: 10px 14px !important;
    border-radius: 18px !important;
    max-width: 58% !important;
    min-width: 60px !important;
    display: inline-block !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    background: rgba(30,35,60,.85) !important;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
    border: 1px solid rgba(139,92,246,.18) !important;
    border-bottom-left-radius: 4px !important;
    box-shadow: 0 4px 18px rgba(59,130,246,.10) !important;
    margin-left: 0 !important;
    margin-right: auto !important;
}

/* ---- CHAT INPUT ---- */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    background:transparent !important; box-shadow:none !important; border:none !important;
}
section[data-testid="stChatInput"] {
    position:fixed; bottom:22px;
    left:50%; transform:translateX(-50%);
    width:62%;
    background:rgba(17,24,39,.90) !important;
    border:1px solid rgba(139,92,246,.35) !important;
    border-radius:28px !important;
    backdrop-filter:blur(20px);
    padding:8px 16px;
    box-shadow:0 0 50px rgba(139,92,246,.15), 0 0 90px rgba(59,130,246,.07);
    transition:all .3s ease;
}
section[data-testid="stChatInput"]:focus-within {
    border-color:rgba(168,85,247,.6) !important;
    box-shadow:0 0 70px rgba(139,92,246,.25);
}
[data-testid="stChatInputTextArea"] textarea {
    color:white !important; caret-color:white !important;
    font-size:15px !important; background:transparent !important;
    -webkit-text-fill-color:white !important;
}
[data-testid="stChatInputTextArea"] textarea::placeholder { color:#6b7280 !important; opacity:1 !important; }
[data-testid="stChatInputTextArea"] { background:transparent !important; }

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#374151; border-radius:20px; }
[data-testid="stSpinner"] p { color:#a78bfa !important; }

/* Hide default stChatMessage for user — we render custom HTML instead */
.hide-user-default [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# TOP-RIGHT BUTTONS
# ===============================
col_clear, col_cancel = st.columns([1, 1])

with col_clear:
    if st.button("🗑 Clear", key="btn_clear"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.context_state = None
        st.session_state.zoom_state = None
        st.session_state.is_generating = False
        st.session_state.cancel_flag.set()
        st.rerun()

with col_cancel:
    # FIX: Stop button always enabled — we rely on cancel_flag to guard misclicks
    if st.button("⏹ Stop", key="btn_cancel"):
        st.session_state.cancel_flag.set()
        st.session_state.is_generating = False
        st.rerun()

# ===============================
# HERO
# ===============================
st.markdown("""
<div class="hero-wrap">
  <span class="hero-icon">📚</span>
  <div class="hero-title">Statistical Data<br>Mining-2 Assistant</div>
  <div class="hero-sub">✨ RHB • Racheal Hageman's Bot</div>
  <div class="hero-desc">Your intelligent companion for all your course-related queries and academic support</div>
</div>
""", unsafe_allow_html=True)

# ===============================
# CAPABILITY CARDS
# ===============================
st.markdown("""
<div class="caps-row">
  <div class="cap-card"><div class="cap-icon purple">🎓</div><div>
    <div class="cap-label">Subject doubts</div><div class="cap-sub">Get clarity on concepts</div></div></div>
  <div class="cap-card"><div class="cap-icon blue">📋</div><div>
    <div class="cap-label">Project evaluation</div><div class="cap-sub">Review &amp; feedback</div></div></div>
  <div class="cap-card"><div class="cap-icon teal">📅</div><div>
    <div class="cap-label">Assignment deadlines</div><div class="cap-sub">Stay on track</div></div></div>
  <div class="cap-card"><div class="cap-icon indigo">🔗</div><div>
    <div class="cap-label">Zoom links etc.</div><div class="cap-sub">Access important links</div></div></div>
</div>
""", unsafe_allow_html=True)

# ===============================
# HELPERS
# ===============================
def contact_info():
    return """For better clarification, you may consider contacting:

👨‍🏫 Professor: Dr. Raacheal Hageman Blair
📧 hageman@buffalo.edu
Office Hours: Monday & Wednesday (1:00 PM - 2:30 PM)

👨‍💻 TA: Nithish Kumar Reddy Yerreddy
📧 nyerredd@buffalo.edu
Office Hours: Monday & Thursday (3:00 PM - 4:00 PM)

🧑‍🏫 Grader: Nuthan Teja Reddy
📧 nuthan@buffalo.edu
🕒 Office Hours: Tuesday & Thursday (12:00 PM – 1:00 PM)
"""

def has_word(q, words):
    pattern = r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b'
    return re.search(pattern, q.lower()) is not None

def detect_normal_intent(q):
    q = q.lower().strip()
    if len(q.split()) <= 3 and has_word(q, ["hi","hello","hey"]): return "greeting"
    if "you too" in q or "you as well" in q:   return "you_too"
    if has_word(q, ["thanks","thank you","thank"]): return "thanks"
    if "how are you" in q:                     return "how_are_you"
    if has_word(q, ["bye","goodbye"]):         return "bye"
    if "who are you" in q:                     return "who_are_you"
    if "your name" in q:                       return "your_name"
    if "what is rhb" in q:                     return "what_is_rhb"
    if has_word(q, ["vacancy","position","openings"]): return "vacancy"
    return "other"

# ===============================
# HELPER: render user bubble as custom HTML (right-aligned)
# ===============================
def render_user_bubble(text):
    safe = text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown(f"""
    <div class="user-bubble-row">
        <div class="user-bubble">{safe}</div>
        <div class="user-avatar">👤</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# RAG
# ===============================
def build_retriever(pdf_list):
    docs_all = []
    for f in pdf_list:
        if os.path.exists(f):
            try: docs_all.extend(PyPDFLoader(f).load())
            except Exception as e: st.error(f"Error loading {f}: {e}")
    if not docs_all: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    docs = splitter.split_documents(docs_all)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever(search_kwargs={"k": 3})

@st.cache_resource
def load_subject_data():
    return build_retriever([
        "14. CART.pdf","biplots_and_outliers.pdf","Overview.pdf",
        "Association_presA.pdf","PGM_I .pdf","PGM_structure.pdf",
        "BN_Prob_Reasoning.pdf","Association_presB.pdf","Association_presD.pdf",
        "Clustering_1-2.pdf","Clustering_2-2.pdf","Clustering_3.pdf",
        "Clustering_4.pdf","Clustering_4b.pdf","SOM.pdf","JC_Clauset.pdf",
        "PCA.pdf","PGM_II_2025.pdf","UDG_A.pdf",
    ])

@st.cache_resource
def load_project_data():
    return build_retriever(["SDM-2 Project proposal guidlines.pdf"])

def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0, max_retries=3,
    )

prompt_template = ChatPromptTemplate.from_template("""
You are the Statistical Data Mining-2 Teaching Assistant.

Rules:
- Answer only from the provided context.
- Explain in simple and student-friendly language.
- By default, give short, clear, and easy-to-understand answers.
- Give detailed, step-by-step, and longer explanations only if the user explicitly asks for:
  "detailed", "long", "in depth", "clearly explain", or "step-by-step".
- Use bullet points when helpful.
- Do not hallucinate or invent information.
- If the answer is not in the context, say:
"I could not find this in the course materials."

Context:
{context}

Question:
{input}

Answer:
""")

@st.cache_resource
def get_subject_chain():
    r = load_subject_data()
    if r is None: st.error("No docs."); st.stop()
    return create_retrieval_chain(r, create_stuff_documents_chain(load_model(), prompt_template))

@st.cache_resource
def get_project_chain():
    r = load_project_data()
    if r is None: st.error("No docs."); st.stop()
    return create_retrieval_chain(r, create_stuff_documents_chain(load_model(), prompt_template))

# ===============================
# THREADED STREAMING
# ===============================
def _stream_worker(chain, query, token_queue, cancel_flag):
    try:
        for chunk in chain.stream({"input": query}):
            if cancel_flag.is_set():
                break
            token = chunk.get("answer", "")
            if token:
                token_queue.put(token)
    except Exception as e:
        msg = "⚠️ Too many requests right now." if "quota" in str(e).lower() else f"⚠️ Error: {e}"
        token_queue.put(msg)
    finally:
        token_queue.put(None)

def run_with_streaming(chain, query):
    cancel_flag = st.session_state.cancel_flag
    cancel_flag.clear()
    st.session_state.is_generating = True

    token_queue = queue.Queue()
    thread = threading.Thread(
        target=_stream_worker,
        args=(chain, query, token_queue, cancel_flag),
        daemon=True,
    )
    thread.start()

    full = ""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        while True:
            try:
                token = token_queue.get(timeout=0.1)
            except queue.Empty:
                if cancel_flag.is_set():
                    placeholder.markdown("⏹️ Generation stopped.")
                    st.session_state.is_generating = False
                    return "⏹️ Generation stopped."
                continue

            if token is None:
                break
            if cancel_flag.is_set():
                placeholder.markdown("⏹️ Generation stopped.")
                st.session_state.is_generating = False
                thread.join(timeout=1)
                return "⏹️ Generation stopped."

            full += token
            placeholder.markdown(full + "▌")

        placeholder.markdown(full)

    thread.join(timeout=1)
    st.session_state.is_generating = False
    st.session_state["_streamed_this_turn"] = True
    return full

# ===============================
# CHAT HISTORY DISPLAY
# ===============================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        render_user_bubble(msg["content"])   # custom right-aligned HTML
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# ===============================
# CHAT INPUT
# ===============================
if "quick_prompt" in st.session_state:
    query = st.session_state.pop("quick_prompt")
else:
    query = st.chat_input("Type your message...")

# ===============================
# HANDLER
# ===============================
if query:
    response = ""

    # Save + display user message as right-aligned bubble
    st.session_state.messages.append({"role": "user", "content": query})
    render_user_bubble(query)

    q = query.lower()

    assign_kw = ["assignment","homework","hw","question","problem","q1","q2",
                 "calculate","find","compute","implement"]
    if st.session_state.context_state == "assignment_doubt" and \
       not any(w in q for w in assign_kw):
        st.session_state.context_state = None

    # ---- ZOOM STATE ----
    if st.session_state.zoom_state == "ask_person":
        if "professor" in q:
            response = "Professor Zoom: https://buffalo.zoom.us/j/7342873196"
            st.session_state.zoom_state = None
        elif re.search(r"\bta\b", q):
            response = "TA Zoom: https://buffalo.zoom.us/j/93740724275"
            st.session_state.zoom_state = None
        elif "grader" in q:
            response = "Grader Zoom: https://buffalo.zoom.us/j/4027519593"
            st.session_state.zoom_state = None
        else:
            response = "Please choose: Professor / TA / Grader"

    elif has_word(q, ["professor"]):
        response = "👨‍🏫 **Professor: Dr. Raacheal Hageman Blair**\n📧 hageman@buffalo.edu\nOffice Hours: Monday & Wednesday (1:00 PM - 2:30 PM)"
    elif re.search(r"\bta\b", q) or "nithish" in q:
        response = "👨‍💻 **TA: Nithish Kumar Reddy Yerreddy**\n📧 nyerredd@buffalo.edu\nOffice Hours: Monday & Thursday (3:00 PM - 4:00 PM)"
    elif "grader" in q or "nuthan" in q:
        response = "🧑‍🏫 **Grader: Nuthan Teja Reddy**\n📧 nuthan@buffalo.edu\nOffice Hours: Tuesday & Thursday (12:00 PM – 1:00 PM)"

    # ---- DOUBT FLOW ----
    elif "doubt" in q and st.session_state.context_state is None:
        st.session_state.context_state = "doubt_type"
        response = "Is your doubt related to subject, assignment, or project?"

    elif st.session_state.context_state == "doubt_type":
        if any(x in q for x in ["nothing","cancel","never mind","clarified","solved","no doubt","something else"]):
            st.session_state.context_state = None
            response = "No problem 😊 What would you like help with now?"
        elif len(q.split()) <= 3 and has_word(q, ["hi","hello","hey"]):
            st.session_state.context_state = None
            response = "Hello 😊 How can I help you today?"
        elif "subject" in q:
            st.session_state.context_state = "subject_doubt"
            response = "Please explain your subject doubt."
        elif "assignment" in q or "homework" in q:
            st.session_state.context_state = "assignment_doubt"
            response = "What exactly is your assignment doubt?"
        elif "project" in q:
            st.session_state.context_state = "project_doubt"
            response = "Please explain your project doubt."
        else:
            response = "Please choose: subject / assignment / project or type cancel."

    # ---- PROJECT DOUBT ----
    elif st.session_state.context_state == "project_doubt":
        ck = q.strip().lower()
        if ck in st.session_state.response_cache:
            response = st.session_state.response_cache[ck]
            with st.chat_message("assistant"): st.markdown(response)
            st.session_state["_streamed_this_turn"] = True
        else:
            response = cached_llm_response(query, "project")
            if response and len(response.strip()) >= 5:
                st.session_state.response_cache[ck] = response
            elif not response:
                response = "I could not find this in the course materials."

    # ---- SUBJECT DOUBT ----
    elif st.session_state.context_state == "subject_doubt":
        response = cached_llm_response(query, "subject")
        if not response or len(response.strip()) < 5:
            response = "I could not find this in the course materials."
        st.session_state.context_state = None

    # ---- NORMAL INTENTS ----
    else:
        intent = detect_normal_intent(q)
        deadlines = {"hw1":"2026-02-12","hw2":"2026-03-03","hw3":"2026-03-31","hw4":"2026-04-20"}
        mq = q.replace("home work","hw").replace("homework","hw").replace(" ","")

        if any(x in q for x in ["deadline","due","date"]):
            found = False
            for k in deadlines:
                if k in mq: response = f"📅 {k.upper()} is due on {deadlines[k]}"; found=True; break
            if not found:
                response = "📅 Upcoming deadlines:\n" + "\n".join([f"{k.upper()}: {v}" for k,v in deadlines.items()])
        elif intent == "greeting":    response = "Hello! 😊 I'm here to help you. What would you like to explore today?"
        elif intent == "you_too":     response = "Thanks 😊"
        elif intent == "thanks":      response = "You're very welcome 😊 Happy to help!"
        elif intent == "how_are_you": response = "I'm doing great! 😊 How can I help you today?"
        elif intent == "bye":         response = "Goodbye! 👋 Have a wonderful day 😊"
        elif intent == "who_are_you": response = "I am the Statistical Data Mining-2 Assistant 😊"
        elif intent == "your_name":   response = "My name is RHB."
        elif intent == "what_is_rhb": response = "RHB stands for Racheal Hageman's Bot."
        elif any(d in q for d in ["monday","tuesday","wednesday","thursday"]):
            response = contact_info()
        elif intent == "vacancy":
            response = "Currently, there are no open positions."
        elif any(x in q for x in ["research","thesis"]) and "project" not in q:
            response = "Professor is not currently working on research projects.\nContact: hageman@buffalo.edu"
        elif "zoom" in q:
            if "professor" in q:         response = "Professor Zoom: https://buffalo.zoom.us/j/7342873196"
            elif re.search(r"\bta\b",q): response = "TA Zoom: https://buffalo.zoom.us/j/93740724275"
            elif "grader" in q:          response = "Grader Zoom: https://buffalo.zoom.us/j/4027519593"
            else:
                st.session_state.zoom_state = "ask_person"
                response = "Whose Zoom link do you need? (Professor / TA / Grader)"
        else:
            pw = ["project","proposal","submission","rate my proposal",
                  "score my proposal","evaluate proposal","dataset idea","project guidelines"]
            mq2 = query + (" Explain in very simple words." if "simple" in q else "")
            chosen_chain = get_project_chain() if any(w in q for w in pw) else get_subject_chain()
            chain_type = "project" if any(w in q for w in pw) else "subject"

            response = cached_llm_response(mq2, chain_type)
            if not response or len(response.strip()) < 5:
                response = "I could not find this in the course materials."

    # ---- RENDER non-streaming responses + SAVE ----
    if not st.session_state.pop("_streamed_this_turn", False) and response:
        with st.chat_message("assistant"):
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history += [("user", query), ("assistant", response)]
    st.rerun()