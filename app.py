import streamlit as st
import json
import base64
import re
from pathlib import Path
from openai import OpenAI

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Academic Image Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --border: #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main { background: var(--bg) !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }

.app-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1040 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.app-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00e5ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.4rem;
    font-family: 'Space Mono', monospace;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem !important; }

.key-locked {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #6ee7b7;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stFileUploader > div {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00c4d4, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100%;
}

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--accent);
    margin-bottom: 1rem;
}

.model-badge {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(0,229,255,0.1));
    border: 1px solid rgba(0,229,255,0.25);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    margin-top: 0.5rem;
    text-align: center;
}

/* Raw JSON output box */
.json-output {
    background: #060d1a;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #a5f3fc;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

hr { border-color: var(--border) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
label, .stSelectbox label, .stTextInput label, .stTextArea label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# VLM MODELS
# ──────────────────────────────────────────────
VLM_MODELS = {
    "⭐ Qwen3 VL 235B A22B Instruct  (Best Quality)":   "qwen/qwen3-vl-235b-a22b-instruct",
    "🧠 Qwen3 VL 235B A22B Thinking  (Deep Reasoning)": "qwen/qwen3-vl-235b-a22b-thinking",
    "🔬 Qwen3 VL 32B Instruct         (High Accuracy)":  "qwen/qwen3-vl-32b-instruct",
    "⚡ Qwen3 VL 30B A3B Instruct     (Fast + Smart)":   "qwen/qwen3-vl-30b-a3b-instruct",
    "💡 Qwen3 VL 30B A3B Thinking     (MoE Reasoning)":  "qwen/qwen3-vl-30b-a3b-thinking",
    "🚀 Qwen3 VL 8B Instruct          (Fast & Cheap)":   "qwen/qwen3-vl-8b-instruct",
    "🔍 Qwen3 VL 8B Thinking          (8B Reasoning)":   "qwen/qwen3-vl-8b-thinking",
    "🏆 Qwen2.5 VL 72B Instruct       (Proven Stable)":  "qwen/qwen2.5-vl-72b-instruct",
    "📦 Qwen2.5 VL 7B Instruct        (Lightweight)":    "qwen/qwen2.5-vl-7b-instruct",
    "🌐 Qwen2 VL 72B Instruct         (Previous Gen)":   "qwen/qwen2-vl-72b-instruct",
}

# ──────────────────────────────────────────────
# PROMPT TEMPLATES
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a world-class academic content analyzer trained on NEET, JEE Main, JEE Advanced,
IPhO, IChO, IBO, IMO, USABO, Science Olympiad, and all major international academic competitions.
You analyze textbook diagrams, equations, and images with precision.
Always respond in valid JSON format only — no markdown, no preamble.
Be structured, precise, and exam-focused for the relevant competition level.
"""

PROMPTS = {
    "📋 Comprehensive Summary": {
        "key": "summary",
        "desc": "Full exam-ready summary with key points, formulas, mnemonics & sample questions",
        "template": """
Generate a complete, exam-ready summary of this academic image suitable for last-minute revision.
Return valid JSON only:
{
  "title": "",
  "subject": "",
  "topic": "",
  "exam_relevance": [],
  "difficulty_level": "Beginner | Intermediate | Advanced | Olympiad",
  "one_line_summary": "",
  "detailed_summary": "",
  "key_points": [],
  "must_remember_facts": [],
  "formulas_or_reactions": [],
  "labeled_diagram_notes": [],
  "mnemonics_or_memory_tips": [],
  "exam_traps_to_avoid": [],
  "likely_question_types": [],
  "sample_questions": [{"question": "", "answer": "", "type": ""}],
  "cross_topic_connections": [],
  "revision_priority": "High | Medium | Low"
}"""
    },
    "⚡ Quick Facts Card": {
        "key": "quick_facts",
        "desc": "Flashcard-style — perfect for last-hour revision",
        "template": """
Create a quick-reference fact card for this academic image — suitable for flashcard or last-hour revision.
Return valid JSON only:
{
  "topic": "",
  "subject": "",
  "exam_tags": [],
  "5_key_facts": [],
  "formula_snapshot": [],
  "diagram_in_words": "",
  "remember_this": "",
  "dont_confuse_with": ""
}"""
    },
    "🔭 Physics Analysis": {
        "key": "physics",
        "desc": "NEET / JEE / IPhO / APhO level deep physics breakdown",
        "template": """
Analyze this Physics diagram for NEET / JEE / International Physics Olympiad (IPhO) level.
Return valid JSON only:
{
  "concept_name": "",
  "chapter": "",
  "exam_relevance": [],
  "difficulty_level": "",
  "summary": "",
  "explanation_step_by_step": [],
  "formulas": [],
  "vector_or_symbol_meanings": {},
  "SI_units_involved": [],
  "derivation_hints": "",
  "common_student_mistakes": [],
  "advanced_conceptual_questions": [],
  "olympiad_extension": ""
}"""
    },
    "⚗️ Chemistry Analysis": {
        "key": "chemistry",
        "desc": "NEET / JEE / IChO reaction mechanisms & organic chemistry",
        "template": """
Analyze this Chemistry diagram or reaction mechanism for NEET / JEE / IChO level.
Return valid JSON only:
{
  "compound_names": [],
  "iupac_names": [],
  "functional_groups": [],
  "reaction_type": "",
  "exam_relevance": [],
  "difficulty_level": "",
  "summary": "",
  "mechanism_explanation": [],
  "reagents_and_products": {"reagents": [], "products": [], "conditions": []},
  "stereochemistry_notes": "",
  "important_exam_points": [],
  "mcq_questions": [],
  "olympiad_extension": ""
}"""
    },
    "🌿 Biology Analysis": {
        "key": "biology",
        "desc": "NEET / IBO / USABO labeled diagrams & biological processes",
        "template": """
Analyze this Biology textbook diagram for NEET / IBO / USABO level.
Return valid JSON only:
{
  "biological_structure": "",
  "organism_or_system": "",
  "exam_relevance": [],
  "difficulty_level": "",
  "summary": "",
  "labeled_parts_with_functions": {},
  "biological_process_explained": "",
  "related_diseases_or_disorders": [],
  "important_exam_points": [],
  "common_confusions": [],
  "assertion_reason_questions": [],
  "olympiad_extension": ""
}"""
    },
    "📐 Mathematics Analysis": {
        "key": "mathematics",
        "desc": "JEE / IMO / USAMO / AMC geometry, algebra, calculus",
        "template": """
Analyze this Mathematics diagram or problem for JEE / IMO / Putnam level.
Return valid JSON only:
{
  "topic": "",
  "branch": "Algebra | Geometry | Calculus | Number Theory | Combinatorics | Trigonometry | Other",
  "exam_relevance": [],
  "difficulty_level": "",
  "summary": "",
  "problem_statement": "",
  "solution_approach": [],
  "key_theorems_used": [],
  "formulas": [],
  "geometric_properties": [],
  "common_mistakes": [],
  "practice_problems": [],
  "olympiad_extension": ""
}"""
    },
    "🔍 Topic Identifier": {
        "key": "topic",
        "desc": "Quickly classify the subject, chapter, topic & exam tags",
        "template": """
Identify the academic classification of this image.
Return valid JSON only:
{
  "subject": "",
  "chapter": "",
  "topic": "",
  "subtopic": "",
  "exam_tags": [],
  "difficulty_level": "",
  "estimated_marks_weightage": ""
}"""
    },
    "📚 General Academic": {
        "key": "general",
        "desc": "General-purpose analysis for any academic image",
        "template": """
Carefully analyze this academic image/diagram and return a structured educational description.
Return valid JSON only:
{
  "subject": "",
  "chapter_name": "",
  "topic_name": "",
  "exam_relevance": [],
  "difficulty_level": "",
  "diagram_type": "",
  "summary": "",
  "detailed_explanation": "",
  "key_labels": [],
  "formulas_or_equations": [],
  "important_concepts": [],
  "common_mistakes": [],
  "possible_exam_questions": []
}"""
    },
}

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_key_locked" not in st.session_state:
    st.session_state.api_key_locked = False
if "last_raw" not in st.session_state:
    st.session_state.last_raw = None

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def clean_url(raw_url: str) -> str:
    """Strip surrounding quotes, commas, spaces from pasted URLs."""
    return raw_url.strip().strip('"').strip("'").strip(',').strip()

def image_to_base64(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix.lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp"}
    mime = mime_map.get(ext, "image/jpeg")
    b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def call_api(api_key: str, model: str, prompt: str, image_url: str) -> str:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    return completion.choices[0].message.content

def clean_and_parse(raw: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
        return json.dumps(parsed, indent=2, ensure_ascii=False), None
    except json.JSONDecodeError as e:
        return cleaned, str(e)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;
                color:#00e5ff;margin-bottom:1.5rem;border-bottom:1px solid #1e2d45;
                padding-bottom:1rem;'>
        ⚙️ Configuration
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔑 OpenRouter API Key</div>", unsafe_allow_html=True)

    if st.session_state.api_key_locked:
        masked = st.session_state.api_key[:8] + "••••••••••••••••" + st.session_state.api_key[-4:]
        st.markdown(f"""
        <div class='key-locked'>
            ✅ &nbsp;<strong>API KEY ACTIVE</strong><br>
            <span style='opacity:0.7;font-size:0.7rem;'>{masked}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔓 Change Key", key="unlock_btn"):
            st.session_state.api_key_locked = False
            st.session_state.api_key = ""
            st.rerun()
    else:
        api_input = st.text_input(
            "Paste your key",
            type="password",
            placeholder="sk-or-v1-...",
            key="api_input_field",
        )
        if st.button("🔒 Lock & Save Key", key="lock_btn"):
            if api_input and api_input.strip():
                st.session_state.api_key = api_input.strip()
                st.session_state.api_key_locked = True
                st.rerun()
            else:
                st.error("Please enter a valid API key")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🤖 Vision Model</div>", unsafe_allow_html=True)
    selected_model_label = st.selectbox(
        "Choose model",
        options=list(VLM_MODELS.keys()),
        index=7,
        label_visibility="collapsed"
    )
    selected_model = VLM_MODELS[selected_model_label]
    st.markdown(f"<div class='model-badge'>{selected_model}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # st.markdown("<div class='section-title'>📝 Analysis Mode</div>", unsafe_allow_html=True)
    # selected_prompt_label = st.selectbox(
    #     "Select mode",
    #     options=list(PROMPTS.keys()),
    #     label_visibility="collapsed"
    # )
    prompt_info = PROMPTS["📋 Comprehensive Summary"]
    st.markdown(f"""
    <div style='background:#0d1f33;border-left:3px solid #7c3aed;border-radius:0 8px 8px 0;
                padding:0.6rem 0.8rem;font-size:0.75rem;color:#94a3b8;margin-top:0.5rem;'>
        {prompt_info["desc"]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem;color:#374151;font-family:Space Mono,monospace;
                text-align:center;border-top:1px solid #1e2d45;padding-top:1rem;'>
        Powered by OpenRouter · Qwen VL Series<br>
        NEET · JEE · IPhO · IChO · IBO · IMO
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN AREA
# ──────────────────────────────────────────────
st.markdown("""
<div class='app-header'>
    <div class='app-title'>🔬 Academic Image Analyzer</div>
    <div class='app-subtitle'>
        NEET · JEE Main · JEE Advanced · IPhO · IChO · IBO · IMO · USABO · Olympiad
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.api_key_locked:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1c1009,#2d1b00);
                border:1px solid #f59e0b;border-radius:12px;
                padding:2rem;text-align:center;margin-top:2rem;'>
        <div style='font-size:2rem;margin-bottom:0.8rem;'>🔑</div>
        <div style='font-family:Space Mono,monospace;color:#fbbf24;font-size:1rem;
                    font-weight:700;margin-bottom:0.5rem;'>API Key Required</div>
        <div style='color:#92400e;font-size:0.85rem;'>
            Enter your OpenRouter API key in the sidebar to begin analyzing images.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── IMAGE INPUT ──
st.markdown("<div class='section-title'>📸 Image Input</div>", unsafe_allow_html=True)

col_input, col_preview = st.columns([1, 1], gap="large")

with col_input:
    input_mode = st.radio(
        "Source",
        ["🔗 URL / S3 Link", "📁 Upload Image"],
        horizontal=True,
        label_visibility="collapsed"
    )

    final_image_url = None
    uploaded_preview = None

    if "🔗" in input_mode:
        url_input = st.text_input(
            "Image URL or S3 link",
            placeholder="https://your-bucket.s3.amazonaws.com/image.jpg",
        )
        if url_input and url_input.strip():
            final_image_url = clean_url(url_input)   # ← strips quotes/commas
            uploaded_preview = final_image_url
    else:
        uploaded_file = st.file_uploader(
            "Drop image here",
            type=["jpg", "jpeg", "png", "webp", "gif", "bmp"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            final_image_url = image_to_base64(uploaded_file)
            uploaded_preview = uploaded_file

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 Analyze Image", disabled=(final_image_url is None), key="analyze_btn")

with col_preview:
    if uploaded_preview:
        st.markdown("<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>Preview</div>", unsafe_allow_html=True)
        st.image(uploaded_preview, width="stretch")
    else:
        st.markdown("""
        <div style='background:#0d1624;border:1px dashed #1e2d45;border-radius:12px;
                    height:200px;display:flex;align-items:center;justify-content:center;
                    color:#1e2d45;font-family:Space Mono,monospace;font-size:0.8rem;'>
            Image preview will appear here
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────
# ANALYSIS
# ──────────────────────────────────────────────
if analyze_clicked and final_image_url:
    with st.spinner(f"Analyzing with {selected_model} ..."):
        try:
            raw = call_api(
                api_key=st.session_state.api_key,
                model=selected_model,
                prompt=prompt_info["template"],
                image_url=final_image_url,
            )
            output, err = clean_and_parse(raw)
            st.session_state.last_raw = output
            st.session_state.last_err = err
        except Exception as e:
            st.error(f"❌ API Error: {str(e)}")
            st.session_state.last_raw = None

# ──────────────────────────────────────────────
# OUTPUT — plain JSON only
# ──────────────────────────────────────────────
if st.session_state.last_raw:
    st.markdown("<div class='section-title'>📊 Analysis Output</div>", unsafe_allow_html=True)

    if hasattr(st.session_state, "last_err") and st.session_state.last_err:
        st.warning(f"JSON parse warning: {st.session_state.last_err}")

    st.code(st.session_state.last_raw, language="json")

    st.download_button(
        label="⬇️ Download JSON",
        data=st.session_state.last_raw,
        file_name=f"analysis_{prompt_info['key']}.json",
        mime="application/json",
    )
