import utils.session
import streamlit as st
import requests
import json
import base64
import io

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Qwen VL Studio", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0d0d14;
    --surface: #16161f;
    --accent: #00e5ff;
    --text: #e8eaf0;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: 'Inter', sans-serif; }

.main-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(0deg, var(--bg) 80%, transparent 100%);
    padding: 1.5rem 15% 2rem 15%;
    z-index: 1000;
}

.chat-container { padding-bottom: 260px; max-width: 900px; margin: 0 auto; }

.msg-row-user { display: flex; justify-content: flex-end; margin: 20px 0; gap: 12px; }
.msg-row-ai { display: flex; justify-content: flex-start; margin: 20px 0; gap: 12px; }

.bubble-user {
    background: #252538;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    max-width: 80%;
    border: 1px solid rgba(255,255,255,0.05);
}

.bubble-ai { background: transparent; padding: 14px 0; max-width: 90%; line-height: 1.6; }

.avatar { width: 35px; height: 35px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0; }
.avatar-user { background: var(--accent); color: #000; }
.avatar-ai { background: #7c3aed; color: #fff; }

#MainMenu, footer, header { visibility: hidden; }
.stTextArea textarea { background: #1a1a28 !important; color: white !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  MODELS & CONSTANTS
# ─────────────────────────────────────────────────────────────────
QWEN_MODELS = {
    "⭐ Qwen3 VL 235B A22B Instruct":  "qwen/qwen3-vl-235b-a22b-instruct",
    "🧠 Qwen3 VL 235B A22B Thinking":  "qwen/qwen3-vl-235b-a22b-thinking",
    "🔬 Qwen3 VL 32B Instruct":        "qwen/qwen3-vl-32b-instruct",
    "⚡ Qwen3 VL 30B A3B Instruct":    "qwen/qwen3-vl-30b-a3b-instruct",
    "💡 Qwen3 VL 30B A3B Thinking":    "qwen/qwen3-vl-30b-a3b-thinking",
    "🚀 Qwen3 VL 8B Instruct":         "qwen/qwen3-vl-8b-instruct",
    "🔍 Qwen3 VL 8B Thinking":         "qwen/qwen3-vl-8b-thinking",
    "🏆 Qwen2.5 VL 72B Instruct":      "qwen/qwen2.5-vl-72b-instruct",
    "📦 Qwen2.5 VL 7B Instruct":       "qwen/qwen2.5-vl-7b-instruct",
    "🌐 Qwen2 VL 72B Instruct":        "qwen/qwen2-vl-72b-instruct",
}

DEEPSEEK_MODELS = {
    "🔥 DeepSeek V3.2 Speciale":       "deepseek/deepseek-v3.2-speciale",
    "⭐ DeepSeek V3.2":                "deepseek/deepseek-v3.2",
    "🧪 DeepSeek V3.2 Experimental":   "deepseek/deepseek-v3.2-exp"
}

MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}

# ─────────────────────────────────────────────────────────────────
#  SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Studio Settings")
    api_key = st.text_input("OpenRouter API Key", type="password")

    st.markdown("### 📝 User Prompts")
    sys_prompt = st.text_area("System Prompt", value="You are a helpful AI assistant.")
    ocr_prompt = "Extract all text and details from this image"

    st.markdown("---")
    use_deepseek = st.toggle("Use DeepSeek Pipeline", value=False)

    if use_deepseek:
        model_name = st.selectbox("Select DeepSeek Model", list(DEEPSEEK_MODELS.keys()))
        model_id = DEEPSEEK_MODELS[model_name]
        stream_choice = False
        st.info("💡 DeepSeek: Streaming disabled for stability.")
    else:
        model_name = st.selectbox("Select Qwen Model", list(QWEN_MODELS.keys()))
        model_id = QWEN_MODELS[model_name]
        stream_choice = st.checkbox("Streaming Mode", value=True)

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)

    if st.button("🗑 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────
#  CORE HELPERS
# ─────────────────────────────────────────────────────────────────
def call_api(messages, model, stream, timeout=180):
    """Call OpenRouter API. Returns response or None on failure."""
    if not api_key:
        st.error("❌ API Key is required. Please enter it in the sidebar.")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://qwen-vl-studio.app",  # helps OpenRouter routing
        "X-Title": "Qwen VL Studio"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=stream,
            timeout=timeout
        )
        return response
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Try a smaller model or shorter input.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🌐 Connection error. Check your internet connection.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def get_mime_type(filename: str) -> str:
    """Return correct MIME type based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    return MIME_MAP.get(ext, "image/jpeg")


def process_file(uploaded_file):
    """
    Returns (ftype, fdata, mime_type).
    ftype: 'image' | 'text'
    fdata: base64 string for images, plain text for text files
    mime_type: proper MIME string for images
    """
    fname = uploaded_file.name.lower()
    content = uploaded_file.read()
    ext = fname.rsplit(".", 1)[-1]

    if ext in ("png", "jpg", "jpeg", "webp", "gif"):
        mime = get_mime_type(fname)
        return "image", base64.b64encode(content).decode(), mime

    elif ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join([page.extract_text() or "" for page in reader.pages[:5]])
            return "text", f"[PDF Content]\n{text}", None
        except ImportError:
            return "text", "[Error: pypdf not installed. Run: pip install pypdf]", None
        except Exception as e:
            return "text", f"[Error reading PDF: {e}]", None

    else:
        try:
            return "text", f"[File: {uploaded_file.name}]\n{content.decode('utf-8')}", None
        except Exception:
            return "text", "[Error: Could not decode file as text]", None


def build_history_for_api(history):
    """
    Convert session history to API messages.
    Image content is stored as text in session for display but
    we only send text for past messages (image already consumed).
    """
    api_msgs = []
    for m in history:
        role = m["role"]
        content = m["content"]
        # content in history is always a display string (never raw image dict)
        api_msgs.append({"role": role, "content": content})
    return api_msgs


def stream_response(res, placeholder):
    """Stream response chunks and return full text."""
    full_res = ""
    try:
        for line in res.iter_lines():
            if line:
                try:
                    raw = line.decode("utf-8").strip()
                    if raw.startswith("data: "):
                        raw = raw[6:]
                    if raw == "[DONE]":
                        break
                    chunk = json.loads(raw)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        full_res += text
                        placeholder.markdown(full_res + "▌")
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Streaming error: {e}")
    placeholder.markdown(full_res)
    return full_res


# ─────────────────────────────────────────────────────────────────
#  CHAT DISPLAY
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    if role == "user":
        st.markdown(
            f"<div class='msg-row-user'>"
            f"<div class='bubble-user'>{content}</div>"
            f"<div class='avatar-user avatar'>U</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='msg-row-ai'>"
            f"<div class='avatar-ai avatar'>AI</div>"
            f"<div class='bubble-ai'>{content}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  FIXED FOOTER INPUT
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-footer">', unsafe_allow_html=True)
with st.container():
    c1, c2 = st.columns([0.85, 0.15])
    with c1:
        u_input = st.text_area(
            "msg",
            placeholder="Ask anything...",
            label_visibility="collapsed",
            height=70
        )
        u_file = st.file_uploader(
            "upload",
            type=["png", "jpg", "jpeg", "pdf", "txt", "py", "js", "html", "json"],
            label_visibility="collapsed"
        )
    with c2:
        st.write(" ")
        send = st.button("Send ➔", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  PROCESS REQUEST
# ─────────────────────────────────────────────────────────────────
if send:
    if not u_input.strip():
        st.warning("Please type a message before sending.")
        st.stop()

    if not api_key:
        st.error("❌ API Key required. Enter it in the sidebar.")
        st.stop()

    # Build API messages: system prompt + history (text only) + new user message
    api_msgs = [{"role": "system", "content": sys_prompt}]
    api_msgs += build_history_for_api(st.session_state.messages)

    # Process uploaded file
    final_content = u_input.strip()
    file_label = ""  # shown in chat history

    if u_file:
        ftype, fdata, fmime = process_file(u_file)

        if ftype == "image":
            file_label = f"\n\n📎 *Image attached: {u_file.name}*"

            if use_deepseek:
                # DeepSeek doesn't support vision → OCR via Qwen first
                with st.spinner("🔍 Reading image via Qwen OCR..."):
                    ocr_payload = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ocr_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{fmime};base64,{fdata}"}}
                        ]
                    }]
                    ocr_res = call_api(ocr_payload, "qwen/qwen3-vl-8b-instruct", False, timeout=120)

                    if ocr_res is None:
                        st.stop()
                    if ocr_res.status_code != 200:
                        try:
                            err = ocr_res.json().get("error", {}).get("message", "Unknown")
                        except Exception:
                            err = ocr_res.text[:300]
                        st.error(f"OCR step failed ({ocr_res.status_code}): {err}")
                        st.stop()

                    img_ctx = (
                        ocr_res.json()
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "OCR returned no text")
                    )
                    final_content = f"Image Context (OCR):\n{img_ctx}\n\nUser Question: {u_input.strip()}"

            else:
                # Vision-capable Qwen model: send image directly
                final_content = [
                    {"type": "text", "text": u_input.strip()},
                    {"type": "image_url", "image_url": {"url": f"data:{fmime};base64,{fdata}"}}
                ]

        else:
            # Text/PDF file
            file_label = f"\n\n📎 *File attached: {u_file.name}*"
            final_content = f"{fdata}\n\nUser Question: {u_input.strip()}"

    # Save user message for display (always plain text)
    display_text = u_input.strip() + file_label
    st.session_state.messages.append({"role": "user", "content": display_text})

    # Add the actual content (may include image dict) to API call only
    api_msgs.append({"role": "user", "content": final_content})

    # Call the main model
    with st.spinner(f"Thinking with {model_name}..."):
        res = call_api(api_msgs, model_id, stream_choice)

    if res is None:
        st.stop()

    if res.status_code != 200:
        try:
            err_body = res.json()
            err_msg = err_body.get("error", {}).get("message", json.dumps(err_body))
        except Exception:
            err_msg = res.text[:500]
        st.error(f"❌ API Error {res.status_code}: {err_msg}")
        st.stop()

    # Parse response
    full_res = ""
    ph = st.empty()

    if stream_choice and not use_deepseek:
        full_res = stream_response(res, ph)
    else:
        try:
            data = res.json()
        except Exception:
            st.error("❌ Could not parse API response as JSON.")
            st.stop()

        if "choices" in data and data["choices"]:
            full_res = data["choices"][0]["message"]["content"]
            ph.markdown(full_res)
        elif "error" in data:
            st.error(f"❌ API Error: {data['error'].get('message', str(data['error']))}")
            st.stop()
        else:
            st.error(f"❌ Unexpected response format: {json.dumps(data)[:300]}")
            st.stop()

    if full_res:
        st.session_state.messages.append({"role": "assistant", "content": full_res})
        st.rerun()
    else:
        st.warning("⚠️ Model returned an empty response. Try again or switch models.")
