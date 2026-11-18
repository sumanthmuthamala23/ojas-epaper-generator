import os
import re
import io
import json
import time
import base64
import tempfile
import datetime
import urllib.parse
import requests
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

# Optional OpenCV for extracting thumbnail frame from uploaded video
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Optional Gemini SDK for Chief-Editor AI bilingual translation & newsroom editing
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="OJAS - Power of Information | Bilingual E-Paper Studio",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="auto"
)

# ----------------- ASSET HELPERS -----------------
def load_image_base64(candidates: list) -> str:
    """Loads the first matching image file from the repository root as base64."""
    for fname in candidates:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    return ""

def image_to_base64(uploaded_file, max_width=1400) -> str:
    """Optimizes uploaded news photos and converts to high-res base64 JPEG."""
    img = Image.open(uploaded_file).convert("RGB")
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int(float(img.height) * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_video_thumbnail_b64(video_file) -> str:
    """Extracts a sharp cover frame from an uploaded video file and returns base64 JPEG."""
    if not HAS_CV2 or video_file is None:
        return ""
    try:
        suffix = os.path.splitext(video_file.name)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_file.getvalue())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = min(30, max(0, total_frames // 5))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        cap.release()
        os.remove(tmp_path)

        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            if pil_img.width > 1280:
                ratio = 1280 / float(pil_img.width)
                pil_img = pil_img.resize((1280, int(pil_img.height * ratio)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        pass
    return ""

def generate_qr_base64(url: str) -> str:
    """Generates a crisp QR code image as base64 PNG for playable video links in PDF/print."""
    if not url or not url.strip():
        return ""
    try:
        qr_api = f"https://api.qrserver.com/v1/create-qr-code/?size=160x160&margin=4&data={urllib.parse.quote(url.strip())}"
        res = requests.get(qr_api, timeout=5)
        if res.status_code == 200:
            return base64.b64encode(res.content).decode("utf-8")
    except Exception:
        pass
    return ""

# ----------------- TEXT & BULLETPROOF TWO-WAY TRANSLATION HELPERS -----------------
def clean_whatsapp_formatting(text: str) -> str:
    """Strips WhatsApp *bold* / _italic_ markers and bullet symbols that break URL translation."""
    cleaned = text.replace("*", "").replace("_", "").replace("◆", "").replace("•", "")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def detect_telugu(text: str) -> bool:
    """Detects whether the input text is predominantly in Telugu script."""
    te_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    en_chars = len(re.findall(r'[A-Za-z]', text))
    return te_chars > en_chars

def chunked_free_translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Bulletproof sentence-by-sentence free translator between Telugu ('te') and English ('en').
    Prevents long-URL truncation so the entire news body translates cleanly every time.
    """
    if not text.strip():
        return ""
    cleaned = clean_whatsapp_formatting(text)
    # Split into manageable sentence chunks (< 400 chars each)
    raw_sentences = re.split(r'(?<=[.!?।])\s+', cleaned)
    chunks = []
    current = ""
    for s in raw_sentences:
        if len(current) + len(s) < 380:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s[:380]
    if current:
        chunks.append(current)

    translated_parts = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for chunk in chunks:
        try:
            url = (
                "https://translate.googleapis.com/translate_a/single"
                f"?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={urllib.parse.quote(chunk)}"
            )
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                piece = "".join([seg[0] for seg in data[0] if seg and seg[0]]).strip()
                translated_parts.append(piece if piece else chunk)
            else:
                translated_parts.append(chunk)
        except Exception:
            translated_parts.append(chunk)
    return " ".join(translated_parts).strip()

def smart_condense_to_word_budget(text: str, max_words: int) -> str:
    """Trims text cleanly at sentence boundaries to fit single-page column budgets."""
    cleaned = re.sub(r'\n{2,}', '\n', text.strip())
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    sentences = re.split(r'(?<=[.!?।])\s+', cleaned)
    out = []
    count = 0
    for s in sentences:
        w = len(s.split())
        if count + w <= max_words:
            out.append(s)
            count += w
        else:
            break
    if not out:
        return " ".join(words[:max_words]) + "..."
    return " ".join(out)

def transliterate_to_telugu(text: str) -> str:
    """Free, unlimited phonetic English-to-Telugu converter (no API key required)."""
    if not text.strip():
        return ""
    lines = text.split("\n")
    converted_lines = []
    url = "https://inputtools.google.com/request?text={}&itc=te-t-i0-und&num=1"
    headers = {"User-Agent": "Mozilla/5.0"}

    for line in lines:
        words = line.split()
        out_words = []
        for word in words:
            if word.isdigit() or word in [",", ".", "!", "?", "-", ":", ";", "(", ")"]:
                out_words.append(word)
                continue
            try:
                res = requests.get(url.format(urllib.parse.quote(word)), headers=headers, timeout=4)
                data = res.json()
                if data[0] == "SUCCESS" and data[1][0][1]:
                    out_words.append(data[1][0][1][0])
                else:
                    out_words.append(word)
            except Exception:
                out_words.append(word)
        converted_lines.append(" ".join(out_words))
    return "\n".join(converted_lines)

# ----------------- PROFESSIONAL GEMINI 3.8 NEWSROOM ENGINE -----------------
def discover_active_gemini_models(client) -> list:
    """Prioritizes gemini-3.8-flash and active models on the user's API key."""
    preferred_order = [
        "gemini-3.8-flash",
        "gemini-3.8-pro",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    discovered = []
    try:
        for m in client.models.list():
            name = m.name.replace("models/", "")
            methods = getattr(m, "supported_generation_methods", []) or []
            if not methods or "generateContent" in methods:
                discovered.append(name)
    except Exception:
        return preferred_order

    ordered = [m for m in preferred_order if m in discovered]
    for m in discovered:
        if m not in ordered and ("flash" in m or "pro" in m) and "tts" not in m and "image" not in m and "live" not in m:
            ordered.append(m)
    return ordered if ordered else preferred_order

def ai_editorial_bilingual_engine(
    keys: list,
    raw_input: str,
    location: str,
    target_te_words: int,
    target_en_words: int,
    editorial_style: str
) -> tuple[dict, str]:
    """
    Chief-Editor AI Newsroom Engine using Gemini 3.8 Flash:
    - Translates and edits BOTH headline AND full body between Telugu and English
    - Never uses markdown asterisks (*) in output
    - Produces authentic Telugu పత్రికా భాష (Eenadu / Sakshi style) & crisp English print copy
    """
    last_err = None

    prompt = f"""
    You are the Chief News Editor & Bilingual Copy Desk Director of 'OJAS - Power of Information' (ఓజస్), a premier Telugu-English daily newspaper in Telangana & Andhra Pradesh.

    TASK:
    Convert the RAW NEWS INPUT below into publication-ready bilingual newspaper copy.
    - If the input is in Telugu -> refine the Telugu copy AND fully translate/transcreate the ENTIRE headline, subhead, and full news body into crisp English print journalism.
    - If the input is in English -> refine the English copy AND fully translate/transcreate the ENTIRE headline, subhead, and full news body into authentic journalistic Telugu (పత్రికా భాష).

    STRICT JOURNALISTIC & FORMATTING RULES:
    1. BOTH `te_body` and `en_body` MUST be completely translated into their respective target languages. NEVER leave Telugu text inside `en_body` or English sentences inside `te_body`.
    2. DO NOT use any markdown symbols like `*`, `**`, `#`, or `_` anywhere in the JSON values. Plain clean text only.
    3. **Telugu Style (`te_headline`, `te_subhead`, `te_body`)**: Authentic Telugu print journalism (*Eenadu / Sakshi / Namasthe Telangana* style). Start body with dateline (e.g., `{location.split('•')[-1].strip()}:`). Use formal journalistic verbs (*పేర్కొన్నారు, వెల్లడించారు, డిమాండ్ చేశారు, విమర్శించారు, ఆదేశించారు*).
    4. **English Style (`en_headline`, `en_subhead`, `en_body`)**: Crisp *The Hindu / Deccan Chronicle* style. Start body with uppercase dateline (e.g., `{location.split('•')[-1].strip().upper()}:`).
    5. **Editing & Word Budget**: {editorial_style}. Keep `te_body` around {target_te_words} words and `en_body` around {target_en_words} words in 1–2 tight paragraphs with NO empty lines.

    RAW NEWS INPUT:
    {raw_input}

    Return STRICTLY valid JSON with these exact keys:
    {{
        "te_headline": "Punchy 5-8 word journalistic Telugu headline (no asterisks)",
        "te_subhead": "Crisp 1-line Telugu deck sub-headline (no asterisks)",
        "te_body": "Full journalistic Telugu report within {target_te_words} words (no asterisks)",
        "en_headline": "Punchy 6-10 word journalistic English headline (no asterisks)",
        "en_subhead": "Crisp 1-line English deck sub-headline (no asterisks)",
        "en_body": "Full journalistic English report within {target_en_words} words (no asterisks)",
        "pull_quote": "1 sharp key takeaway quote (max 14 words, no asterisks)"
    }}
    """

    for key in keys:
        if not key or not key.strip():
            continue
        try:
            client = genai.Client(api_key=key.strip())
            models_pool = discover_active_gemini_models(client)
            for m_name in models_pool:
                try:
                    res = client.models.generate_content(
                        model=m_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.25,
                            response_mime_type="application/json"
                        )
                    )
                    if res and res.text:
                        parsed = json.loads(res.text)
                        for k_clean in parsed:
                            if isinstance(parsed[k_clean], str):
                                parsed[k_clean] = parsed[k_clean].replace("*", "").strip()
                        return parsed, m_name
                except Exception as e:
                    last_err = e
                    time.sleep(0.4)
                    continue
        except Exception as ce:
            last_err = ce
            continue

    if last_err:
        raise last_err
    raise RuntimeError("No valid Gemini API key provided.")

def process_bilingual_auto_fit(
    keys: list,
    raw_text: str,
    location: str,
    target_te_words: int,
    target_en_words: int,
    editorial_style: str
) -> tuple[dict, str]:
    """Runs Gemini 3.8 Chief-Editor AI if API key is available; otherwise uses chunked sentence-by-sentence free translator."""
    valid_keys = [k for k in keys if k and k.strip()]
    if valid_keys and HAS_GENAI:
        try:
            return ai_editorial_bilingual_engine(
                valid_keys, raw_text, location, target_te_words, target_en_words, editorial_style
            )
        except Exception as e:
            st.warning(f"Gemini API notice ({e}) — using chunked free bilingual translator.")

    # Bulletproof Chunked Free Fallback (Translates EVERY sentence in both headline & body)
    cleaned_raw = clean_whatsapp_formatting(raw_text)
    is_te = detect_telugu(cleaned_raw)
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]

    if len(lines) > 1:
        first_line = clean_whatsapp_formatting(lines[0])
        body_raw = clean_whatsapp_formatting(" ".join(lines[1:]))
    else:
        sents = re.split(r'(?<=[.!?।])\s+', cleaned_raw)
        first_line = sents[0][:95] if sents else cleaned_raw[:85]
        body_raw = cleaned_raw

    if is_te:
        te_head = first_line[:95]
        te_body = smart_condense_to_word_budget(body_raw, target_te_words)
        en_head = chunked_free_translate(te_head, "te", "en")
        en_body_full = chunked_free_translate(te_body, "te", "en")
        en_body = smart_condense_to_word_budget(en_body_full, target_en_words)
    else:
        en_head = first_line[:100]
        en_body = smart_condense_to_word_budget(body_raw, target_en_words)
        te_head = chunked_free_translate(en_head, "en", "te")
        te_body_full = chunked_free_translate(en_body, "en", "te")
        te_body = smart_condense_to_word_budget(te_body_full, target_te_words)

    return {
        "te_headline": te_head,
        "te_subhead": "",
        "te_body": te_body,
        "en_headline": en_head,
        "en_subhead": "",
        "en_body": en_body,
        "pull_quote": ""
    }, "Chunked Bilingual Translator"

# ----------------- CUSTOM RESPONSIVE APP STYLING -----------------
bg_b64 = load_image_base64(["background.png"])
dev_b64 = load_image_base64(["Sumanth.jpg", "sumanth.jpg"])

app_bg_css = (
    f"""
    .stApp {{
        background: linear-gradient(rgba(248, 250, 252, 0.94), rgba(241, 245, 249, 0.96)),
                    url("data:image/png;base64,{bg_b64}");
        background-size: cover;
        background-attachment: fixed;
    }}
    """
    if bg_b64 else
    """
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f6 100%);
    }
    """
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    {app_bg_css}

    html, body, p, div:not([data-testid="stIconMaterial"]), input, textarea, button {{
        font-family: 'Plus Jakarta Sans', 'Noto Sans Telugu', sans-serif !important;
    }}
    span[data-testid="stIconMaterial"], .material-symbols-rounded {{
        font-family: 'Material Symbols Rounded' !important;
    }}
    .top-brand-bar {{
        background: linear-gradient(90deg, #7f1d1d 0%, #b91c1c 58%, #d97706 100%);
        color: white;
        padding: 14px 22px;
        border-radius: 14px;
        margin-bottom: 16px;
        box-shadow: 0 8px 22px rgba(127, 29, 29, 0.22);
        border: 1px solid rgba(255,255,255,0.18);
    }}
    .top-brand-bar h1 {{
        color: white !important;
        margin: 0 !important;
        font-size: 21px !important;
        font-weight: 800 !important;
    }}
    .top-brand-bar p {{
        color: #fef3c7 !important;
        margin: 3px 0 0 0 !important;
        font-size: 13px !important;
    }}
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #991b1b 0%, #dc2626 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 11px 20px !important;
        box-shadow: 0 4px 12px rgba(185, 28, 28, 0.25) !important;
    }}
    @media screen and (max-width: 768px) {{
        .top-brand-bar {{ padding: 11px 14px !important; }}
        .top-brand-bar h1 {{ font-size: 17px !important; }}
        .block-container {{ padding-left: 0.65rem !important; padding-right: 0.65rem !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
defaults = {
    "te_headline": "ప్రజా సమస్యల పరిష్కారమే లక్ష్యంగా ప్రత్యేక కార్యాచరణ",
    "te_subhead": "అధికారులు క్షేత్రస్థాయిలో పర్యటించి తక్షణ చర్యలు చేపట్టాలి",
    "te_body": "కోదాడ, న్యూస్‌టుడే: పట్టణ, గ్రామీణ ప్రాంతాల్లో ప్రజలు ఎదుర్కొంటున్న మౌలిక వసతుల సమస్యలను యుద్ధప్రాతిపదికన పరిష్కరించేందుకు ప్రత్యేక ప్రణాళిక సిద్ధం చేసినట్లు అధికారులు వెల్లడించారు. తాగునీరు, రహదారులు, పారిశుద్ధ్య పనులపై ప్రత్యేక దృష్టి సారించి నిర్ణీత గడువులోగా పూర్తి చేయాలని ఆదేశించారు.",
    "en_headline": "Special Action Plan Launched to Resolve Public Grievances",
    "en_subhead": "Officials directed to conduct field visits and ensure swift redressal",
    "en_body": "KODAD: Authorities have rolled out a dedicated action plan to resolve civic and infrastructure issues across urban and rural pockets on a war footing. Officials have been instructed to prioritize drinking water supply, road repairs, and sanitation works with strict timelines.",
    "pull_quote": "",
    "raw_bilingual_input": "",
    "last_engine_used": "",
    # Secondary News Item (Optional to fill lower broadsheet space)
    "enable_story_2": False,
    "s2_raw_input": "",
    "s2_te_headline": "రైతులకు సకాలంలో ఎరువులు, విత్తనాలు అందించాలి",
    "s2_te_body": "ఖమ్మం: జిల్లాలోని అన్ని సహకార సంఘాల్లో సరిపడా యూరియా, విత్తనాలను అందుబాటులో ఉంచాలని నాయకులు అధికారులను కోరారు.",
    "s2_en_headline": "Ensure Timely Supply of Fertilizers and Seeds to Farmers",
    "s2_en_body": "KHAMMAM: Leaders urged agricultural officials to maintain adequate stocks of urea and quality seeds across all cooperative societies.",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- SIDEBAR: EDITION & VIEWPORT SETTINGS -----------------
with st.sidebar:
    logo_b64 = load_image_base64(["OJAS.png", "ojas.png", "OJAS.jpg", "ojas.jpg"])
    if logo_b64:
        st.markdown(
            f'<div style="background:#fff; padding:8px; border-radius:10px; border:1px solid #e2e8f0; text-align:center; margin-bottom:10px;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="max-width:100%; height:auto;"></div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Upload `OJAS.png` to your GitHub repo to display the masthead logo.")

    st.subheader("📐 1. Page Format & Blank-Space Control")
    aspect_preset = st.radio(
        "Choose Edition Format:",
        [
            "✂️ Auto-Trim Clipping (Zero Blank Space — Tight Fit)",
            "📰 Full A4 Broadsheet (Multi-Story / Full Page)",
            "📱 9:16 Vertical E-Paper (Mobile / WhatsApp Status)"
        ],
        index=0,
        help="Auto-Trim removes all leftover blank space at the bottom so your JPEG & PDF hug the news clipping cleanly."
    )

    editorial_style = st.selectbox(
        "🧠 AI Newsroom Editing Style",
        [
            "Print Daily Standard • Eenadu / Sakshi & The Hindu Style",
            "Inverted Pyramid • Crisp Breaking News",
            "Official Statement • Authoritative Press Release",
            "Civic & Welfare Spotlight • Human-Interest Lead"
        ]
    )

    st.subheader("🗞️ 2. Edition & Masthead Details")
    pub_date = st.date_input("Publication Date", value=datetime.date.today())
    formatted_date = pub_date.strftime("%A, %d %B %Y")
    telugu_date_str = pub_date.strftime("%d-%m-%Y")

    edition_name = st.text_input("Edition / Bureau", value="TELANGANA EDITION • KHAMMAM")
    vol_issue = st.text_input("Volume / Issue", value="VOL. 01 | ISSUE 142 | DAILY DIGITAL")
    category_tag = st.selectbox(
        "Section Ribbon",
        [
            "EXCLUSIVE REPORT • ప్రత్యేక కథనం",
            "BREAKING NEWS • తాజా వార్త",
            "DISTRICT FOCUS • జిల్లా సమాచారం",
            "STATE POLITICS • రాష్ట్ర రాజకీయాలు",
            "PUBLIC WELFARE • ప్రజా సంక్షేమం",
            "SPECIAL STORY • విశ్లేషణ"
        ]
    )

    layout_mode = st.radio(
        "Bilingual Column Arrangement",
        [
            "Bilingual Side-by-Side (Telugu + English)",
            "Telugu Lead + English Box",
            "English Lead + Telugu Box"
        ]
    )

    paper_theme = st.selectbox(
        "Paper Graphic Theme",
        [
            "Classic Editorial Newsprint (Warm Ivory & Crimson)",
            "Royal Broadsheet White (Crisp White & Maroon)",
            "Sunday Magazine Edition (Cream & Gold Accent)"
        ]
    )

    st.divider()
    with st.expander("🔑 Gemini AI Key (For Chief-Editor Newsroom Translation)", expanded=True):
        st.caption("Uses Gemini 3.8 Flash for full headline + body translation in authentic Telugu (పత్రికా భాష) & English.")
        secret_key = ""
        backup_secret = ""
        try:
            if hasattr(st, "secrets"):
                secret_key = st.secrets.get("GEMINI_API_KEY", "")
                backup_secret = st.secrets.get("BACKUP_API_KEY", "")
        except Exception:
            pass
        gemini_key = st.text_input("Gemini API Key", value=secret_key, type="password")
        if st.session_state["last_engine_used"]:
            st.caption(f"⚡ Active Engine: `{st.session_state['last_engine_used']}`")

    if dev_b64:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:10px; background:#fff; padding:8px 10px; border-radius:10px; border:1px solid #e2e8f0; margin-top:10px;">'
            f'<img src="data:image/jpeg;base64,{dev_b64}" style="width:40px; height:40px; border-radius:50%; object-fit:cover; border:2px solid #991b1b;">'
            f'<div style="font-size:11.5px; font-weight:700; color:#1e293b; line-height:1.25;">OJAS Digital Studio<br><span style="color:#991b1b; font-size:10.5px;">By Sumanth Muthamala</span></div></div>',
            unsafe_allow_html=True
        )

active_keys = [gemini_key]
if backup_secret and backup_secret != gemini_key:
    active_keys.append(backup_secret)

# ----------------- MAIN WORKSPACE -----------------
st.markdown("""
<div class="top-brand-bar">
    <h1>📰 OJAS — Power of Information (ఓజస్) E-Paper Studio</h1>
    <p>Full Telugu ↔ English Newsroom Translation • Auto-Trim Zero Blank Space • Multi-Story & Video PDF</p>
</div>
""", unsafe_allow_html=True)

col_editor, col_preview = st.columns([0.95, 1.35], gap="large")

with col_editor:
    st.subheader("✍️ 1. Lead News & Bilingual Desk")

    with st.expander("🔤 Type English Phonetics ➔ Instant Telugu (100% Free)", expanded=False):
        phonetic_in = st.text_input(
            "Type Telugu in English letters:",
            placeholder="e.g., Kodad lo nedu pratyeka samavesam jarigindi"
        )
        if st.button("Convert to Telugu Script"):
            if phonetic_in.strip():
                st.code(transliterate_to_telugu(phonetic_in), language=None)

    st.markdown("##### ⚡ Paste Lead News in Telugu OR English (Translates Both Headline & Full Body)")
    raw_news_input = st.text_area(
        "Paste raw report, press note, or WhatsApp news in Telugu or English:",
        value=st.session_state["raw_bilingual_input"],
        height=130,
        placeholder="Paste news in Telugu OR English here. Clicking below translates BOTH the headline and the entire body text into the other language..."
    )

    is_9_16 = "9:16" in aspect_preset
    is_auto_trim = "Auto-Trim" in aspect_preset
    target_te_words = 75 if is_9_16 else 125
    target_en_words = 85 if is_9_16 else 140

    if st.button("⚡ AI Newsroom Translate (Telugu ↔ English) & Format", type="primary", use_container_width=True):
        if not raw_news_input.strip():
            st.warning("Please enter or paste your news text in Telugu or English first.")
        else:
            st.session_state["raw_bilingual_input"] = raw_news_input
            with st.spinner("Translating headline & full body into bilingual newspaper format..."):
                result, used_engine = process_bilingual_auto_fit(
                    active_keys,
                    raw_news_input,
                    edition_name,
                    target_te_words,
                    target_en_words,
                    editorial_style
                )
                st.session_state["last_engine_used"] = used_engine
                for k in ["te_headline", "te_subhead", "te_body", "en_headline", "en_subhead", "en_body", "pull_quote"]:
                    if k in result:
                        st.session_state[k] = result[k]
                st.success(f"✅ Headline & full body translated via {used_engine}!")
                st.rerun()

    st.write("")
    tab_te, tab_en, tab_photos, tab_video, tab_story2 = st.tabs([
        "తెలుగు (Edit)",
        "🇬🇧 English (Edit)",
        "📸 Photos (Max 2)",
        "🎬 Video",
        "➕ 2nd News"
    ])

    with tab_te:
        st.session_state["te_headline"] = st.text_input(
            "తెలుగు ప్రధాన శీర్షిక (Telugu Headline)",
            value=st.session_state["te_headline"]
        )
        st.session_state["te_subhead"] = st.text_input(
            "ఉప శీర్షిక (Telugu Sub-headline - Optional)",
            value=st.session_state["te_subhead"]
        )
        st.session_state["te_body"] = st.text_area(
            "తెలుగు వార్తా వివరాలు (Telugu News Body)",
            value=st.session_state["te_body"],
            height=150
        )
        st.session_state["pull_quote"] = st.text_input(
            "💎 ముఖ్యాంశం / Quote Ribbon (Optional)",
            value=st.session_state["pull_quote"]
        )

    with tab_en:
        st.session_state["en_headline"] = st.text_input(
            "English Main Headline",
            value=st.session_state["en_headline"]
        )
        st.session_state["en_subhead"] = st.text_input(
            "English Sub-headline (Optional)",
            value=st.session_state["en_subhead"]
        )
        st.session_state["en_body"] = st.text_area(
            "English News Body",
            value=st.session_state["en_body"],
            height=150
        )

    with tab_photos:
        st.info("Attach **1 Photo** (Wide Hero Banner) or **2 Photos** (Balanced Dual-Frame Grid). Maximum 2 photos.")
        photo_b64_list = []
        captions = []

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            photo_1 = st.file_uploader(
                "📷 Photo 1 (Primary)",
                type=["jpg", "jpeg", "png", "webp"],
                key="photo_slot_1"
            )
            if photo_1 is not None:
                photo_b64_list.append(image_to_base64(photo_1))
                cap1 = st.text_input(
                    "Photo 1 Caption",
                    value="కార్యక్రమంలో పాల్గొన్న దృశ్యం / Event highlight (1)",
                    key="cap_slot_1"
                )
                captions.append(cap1)

        with p_col2:
            photo_2 = st.file_uploader(
                "📷 Photo 2 (Optional)",
                type=["jpg", "jpeg", "png", "webp"],
                key="photo_slot_2"
            )
            if photo_2 is not None:
                photo_b64_list.append(image_to_base64(photo_2))
                cap2 = st.text_input(
                    "Photo 2 Caption",
                    value="సమావేశంలో మాట్లాడుతున్న దృశ్యం / Event highlight (2)",
                    key="cap_slot_2"
                )
                captions.append(cap2)

    with tab_video:
        st.markdown("##### 🎥 Attach News Video (Clickable Playable Frame in PDF)")
        uploaded_video = st.file_uploader(
            "Upload Video File (MP4, MOV, WEBM)",
            type=["mp4", "mov", "webm"],
            key="news_video_file"
        )
        video_play_url = st.text_input(
            "🔗 Video Play Link (YouTube / Google Drive / Facebook / X URL)",
            placeholder="https://youtu.be/... or https://drive.google.com/file/d/.../view",
            help="Tapping the video frame inside the exported PDF immediately opens and plays this video."
        )
        video_caption = st.text_input(
            "Video News Strip Label",
            value="వీడియో కథనం చూడటానికి ఇక్కడ క్లిక్ చేయండి • TAP TO WATCH VIDEO"
        )

        video_thumb_b64 = ""
        if uploaded_video is not None:
            st.video(uploaded_video)
            video_thumb_b64 = extract_video_thumbnail_b64(uploaded_video)

        custom_video_thumb = st.file_uploader(
            "🖼️ Custom Video Cover Image (Optional)",
            type=["jpg", "jpeg", "png", "webp"],
            key="custom_vid_thumb"
        )
        if custom_video_thumb is not None:
            video_thumb_b64 = image_to_base64(custom_video_thumb)

    # TAB 5: OPTIONAL 2ND NEWS STORY TO FILL LOWER BROADSHEET SPACE
    with tab_story2:
        st.session_state["enable_story_2"] = st.checkbox(
            "➕ Include 2nd News Item on This Page (Fills Lower Broadsheet Space)",
            value=st.session_state["enable_story_2"]
        )
        s2_photo_b64 = ""
        s2_photo_cap = ""
        if st.session_state["enable_story_2"]:
            st.session_state["s2_raw_input"] = st.text_area(
                "Paste 2nd News in Telugu or English:",
                value=st.session_state["s2_raw_input"],
                height=90,
                placeholder="Paste brief 2nd news story here..."
            )
            if st.button("⚡ Auto-Translate 2nd News (Telugu ↔ English)", use_container_width=True):
                if st.session_state["s2_raw_input"].strip():
                    with st.spinner("Translating 2nd news item..."):
                        res2, _ = process_bilingual_auto_fit(
                            active_keys,
                            st.session_state["s2_raw_input"],
                            edition_name,
                            55,
                            65,
                            editorial_style
                        )
                        st.session_state["s2_te_headline"] = res2.get("te_headline", "")
                        st.session_state["s2_te_body"] = res2.get("te_body", "")
                        st.session_state["s2_en_headline"] = res2.get("en_headline", "")
                        st.session_state["s2_en_body"] = res2.get("en_body", "")
                        st.rerun()

            st.session_state["s2_te_headline"] = st.text_input("2nd News Telugu Headline", value=st.session_state["s2_te_headline"])
            st.session_state["s2_te_body"] = st.text_area("2nd News Telugu Body", value=st.session_state["s2_te_body"], height=80)
            st.session_state["s2_en_headline"] = st.text_input("2nd News English Headline", value=st.session_state["s2_en_headline"])
            st.session_state["s2_en_body"] = st.text_area("2nd News English Body", value=st.session_state["s2_en_body"], height=80)

            s2_photo = st.file_uploader("📷 Optional Photo for 2nd News", type=["jpg", "jpeg", "png", "webp"], key="s2_photo_slot")
            if s2_photo is not None:
                s2_photo_b64 = image_to_base64(s2_photo, max_width=900)
                s2_photo_cap = st.text_input("2nd News Photo Caption", value="సమావేశ దృశ్యం / Event glimpse")

    footer_credit = st.text_input(
        "Reporter / Bureau Footer Line",
        value="OJAS DIGITAL NEWS NETWORK • POWER OF INFORMATION"
    )

# ----------------- PRINT GRAPHICS & THEME PALETTE -----------------
if "Classic Editorial Newsprint" in paper_theme:
    bg_paper = "#fcfaf5"
    card_bg = "#ffffff"
    accent_primary = "#991b1b"
    accent_gold = "#b45309"
    quote_bg = "#fffbeb"
    quote_border = "#d97706"
elif "Sunday Magazine" in paper_theme:
    bg_paper = "#fffdf7"
    card_bg = "#ffffff"
    accent_primary = "#7f1d1d"
    accent_gold = "#ca8a04"
    quote_bg = "#fef9c3"
    quote_border = "#ca8a04"
else:
    bg_paper = "#ffffff"
    card_bg = "#f8fafc"
    accent_primary = "#991b1b"
    accent_gold = "#b45309"
    quote_bg = "#fef2f2"
    quote_border = "#991b1b"

has_video_section = bool(uploaded_video is not None or video_play_url.strip() or video_thumb_b64)

if is_9_16:
    sheet_width_px = 720
    # 9:16 mobile canvas (minimum 1280px height, or hugs content if Auto-Trim is selected)
    sheet_height_css = "height: auto;" if is_auto_trim else "min-height: 1280px; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "84px"
    photo_single_h = "205px" if not has_video_section else "160px"
    photo_dual_h = "155px" if not has_video_section else "128px"
    video_thumb_h = "145px"
    te_head_size = "20.5px"
    en_head_size = "18px"
    te_body_size = "13.6px"
    en_body_size = "12.8px"
elif is_auto_trim:
    # Auto-Trim Clipping: Wide 980px Broadsheet width, height hugs the exact content with zero blank bottom gap!
    sheet_width_px = 980
    sheet_height_css = "height: auto;"
    logo_max_h = "108px"
    photo_single_h = "275px" if not has_video_section else "210px"
    photo_dual_h = "205px" if not has_video_section else "162px"
    video_thumb_h = "175px"
    te_head_size = "25.5px"
    en_head_size = "22px"
    te_body_size = "15px"
    en_body_size = "14px"
else:
    # Full A4 Broadsheet (980 x 1386 master -> 1960 x 2772 HD)
    sheet_width_px = 980
    sheet_height_css = "min-height: 1386px; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "108px"
    photo_single_h = "275px" if not has_video_section else "210px"
    photo_dual_h = "205px" if not has_video_section else "162px"
    video_thumb_h = "175px"
    te_head_size = "25.5px"
    en_head_size = "22px"
    te_body_size = "15px"
    en_body_size = "14px"

logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="max-height: {logo_max_h}; max-width: 84%; object-fit: contain; display: block; margin: 0 auto;">'
    if logo_b64 else
    f'<div style="font-size:44px; font-weight:900; color:{accent_primary}; letter-spacing:3px; line-height:1;">OJAS</div>'
    f'<div style="font-size:12px; font-weight:800; color:#1e293b; letter-spacing:1.5px; margin-top:2px;">POWER OF INFORMATION • ఓజస్</div>'
)

# Photo Grid HTML
photos_html = ""
if len(photo_b64_list) == 1:
    photos_html = f"""
    <div style="margin: 6px 0 8px 0; border: 1.5px solid #94a3b8; padding: 5px; background: {card_bg}; box-shadow: 0 2px 6px rgba(0,0,0,0.06); position: relative;">
        <div style="position: relative;">
            <img src="data:image/jpeg;base64,{photo_b64_list[0]}" style="width: 100%; height: {photo_single_h}; object-fit: cover; display: block; border: 1px solid #cbd5e1;">
            <div style="position: absolute; bottom: 6px; right: 6px; background: rgba(15, 23, 42, 0.82); color: #fff; font-size: 9px; font-weight: 800; padding: 2px 7px; letter-spacing: 0.6px; border-left: 3px solid {accent_gold};">
                OJAS PHOTO DESK
            </div>
        </div>
        <div style="font-size: 11.5px; color: #1e293b; padding: 4px 3px 0 3px; font-weight: 700; display: flex; align-items: center; gap: 5px;">
            <span style="color:{accent_primary}; font-size:10px;">■</span> {captions[0]}
        </div>
    </div>
    """
elif len(photo_b64_list) >= 2:
    photos_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 6px 0 8px 0;">
        <div style="border: 1.5px solid #94a3b8; padding: 4px; background: {card_bg}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="position: relative;">
                <img src="data:image/jpeg;base64,{photo_b64_list[0]}" style="width: 100%; height: {photo_dual_h}; object-fit: cover; display: block; border: 1px solid #cbd5e1;">
                <div style="position: absolute; top: 5px; left: 5px; background: {accent_primary}; color: #fff; font-size: 8.5px; font-weight: 800; padding: 1.5px 6px;">01</div>
            </div>
            <div style="font-size: 11px; color: #1e293b; padding-top: 3px; font-weight: 700; line-height: 1.25;">
                <span style="color:{accent_primary};">■</span> {captions[0]}
            </div>
        </div>
        <div style="border: 1.5px solid #94a3b8; padding: 4px; background: {card_bg}; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <div style="position: relative;">
                <img src="data:image/jpeg;base64,{photo_b64_list[1]}" style="width: 100%; height: {photo_dual_h}; object-fit: cover; display: block; border: 1px solid #cbd5e1;">
                <div style="position: absolute; top: 5px; left: 5px; background: #1e293b; color: #fff; font-size: 8.5px; font-weight: 800; padding: 1.5px 6px;">02</div>
            </div>
            <div style="font-size: 11px; color: #1e293b; padding-top: 3px; font-weight: 700; line-height: 1.25;">
                <span style="color:{accent_gold};">■</span> {captions[1]}
            </div>
        </div>
    </div>
    """

# Video Box HTML
qr_b64 = generate_qr_base64(video_play_url) if video_play_url.strip() else ""
video_box_html = ""
if has_video_section:
    thumb_visual = (
        f'<img src="data:image/jpeg;base64,{video_thumb_b64}" style="width:100%; height:{video_thumb_h}; object-fit:cover; display:block; filter: brightness(0.84);">'
        if video_thumb_b64 else
        '<div style="width:100%; height:105px; background: linear-gradient(135deg, #0f172a 0%, #991b1b 100%);"></div>'
    )
    qr_html = (
        f'<div style="background:#fff; padding:3px; border-radius:4px; text-align:center; border:1px solid #cbd5e1; flex-shrink:0;">'
        f'<img src="data:image/png;base64,{qr_b64}" style="width:46px; height:46px; display:block;">'
        f'<div style="font-size:7px; font-weight:800; color:#0f172a; margin-top:1px;">SCAN QR</div></div>'
        if qr_b64 else ""
    )
    clean_url_display = video_play_url.strip() if video_play_url.strip() else "Attached Digital Video Clip • OJAS E-Paper"

    video_box_html = f"""
    <div id="video-hotspot" style="margin: 6px 0 8px 0; border: 2px solid {accent_primary}; background: #0f172a; border-radius: 4px; overflow: hidden; position: relative;">
        <div style="position: relative; width: 100%;">
            {thumb_visual}
            <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.22);">
                <div style="width: 46px; height: 46px; border-radius: 50%; background: rgba(220, 38, 38, 0.95); border: 2.5px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.45);">
                    <div style="width: 0; height: 0; border-top: 9px solid transparent; border-bottom: 9px solid transparent; border-left: 15px solid #ffffff; margin-left: 3px;"></div>
                </div>
                <div style="margin-top: 5px; background: rgba(15, 23, 42, 0.92); color: #fef3c7; font-size: 10.5px; font-weight: 800; padding: 3px 11px; border-radius: 16px; border: 1px solid #f59e0b;">
                    ▶ {video_caption}
                </div>
            </div>
        </div>
        <div style="background: #fffbeb; padding: 5px 10px; display: flex; justify-content: space-between; align-items: center; gap: 8px; border-top: 1.5px solid {accent_primary};">
            <div style="flex: 1;">
                <div style="font-size: 9.5px; font-weight: 800; color: {accent_primary}; letter-spacing: 0.4px;">🎬 OJAS MULTIMEDIA VIDEO REPORT</div>
                <div style="font-size: 10.5px; color: #1e293b; font-weight: 600; word-break: break-all;">{clean_url_display}</div>
            </div>
            {qr_html}
        </div>
    </div>
    """

# Pull-Quote / Key Highlight Strip HTML
pull_quote_text = st.session_state.get("pull_quote", "").strip()
pull_quote_html = (
    f"""
    <div style="margin: 6px 0 8px 0; background: {quote_bg}; border-left: 4px solid {quote_border}; border-right: 4px solid {quote_border}; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; padding: 5px 12px; text-align: center; font-size: 12.5px; font-weight: 800; color: #0f172a; letter-spacing: 0.2px;">
        ❝ {pull_quote_text} ❞
    </div>
    """
    if pull_quote_text else ""
)

# Clean any stray asterisks from displayed text
te_head_clean = st.session_state["te_headline"].replace("*", "").strip()
en_head_clean = st.session_state["en_headline"].replace("*", "").strip()
te_sub_clean = st.session_state["te_subhead"].replace("*", "").strip()
en_sub_clean = st.session_state["en_subhead"].replace("*", "").strip()

te_clean_lines = [line.replace("*", "").strip() for line in st.session_state["te_body"].split("\n") if line.strip()]
en_clean_lines = [line.replace("*", "").strip() for line in st.session_state["en_body"].split("\n") if line.strip()]
te_body_formatted = "<br>".join(te_clean_lines)
en_body_formatted = "<br>".join(en_clean_lines)

te_sub_html = f'<div style="font-size: 13.5px; font-weight: 700; color: {accent_primary}; margin-bottom: 5px; line-height: 1.3; border-left: 3px solid {accent_primary}; padding-left: 7px;">{te_sub_clean}</div>' if te_sub_clean else ""
en_sub_html = f'<div style="font-size: 12.8px; font-weight: 700; color: {accent_gold}; margin-bottom: 5px; line-height: 1.3; border-left: 3px solid {accent_gold}; padding-left: 7px;">{en_sub_clean}</div>' if en_sub_clean else ""

if layout_mode == "Bilingual Side-by-Side (Telugu + English)":
    news_columns_html = f"""
    <div style="display: grid; grid-template-columns: 1.08fr 1fr; gap: 16px; margin-top: 6px;">
        <div style="border-right: 1.5px solid #cbd5e1; padding-right: 14px;">
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                <span style="background:{accent_primary}; color:#fff; font-size:9.5px; font-weight:800; padding:1.5px 7px; border-radius:2px;">తెలుగు వార్త</span>
                <span style="flex:1; height:1px; background:#cbd5e1;"></span>
            </div>
            <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.24; margin: 0 0 5px 0;">
                {te_head_clean}
            </h2>
            {te_sub_html}
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.52; color: #1e293b; text-align: justify;">
                {te_body_formatted}
            </div>
        </div>
        <div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
                <span style="background:#1e293b; color:#fff; font-size:9.5px; font-weight:800; padding:1.5px 7px; border-radius:2px; letter-spacing:0.4px;">ENGLISH REPORT</span>
                <span style="flex:1; height:1px; background:#cbd5e1;"></span>
            </div>
            <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.22; margin: 0 0 5px 0;">
                {en_head_clean}
            </h2>
            {en_sub_html}
            <div class="en-dropcap" style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: {en_body_size}; line-height: 1.46; color: #1e293b; text-align: justify;">
                {en_body_formatted}
            </div>
        </div>
    </div>
    """
elif layout_mode == "Telugu Lead + English Box":
    col_count_te = 2 if not is_9_16 else 1
    news_columns_html = f"""
    <div style="margin-top: 6px;">
        <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.24; margin: 0 0 5px 0;">
            {te_head_clean}
        </h2>
        {te_sub_html}
        <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.54; color: #1e293b; text-align: justify; column-count: {col_count_te}; column-gap: 18px; column-rule: 1px solid #cbd5e1;">
            {te_body_formatted}
        </div>
        <div style="margin-top: 10px; background: {card_bg}; border-left: 4px solid {accent_primary}; border-top: 1px solid #cbd5e1; border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 9px 12px;">
            <div style="font-size: 9.5px; font-weight: 800; color: {accent_primary}; letter-spacing: 0.8px; margin-bottom: 2px;">ENGLISH EDITION</div>
            <div style="font-family: 'Merriweather', Georgia, serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 3px; line-height: 1.22;">{en_head_clean}</div>
            <div style="font-size: {en_body_size}; line-height: 1.44; color: #334155; text-align: justify;">{en_body_formatted}</div>
        </div>
    </div>
    """
else:
    col_count_en = 2 if not is_9_16 else 1
    news_columns_html = f"""
    <div style="margin-top: 6px;">
        <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.22; margin: 0 0 5px 0;">
            {en_head_clean}
        </h2>
        {en_sub_html}
        <div class="en-dropcap" style="font-size: {en_body_size}; line-height: 1.48; color: #1e293b; text-align: justify; column-count: {col_count_en}; column-gap: 18px; column-rule: 1px solid #cbd5e1;">
            {en_body_formatted}
        </div>
        <div style="margin-top: 10px; background: {quote_bg}; border-left: 4px solid {accent_gold}; border-top: 1px solid #fde68a; border-right: 1px solid #fde68a; border-bottom: 1px solid #fde68a; padding: 9px 12px;">
            <div style="font-size: 10px; font-weight: 800; color: {accent_gold}; margin-bottom: 2px;">తెలుగు సంచిక</div>
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: 17px; font-weight: 800; color: #0f172a; margin-bottom: 3px; line-height: 1.24;">{te_head_clean}</div>
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.5; color: #334155; text-align: justify;">{te_body_formatted}</div>
        </div>
    </div>
    """

# Optional 2nd News Item HTML Block (Fills lower broadsheet space seamlessly)
story2_html = ""
if st.session_state.get("enable_story_2", False):
    s2_te_h = st.session_state.get("s2_te_headline", "").replace("*", "").strip()
    s2_te_b = st.session_state.get("s2_te_body", "").replace("*", "").strip()
    s2_en_h = st.session_state.get("s2_en_headline", "").replace("*", "").strip()
    s2_en_b = st.session_state.get("s2_en_body", "").replace("*", "").strip()

    s2_img_block = (
        f'<div style="margin-bottom:6px; border:1px solid #cbd5e1; padding:3px; background:#fff;">'
        f'<img src="data:image/jpeg;base64,{s2_photo_b64}" style="width:100%; height:130px; object-fit:cover; display:block;">'
        f'<div style="font-size:10px; font-weight:700; color:#334155; padding-top:2px;">▲ {s2_photo_cap}</div></div>'
        if s2_photo_b64 else ""
    )

    story2_html = f"""
    <div style="margin-top: 12px; padding-top: 10px; border-top: 2.5px double #1e293b;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="background:#1e293b; color:#fff; font-size:9.5px; font-weight:800; padding:2px 8px; letter-spacing:0.5px;">OTHER KEY REPORT • మరిన్ని ముఖ్య వార్తలు</span>
            <span style="flex:1; height:1px; background:#94a3b8;"></span>
        </div>
        {s2_img_block}
        <div style="display: grid; grid-template-columns: 1.08fr 1fr; gap: 16px;">
            <div style="border-right: 1px solid #cbd5e1; padding-right: 14px;">
                <h3 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: 17px; font-weight: 800; color: #0f172a; margin: 0 0 4px 0; line-height: 1.25;">{s2_te_h}</h3>
                <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: 13.5px; line-height: 1.48; color: #1e293b; text-align: justify;">{s2_te_b}</div>
            </div>
            <div>
                <h3 style="font-family: 'Merriweather', Georgia, serif; font-size: 15.5px; font-weight: 900; color: #0f172a; margin: 0 0 4px 0; line-height: 1.24;">{s2_en_h}</h3>
                <div style="font-size: 12.8px; line-height: 1.44; color: #1e293b; text-align: justify;">{s2_en_b}</div>
            </div>
        </div>
    </div>
    """

safe_filename = f"OJAS_News_{pub_date.strftime('%Y%m%d')}"
escaped_video_url = json.dumps(video_play_url.strip())

# ----------------- RIGHT COLUMN: LIVE PREVIEW & AUTO-SIZED JPEG / PDF EXPORT -----------------
with col_preview:
    st.subheader("🗞️ 2. Print-Grade E-Paper Preview & Export")

    newspaper_widget_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Merriweather:wght@700;900&family=Noto+Sans+Telugu:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            margin: 0;
            padding: 4px;
            background: transparent;
            font-family: 'Plus Jakarta Sans', 'Noto Sans Telugu', sans-serif;
            overflow-x: hidden;
        }}
        .toolbar {{
            display: flex;
            gap: 10px;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }}
        .btn {{
            flex: 1;
            min-width: 150px;
            padding: 11px 14px;
            border: none;
            border-radius: 8px;
            font-size: 13.5px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.12);
        }}
        .btn-jpg {{
            background: linear-gradient(90deg, #991b1b, #dc2626);
            color: #ffffff;
        }}
        .btn-pdf {{
            background: linear-gradient(90deg, #1e293b, #334155);
            color: #ffffff;
        }}
        #status-msg {{
            font-size: 12px;
            font-weight: 700;
            color: #15803d;
            margin-bottom: 6px;
            text-align: center;
            min-height: 16px;
        }}
        #preview-viewport {{
            width: 100%;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }}
        #scale-holder {{
            transform-origin: top left;
        }}
        /* Authentic Print-Grade Newspaper Broadsheet Frame */
        #newspaper-sheet {{
            width: {sheet_width_px}px;
            {sheet_height_css}
            background-color: {bg_paper};
            border: 3px double #1e293b;
            outline: 1px solid {accent_primary};
            outline-offset: -6px;
            padding: 16px 22px 12px 22px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.16);
            color: #0f172a;
        }}
        .masthead-header {{
            display: grid;
            grid-template-columns: 110px 1fr 110px;
            align-items: center;
            gap: 8px;
            padding-bottom: 5px;
        }}
        .edition-box-left {{
            border: 1px solid #94a3b8;
            background: {card_bg};
            padding: 4px 6px;
            font-size: 9px;
            font-weight: 800;
            color: #1e293b;
            line-height: 1.3;
            text-align: left;
        }}
        .edition-box-right {{
            border: 1px solid #94a3b8;
            background: {card_bg};
            padding: 4px 6px;
            font-size: 9px;
            font-weight: 800;
            color: {accent_primary};
            line-height: 1.3;
            text-align: right;
        }}
        .masthead-wrap {{
            text-align: center;
        }}
        .dateline-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 3px solid {accent_primary};
            border-bottom: 1.5px solid #1e293b;
            background: linear-gradient(90deg, rgba(153,27,27,0.05) 0%, rgba(217,119,6,0.06) 50%, rgba(153,27,27,0.05) 100%);
            padding: 4px 6px;
            font-size: 11px;
            font-weight: 800;
            color: #1e293b;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
            gap: 6px;
        }}
        .section-pill {{
            background: linear-gradient(90deg, {accent_primary}, #b91c1c);
            color: #ffffff;
            padding: 2px 10px;
            font-size: 10px;
            font-weight: 800;
            border-radius: 2px;
            letter-spacing: 0.5px;
        }}
        .en-dropcap::first-letter {{
            float: left;
            font-family: 'Merriweather', Georgia, serif;
            font-size: 36px;
            line-height: 0.85;
            padding-top: 3px;
            padding-right: 6px;
            padding-left: 1px;
            font-weight: 900;
            color: {accent_primary};
        }}
        .ornament-rule {{
            text-align: center;
            font-size: 10px;
            color: {accent_gold};
            letter-spacing: 4px;
            margin-top: 8px;
            line-height: 1;
        }}
        .footer-strip {{
            margin-top: 4px;
            padding-top: 5px;
            border-top: 2px solid {accent_primary};
            display: flex;
            justify-content: space-between;
            font-size: 10.5px;
            font-weight: 800;
            color: #334155;
        }}
    </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn btn-jpg" onclick="downloadJPEG()">🖼️ Download HD JPEG</button>
            <button class="btn btn-pdf" onclick="downloadPDF()">📄 Download Exact-Fit PDF</button>
        </div>
        <div id="status-msg"></div>

        <div id="preview-viewport">
            <div id="scale-holder">
                <div id="newspaper-sheet">
                    <div>
                        <div class="masthead-header">
                            <div class="edition-box-left">
                                DIGITAL BROADSHEET<br>
                                <span style="color:{accent_primary};">BILINGUAL EDITION</span><br>
                                TELUGU • ENGLISH
                            </div>
                            <div class="masthead-wrap">
                                {logo_html}
                            </div>
                            <div class="edition-box-right">
                                OJAS MEDIA<br>
                                <span style="color:#1e293b;">VERIFIED NEWS</span><br>
                                {telugu_date_str}
                            </div>
                        </div>

                        <div class="dateline-bar">
                            <span>{edition_name}</span>
                            <span class="section-pill">{category_tag}</span>
                            <span>{formatted_date}</span>
                        </div>

                        {photos_html}

                        {video_box_html}

                        {pull_quote_html}

                        {news_columns_html}

                        {story2_html}
                    </div>

                    <div>
                        <div class="ornament-rule">❖ ◆ ❖</div>
                        <div class="footer-strip">
                            <span>{vol_issue}</span>
                            <span>{footer_credit}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const MASTER_WIDTH = {sheet_width_px};
            const VIDEO_URL = {escaped_video_url};

            function fitPreviewToViewport() {{
                const viewport = document.getElementById('preview-viewport');
                const holder = document.getElementById('scale-holder');
                const sheet = document.getElementById('newspaper-sheet');
                if (!viewport || !holder || !sheet) return;

                const availW = viewport.clientWidth - 4;
                const scale = Math.min(1, availW / MASTER_WIDTH);
                holder.style.transform = `scale(${{scale}})`;
                holder.style.width = MASTER_WIDTH + 'px';
                const scaledH = sheet.offsetHeight * scale;
                holder.style.height = scaledH + 'px';
                viewport.style.height = (scaledH + 10) + 'px';
            }}

            window.addEventListener('load', () => {{
                fitPreviewToViewport();
                setTimeout(fitPreviewToViewport, 350);
            }});
            window.addEventListener('resize', fitPreviewToViewport);

            async function renderMasterCanvas() {{
                const status = document.getElementById('status-msg');
                status.innerText = "⏳ Rendering high-resolution print edition...";
                await document.fonts.ready;

                const holder = document.getElementById('scale-holder');
                const prevTransform = holder.style.transform;
                holder.style.transform = "scale(1)";

                const element = document.getElementById('newspaper-sheet');
                const exactHeight = element.offsetHeight;
                const canvas = await html2canvas(element, {{
                    scale: 2,
                    width: MASTER_WIDTH,
                    height: exactHeight,
                    windowWidth: MASTER_WIDTH + 40,
                    useCORS: true,
                    backgroundColor: "{bg_paper}",
                    logging: false
                }});

                holder.style.transform = prevTransform;
                fitPreviewToViewport();
                return canvas;
            }}

            async function downloadJPEG() {{
                try {{
                    const canvas = await renderMasterCanvas();
                    const link = document.createElement('a');
                    link.download = "{safe_filename}.jpg";
                    link.href = canvas.toDataURL("image/jpeg", 0.96);
                    link.click();
                    document.getElementById('status-msg').innerText = "✅ Zero-blank-space HD JPEG downloaded!";
                }} catch (err) {{
                    document.getElementById('status-msg').innerText = "⚠️ Error: " + err;
                }}
            }}

            async function downloadPDF() {{
                try {{
                    const holder = document.getElementById('scale-holder');
                    holder.style.transform = "scale(1)";

                    const sheetEl = document.getElementById('newspaper-sheet');
                    const sheetRect = sheetEl.getBoundingClientRect();
                    const videoEl = document.getElementById('video-hotspot');
                    let videoRect = videoEl ? videoEl.getBoundingClientRect() : null;

                    const canvas = await renderMasterCanvas();
                    const imgData = canvas.toDataURL("image/jpeg", 0.96);
                    const {{ jsPDF }} = window.jspdf;

                    // Dynamically match PDF page height to the exact captured sheet ratio so zero blank space remains!
                    const pdfWidthMm = 210; // Standard A4 width in mm
                    const pdfHeightMm = (canvas.height * pdfWidthMm) / canvas.width;

                    const pdf = new jsPDF({{
                        orientation: pdfHeightMm >= pdfWidthMm ? "portrait" : "landscape",
                        unit: "mm",
                        format: [pdfWidthMm, pdfHeightMm]
                    }});

                    pdf.addImage(imgData, "JPEG", 0, 0, pdfWidthMm, pdfHeightMm);

                    if (videoRect && VIDEO_URL) {{
                        const scaleX = pdfWidthMm / sheetRect.width;
                        const scaleY = pdfHeightMm / sheetRect.height;
                        pdf.link(
                            (videoRect.left - sheetRect.left) * scaleX,
                            (videoRect.top - sheetRect.top) * scaleY,
                            videoRect.width * scaleX,
                            videoRect.height * scaleY,
                            {{ url: VIDEO_URL }}
                        );
                    }}

                    pdf.save("{safe_filename}.pdf");
                    document.getElementById('status-msg').innerText = "✅ Exact-fit PDF downloaded (zero blank gap)!";
                }} catch (err) {{
                    document.getElementById('status-msg').innerText = "⚠️ Error: " + err;
                }}
            }}
        </script>
    </body>
    </html>
    """

    components.html(newspaper_widget_html, height=1180, scrolling=True)
