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

# Optional Gemini SDK for Chief-Editor AI bilingual translation & condensation
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="OJAS - Power of Information | E-Paper Studio",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="auto"
)

# ----------------- HELPER FUNCTIONS -----------------
def load_logo_base64() -> str:
    """Loads OJAS.png from repository root and converts to base64."""
    for fname in ["OJAS.png", "ojas.png", "OJAS.jpg", "ojas.jpg"]:
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
    """Generates a clean QR code image as base64 PNG for playable video links in PDF/print."""
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

def detect_telugu(text: str) -> bool:
    """Detects whether the input text is predominantly in Telugu script."""
    te_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    en_chars = len(re.findall(r'[A-Za-z]', text))
    return te_chars > en_chars

def free_translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Free unlimited fallback translator between Telugu ('te') and English ('en')."""
    if not text.strip():
        return ""
    try:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={urllib.parse.quote(text[:4500])}"
        )
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        data = res.json()
        return "".join([seg[0] for seg in data[0] if seg[0]]).strip()
    except Exception:
        return text

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

def ai_editorial_bilingual_engine(
    keys: list,
    raw_input: str,
    location: str,
    target_te_words: int,
    target_en_words: int,
    editorial_style: str
) -> dict:
    """
    Chief-Editor AI Newsroom Engine:
    - Auto-detects Telugu or English input
    - Translates & transcreates into authentic Telugu (పత్రికా భాష) AND crisp English
    - Applies Inverted Pyramid editing & word-budget condensation to fit 1 single page cleanly
    """
    last_err = None
    models_pool = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-3.5-flash-lite"]

    prompt = f"""
    You are the Chief News Editor & Copy Desk Director of 'OJAS - Power of Information' (ఓజస్), a premier Telugu-English bilingual daily newspaper.

    TASK:
    1. Detect the language of the RAW NEWS INPUT below (it may be in Telugu, English, or mixed Tanglish).
    2. If the input is in Telugu -> retain/refine the Telugu into crisp journalistic Telugu (పత్రికా భాష) AND translate/transcreate it into sharp English newspaper copy.
    3. If the input is in English -> refine it into sharp English newspaper copy AND translate/transcreate it into authentic journalistic Telugu (పత్రికా భాష).
    4. SUPERFICIAL & STRUCTURAL NEWS EDITING TECHNIQUES:
       - Apply the Inverted Pyramid rule (Lead with Who, What, Where, When, Impact in sentence 1).
       - Cut redundant filler words, passive phrasing, and repetitive greetings so the entire bilingual story fits comfortably on ONE SINGLE PAGE without overflowing.
       - Editorial Angle: {editorial_style}
       - Dateline / Place: {location}
       - STRICT WORD BUDGET: Keep `te_body` around {target_te_words} words (maximum {target_te_words + 15} words) and `en_body` around {target_en_words} words (maximum {target_en_words + 15} words). Use 1 or 2 tight paragraphs with NO blank lines between them.

    RAW NEWS INPUT:
    {raw_input}

    Return STRICTLY valid JSON with these exact keys (no markdown fences):
    {{
        "te_headline": "Punchy 5-8 word journalistic Telugu headline",
        "te_subhead": "Crisp 1-line Telugu deck sub-headline",
        "te_body": "Edited journalistic Telugu body within {target_te_words} words",
        "en_headline": "Punchy 6-10 word journalistic English headline",
        "en_subhead": "Crisp 1-line English deck sub-headline",
        "en_body": "Edited journalistic English body within {target_en_words} words"
    }}
    """

    for key in keys:
        if not key or not key.strip():
            continue
        try:
            client = genai.Client(api_key=key.strip())
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
                        return json.loads(res.text)
                except Exception as e:
                    last_err = e
                    time.sleep(0.6)
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
) -> dict:
    """Runs AI Chief Editor if API key is present; otherwise runs free two-way translation + smart single-page condenser."""
    valid_keys = [k for k in keys if k and k.strip()]
    if valid_keys and HAS_GENAI:
        try:
            return ai_editorial_bilingual_engine(
                valid_keys, raw_text, location, target_te_words, target_en_words, editorial_style
            )
        except Exception as e:
            st.warning(f"AI key busy ({e}) — switching to instant free bilingual translator & condenser.")

    # Free Unlimited Fallback (Zero API Key Needed)
    is_te = detect_telugu(raw_text)
    lines = [l.strip() for l in raw_text.strip().split("\n") if l.strip()]
    first_line = lines[0] if lines else raw_text.strip()[:70]
    body_raw = " ".join(lines[1:]) if len(lines) > 1 else raw_text.strip()

    if is_te:
        te_head = first_line[:90]
        te_body = smart_condense_to_word_budget(body_raw, target_te_words)
        en_head = free_translate_text(te_head, "te", "en")
        en_body_full = free_translate_text(body_raw, "te", "en")
        en_body = smart_condense_to_word_budget(en_body_full, target_en_words)
    else:
        en_head = first_line[:95]
        en_body = smart_condense_to_word_budget(body_raw, target_en_words)
        te_head = free_translate_text(en_head, "en", "te")
        te_body_full = free_translate_text(body_raw, "en", "te")
        te_body = smart_condense_to_word_budget(te_body_full, target_te_words)

    return {
        "te_headline": te_head,
        "te_subhead": "",
        "te_body": te_body,
        "en_headline": en_head,
        "en_subhead": "",
        "en_body": en_body,
    }

# ----------------- CUSTOM RESPONSIVE APP STYLING -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    html, body, p, div:not([data-testid="stIconMaterial"]), input, textarea, button {
        font-family: 'Plus Jakarta Sans', 'Noto Sans Telugu', sans-serif !important;
    }
    span[data-testid="stIconMaterial"], .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }
    .top-brand-bar {
        background: linear-gradient(90deg, #991b1b 0%, #b91c1c 60%, #d97706 100%);
        color: white;
        padding: 14px 22px;
        border-radius: 12px;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(153, 27, 27, 0.2);
    }
    .top-brand-bar h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 21px !important;
        font-weight: 800 !important;
    }
    .top-brand-bar p {
        color: #fef3c7 !important;
        margin: 3px 0 0 0 !important;
        font-size: 13px !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #991b1b 0%, #dc2626 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 20px !important;
    }
    @media screen and (max-width: 768px) {
        .top-brand-bar { padding: 11px 14px !important; }
        .top-brand-bar h1 { font-size: 17px !important; }
        .block-container { padding-left: 0.65rem !important; padding-right: 0.65rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
defaults = {
    "te_headline": "ప్రజా సమస్యల పరిష్కారమే లక్ష్యంగా ప్రత్యేక కార్యాచరణ",
    "te_subhead": "అధికారులు క్షేత్రస్థాయిలో పర్యటించి తక్షణ చర్యలు చేపట్టాలి",
    "te_body": "కోదాడ, న్యూస్‌టుడే: పట్టణ, గ్రామీణ ప్రాంతాల్లో ప్రజలు ఎదుర్కొంటున్న మౌలిక వసతుల సమస్యలను యుద్ధప్రాతిపదికన పరిష్కరించేందుకు ప్రత్యేక ప్రణాళిక సిద్ధం చేసినట్లు అధికారులు వెల్లడించారు. తాగునీరు, రహదారులు, పారిశుద్ధ్య పనులపై ప్రత్యేక దృష్టి సారించి గడువులోగా పూర్తి చేయాలని ఆదేశించారు.",
    "en_headline": "Special Action Plan Launched to Resolve Public Grievances",
    "en_subhead": "Officials directed to conduct field visits and ensure swift redressal",
    "en_body": "KODAD: Authorities have rolled out a dedicated action plan to resolve civic and infrastructure issues across urban and rural pockets on a war footing. Officials have been instructed to prioritize drinking water supply, road repairs, and sanitation works with strict timelines.",
    "raw_bilingual_input": ""
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- SIDEBAR: EDITION & VIEWPORT SETTINGS -----------------
with st.sidebar:
    logo_b64 = load_logo_base64()
    if logo_b64:
        st.markdown(
            f'<div style="background:#fff; padding:8px; border-radius:10px; border:1px solid #e2e8f0; text-align:center; margin-bottom:10px;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="max-width:100%; height:auto;"></div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Upload `OJAS.png` to your GitHub repo to display the masthead logo.")

    st.subheader("📐 1. Single-Page Format & Ratio")
    aspect_preset = st.radio(
        "Choose Edition Ratio:",
        [
            "📰 Broadsheet E-Paper (A4 / iPad / Computer — 1 Page)",
            "📱 9:16 Vertical E-Paper (Mobile / WhatsApp Status — 1 Page)"
        ],
        index=0
    )

    editorial_style = st.selectbox(
        "🧠 AI News Editing Technique",
        [
            "Inverted Pyramid • Crisp Breaking News",
            "Print Daily Standard • Eenadu / Sakshi Style",
            "Official Statement • Authoritative Press Tone",
            "Civic & Welfare Spotlight • Human-Interest Lead"
        ]
    )

    st.subheader("🗞️ 2. Edition & Masthead Details")
    pub_date = st.date_input("Publication Date", value=datetime.date.today())
    formatted_date = pub_date.strftime("%A, %d %B %Y")
    telugu_date_str = pub_date.strftime("%d-%m-%Y")

    edition_name = st.text_input("Edition / Bureau", value="TELANGANA EDITION • KODAD")
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
        "Paper Tone",
        ["Classic Crisp White", "Warm Editorial Newsprint"]
    )

    st.divider()
    with st.expander("🔑 Gemini API Key (Optional for AI Copy-Editor)"):
        st.caption("Uses secret key automatically if configured. Even without a key, free Telugu ↔ English auto-translation works unlimitedly.")
        secret_key = ""
        backup_secret = ""
        try:
            if hasattr(st, "secrets"):
                secret_key = st.secrets.get("GEMINI_API_KEY", "")
                backup_secret = st.secrets.get("BACKUP_API_KEY", "")
        except Exception:
            pass
        gemini_key = st.text_input("Gemini API Key", value=secret_key, type="password")

active_keys = [gemini_key]
if backup_secret and backup_secret != gemini_key:
    active_keys.append(backup_secret)

# ----------------- MAIN WORKSPACE -----------------
st.markdown("""
<div class="top-brand-bar">
    <h1>📰 OJAS — Power of Information (ఓజస్) E-Paper Studio</h1>
    <p>Auto Telugu ↔ English Translation • AI Single-Page Condenser • 1–2 Photos + Playable Video PDF</p>
</div>
""", unsafe_allow_html=True)

col_editor, col_preview = st.columns([0.95, 1.35], gap="large")

with col_editor:
    st.subheader("✍️ 1. Smart Bilingual News Desk")

    # Quick Tanglish-to-Telugu Converter
    with st.expander("🔤 Type English Phonetics ➔ Instant Telugu (100% Free)", expanded=False):
        phonetic_in = st.text_input(
            "Type Telugu in English letters:",
            placeholder="e.g., Kodad lo nedu pratyeka samavesam jarigindi"
        )
        if st.button("Convert to Telugu Script"):
            if phonetic_in.strip():
                st.code(transliterate_to_telugu(phonetic_in), language=None)

    # ONE-BOX AUTO-TRANSLATE & SINGLE-PAGE CONDENSER
    st.markdown("##### ⚡ Paste News in Telugu OR English (Auto-Translates Both & Fits 1 Page)")
    raw_news_input = st.text_area(
        "Paste raw report in Telugu or English:",
        value=st.session_state["raw_bilingual_input"],
        height=135,
        placeholder="Paste long or short news in Telugu OR English here. Clicking below automatically translates into the other language, removes fluff, and condenses to fit cleanly on 1 single page..."
    )

    is_9_16 = "9:16" in aspect_preset
    # Calculate tight word budgets based on selected page format
    target_te_words = 75 if is_9_16 else 115
    target_en_words = 85 if is_9_16 else 130

    if st.button("⚡ Auto-Translate (Telugu ↔ English) & Fit to Single Page", type="primary", use_container_width=True):
        if not raw_news_input.strip():
            st.warning("Please enter or paste your news text in Telugu or English first.")
        else:
            st.session_state["raw_bilingual_input"] = raw_news_input
            with st.spinner("Applying newsroom editing, two-way translation & single-page fit..."):
                result = process_bilingual_auto_fit(
                    active_keys,
                    raw_news_input,
                    edition_name,
                    target_te_words,
                    target_en_words,
                    editorial_style
                )
                for k in ["te_headline", "te_subhead", "te_body", "en_headline", "en_subhead", "en_body"]:
                    if k in result:
                        st.session_state[k] = result[k]
                st.success("✅ Translated & condensed to fit 1 single page!")
                st.rerun()

    st.write("")
    tab_te, tab_en, tab_photos, tab_video = st.tabs([
        "తెలుగు (Edit)",
        "🇬🇧 English (Edit)",
        "📸 Photos (Max 2)",
        "🎬 Video"
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
            height=165
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
            height=165
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

    footer_credit = st.text_input(
        "Reporter / Bureau Footer Line",
        value="OJAS DIGITAL NEWS NETWORK • POWER OF INFORMATION"
    )

# ----------------- COMPACT SINGLE-PAGE GEOMETRY & TYPOGRAPHY -----------------
has_video_section = bool(uploaded_video is not None or video_play_url.strip() or video_thumb_b64)

if is_9_16:
    # Exact 9:16 Single-Page Mobile / WhatsApp Status Canvas (720 x 1280 master -> 1440 x 2560 HD)
    sheet_width_px = 720
    sheet_height_css = "height: 1280px; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "86px"
    photo_single_h = "215px" if not has_video_section else "170px"
    photo_dual_h = "165px" if not has_video_section else "135px"
    video_thumb_h = "155px"
    te_head_size = "21px"
    en_head_size = "18.5px"
    te_body_size = "13.8px"
    en_body_size = "13px"
    pdf_format_js = "[144, 256]"
else:
    # Exact A4 Proportion Single-Page Broadsheet (980 x 1386 master -> 1960 x 2772 HD)
    sheet_width_px = 980
    sheet_height_css = "height: 1386px; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "112px"
    photo_single_h = "290px" if not has_video_section else "220px"
    photo_dual_h = "215px" if not has_video_section else "170px"
    video_thumb_h = "185px"
    te_head_size = "26px"
    en_head_size = "22.5px"
    te_body_size = "15.2px"
    en_body_size = "14.2px"
    pdf_format_js = '"a4"'

bg_paper = "#ffffff" if paper_theme == "Classic Crisp White" else "#fdfbf7"

logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="max-height: {logo_max_h}; max-width: 86%; object-fit: contain; display: block; margin: 0 auto;">'
    if logo_b64 else
    '<div style="font-size:42px; font-weight:900; color:#991b1b; letter-spacing:2px;">OJAS</div><div style="font-size:12px; font-weight:700;">POWER OF INFORMATION • ఓజస్</div>'
)

# Photo Grid HTML (Compact Spacing)
photos_html = ""
if len(photo_b64_list) == 1:
    photos_html = f"""
    <div style="margin: 6px 0 8px 0; border: 1.5px solid #cbd5e1; padding: 4px; background: #fff;">
        <img src="data:image/jpeg;base64,{photo_b64_list[0]}" style="width: 100%; height: {photo_single_h}; object-fit: cover; display: block;">
        <div style="font-size: 11.5px; color: #334155; padding: 3px 2px 0 2px; font-weight: 700;">
            ▲ {captions[0]}
        </div>
    </div>
    """
elif len(photo_b64_list) >= 2:
    photos_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 6px 0 8px 0;">
        <div style="border: 1.5px solid #cbd5e1; padding: 4px; background: #fff;">
            <img src="data:image/jpeg;base64,{photo_b64_list[0]}" style="width: 100%; height: {photo_dual_h}; object-fit: cover; display: block;">
            <div style="font-size: 11px; color: #334155; padding-top: 3px; font-weight: 700; line-height: 1.25;">▲ {captions[0]}</div>
        </div>
        <div style="border: 1.5px solid #cbd5e1; padding: 4px; background: #fff;">
            <img src="data:image/jpeg;base64,{photo_b64_list[1]}" style="width: 100%; height: {photo_dual_h}; object-fit: cover; display: block;">
            <div style="font-size: 11px; color: #334155; padding-top: 3px; font-weight: 700; line-height: 1.25;">▲ {captions[1]}</div>
        </div>
    </div>
    """

# Video Box HTML (Compact Single-Page Proportions)
qr_b64 = generate_qr_base64(video_play_url) if video_play_url.strip() else ""
video_box_html = ""
if has_video_section:
    thumb_visual = (
        f'<img src="data:image/jpeg;base64,{video_thumb_b64}" style="width:100%; height:{video_thumb_h}; object-fit:cover; display:block; filter: brightness(0.84);">'
        if video_thumb_b64 else
        '<div style="width:100%; height:110px; background: linear-gradient(135deg, #0f172a 0%, #991b1b 100%);"></div>'
    )
    qr_html = (
        f'<div style="background:#fff; padding:3px; border-radius:5px; text-align:center; border:1px solid #cbd5e1; flex-shrink:0;">'
        f'<img src="data:image/png;base64,{qr_b64}" style="width:48px; height:48px; display:block;">'
        f'<div style="font-size:7.5px; font-weight:800; color:#0f172a; margin-top:1px;">SCAN</div></div>'
        if qr_b64 else ""
    )
    clean_url_display = video_play_url.strip() if video_play_url.strip() else "Attached Digital Video Clip • OJAS E-Paper"

    video_box_html = f"""
    <div id="video-hotspot" style="margin: 6px 0 8px 0; border: 2px solid #991b1b; background: #0f172a; border-radius: 5px; overflow: hidden; position: relative;">
        <div style="position: relative; width: 100%;">
            {thumb_visual}
            <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.24);">
                <div style="width: 48px; height: 48px; border-radius: 50%; background: rgba(220, 38, 38, 0.95); border: 2.5px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 14px rgba(0,0,0,0.45);">
                    <div style="width: 0; height: 0; border-top: 9px solid transparent; border-bottom: 9px solid transparent; border-left: 16px solid #ffffff; margin-left: 3px;"></div>
                </div>
                <div style="margin-top: 6px; background: rgba(15, 23, 42, 0.9); color: #fef3c7; font-size: 11px; font-weight: 800; padding: 3px 11px; border-radius: 16px; border: 1px solid #f59e0b;">
                    ▶ {video_caption}
                </div>
            </div>
        </div>
        <div style="background: #fffbeb; padding: 5px 10px; display: flex; justify-content: space-between; align-items: center; gap: 8px; border-top: 1.5px solid #991b1b;">
            <div style="flex: 1;">
                <div style="font-size: 10px; font-weight: 800; color: #991b1b;">🎬 OJAS MULTIMEDIA VIDEO</div>
                <div style="font-size: 11px; color: #1e293b; font-weight: 600; word-break: break-all;">{clean_url_display}</div>
            </div>
            {qr_html}
        </div>
    </div>
    """

# Remove extra blank lines so text stays crisp & compact on a single page
te_clean_lines = [line.strip() for line in st.session_state["te_body"].split("\n") if line.strip()]
en_clean_lines = [line.strip() for line in st.session_state["en_body"].split("\n") if line.strip()]
te_body_formatted = "<br>".join(te_clean_lines)
en_body_formatted = "<br>".join(en_clean_lines)

te_sub_html = f'<div style="font-size: 13.5px; font-weight: 700; color: #991b1b; margin-bottom: 5px; line-height: 1.3;">◆ {st.session_state["te_subhead"]}</div>' if st.session_state["te_subhead"].strip() else ""
en_sub_html = f'<div style="font-size: 12.8px; font-weight: 700; color: #b45309; margin-bottom: 5px; line-height: 1.3;">◆ {st.session_state["en_subhead"]}</div>' if st.session_state["en_subhead"].strip() else ""

if layout_mode == "Bilingual Side-by-Side (Telugu + English)":
    news_columns_html = f"""
    <div style="display: grid; grid-template-columns: 1.08fr 1fr; gap: 16px; margin-top: 6px;">
        <div style="border-right: 1.5px solid #cbd5e1; padding-right: 14px;">
            <div style="display:inline-block; background:#991b1b; color:#fff; font-size:9.5px; font-weight:800; padding:1.5px 7px; border-radius:2px; margin-bottom:4px;">తెలుగు</div>
            <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.24; margin: 0 0 4px 0;">
                {st.session_state["te_headline"]}
            </h2>
            {te_sub_html}
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.52; color: #1e293b; text-align: justify;">
                {te_body_formatted}
            </div>
        </div>
        <div>
            <div style="display:inline-block; background:#1e293b; color:#fff; font-size:9.5px; font-weight:800; padding:1.5px 7px; border-radius:2px; margin-bottom:4px; letter-spacing:0.4px;">ENGLISH</div>
            <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.22; margin: 0 0 4px 0;">
                {st.session_state["en_headline"]}
            </h2>
            {en_sub_html}
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: {en_body_size}; line-height: 1.46; color: #1e293b; text-align: justify;">
                {en_body_formatted}
            </div>
        </div>
    </div>
    """
elif layout_mode == "Telugu Lead + English Box":
    col_count_te = 2 if not is_9_16 else 1
    news_columns_html = f"""
    <div style="margin-top: 6px;">
        <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.24; margin: 0 0 4px 0;">
            {st.session_state["te_headline"]}
        </h2>
        {te_sub_html}
        <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.54; color: #1e293b; text-align: justify; column-count: {col_count_te}; column-gap: 18px; column-rule: 1px solid #cbd5e1;">
            {te_body_formatted}
        </div>
        <div style="margin-top: 10px; background: #f8fafc; border-left: 4px solid #991b1b; border-top: 1px solid #cbd5e1; border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 9px 12px;">
            <div style="font-size: 9.5px; font-weight: 800; color: #991b1b; letter-spacing: 0.8px; margin-bottom: 2px;">ENGLISH EDITION</div>
            <div style="font-family: 'Merriweather', Georgia, serif; font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 3px; line-height: 1.22;">{st.session_state["en_headline"]}</div>
            <div style="font-size: {en_body_size}; line-height: 1.44; color: #334155; text-align: justify;">{en_body_formatted}</div>
        </div>
    </div>
    """
else:
    col_count_en = 2 if not is_9_16 else 1
    news_columns_html = f"""
    <div style="margin-top: 6px;">
        <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.22; margin: 0 0 4px 0;">
            {st.session_state["en_headline"]}
        </h2>
        {en_sub_html}
        <div style="font-size: {en_body_size}; line-height: 1.48; color: #1e293b; text-align: justify; column-count: {col_count_en}; column-gap: 18px; column-rule: 1px solid #cbd5e1;">
            {en_body_formatted}
        </div>
        <div style="margin-top: 10px; background: #fffbeb; border-left: 4px solid #b45309; border-top: 1px solid #fde68a; border-right: 1px solid #fde68a; border-bottom: 1px solid #fde68a; padding: 9px 12px;">
            <div style="font-size: 10px; font-weight: 800; color: #b45309; margin-bottom: 2px;">తెలుగు సంచిక</div>
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: 17px; font-weight: 800; color: #0f172a; margin-bottom: 3px; line-height: 1.24;">{st.session_state["te_headline"]}</div>
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.5; color: #334155; text-align: justify;">{te_body_formatted}</div>
        </div>
    </div>
    """

safe_filename = f"OJAS_News_{pub_date.strftime('%Y%m%d')}"
escaped_video_url = json.dumps(video_play_url.strip())

# ----------------- RIGHT COLUMN: SINGLE-PAGE LIVE PREVIEW & HD JPEG / PDF EXPORT -----------------
with col_preview:
    st.subheader("🗞️ 2. Single-Page Newspaper Preview & Export")

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
        /* Fixed Single-Page Master Canvas */
        #newspaper-sheet {{
            width: {sheet_width_px}px;
            {sheet_height_css}
            background-color: {bg_paper};
            border: 2.5px solid #1e293b;
            padding: 16px 22px 12px 22px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.14);
            color: #0f172a;
        }}
        .masthead-wrap {{
            text-align: center;
            padding-bottom: 6px;
        }}
        .dateline-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 2.5px solid #991b1b;
            border-bottom: 1.5px solid #1e293b;
            padding: 4px 4px;
            font-size: 11.5px;
            font-weight: 800;
            color: #1e293b;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
            gap: 6px;
        }}
        .section-pill {{
            background: #991b1b;
            color: #ffffff;
            padding: 2px 9px;
            font-size: 10.5px;
            font-weight: 800;
            border-radius: 2px;
        }}
        .footer-strip {{
            margin-top: 8px;
            padding-top: 6px;
            border-top: 2px solid #991b1b;
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            font-weight: 700;
            color: #475569;
        }}
    </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn btn-jpg" onclick="downloadJPEG()">🖼️ Download Single-Page HD JPEG</button>
            <button class="btn btn-pdf" onclick="downloadPDF()">📄 Download Single-Page PDF</button>
        </div>
        <div id="status-msg"></div>

        <div id="preview-viewport">
            <div id="scale-holder">
                <div id="newspaper-sheet">
                    <div>
                        <div class="masthead-wrap">
                            {logo_html}
                        </div>

                        <div class="dateline-bar">
                            <span>{edition_name}</span>
                            <span class="section-pill">{category_tag}</span>
                            <span>{formatted_date} ({telugu_date_str})</span>
                        </div>

                        {photos_html}

                        {video_box_html}

                        {news_columns_html}
                    </div>

                    <div class="footer-strip">
                        <span>{vol_issue}</span>
                        <span>{footer_credit}</span>
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
                status.innerText = "⏳ Rendering single-page print edition...";
                await document.fonts.ready;

                const holder = document.getElementById('scale-holder');
                const prevTransform = holder.style.transform;
                holder.style.transform = "scale(1)";

                const element = document.getElementById('newspaper-sheet');
                const canvas = await html2canvas(element, {{
                    scale: 2,
                    width: MASTER_WIDTH,
                    height: element.offsetHeight,
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
                    document.getElementById('status-msg').innerText = "✅ Single-page HD JPEG downloaded!";
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

                    const pdf = new jsPDF({{
                        orientation: "portrait",
                        unit: "mm",
                        format: {pdf_format_js}
                    }});

                    const pageWidth = pdf.internal.pageSize.getWidth();
                    const pageHeight = pdf.internal.pageSize.getHeight();

                    // Strict 1-Page Edge-to-Edge Fit
                    pdf.addImage(imgData, "JPEG", 0, 0, pageWidth, pageHeight);

                    if (videoRect && VIDEO_URL) {{
                        const scaleX = pageWidth / sheetRect.width;
                        const scaleY = pageHeight / sheetRect.height;
                        pdf.link(
                            (videoRect.left - sheetRect.left) * scaleX,
                            (videoRect.top - sheetRect.top) * scaleY,
                            videoRect.width * scaleX,
                            videoRect.height * scaleY,
                            {{ url: VIDEO_URL }}
                        );
                    }}

                    pdf.save("{safe_filename}.pdf");
                    document.getElementById('status-msg').innerText = "✅ Single-page PDF downloaded!";
                }} catch (err) {{
                    document.getElementById('status-msg').innerText = "⚠️ Error: " + err;
                }}
            }}
        </script>
    </body>
    </html>
    """

    components.html(newspaper_widget_html, height=1120, scrolling=False)
