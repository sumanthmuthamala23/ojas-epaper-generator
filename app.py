import os
import io
import json
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

# Optional Gemini SDK for instant bilingual translation/headline polish
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

def ai_bilingual_assist(api_key: str, source_text: str, location: str):
    """Generates crisp newspaper headlines and bilingual Telugu + English body."""
    client = genai.Client(api_key=api_key.strip())
    prompt = f"""
    You are the Chief Editor of 'OJAS - Power of Information' (ఓజస్), a premier Telugu-English bilingual newspaper.
    Convert the following raw news notes into a polished, publication-ready bilingual newspaper report.
    Location/Dateline: {location}

    Raw Input:
    {source_text}

    Return STRICTLY valid JSON with these exact keys (no markdown fences):
    {{
        "te_headline": "Punchy journalistic Telugu headline",
        "te_subhead": "Short Telugu sub-headline (1 line)",
        "te_body": "Standard journalistic Telugu news report (2-3 crisp paragraphs)",
        "en_headline": "Punchy journalistic English headline",
        "en_subhead": "Short English sub-headline (1 line)",
        "en_body": "Standard journalistic English news report (2-3 crisp paragraphs)"
    }}
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text)

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
        padding: 16px 24px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(153, 27, 27, 0.22);
    }
    .top-brand-bar h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 22px !important;
        font-weight: 800 !important;
    }
    .top-brand-bar p {
        color: #fef3c7 !important;
        margin: 4px 0 0 0 !important;
        font-size: 13.5px !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #991b1b 0%, #dc2626 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 22px !important;
    }
    @media screen and (max-width: 768px) {
        .top-brand-bar { padding: 12px 15px !important; }
        .top-brand-bar h1 { font-size: 18px !important; }
        .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INIT -----------------
defaults = {
    "te_headline": "ప్రజా సమస్యల పరిష్కారమే లక్ష్యంగా ప్రత్యేక కార్యాచరణ",
    "te_subhead": "అధికారులు క్షేత్రస్థాయిలో పర్యటించి తక్షణ చర్యలు చేపట్టాలి",
    "te_body": "కోదాడ, న్యూస్‌టుడే: పట్టణ మరియు గ్రామీణ ప్రాంతాల్లో ప్రజలు ఎదుర్కొంటున్న మౌలిక వసతుల సమస్యలను యుద్ధప్రాతిపదికన పరిష్కరించేందుకు ప్రత్యేక ప్రణాళిక సిద్ధం చేసినట్లు అధికారులు వెల్లడించారు. తాగునీరు, రహదారులు, పారిశుద్ధ్య పనులపై ప్రత్యేక దృష్టి సారించామన్నారు.",
    "en_headline": "Special Action Plan Launched to Resolve Public Grievances",
    "en_subhead": "Officials directed to conduct field visits and ensure swift redressal",
    "en_body": "KODAD: Authorities have rolled out a dedicated action plan aimed at resolving civic and infrastructure challenges across urban and rural pockets on a war footing. Priority is being accorded to drinking water supply, road repairs, and sanitation works.",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- SIDEBAR: EDITION & VIEWPORT SETTINGS -----------------
with st.sidebar:
    logo_b64 = load_logo_base64()
    if logo_b64:
        st.markdown(
            f'<div style="background:#fff; padding:10px; border-radius:10px; border:1px solid #e2e8f0; text-align:center; margin-bottom:12px;">'
            f'<img src="data:image/png;base64,{logo_b64}" style="max-width:100%; height:auto;"></div>',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Upload `OJAS.png` to your GitHub repo to display the masthead logo.")

    st.subheader("📐 1. Page Aspect Ratio & Screen Fit")
    aspect_preset = st.radio(
        "Choose Edition Aspect Ratio:",
        [
            "📰 Broadsheet E-Paper (A4 / iPad / Computer — Wide)",
            "📱 9:16 Vertical E-Paper (Mobile / WhatsApp Status)",
            "📐 9:16 Tall Broadsheet (Full Article Vertical)"
        ],
        index=0,
        help="Broadsheet gives a wide, balanced newspaper page on iPad, PC, and A4 PDF without thin vertical stretching. 9:16 optimizes for phone screens & WhatsApp Status."
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
    with st.expander("✨ Optional AI Bilingual Auto-Writer"):
        st.caption("Paste rough notes in English or Telugu to auto-generate both languages. Leave blank for 100% manual unlimited mode.")
        secret_key = ""
        try:
            if hasattr(st, "secrets"):
                secret_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
        gemini_key = st.text_input("Gemini API Key (Optional)", value=secret_key, type="password")

# ----------------- MAIN WORKSPACE -----------------
st.markdown("""
<div class="top-brand-bar">
    <h1>📰 OJAS — Power of Information (ఓజస్) E-Paper Studio</h1>
    <p>Broadsheet (iPad / Desktop / A4 PDF) & 9:16 Mobile E-Paper • 1–2 Photos + Playable Video PDF</p>
</div>
""", unsafe_allow_html=True)

col_editor, col_preview = st.columns([0.95, 1.35], gap="large")

with col_editor:
    st.subheader("✍️ 1. News Content, Photos & Video")

    if gemini_key and HAS_GENAI:
        with st.expander("🤖 Quick AI Bilingual Translator & Editor", expanded=False):
            raw_story = st.text_area(
                "Enter rough news notes (in English, Telugu, or Tanglish):",
                placeholder="Example: Today MLA inaugurated new CC road in Kodad town, many locals attended...",
                height=90
            )
            if st.button("⚡ Auto-Fill Telugu & English Columns", use_container_width=True):
                if raw_story.strip():
                    with st.spinner("Drafting bilingual newspaper report..."):
                        try:
                            res_json = ai_bilingual_assist(gemini_key, raw_story, edition_name)
                            for key in defaults.keys():
                                if key in res_json:
                                    st.session_state[key] = res_json[key]
                            st.success("✅ Bilingual news populated! Review below.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI Error: {e}")

    with st.expander("🔤 Type English Phonetics ➔ Instant Telugu (100% Free)", expanded=False):
        phonetic_in = st.text_input(
            "Type Telugu in English letters:",
            placeholder="e.g., Kodad lo nedu pratyeka samavesam jarigindi"
        )
        if st.button("Convert to Telugu Script"):
            if phonetic_in.strip():
                st.code(transliterate_to_telugu(phonetic_in), language=None)

    tab_te, tab_en, tab_photos, tab_video = st.tabs([
        "తెలుగు వార్త",
        "🇬🇧 English News",
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
            height=210
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
            height=210
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
            value="వీడియో కథనం చూడటానికి ఇక్కడ క్లిక్ చేయండి • TAP TO WATCH FULL VIDEO REPORT"
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

# ----------------- CANVAS GEOMETRY & ASPECT RATIO CONFIGURATION -----------------
is_9_16_fixed = "9:16 Vertical E-Paper" in aspect_preset
is_9_16_tall = "9:16 Tall Broadsheet" in aspect_preset

if is_9_16_fixed:
    # Exact 9:16 Mobile / WhatsApp Status Canvas (720 x 1280 master -> exports at 1440 x 2560 HD)
    sheet_width_px = 720
    sheet_min_height_css = "height: 1280px; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "105px"
    photo_single_h = "260px"
    photo_dual_h = "195px"
    te_head_size = "24px"
    en_head_size = "21px"
    te_body_size = "15px"
    en_body_size = "14px"
    pdf_format_js = "[144, 256]"  # Exact 9:16 mm page in PDF!
elif is_9_16_tall:
    # 9:16 Minimum Aspect Ratio (760px wide, min-height 1351px, grows cleanly with long articles)
    sheet_width_px = 760
    sheet_min_height_css = "min-height: 1351px; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "115px"
    photo_single_h = "300px"
    photo_dual_h = "220px"
    te_head_size = "26px"
    en_head_size = "22px"
    te_body_size = "15.5px"
    en_body_size = "14.5px"
    pdf_format_js = '"a4"'
else:
    # Broadsheet E-Paper (980px wide — Authentic Newspaper Proportion for iPad, Computer & A4 PDF)
    sheet_width_px = 980
    sheet_min_height_css = "min-height: 1120px; display: flex; flex-direction: column; justify-content: space-between;"
    logo_max_h = "132px"
    photo_single_h = "340px"
    photo_dual_h = "245px"
    te_head_size = "28px"
    en_head_size = "24px"
    te_body_size = "16px"
    en_body_size = "14.8px"
    pdf_format_js = '"a4"'

bg_paper = "#ffffff" if paper_theme == "Classic Crisp White" else "#fdfbf7"

logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" style="max-height: {logo_max_h}; max-width: 88%; object-fit: contain; display: block; margin: 0 auto;">'
    if logo_b64 else
    '<div style="font-size:48px; font-weight:900; color:#991b1b; letter-spacing:2px;">OJAS</div><div style="font-size:14px; font-weight:700;">POWER OF INFORMATION • ఓజస్</div>'
)

# Photo Grid HTML (1 or 2 photos)
photos_html = ""
if len(photo_b64_list) == 1:
    photos_html = f"""
    <div style="margin: 12px 0 14px 0; border: 1.5px solid #cbd5e1; padding: 6px; background: #fff;">
        <img src="data:image/jpeg;base64,{photo_b64_list[0]}" style="width: 100%; height: {photo_single_h}; object-fit: cover; display: block;">
        <div style="font-size: 12.5px; color: #334155; padding: 6px 4px 2px 4px; font-weight: 700; border-top: 1px solid #e2e8f0; margin-top: 5px;">
            ▲ {captions[0]}
        </div>
    </div>
    """
elif len(photo_b64_list) >= 2:
    photos_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 12px 0 14px 0;">
        <div style="border: 1.5px solid #cbd5e1; padding: 6px; background: #fff;">
            <img src="data:image/jpeg;base64,{photo_b64_list[0]}" style="width: 100%; height: {photo_dual_h}; object-fit: cover; display: block;">
            <div style="font-size: 12px; color: #334155; padding-top: 5px; font-weight: 700; line-height: 1.35;">▲ {captions[0]}</div>
        </div>
        <div style="border: 1.5px solid #cbd5e1; padding: 6px; background: #fff;">
            <img src="data:image/jpeg;base64,{photo_b64_list[1]}" style="width: 100%; height: {photo_dual_h}; object-fit: cover; display: block;">
            <div style="font-size: 12px; color: #334155; padding-top: 5px; font-weight: 700; line-height: 1.35;">▲ {captions[1]}</div>
        </div>
    </div>
    """

# Video Box HTML
has_video_section = bool(uploaded_video is not None or video_play_url.strip() or video_thumb_b64)
qr_b64 = generate_qr_base64(video_play_url) if video_play_url.strip() else ""

video_box_html = ""
if has_video_section:
    thumb_visual = (
        f'<img src="data:image/jpeg;base64,{video_thumb_b64}" style="width:100%; height:230px; object-fit:cover; display:block; filter: brightness(0.84);">'
        if video_thumb_b64 else
        '<div style="width:100%; height:150px; background: linear-gradient(135deg, #0f172a 0%, #991b1b 100%);"></div>'
    )
    qr_html = (
        f'<div style="background:#fff; padding:4px; border-radius:6px; text-align:center; border:1px solid #cbd5e1; flex-shrink:0;">'
        f'<img src="data:image/png;base64,{qr_b64}" style="width:62px; height:62px; display:block;">'
        f'<div style="font-size:8.5px; font-weight:800; color:#0f172a; margin-top:2px;">SCAN TO PLAY</div></div>'
        if qr_b64 else ""
    )
    clean_url_display = video_play_url.strip() if video_play_url.strip() else "Attached Digital Video Clip • OJAS E-Paper"

    video_box_html = f"""
    <div id="video-hotspot" style="margin: 12px 0 14px 0; border: 2px solid #991b1b; background: #0f172a; border-radius: 6px; overflow: hidden; position: relative;">
        <div style="position: relative; width: 100%;">
            {thumb_visual}
            <div style="position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.26);">
                <div style="width: 62px; height: 62px; border-radius: 50%; background: rgba(220, 38, 38, 0.95); border: 3px solid #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 18px rgba(0,0,0,0.45);">
                    <div style="width: 0; height: 0; border-top: 12px solid transparent; border-bottom: 12px solid transparent; border-left: 20px solid #ffffff; margin-left: 4px;"></div>
                </div>
                <div style="margin-top: 9px; background: rgba(15, 23, 42, 0.9); color: #fef3c7; font-size: 12px; font-weight: 800; padding: 5px 14px; border-radius: 20px; border: 1px solid #f59e0b; letter-spacing: 0.4px;">
                    ▶ {video_caption}
                </div>
            </div>
        </div>
        <div style="background: #fffbeb; padding: 8px 14px; display: flex; justify-content: space-between; align-items: center; gap: 12px; border-top: 1.5px solid #991b1b;">
            <div style="flex: 1;">
                <div style="font-size: 11px; font-weight: 800; color: #991b1b; text-transform: uppercase;">🎬 OJAS MULTIMEDIA VIDEO BULLETIN</div>
                <div style="font-size: 12px; color: #1e293b; font-weight: 600; word-break: break-all;">{clean_url_display}</div>
            </div>
            {qr_html}
        </div>
    </div>
    """

te_body_formatted = st.session_state["te_body"].replace("\n", "<br><br>")
en_body_formatted = st.session_state["en_body"].replace("\n", "<br><br>")

te_sub_html = f'<div style="font-size: 15.5px; font-weight: 700; color: #991b1b; margin-bottom: 10px; line-height: 1.45;">◆ {st.session_state["te_subhead"]}</div>' if st.session_state["te_subhead"].strip() else ""
en_sub_html = f'<div style="font-size: 14.5px; font-weight: 700; color: #b45309; margin-bottom: 10px; line-height: 1.45;">◆ {st.session_state["en_subhead"]}</div>' if st.session_state["en_subhead"].strip() else ""

# Newspaper Column Layouts (Broadsheet Multi-Column vs 9:16 Stacked Vertical Flow)
if is_9_16_fixed and layout_mode == "Bilingual Side-by-Side (Telugu + English)":
    # On narrow 9:16 mobile view, balanced 2-column grid with crisp readable sizing
    news_columns_html = f"""
    <div style="display: grid; grid-template-columns: 1.08fr 1fr; gap: 18px; margin-top: 10px; flex: 1;">
        <div style="border-right: 1.5px solid #cbd5e1; padding-right: 16px;">
            <div style="display:inline-block; background:#991b1b; color:#fff; font-size:11px; font-weight:800; padding:2px 9px; border-radius:3px; margin-bottom:7px;">తెలుగు</div>
            <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.32; margin: 0 0 7px 0;">
                {st.session_state["te_headline"]}
            </h2>
            {te_sub_html}
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.72; color: #1e293b; text-align: justify;">
                {te_body_formatted}
            </div>
        </div>
        <div>
            <div style="display:inline-block; background:#1e293b; color:#fff; font-size:11px; font-weight:800; padding:2px 9px; border-radius:3px; margin-bottom:7px; letter-spacing:0.5px;">ENGLISH</div>
            <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.28; margin: 0 0 7px 0;">
                {st.session_state["en_headline"]}
            </h2>
            {en_sub_html}
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: {en_body_size}; line-height: 1.65; color: #1e293b; text-align: justify;">
                {en_body_formatted}
            </div>
        </div>
    </div>
    """
elif layout_mode == "Bilingual Side-by-Side (Telugu + English)":
    news_columns_html = f"""
    <div style="display: grid; grid-template-columns: 1.12fr 1fr; gap: 24px; margin-top: 12px; flex: 1;">
        <div style="border-right: 1.5px solid #cbd5e1; padding-right: 22px;">
            <div style="display:inline-block; background:#991b1b; color:#fff; font-size:11px; font-weight:800; padding:3px 10px; border-radius:3px; margin-bottom:8px;">తెలుగు</div>
            <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.32; margin: 0 0 8px 0;">
                {st.session_state["te_headline"]}
            </h2>
            {te_sub_html}
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.76; color: #1e293b; text-align: justify;">
                {te_body_formatted}
            </div>
        </div>
        <div>
            <div style="display:inline-block; background:#1e293b; color:#fff; font-size:11px; font-weight:800; padding:3px 10px; border-radius:3px; margin-bottom:8px; letter-spacing:0.5px;">ENGLISH</div>
            <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.28; margin: 0 0 8px 0;">
                {st.session_state["en_headline"]}
            </h2>
            {en_sub_html}
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: {en_body_size}; line-height: 1.68; color: #1e293b; text-align: justify;">
                {en_body_formatted}
            </div>
        </div>
    </div>
    """
elif layout_mode == "Telugu Lead + English Box":
    col_count_te = 2 if sheet_width_px >= 900 else 1
    news_columns_html = f"""
    <div style="margin-top: 10px; flex: 1;">
        <h2 style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_head_size}; font-weight: 800; color: #0f172a; line-height: 1.32; margin: 0 0 8px 0;">
            {st.session_state["te_headline"]}
        </h2>
        {te_sub_html}
        <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.78; color: #1e293b; text-align: justify; column-count: {col_count_te}; column-gap: 24px; column-rule: 1px solid #cbd5e1;">
            {te_body_formatted}
        </div>
        <div style="margin-top: 18px; background: #f8fafc; border-left: 5px solid #991b1b; border-top: 1px solid #cbd5e1; border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; padding: 14px 18px;">
            <div style="font-size: 11px; font-weight: 800; color: #991b1b; letter-spacing: 1px; margin-bottom: 5px;">ENGLISH SUMMARY</div>
            <div style="font-family: 'Merriweather', Georgia, serif; font-size: 19px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">{st.session_state["en_headline"]}</div>
            <div style="font-size: {en_body_size}; line-height: 1.65; color: #334155; text-align: justify;">{en_body_formatted}</div>
        </div>
    </div>
    """
else:
    col_count_en = 2 if sheet_width_px >= 900 else 1
    news_columns_html = f"""
    <div style="margin-top: 10px; flex: 1;">
        <h2 style="font-family: 'Merriweather', Georgia, serif; font-size: {en_head_size}; font-weight: 900; color: #0f172a; line-height: 1.28; margin: 0 0 8px 0;">
            {st.session_state["en_headline"]}
        </h2>
        {en_sub_html}
        <div style="font-size: {en_body_size}; line-height: 1.72; color: #1e293b; text-align: justify; column-count: {col_count_en}; column-gap: 24px; column-rule: 1px solid #cbd5e1;">
            {en_body_formatted}
        </div>
        <div style="margin-top: 18px; background: #fffbeb; border-left: 5px solid #b45309; border-top: 1px solid #fde68a; border-right: 1px solid #fde68a; border-bottom: 1px solid #fde68a; padding: 14px 18px;">
            <div style="font-size: 12px; font-weight: 800; color: #b45309; margin-bottom: 5px;">తెలుగు సారాంశం</div>
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">{st.session_state["te_headline"]}</div>
            <div style="font-family: 'Noto Sans Telugu', sans-serif; font-size: {te_body_size}; line-height: 1.72; color: #334155; text-align: justify;">{te_body_formatted}</div>
        </div>
    </div>
    """

safe_filename = f"OJAS_News_{pub_date.strftime('%Y%m%d')}"
escaped_video_url = json.dumps(video_play_url.strip())

# ----------------- RIGHT COLUMN: RESPONSIVE SCALED PREVIEW + FULL-RES JPEG & PDF EXPORT -----------------
with col_preview:
    st.subheader("🗞️ 2. Live Newspaper Preview & Export")

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
            padding: 6px;
            background: transparent;
            font-family: 'Plus Jakarta Sans', 'Noto Sans Telugu', sans-serif;
            overflow-x: hidden;
        }}
        .toolbar {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}
        .btn {{
            flex: 1;
            min-width: 155px;
            padding: 12px 16px;
            border: none;
            border-radius: 9px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: transform 0.15s ease, opacity 0.15s ease;
            box-shadow: 0 3px 10px rgba(0,0,0,0.12);
        }}
        .btn:hover {{
            transform: translateY(-1px);
            opacity: 0.95;
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
            font-size: 12.5px;
            font-weight: 700;
            color: #15803d;
            margin-bottom: 8px;
            text-align: center;
            min-height: 18px;
        }}
        /* Responsive Auto-Fit Wrapper for Phone, iPad & Desktop */
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
        /* Fixed Master Print Sheet (Prevents thin/squished exports on mobile & iPad) */
        #newspaper-sheet {{
            width: {sheet_width_px}px;
            {sheet_min_height_css}
            background-color: {bg_paper};
            border: 2.5px solid #1e293b;
            padding: 24px 28px 18px 28px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.14);
            color: #0f172a;
        }}
        .masthead-wrap {{
            text-align: center;
            padding-bottom: 10px;
        }}
        .dateline-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top: 3px solid #991b1b;
            border-bottom: 2px solid #1e293b;
            padding: 7px 6px;
            font-size: 12.5px;
            font-weight: 800;
            color: #1e293b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            gap: 8px;
        }}
        .section-pill {{
            background: #991b1b;
            color: #ffffff;
            padding: 3px 10px;
            font-size: 11.5px;
            font-weight: 800;
            border-radius: 2px;
        }}
        .footer-strip {{
            margin-top: 20px;
            padding-top: 9px;
            border-top: 2.5px solid #991b1b;
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 700;
            color: #475569;
        }}
    </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn btn-jpg" onclick="downloadJPEG()">🖼️ Download HD JPEG</button>
            <button class="btn btn-pdf" onclick="downloadPDF()">📄 Download Interactive PDF</button>
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

            // Automatically scales the fixed broadsheet/9:16 master canvas to fit Mobile, iPad, or Desktop viewport cleanly
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
                viewport.style.height = (scaledH + 16) + 'px';
            }}

            window.addEventListener('load', () => {{
                fitPreviewToViewport();
                setTimeout(fitPreviewToViewport, 400);
            }});
            window.addEventListener('resize', fitPreviewToViewport);

            async function renderMasterCanvas() {{
                const status = document.getElementById('status-msg');
                status.innerText = "⏳ Rendering full-resolution broadsheet edition...";
                await document.fonts.ready;

                const holder = document.getElementById('scale-holder');
                const prevTransform = holder.style.transform;
                // Temporarily reset CSS preview scale to 1.0 so html2canvas captures un-scaled full-width HD pixels
                holder.style.transform = "scale(1)";

                const element = document.getElementById('newspaper-sheet');
                const canvas = await html2canvas(element, {{
                    scale: 2,
                    width: MASTER_WIDTH,
                    windowWidth: MASTER_WIDTH + 60,
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
                    document.getElementById('status-msg').innerText = "✅ Full-width HD JPEG downloaded!";
                }} catch (err) {{
                    document.getElementById('status-msg').innerText = "⚠️ Error: " + err;
                }}
            }}

            async function downloadPDF() {{
                try {{
                    const holder = document.getElementById('scale-holder');
                    const prevTransform = holder.style.transform;
                    holder.style.transform = "scale(1)";

                    const sheetEl = document.getElementById('newspaper-sheet');
                    const sheetRect = sheetEl.getBoundingClientRect();
                    const videoEl = document.getElementById('video-hotspot');
                    let videoRect = videoEl ? videoEl.getBoundingClientRect() : null;

                    const canvas = await renderMasterCanvas();
                    const imgData = canvas.toDataURL("image/jpeg", 0.96);
                    const {{ jsPDF }} = window.jspdf;

                    const isCustom916 = {str(is_9_16_fixed).lower()};
                    const pdf = new jsPDF({{
                        orientation: "portrait",
                        unit: "mm",
                        format: {pdf_format_js}
                    }});

                    const pageWidth = pdf.internal.pageSize.getWidth();
                    const pageHeight = pdf.internal.pageSize.getHeight();
                    const margin = isCustom916 ? 0 : 6;
                    const usableWidth = pageWidth - (margin * 2);
                    const imgHeight = (canvas.height * usableWidth) / canvas.width;

                    if (isCustom916) {{
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
                    }} else {{
                        // Full-width A4 PDF layout (scales cleanly across 1 or multiple pages without ever becoming a thin strip)
                        const usablePageHeight = pageHeight - (margin * 2);
                        let heightLeft = imgHeight;
                        let position = margin;

                        pdf.addImage(imgData, "JPEG", margin, position, usableWidth, imgHeight);
                        if (videoRect && VIDEO_URL) {{
                            const scale = usableWidth / sheetRect.width;
                            pdf.link(
                                margin + (videoRect.left - sheetRect.left) * scale,
                                margin + (videoRect.top - sheetRect.top) * scale,
                                videoRect.width * scale,
                                videoRect.height * scale,
                                {{ url: VIDEO_URL }}
                            );
                        }}
                        heightLeft -= usablePageHeight;

                        while (heightLeft > 2) {{
                            position = heightLeft - imgHeight + margin;
                            pdf.addPage();
                            pdf.addImage(imgData, "JPEG", margin, position, usableWidth, imgHeight);
                            heightLeft -= usablePageHeight;
                        }}
                    }}

                    pdf.save("{safe_filename}.pdf");
                    document.getElementById('status-msg').innerText = "✅ Full-width Print & Video PDF downloaded!";
                }} catch (err) {{
                    document.getElementById('status-msg').innerText = "⚠️ Error: " + err;
                }}
            }}
        </script>
    </body>
    </html>
    """

    components.html(newspaper_widget_html, height=1260, scrolling=True)
