"""Streamlit interface for the Listing Lens used-car photo checker."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.cnn_inference import AI_THRESHOLD, MODEL_PATH, load_model, prepare_input, predict


UI_CSS = r"""
<style>
:root {
  --ink: #1b2928;
  --muted: #687875;
  --line: #e4e9e5;
  --paper: #ffffff;
  --canvas: #f6f8f5;
  --forest: #1d5b50;
  --forest-dark: #16483f;
  --forest-soft: #eaf3ef;
  --amber: #a56d23;
  --amber-soft: #fff5e7;
}
.stApp { background: var(--canvas); color: var(--ink); }
[data-testid="stHeader"] { background: rgba(246,248,245,.94); }
.block-container { max-width: 1180px; padding-top: 1.55rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid var(--line); }
[data-testid="stFileUploaderDropzone"] {
  background: #fbfcfa !important; border: 1.5px dashed #aebfb5 !important;
  border-radius: 15px !important; min-height: 132px;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--forest) !important; background: #f4f8f5 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #71817c !important; }
.stButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 10px !important; min-height: 2.8rem; font-weight: 700 !important;
  border: 1px solid var(--forest) !important; transition: all .15s ease;
}
[data-testid="stFormSubmitButton"] > button[kind="primary"],
.stButton > button[kind="primary"] {
  background: var(--forest) !important; color: #fff !important;
}
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
  background: var(--forest-dark) !important; border-color: var(--forest-dark) !important;
}
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #fff; }
[data-testid="stAlert"] { border-radius: 12px; }
.brand-row { display:flex; align-items:center; gap:10px; margin-bottom:1.1rem; }
.brand-mark {
  width:38px; height:38px; display:flex; align-items:center; justify-content:center;
  border-radius:12px; background:#e6f0eb; color:var(--forest); font-size:20px;
}
.brand-name { color:#263c37; font-size:13px; letter-spacing:.12em; font-weight:850; text-transform:uppercase; }
.brand-sub { color:#7b8a84; font-size:11px; margin-left:2px; }
.hero-kicker {
  display:inline-block; padding:6px 10px; margin-bottom:13px; border-radius:999px;
  background:#e9f1ec; color:#315f51; font-size:10px; font-weight:850;
  letter-spacing:.13em; text-transform:uppercase;
}
.hero-title {
  color:var(--ink); font-size:clamp(2.05rem,4.2vw,3.05rem); line-height:1.08;
  letter-spacing:-.052em; font-weight:850; margin:0 0 .75rem 0; max-width:850px;
}
.hero-subtitle { color:var(--muted); font-size:1.04rem; line-height:1.68; margin:0; max-width:810px; }
.section-kicker { font-size:10px; letter-spacing:.13em; text-transform:uppercase; color:#7a8a83; font-weight:850; margin-bottom:8px; }
.panel { border:1px solid var(--line); background:#fff; border-radius:16px; padding:20px 21px; height:100%; }
.panel-title { color:var(--ink); font-size:16px; font-weight:800; margin-bottom:5px; }
.panel-copy { color:var(--muted); font-size:13px; line-height:1.6; }
.tip-row { display:flex; gap:11px; align-items:flex-start; margin-top:14px; }
.tip-num {
  flex:0 0 26px; width:26px; height:26px; border-radius:9px; display:flex;
  align-items:center; justify-content:center; background:#edf3ef; color:var(--forest);
  font-weight:800; font-size:12px;
}
.tip-text { color:#54655f; font-size:13px; line-height:1.5; }
.result-card { border:1px solid var(--line); border-radius:17px; background:#fff; padding:22px 23px; border-left:5px solid #6e8b7d; }
.result-card.ai { border-left-color:#b77a2e; }
.result-eyebrow { color:#788780; font-size:10px; font-weight:850; letter-spacing:.13em; text-transform:uppercase; margin-bottom:10px; }
.result-title { color:var(--ink); font-weight:850; font-size:clamp(1.28rem,2.2vw,1.72rem); letter-spacing:-.03em; line-height:1.2; margin-bottom:9px; }
.result-copy { color:#60716b; font-size:14px; line-height:1.65; }
.result-chip { display:inline-block; padding:5px 9px; border-radius:999px; font-size:10px; font-weight:850; letter-spacing:.08em; text-transform:uppercase; background:#edf3ef; color:#35614f; margin-bottom:12px; }
.result-chip.ai { background:var(--amber-soft); color:#94601d; }
.score-panel { margin-top:13px; border:1px solid var(--line); border-radius:13px; padding:15px 16px; background:#fbfcfa; }
.score-top { display:flex; justify-content:space-between; align-items:baseline; gap:10px; margin-bottom:10px; }
.score-label { color:#52635d; font-size:13px; font-weight:750; }
.score-value { color:var(--ink); font-size:23px; font-weight:850; letter-spacing:-.03em; }
.score-track { position:relative; background:#e9eeea; height:9px; border-radius:99px; overflow:visible; }
.score-fill { height:9px; border-radius:99px; background:#6c9480; }
.score-fill.ai { background:#c08a43; }
.score-threshold { position:absolute; top:-4px; height:17px; width:2px; background:#344640; border-radius:2px; }
.score-foot { display:flex; justify-content:space-between; gap:8px; color:#7b8983; font-size:11px; margin-top:9px; }
.action-card { background:#f0f5f1; border:1px solid #dfe9e2; border-radius:14px; padding:16px 17px; }
.action-title { color:#29443a; font-size:14px; font-weight:850; margin-bottom:5px; }
.action-copy { color:#5f7168; font-size:13px; line-height:1.6; }
.caution-card { background:#fff8ee; border:1px solid #f0e2cc; border-radius:14px; padding:15px 16px; }
.caution-title { color:#79521e; font-size:13px; font-weight:850; margin-bottom:4px; }
.caution-copy { color:#78684f; font-size:12px; line-height:1.55; }
@media (max-width: 700px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:1.1rem; }
  .hero-title { font-size:2rem; }
  .panel { padding:17px; }
  .result-card { padding:18px; }
}
</style>
"""


@st.cache_resource(show_spinner="Loading CNN...")
def cached_cnn_model(model_path: str):
    return load_model(model_path)


def _image_metadata(image_bytes: bytes) -> tuple[Image.Image, str]:
    """Create a correctly oriented preview and human-readable image metadata."""
    with Image.open(io.BytesIO(image_bytes)) as opened:
        opened.seek(0)
        preview = ImageOps.exif_transpose(opened).convert("RGB").copy()
    width, height = preview.size
    size_mb = len(image_bytes) / (1024 * 1024)
    return preview, f"{width:,} × {height:,} px  ·  {size_mb:.2f} MB"


def _render_score(result: dict[str, Any]) -> None:
    """Display the uncalibrated AI-pattern model score and decision threshold."""
    score = float(result["ai_score"])
    score_percent = max(0.0, min(100.0, score * 100.0))
    is_ai = int(result["predicted_id"]) == 1
    fill_class = "ai" if is_ai else ""
    threshold = float(result.get("threshold", AI_THRESHOLD))

    st.markdown(
        f'''<div class="score-panel">
          <div class="score-top">
            <span class="score-label">AI-pattern model score</span>
            <span class="score-value">{score_percent:.1f}/100</span>
          </div>
          <div class="score-track">
            <div class="score-fill {fill_class}" style="width:{score_percent:.1f}%"></div>
            <div class="score-threshold" style="left:{threshold * 100:.1f}%"></div>
          </div>
          <div class="score-foot"><span>Lower AI signal</span><span>Decision boundary: {threshold:.0%}</span><span>Higher AI signal</span></div>
        </div>''',
        unsafe_allow_html=True,
    )
    st.caption(
        "This is an uncalibrated model score, not the probability that a listing is fake. "
        f"Scores at or above the {threshold:.0%} threshold are labelled AI-generated-looking."
    )


def _render_result(result: dict[str, Any]) -> None:
    is_ai = int(result["predicted_id"]) == 1
    kind = "ai" if is_ai else "real"

    if is_ai:
        chip = "Needs a closer look"
        title = "This photo has AI-generated-looking patterns"
        explanation = (
            "The model found visual patterns that are more consistent with AI-generated imagery. "
            "That does not prove the listing is fake: the image may be edited, heavily processed, "
            "or simply misclassified."
        )
    else:
        chip = "Looks more like a photograph"
        title = "This photo looks more like a standard photograph"
        explanation = (
            "The model found visual patterns that are more consistent with an ordinary photograph. "
            "This does not confirm that the photo is current, that the car is available, "
            "or that the seller owns it."
        )

    st.markdown(
        f'''<div class="result-card {kind}">
          <div class="result-eyebrow">Your listing-photo check · CNN</div>
          <div class="result-chip {kind}">{chip}</div>
          <div class="result-title">{title}</div>
          <div class="result-copy">{explanation}</div>
        </div>''',
        unsafe_allow_html=True,
    )
    _render_score(result)

    if is_ai:
        st.markdown(
            '''<div class="caution-card">
              <div class="caution-title">What to do before you go further</div>
              <div class="caution-copy">Ask the seller for fresh, original photos of specific details and a short live walk-around video. Do not pay a deposit based on this image or this result.</div>
            </div>''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '''<div class="action-card">
              <div class="action-title">Still verify the car and seller</div>
              <div class="action-copy">Ask for a live walk-around video, check the vehicle documents and service history, and arrange an in-person inspection. A photo that looks real is not proof that the listing is genuine.</div>
            </div>''',
            unsafe_allow_html=True,
        )


def render_app() -> None:
    st.set_page_config(
        page_title="Used Car Photo Check | Listing Lens",
        page_icon="🚘",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(UI_CSS, unsafe_allow_html=True)

    st.markdown(
        '''<div class="brand-row">
          <div class="brand-mark">◈</div>
          <div><div class="brand-name">Listing Lens</div>
          <div class="brand-sub">A smarter first look at used-car listings</div></div>
        </div>
        <div class="hero-kicker">A buyer's photo-check assistant</div>
        <h1 class="hero-title">Found a used car online? Check its photo before you make the trip.</h1>
        <p class="hero-subtitle">Upload a photograph from a used-car listing. Listing Lens checks whether its visual patterns look more like an ordinary photograph or AI-generated imagery, then helps you decide what to ask the seller next.</p>''',
        unsafe_allow_html=True,
    )
    st.write("")

    upload_col, guide_col = st.columns([1.45, 0.85], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown('<div class="section-kicker">01 · Check a listing photo</div>', unsafe_allow_html=True)
            st.markdown("### Upload the car photo")
            st.caption(
                "Choose an image you saved from the used-car listing. For best results, "
                "use the original image rather than a screenshot or a tiny thumbnail."
            )
            with st.form("listing_photo_form", clear_on_submit=False):
                uploaded = st.file_uploader(
                    "Drop a listing photo here or browse your device",
                    type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
                    help="Supported formats: JPG, JPEG, PNG, WebP, BMP and GIF. Animated GIFs use their first frame.",
                    label_visibility="visible",
                )
                submitted = st.form_submit_button(
                    "Check this photo", type="primary", use_container_width=True
                )
            st.caption("No listing link or personal details needed. The app analyses the image you upload.")

    with guide_col:
        with st.container(border=True):
            st.markdown('<div class="section-kicker">02 · What happens next</div>', unsafe_allow_html=True)
            st.markdown("### A quick check, then smarter questions")
            tips = [
                ("1", "Upload", "Add one clear photo from the online car listing."),
                ("2", "Review", "See whether the image looks more photo-like or AI-generated to the model."),
                ("3", "Verify", "Use the result to decide what to ask the seller—not as a final verdict."),
            ]
            for number, title, copy in tips:
                st.markdown(
                    f'''<div class="tip-row"><div class="tip-num">{number}</div>
                    <div class="tip-text"><strong>{title}</strong><br>{copy}</div></div>''',
                    unsafe_allow_html=True,
                )

    if submitted:
        st.session_state.pop("listing_analysis", None)
        st.session_state.pop("listing_analysis_error", None)
        if uploaded is None:
            st.warning("Upload a photo from the car listing first.")
        else:
            image_bytes = uploaded.getvalue()
            extension = Path(uploaded.name).suffix.lower()
            try:
                _image_metadata(image_bytes)  # validate before loading the CNN
                with st.spinner("Checking the listing photo with CNN..."):
                    model = cached_cnn_model(str(MODEL_PATH))
                    batch = prepare_input(image_bytes, extension)
                    result = predict(model, batch)
                st.session_state["listing_analysis"] = {
                    "image_bytes": image_bytes,
                    "filename": uploaded.name,
                    "result": result,
                }
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                st.session_state["listing_analysis_error"] = (
                    f"This file could not be read as a supported image: {exc}"
                )
            except Exception as exc:
                st.session_state["listing_analysis_error"] = (
                    "The photo check could not be completed. Confirm that the project "
                    "dependencies and CNN model file are installed. "
                    f"Details: {type(exc).__name__}: {exc}"
                )

    if st.session_state.get("listing_analysis_error"):
        st.error(st.session_state["listing_analysis_error"])

    payload = st.session_state.get("listing_analysis")
    if payload:
        st.write("")
        st.markdown('<div class="section-kicker">03 · Your photo check</div>', unsafe_allow_html=True)
        st.markdown("### What the model noticed")
        image_col, result_col = st.columns([0.95, 1.05], gap="large")
        with image_col:
            with st.container(border=True):
                preview, metadata = _image_metadata(payload["image_bytes"])
                st.image(preview, use_container_width=True)
                st.caption(f"{payload['filename']}  ·  {metadata}")
        with result_col:
            _render_result(payload["result"])
            st.caption("Model used: CNN")

    st.write("")
    st.markdown('<div class="section-kicker">Before you contact the seller</div>', unsafe_allow_html=True)
    st.markdown("### A photo check is only one part of buying a used car")
    advice_columns = st.columns(3, gap="medium")
    advice = [
        (
            "Ask for a fresh walk-around",
            "Request a short video showing the same car from several angles, including details you choose.",
        ),
        (
            "Check the vehicle details",
            "Match registration or VIN/chassis details and ask for service and ownership documents through appropriate channels.",
        ),
        (
            "Inspect before paying",
            "Arrange an in-person viewing and an independent inspection. Avoid deposits until you have verified the seller and vehicle.",
        ),
    ]
    for column, (title, copy) in zip(advice_columns, advice):
        with column:
            st.markdown(
                f'''<div class="panel"><div class="panel-title">{title}</div>
                <div class="panel-copy">{copy}</div></div>''',
                unsafe_allow_html=True,
            )

    with st.expander("What this check can—and cannot—tell you"):
        st.markdown(
            """
            **What it checks:** The uploaded image is classified as more consistent with a real photograph or an AI-generated image, based on visual patterns learned from the project's labelled dataset.

            **What it does not check:** It does not verify the listing, seller identity, vehicle ownership, registration, mileage, condition, availability, photo date, or whether the image was copied from another advertisement. A real-looking image can still be used in a misleading listing, and a genuine car can appear in an AI-looking or heavily edited image.

            **How to read the score:** CNN uses a decision threshold of 0.50. The displayed score is an uncalibrated model output, not the probability that the listing is fake.

            **Privacy:** The application processes the uploaded image for inference and keeps it in the active Streamlit session to display the result. This project code does not intentionally save uploaded images to the project directory. Do not upload personal documents or sensitive information.
            """
        )

    st.divider()
    st.markdown(
        '<div style="color:#7a8982;font-size:12px;line-height:1.6;">'
        '<strong>Listing Lens</strong> · An experimental used-car listing photo screening tool. '
        'It checks image appearance only and is not a vehicle-history, seller-verification, '
        'fraud-detection, or forensic authenticity service.'
        '</div>',
        unsafe_allow_html=True,
    )
