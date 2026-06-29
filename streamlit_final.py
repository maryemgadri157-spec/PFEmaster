from __future__ import annotations
import os
import io
import json
import uuid
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

_genai_import_error = None
try:
    from google import genai
    from google.genai import types as genai_types
except Exception as e:
    genai = None
    genai_types = None
    _genai_import_error = str(e)

from ultralytics import YOLOWorld
from equipface import EquipFace
from chroma_bridge import RealTimeEquipmentRetriever
from pipeline_orchestrator import PipelineOrchestrator
from video_processing import VideoProcessor
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_HERE / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equipment Recognition System",
    page_icon="🎯",
    layout="wide",
)

LEVELS = {
    "🟢 V1 — RAG  (Fast)":                 "v1",
    "🟡 V2 — EquipFace  (Balanced)":       "v2",
    "🔴 V3 — SAM2 + EquipFace  (Precise)": "v3",
}
LEVEL_DESC = {
    "v1": "ChromaDB image-embedding retrieval — fast, uses CLIP to query the catalog.",
    "v2": "EquipFace few-shot recognition — CLIP image embeddings vs enrolled registry.",
    "v3": "SAM2 segmentation → clean crop → EquipFace. Falls back to V2 if SAM2 unavailable.",
}

# =============================================================================
# Rendering helpers — DEFINED FIRST so they can be called anywhere below
# =============================================================================
def _render_single_result(det: dict, temp_id: str, key_suffix: str):
    """Display recognition result for one crop."""
    match      = det.get("match")
    confidence = det.get("confidence", 0)
    candidates = det.get("candidates", [])
    report     = det.get("gemini_report")
    pipeline   = det.get("pipeline_used", "?").upper()

    if not match:
        st.warning(f"⚠️ No match above threshold (best: `{confidence}%`)")
        if candidates:
            with st.expander("📊 Candidates (below threshold)"):
                for c in candidates:
                    st.write(
                        f"**{c.get('brand','?')} {c.get('model','?')}** "
                        f"— `{c.get('confidence','?')}%`"
                    )
        return

    st.success(f"✅ **{match}** | `{confidence}%` | Pipeline: **{pipeline}**")
    st.progress(min(int(confidence), 100))

    with st.expander("📊 Top Candidates", expanded=True):
        for rank, c in enumerate(candidates, 1):
            cf    = c.get("confidence", 0)
            badge = "🟢" if cf >= 70 else ("🟡" if cf >= 50 else "🔴")
            st.write(
                f"{badge} **#{rank}** `{cf}%` "
                f"— **{c.get('brand','?')}** {c.get('model','?')}"
            )

    if report:
        if report.get("_parse_error"):
            st.error("Gemini returned invalid JSON.")
            st.code(report.get("_raw", ""), language="text")
        else:
            st.markdown("#### 📋 Asset Report")
            st.json(report)
            save_data = {
                **report,
                "_match":      match,
                "_confidence": confidence,
                "_pipeline":   pipeline,
                "_sam2":       det.get("sam2_used", False),
                "_fallback":   det.get("fallback_from"),
                "_scan_id":    temp_id,
            }
            if st.button(f"💾 Save asset", key=f"save_{temp_id}_{key_suffix}"):
                st.session_state.confirmed_assets.append(save_data)
                st.success(
                    f"✅ Saved! Total: {len(st.session_state.confirmed_assets)}"
                )
    else:
        st.info("ℹ️ No Gemini report (no match above threshold or API error).")


def _render_detections(dets: list, temp_id: str):
    """Display all YOLO detections with their recognition results."""
    for det in dets:
        idx       = det["asset_index"]
        cls       = det["class"]
        yolo_conf = det["yolo_conf"]
        pipeline  = det["pipeline_used"].upper()
        fallback  = det["fallback_from"]
        sam2_used = det["sam2_used"]

        badge = {"V1": "🟢", "V2": "🟡", "V3": "🔴"}.get(pipeline, "⚪")
        parts = [
            f"### 📦 Asset #{idx}: `{cls.upper()}` — YOLO `{yolo_conf}%`",
            f"{badge} **{pipeline}**",
        ]
        if fallback:
            parts.append(f"⚠️ *Fell back from {fallback.upper()}*")
        if sam2_used:
            parts.append("✂️ *SAM2 segmented*")
        st.markdown("  |  ".join(parts))

        img_col, rep_col = st.columns([1, 2])

        with img_col:
            if sam2_used:
                t1, t2 = st.tabs(["YOLO crop", "SAM2 crop"])
                with t1:
                    st.image(det["yolo_crop"], use_container_width=True)
                with t2:
                    st.image(det["final_crop"], use_container_width=True)
            else:
                st.image(
                    det["final_crop"],
                    caption=f"Crop #{idx}",
                    use_container_width=True,
                )

        with rep_col:
            _render_single_result(det, temp_id, str(idx))

        st.markdown("---")


# =============================================================================
# Load all models once at startup
# =============================================================================
@st.cache_resource(show_spinner="Loading models — ~30 s on first run...")
def load_all():
    # Gemini client
    gemini_client = None
    if genai is not None:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.error("❌ GEMINI_API_KEY missing from .streamlit/secrets.toml")
            st.stop()
        gemini_client = genai.Client(api_key=api_key)

    # YOLO — confidence threshold raised to 0.25 to reduce false positives
    yolo = YOLOWorld("yolov8s-world.pt")
    yolo.set_classes([
        "laptop", "computer", "refrigerator",
        "air conditioner", "printer", "desktop computer",
    ])

    ef     = EquipFace(db_path=str(_HERE / "equipface_db"), threshold=50.0)
    chroma = RealTimeEquipmentRetriever()

    # SAM2 — optional, graceful failure
    sam2_predictor, sam2_error = None, None
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        ckpt = str(_HERE / "sam2_checkpoints" / "sam2.1_hiera_small.pt")
        cfg  = "configs/sam2.1/sam2.1_hiera_s.yaml"
        if Path(ckpt).exists():
            device         = "cuda" if torch.cuda.is_available() else "cpu"
            sam2_predictor = SAM2ImagePredictor(
                build_sam2(cfg, ckpt, device=device)
            )
        else:
            sam2_error = (
                f"Checkpoint not found at:\n`{ckpt}`\n\n"
                "Run `python setup_sam2.py` to download it."
            )
    except ImportError:
        sam2_error = "sam2 not installed. Run `python setup_sam2.py`."
    except Exception as e:
        sam2_error = str(e)

    # Orchestrator — no pil_to_part_fn needed, conversion is built-in
    orch = PipelineOrchestrator(
        gemini_client    = gemini_client,
        yolo_model       = yolo,
        equipface        = ef,
        chroma_retriever = chroma,
        sam2_predictor   = sam2_predictor,
    )

    return orch, ef, chroma, sam2_error


orchestrator, equipface, chroma_retriever, sam2_error = load_all()

if "confirmed_assets" not in st.session_state:
    st.session_state.confirmed_assets = []

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Equipment Recognition & Asset Management")
st.caption("Supports images and videos  ·  V1 RAG  ·  V2 EquipFace  ·  V3 SAM2")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("## ⚙️ Pipeline Level")
    selected_label = st.radio(
        "Recognition mode:", list(LEVELS.keys()), index=1
    )
    selected_level = LEVELS[selected_label]
    st.caption(LEVEL_DESC[selected_level])

    if genai is None:
        st.warning(
            "⚠️ google.genai SDK not installed — Gemini reports disabled."
            + (f"\n\nError: {_genai_import_error}" if _genai_import_error else "")
        )

    if selected_level == "v3":
        if sam2_error:
            st.warning(
                f"⚠️ SAM2 unavailable — will fall back to V2.\n\n{sam2_error}"
            )
        else:
            st.success("✅ SAM2 ready.")

    st.markdown("---")
    st.markdown("## 🎬 Video Settings")
    vid_fps  = st.slider("Extraction rate (fps)", 1, 5, 2)
    vid_mode = st.selectbox(
        "Extraction mode",
        ["smart", "fps", "scene"],
        help=(
            "smart = scene-change + min fps (recommended)\n"
            "fps   = fixed rate only\n"
            "scene = scene-change only (static cameras)"
        ),
    )
    vid_dedup = st.slider(
        "Re-ID dedup threshold", 0.70, 0.99, 0.88, 0.01,
        help=(
            "Cosine similarity above which two crops = same item.\n"
            "Higher → stricter.  Lower → more aggressive merging."
        ),
    )
    st.caption(f"{vid_fps} fps · {vid_mode} mode · dedup >{vid_dedup:.2f}")

    st.markdown("---")
    st.markdown("## 🗃️ EquipFace Registry")
    enrolled    = equipface.list_enrolled()
    type_counts: dict = {}
    for item in enrolled:
        t = item["metadata"].get("equipment_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    st.metric("Total Enrolled", len(enrolled))
    icons = {
        "laptop": "💻", "refrigerator": "🧊",
        "desktop computer": "🖥️", "printer": "🖨️",
        "air conditioner": "🌬️",
    }
    for eq_type, count in sorted(type_counts.items()):
        st.metric(f"{icons.get(eq_type,'📦')} {eq_type.title()}", count)

    st.markdown("---")
    st.markdown("## 🗄️ ChromaDB")
    st.metric("Indexed items", chroma_retriever.count())
    db_types = chroma_retriever.list_equipment_types()
    if db_types:
        st.caption("Types: " + ", ".join(db_types))

    st.markdown("---")
    st.markdown("## 💾 Saved Assets")
    st.metric("Confirmed", len(st.session_state.confirmed_assets))
    if st.session_state.confirmed_assets:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📥 Export"):
                st.download_button(
                    "⬇️ JSON",
                    json.dumps(st.session_state.confirmed_assets, indent=2),
                    "assets.json",
                    "application/json",
                )
        with c2:
            if st.button("🗑️ Clear"):
                st.session_state.confirmed_assets = []
                st.rerun()

st.markdown("---")

# =============================================================================
# Input tabs
# =============================================================================
tab_img, tab_vid = st.tabs(["🖼️  Image", "🎬  Video"])

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_img:
    uploaded_img = st.file_uploader(
        "Upload a workspace image:",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_up",
    )

    if uploaded_img:
        pil_src = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(pil_src, caption="Source", use_container_width=True)
            run_img = st.button(
                f"🚀 Run {selected_label.split('—')[0].strip()}",
                type="primary",
                key="run_img",
                use_container_width=True,
            )

        with col2:
            if run_img:
                temp_id = uuid.uuid4().hex[:6]
                with st.spinner(f"Running {selected_level.upper()} pipeline..."):
                    dets = orchestrator.process_image(
                        pil_src, level=selected_level
                    )

                if not dets:
                    st.warning(
                        "⚠️ No equipment detected. Try a clearer image "
                        "or a closer shot of the device."
                    )
                else:
                    st.success(f"👁️ {len(dets)} asset(s) detected.")
                    _render_detections(dets, temp_id)

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Upload a video file:",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_up",
    )

    if uploaded_vid:
        vid_suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(
            suffix=vid_suffix, delete=False
        ) as tmp_vid:
            tmp_vid.write(uploaded_vid.read())
            tmp_vid_path = tmp_vid.name

        vp_probe = VideoProcessor(equipface=equipface)
        info     = vp_probe.get_video_info(tmp_vid_path)
        if info:
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Duration",   f"{info.get('duration_s','?')} s")
            i2.metric("Native FPS", info.get("native_fps", "?"))
            i3.metric("Resolution", f"{info.get('width','?')}×{info.get('height','?')}")
            i4.metric("Size",       f"{info.get('size_mb','?')} MB")

        st.video(uploaded_vid)

        run_vid = st.button(
            f"🚀 Process Video — {selected_label.split('—')[0].strip()}",
            type="primary",
            key="run_vid",
            use_container_width=True,
        )

        if run_vid:
            temp_id  = uuid.uuid4().hex[:6]
            prog_bar = st.progress(0.0)
            prog_txt = st.empty()

            def _on_progress(step: str, pct: float):
                prog_bar.progress(min(pct, 1.0))
                prog_txt.caption(f"⏳ {step}")

            vp = VideoProcessor(
                equipface         = equipface,
                fps               = vid_fps,
                extraction_mode   = vid_mode,
                dedup_threshold   = vid_dedup,
                progress_callback = _on_progress,
            )

            vid_results = vp.process_video(
                tmp_vid_path, orchestrator, level=selected_level
            )

            prog_bar.progress(1.0)
            prog_txt.empty()

            try:
                os.remove(tmp_vid_path)
            except Exception:
                pass

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames",         vid_results["frames_extracted"])
            m2.metric("Raw detections", vid_results["detections_raw"])
            m3.metric("Unique items",   vid_results["unique_equipment"])
            m4.metric("Time",           f"{vid_results['processing_time']} s")

            if "note" in vid_results:
                st.warning(vid_results["note"])
            elif not vid_results["items"]:
                st.warning("No equipment found after deduplication.")
            else:
                st.success(
                    f"✅ {vid_results['unique_equipment']} unique item(s) "
                    f"after CLIP Re-ID deduplication."
                )
                st.markdown("---")
                for item in vid_results["items"]:
                    pr  = item.get("pipeline_result", {})
                    cls = item["class"]
                    gid = item["dedup_group"]
                    st.markdown(
                        f"#### 📦 Item #{gid+1} | `{cls.upper()}` "
                        f"| Frame {item['frame_index']} "
                        f"| YOLO `{item['yolo_conf']}%`"
                    )
                    img_c, rep_c = st.columns([1, 2])
                    with img_c:
                        crop_img = pr.get("final_crop") or pr.get("yolo_crop")
                        if crop_img:
                            st.image(crop_img, use_container_width=True)
                    with rep_c:
                        _render_single_result(pr, temp_id, f"vid_{gid}")
                    st.markdown("---")

# =============================================================================
# Saved assets viewer
# =============================================================================
if st.session_state.confirmed_assets:
    st.markdown("---")
    st.markdown("## 🗄️ Company Asset Database (This Session)")
    for i, asset in enumerate(st.session_state.confirmed_assets, 1):
        label = (
            f"Asset #{i} — "
            f"{asset.get('confirmed_model','Unknown')} | "
            f"{asset.get('equipment_type','').upper()} | "
            f"Pipeline: {asset.get('_pipeline','?')}"
        )
        with st.expander(label):
            st.json(asset)