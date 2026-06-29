"""
video_processing.py
====================
Processes a video file into unique equipment crops, then runs them through
the pipeline orchestrator.

Pipeline
--------
Video → FFmpeg frames → YOLO detection → CLIP-based Re-ID deduplication
      → unique crops → PipelineOrchestrator → results

Re-ID strategy
--------------
Uses the same OpenCLIP ViT-B-32 model that EquipFace already loads.
If two equipment crops have cosine similarity > DEDUP_THRESHOLD they are
considered the same physical item and the duplicate is discarded.
No extra model needed.

FFmpeg extraction modes
-----------------------
"fps"         : fixed rate (default 2 fps) — simple and predictable
"scene"       : scene-change detection only — good for static cameras
"smart"       : scene-change + minimum fps cap — best for walkthroughs

Usage (standalone)
------------------
    python video_processing.py --video path/to/video.mp4 --level v2 --fps 2

Usage (from Streamlit / API)
----------------------------
    from video_processing import VideoProcessor
    vp      = VideoProcessor(equipface=ef)
    results = vp.process_video(video_path, orchestrator, level="v2")
"""

import os
import re
import json
import time
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_FPS            = 2         # frames per second to extract
DEFAULT_SCENE_THRESH   = 0.30      # scene-change sensitivity (0-1, lower = more)
DEFAULT_DEDUP_THRESH   = 0.88      # cosine similarity above which two crops are
                                   # considered the same physical item
MAX_FRAMES_PER_VIDEO   = 300       # safety cap — prevents OOM on long recordings
YOLO_CONF              = 0.15


# =============================================================================
class VideoProcessor:
    """
    Parameters
    ----------
    equipface         : EquipFace instance (provides the shared CLIP model
                        for Re-ID embeddings — no extra model needed)
    fps               : frames per second to extract in "fps" or "smart" mode
    scene_threshold   : scene-change sensitivity for "scene" and "smart" modes
    dedup_threshold   : cosine similarity above which a crop is a duplicate
    extraction_mode   : "fps" | "scene" | "smart"
    progress_callback : optional fn(step: str, pct: float) for UI progress bars
    """

    def __init__(
        self,
        equipface,
        fps:               float    = DEFAULT_FPS,
        scene_threshold:   float    = DEFAULT_SCENE_THRESH,
        dedup_threshold:   float    = DEFAULT_DEDUP_THRESH,
        extraction_mode:   str      = "smart",
        progress_callback: Optional[Callable] = None,
    ):
        self.ef              = equipface
        self.fps             = fps
        self.scene_threshold = scene_threshold
        self.dedup_threshold = dedup_threshold
        self.mode            = extraction_mode
        self._progress       = progress_callback or (lambda s, p: None)

        # Check FFmpeg availability once at init
        self._ffmpeg_ok = shutil.which("ffmpeg") is not None
        if not self._ffmpeg_ok:
            print("[VideoProcessor] WARNING: ffmpeg not found on PATH. "
                  "Frame extraction will fail. Install from https://ffmpeg.org")

    # =========================================================================
    # Public API
    # =========================================================================

    def process_video(
        self,
        video_path: str,
        orchestrator,
        level:      str = "v2",
    ) -> dict:
        """
        Full pipeline: extract frames → detect → Re-ID dedup → recognize.

        Returns
        -------
        {
            "video_path":      str,
            "level":           str,
            "frames_extracted":int,
            "detections_raw":  int,   # before dedup
            "unique_equipment":int,   # after dedup
            "processing_time": float, # seconds
            "items":           list,  # one entry per unique equipment item
        }

        Each item in "items":
        {
            "frame_index":   int,
            "frame_path":    str,
            "class":         str,
            "yolo_conf":     float,
            "dedup_group":   int,     # Re-ID cluster ID
            "pipeline_result": dict,  # from PipelineOrchestrator
        }
        """
        t0 = time.time()

        # ── Step 1: extract frames ────────────────────────────────────────────
        self._progress("Extracting frames from video...", 0.05)
        with tempfile.TemporaryDirectory() as frame_dir:
            frame_paths = self._extract_frames(video_path, frame_dir)
            n_frames    = len(frame_paths)

            if n_frames == 0:
                return self._empty_result(video_path, level, 0, "No frames extracted.")

            self._progress(f"Extracted {n_frames} frames.", 0.15)

            # ── Step 2: YOLO detection on every frame ─────────────────────────
            self._progress("Running YOLO detection on frames...", 0.20)
            all_crops = self._detect_all_frames(
                frame_paths, orchestrator.yolo
            )
            n_raw = len(all_crops)
            self._progress(f"Detected {n_raw} equipment instances.", 0.45)

            if n_raw == 0:
                return self._empty_result(video_path, level, n_frames, "No equipment detected.")

            # ── Step 3: CLIP Re-ID deduplication ─────────────────────────────
            self._progress("Running CLIP Re-ID deduplication...", 0.50)
            unique_crops = self._dedup(all_crops)
            n_unique     = len(unique_crops)
            self._progress(
                f"Deduplicated: {n_raw} → {n_unique} unique items.", 0.60
            )

            # ── Step 4: run pipeline on each unique crop ─────────────────────
            items = []
            for i, crop_info in enumerate(unique_crops):
                pct = 0.60 + 0.38 * (i / max(n_unique, 1))
                self._progress(
                    f"Recognising item {i+1}/{n_unique} "
                    f"({crop_info['class']})...",
                    pct,
                )
                pipeline_results = orchestrator.process_image(
                    crop_info["crop_pil"], level=level
                )
                # process_image runs YOLO again on the crop — the crop IS
                # the equipment, so YOLO may or may not detect it.
                # Use the top result if any, else package the EquipFace
                # result directly.
                pipeline_result = pipeline_results[0] if pipeline_results else \
                    self._fallback_recognize(crop_info["crop_pil"], orchestrator, level)

                items.append({
                    "frame_index":    crop_info["frame_index"],
                    "frame_path":     crop_info["frame_path"],
                    "class":          crop_info["class"],
                    "yolo_conf":      crop_info["yolo_conf"],
                    "dedup_group":    crop_info["dedup_group"],
                    "pipeline_result": pipeline_result,
                })

            self._progress("Done.", 1.0)
            elapsed = round(time.time() - t0, 1)

            return {
                "video_path":       video_path,
                "level":            level,
                "frames_extracted": n_frames,
                "detections_raw":   n_raw,
                "unique_equipment": n_unique,
                "processing_time":  elapsed,
                "items":            items,
            }

    # =========================================================================
    # Frame extraction
    # =========================================================================

    def _extract_frames(self, video_path: str, out_dir: str) -> list:
        """Run FFmpeg and return sorted list of extracted frame paths."""
        if not self._ffmpeg_ok:
            raise RuntimeError("ffmpeg is not installed or not on PATH.")

        out_pattern = str(Path(out_dir) / "frame_%04d.jpg")

        if self.mode == "fps":
            vf = f"fps={self.fps}"

        elif self.mode == "scene":
            # Extract only on scene change — good for static/surveillance cameras
            vf = f"select='gt(scene\\,{self.scene_threshold})',fps=1"

        else:  # "smart" — scene change + minimum fps
            # Always extract at least self.fps frames/sec, plus extra on changes
            vf = (
                f"select='gt(scene\\,{self.scene_threshold})+not(mod(n\\,"
                f"round(TB*{self.fps})))',fps={self.fps}"
            )

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", vf,
            "-vsync", "vfr",          # variable frame rate — avoids duplicates
            "-q:v", "2",              # JPEG quality (2 = high)
            "-frames:v", str(MAX_FRAMES_PER_VIDEO),
            out_pattern,
            "-y", "-loglevel", "error",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[FFmpeg] Error: {result.stderr}")

        frames = sorted(Path(out_dir).glob("frame_*.jpg"))
        return [str(f) for f in frames]

    # =========================================================================
    # YOLO detection across all frames
    # =========================================================================

    def _detect_all_frames(self, frame_paths: list, yolo_model) -> list:
        """
        Run YOLO on every frame, collect all (crop, class, conf) tuples.
        Returns list of crop_info dicts.
        """
        crops = []
        for frame_idx, fp in enumerate(frame_paths):
            try:
                pil_frame  = Image.open(fp).convert("RGB")
                detections = yolo_model(fp, conf=YOLO_CONF, verbose=False)[0]

                for box in detections.boxes:
                    class_idx   = int(box.cls[0])
                    class_label = detections.names[class_idx]
                    yolo_conf   = round(float(box.conf[0]) * 100, 1)
                    xyxy        = box.xyxy[0].tolist()
                    crop_pil    = pil_frame.crop(
                        (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                    )
                    crops.append({
                        "frame_index": frame_idx,
                        "frame_path":  fp,
                        "class":       class_label,
                        "yolo_conf":   yolo_conf,
                        "crop_pil":    crop_pil,
                        "embedding":   None,   # filled in by _dedup
                        "dedup_group": -1,
                    })
            except Exception as e:
                print(f"[VideoProcessor] Frame {frame_idx} failed: {e}")

        return crops

    # =========================================================================
    # CLIP Re-ID deduplication
    # =========================================================================

    def _embed_crop(self, crop_pil: Image.Image) -> np.ndarray:
        """
        Produce a normalised CLIP embedding for one crop.
        Reuses EquipFace's already-loaded model — no extra model needed.
        """
        return self.ef._embed(crop_pil)   # (1, 512), L2-normalised

    def _dedup(self, crops: list) -> list:
        """
        CLIP-based Re-ID: group crops by identity, keep one per group.

        Algorithm (greedy nearest-neighbour clustering):
        - Embed every crop.
        - For each crop, compare with all existing group centroids.
        - If similarity > threshold → same item, skip.
        - Otherwise → new unique item, start new group.

        Returns list of unique crop_info dicts (one per group, the first
        occurrence in video order, which is usually the clearest view).
        """
        # Embed all crops
        for c in crops:
            try:
                c["embedding"] = self._embed_crop(c["crop_pil"])
            except Exception:
                c["embedding"] = None

        groups         = []    # list of (centroid_embedding, group_id)
        unique_crops   = []
        group_id_counter = 0

        for c in crops:
            emb = c["embedding"]
            if emb is None:
                continue

            matched_group = None

            for centroid, gid in groups:
                # Cosine similarity — embeddings are already L2-normalised
                sim = float((centroid @ emb.T).squeeze())
                if sim >= self.dedup_threshold:
                    matched_group = gid
                    break

            if matched_group is None:
                # New unique item
                c["dedup_group"] = group_id_counter
                groups.append((emb, group_id_counter))
                unique_crops.append(c)
                group_id_counter += 1
            else:
                c["dedup_group"] = matched_group
                # Skip — already seen this equipment

        return unique_crops

    # =========================================================================
    # Fallback: direct EquipFace recognition if YOLO re-detection misses
    # =========================================================================

    def _fallback_recognize(self, crop_pil, orchestrator, level) -> dict:
        """
        When YOLO doesn't re-detect on an already-cropped image,
        run EquipFace directly on the crop.
        """
        try:
            result = orchestrator.ef.recognize(crop_pil, top_k=3)
            candidates = [
                {
                    "brand":      s["metadata"].get("brand", "?"),
                    "model":      s["metadata"].get("model", "?"),
                    "price":      s["metadata"].get("price", "N/A"),
                    "confidence": s["confidence"],
                    "metadata":   s["metadata"],
                }
                for s in result["all_scores"]
            ]
            return {
                "pipeline_used":  level,
                "fallback_from":  None,
                "sam2_used":      False,
                "final_crop":     crop_pil,
                "yolo_crop":      crop_pil,
                "match":          result["match"],
                "confidence":     result["confidence"],
                "equipment_type": result["metadata"].get("equipment_type", "unknown"),
                "metadata":       result["metadata"],
                "candidates":     candidates,
                "gemini_report":  None,
                "asset_index":    1,
                "class":          "unknown",
                "yolo_conf":      0.0,
            }
        except Exception as e:
            return {"error": str(e), "pipeline_used": level}

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _empty_result(video_path, level, n_frames, reason):
        return {
            "video_path":       video_path,
            "level":            level,
            "frames_extracted": n_frames,
            "detections_raw":   0,
            "unique_equipment": 0,
            "processing_time":  0.0,
            "items":            [],
            "note":             reason,
        }

    def get_video_info(self, video_path: str) -> dict:
        """Return basic video metadata using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(out.stdout)
            fmt  = data.get("format", {})
            vstream = next(
                (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
                {}
            )
            # Parse fps from "30/1" or "30000/1001" strings
            fps_str = vstream.get("r_frame_rate", "0/1")
            num, den = fps_str.split("/")
            native_fps = round(int(num) / max(int(den), 1), 2)

            return {
                "duration_s":  round(float(fmt.get("duration", 0)), 1),
                "native_fps":  native_fps,
                "width":       vstream.get("width"),
                "height":      vstream.get("height"),
                "codec":       vstream.get("codec_name"),
                "size_mb":     round(int(fmt.get("size", 0)) / 1e6, 1),
            }
        except Exception:
            return {}


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video for equipment recognition.")
    parser.add_argument(
        "--video",
        default=r"C:\Users\marye\Desktop\PFEMASTER\core_ai_setup\exploration\input_video.mp4",
        help="Path to video file"
    )
    parser.add_argument("--level",  default="v2",  choices=["v1","v2","v3"])
    parser.add_argument("--fps",    default=2,     type=float)
    parser.add_argument("--mode",   default="smart",choices=["fps","scene","smart"])
    parser.add_argument("--dedup",  default=0.88,  type=float)
    parser.add_argument("--output", default=None,  help="Save JSON results to file")
    args = parser.parse_args()

    # Minimal imports for CLI usage
    from equipface import EquipFace
    from chroma_bridge import RealTimeEquipmentRetriever
    from pipeline_orchestrator import PipelineOrchestrator
    from google import genai
    from ultralytics import YOLOWorld
    import os

    print(f"Loading models...")
    ef     = EquipFace()
    chroma = RealTimeEquipmentRetriever()
    yolo   = YOLOWorld("yolov8s-world.pt")
    yolo.set_classes(["laptop","computer","refrigerator",
                      "air conditioner","printer","desktop computer"])

    api_key = os.getenv("GEMINI_API_KEY", "")
    gemini  = genai.Client(api_key=api_key) if api_key else None

    orch = PipelineOrchestrator(
        gemini_client    = gemini,
        yolo_model       = yolo,
        equipface        = ef,
        chroma_retriever = chroma,
    )

    vp = VideoProcessor(
        equipface       = ef,
        fps             = args.fps,
        extraction_mode = args.mode,
        dedup_threshold = args.dedup,
        progress_callback = lambda s, p: print(f"  [{int(p*100):3d}%] {s}"),
    )

    print(f"\nProcessing: {args.video}")
    results = vp.process_video(args.video, orch, level=args.level)

    print(f"\n{'='*50}")
    print(f"  Frames extracted  : {results['frames_extracted']}")
    print(f"  Raw detections    : {results['detections_raw']}")
    print(f"  Unique items      : {results['unique_equipment']}")
    print(f"  Processing time   : {results['processing_time']}s")
    print(f"{'='*50}")

    for item in results["items"]:
        pr = item.get("pipeline_result", {})
        print(f"\n  Item #{item['dedup_group']+1} | {item['class']} "
              f"(frame {item['frame_index']})")
        print(f"    Match      : {pr.get('match', 'N/A')}")
        print(f"    Confidence : {pr.get('confidence', 0)}%")
        print(f"    Pipeline   : {pr.get('pipeline_used','?').upper()}")

    if args.output:
        safe_results = {
            k: v for k, v in results.items() if k != "items"
        }
        safe_results["items"] = [
            {k: v for k, v in item.items() if k not in ("crop_pil",)}
            for item in results["items"]
        ]
        with open(args.output, "w") as f:
            json.dump(safe_results, f, indent=2, default=str)
        print(f"\nResults saved → {args.output}")