"""
pipeline_orchestrator.py
========================
Unified pipeline for V1 (RAG), V2 (EquipFace), V3 (SAM2 + EquipFace).

Fixes applied
-------------
1. YOLO confidence raised to 0.25 + minimum crop size 80x80 px
2. V1 RAG uses CLIP image embedding to query ChromaDB — fixes 0.0% scores
3. V3 SAM2 uses softer masking with coverage check — preserves logos
4. Gemini report schema is unique per equipment type
"""

import os
import io
import json
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

YOLO_TO_TYPE = {
    "laptop":           "laptop",
    "computer":         "desktop computer",
    "desktop computer": "desktop computer",
    "desktop":          "desktop computer",
    "refrigerator":     "refrigerator",
    "fridge":           "refrigerator",
    "printer":          "printer",
    "air conditioner":  "air conditioner",
    "air-conditioner":  "air conditioner",
    "ac":               "air conditioner",
    "microwave":        "microwave",
}

MIN_CROP_PX = 80   # minimum bounding-box size to accept a YOLO detection


def _pil_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class PipelineOrchestrator:

    def __init__(
        self,
        gemini_client,
        yolo_model,
        equipface,
        chroma_retriever,
        sam2_predictor = None,
        pil_to_part_fn = None,   # kept for backwards compat, not used internally
    ):
        self.gemini = gemini_client
        self.yolo   = yolo_model
        self.ef     = equipface
        self.chroma = chroma_retriever
        self.sam2   = sam2_predictor

    # =========================================================================
    # Public API
    # =========================================================================

    def process_image(self, pil_image: Image.Image, level: str = "v2") -> list:
        level = level.lower().strip()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_image.save(tmp.name, format="JPEG", quality=92)
            tmp_path = tmp.name

        try:
            detections = self.yolo(tmp_path, conf=0.25, verbose=False)[0]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if len(detections.boxes) == 0:
            return []

        results = []
        for idx, box in enumerate(detections.boxes):
            class_idx   = int(box.cls[0])
            class_label = detections.names[class_idx]
            yolo_conf   = float(box.conf[0]) * 100
            xyxy        = box.xyxy[0].tolist()

            # Skip crops that are too small — likely false positives
            crop_w = xyxy[2] - xyxy[0]
            crop_h = xyxy[3] - xyxy[1]
            if crop_w < MIN_CROP_PX or crop_h < MIN_CROP_PX:
                print(
                    f"  [YOLO] Skipping small box "
                    f"({crop_w:.0f}x{crop_h:.0f} px) — likely false positive"
                )
                continue

            yolo_crop = pil_image.crop((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))
            result    = self._route(level, yolo_crop, pil_image, xyxy, class_label)

            result["asset_index"] = idx + 1
            result["class"]       = class_label
            result["yolo_conf"]   = round(yolo_conf, 1)
            result["yolo_crop"]   = yolo_crop
            results.append(result)

        return results

    # =========================================================================
    # Router
    # =========================================================================

    def _route(self, level, yolo_crop, pil_image, xyxy, class_label):
        if level == "v3":
            result = self._run_v3(yolo_crop, pil_image, xyxy, class_label)
            if result is not None:
                return result
            result = self._run_v2(yolo_crop, class_label)
            result["fallback_from"] = "v3"
            return result
        if level == "v2":
            return self._run_v2(yolo_crop, class_label)
        return self._run_v1(yolo_crop, class_label)

    # =========================================================================
    # V1 — RAG (ChromaDB) with CLIP image embedding
    # =========================================================================

    def _run_v1(self, crop: Image.Image, class_label: str) -> dict:
        eq_type = YOLO_TO_TYPE.get(class_label.lower(), class_label)

        if hasattr(self.chroma, "retrieve_by_image"):
            candidates = self.chroma.retrieve_by_image(
                image               = crop,
                detected_class_name = class_label,
                embed_fn            = self.ef._embed,
                n_results           = 3,
            )
        else:
            candidates = self.chroma.retrieve_matching_equipment(
                crop_image_path     = "",
                detected_class_name = class_label,
                n_results           = 3,
            )

        match, confidence, metadata = None, 0.0, {}
        if candidates:
            top        = candidates[0]
            confidence = top.get(
                "confidence",
                round(max(0.0, (1.0 - top.get("distance", 1.0) / 2.0) * 100), 2)
            )
            if confidence >= 20.0: 
                match    = top.get("model")
                metadata = top.get("metadata", {
                    "brand":          top.get("brand"),
                    "model":          top.get("model"),
                    "price":          top.get("price"),
                    "equipment_type": eq_type,
                })

        gemini_report = self._gemini_report(crop, candidates, class_label, "v1")

        return {
            "pipeline_used": "v1", "fallback_from": None, "sam2_used": False,
            "final_crop": crop, "match": match, "confidence": confidence,
            "equipment_type": eq_type, "metadata": metadata,
            "candidates": candidates, "gemini_report": gemini_report,
        }

    # =========================================================================
    # V2 — EquipFace
    # =========================================================================

    def _run_v2(self, crop: Image.Image, class_label: str) -> dict:
        eq_type    = YOLO_TO_TYPE.get(class_label.lower(), class_label)
        result     = self.ef.recognize(crop, top_k=3)
        match      = result["match"]
        confidence = result["confidence"]
        metadata   = result["metadata"]
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
        gemini_report = self._gemini_report(crop, candidates, class_label, "v2")

        return {
            "pipeline_used": "v2", "fallback_from": None, "sam2_used": False,
            "final_crop": crop, "match": match, "confidence": confidence,
            "equipment_type": eq_type, "metadata": metadata,
            "candidates": candidates, "gemini_report": gemini_report,
        }

    # =========================================================================
    # V3 — SAM2 + EquipFace with softer masking
    # =========================================================================

    def _run_v3(self, yolo_crop, pil_image, xyxy, class_label):
        if self.sam2 is None:
            return None

        import torch

        bx0, by0, bx1, by1 = [int(v) for v in xyxy]
        np_img = np.array(pil_image.convert("RGB"))

        try:
            self.sam2.set_image(np_img)
            with torch.inference_mode():
                masks, scores, _ = self.sam2.predict(
                    point_coords     = None,
                    point_labels     = None,
                    box              = np.array([[bx0, by0, bx1, by1]]),
                    multimask_output = False,
                )

            if masks is None or len(masks) == 0:
                return None

            best_mask = masks[np.argmax(scores)]

            # Coverage check — if mask removed too much or too little,
            # segmentation is unreliable → fall back to YOLO crop
            box_area  = max((bx1 - bx0) * (by1 - by0), 1)
            mask_area = int(best_mask.sum())
            coverage  = mask_area / box_area

            if coverage < 0.30 or coverage > 0.97:
                print(
                    f"  [V3] SAM2 mask coverage {coverage:.0%} — "
                    f"unreliable, using YOLO crop instead."
                )
                sam2_crop = pil_image.crop((bx0, by0, bx1, by1))
            else:
                # Shrink background mask inward 3 px to preserve
                # edge details like engraved logos and bezels
                shrunk_bg       = _shrink_mask(~best_mask, pixels=3)
                rgb             = np_img.copy()
                rgb[shrunk_bg]  = 255
                sam2_crop       = Image.fromarray(rgb).crop(
                    (bx0, by0, bx1, by1)
                )

        except Exception as e:
            print(f"  [V3] SAM2 failed: {e}")
            return None

        eq_type    = YOLO_TO_TYPE.get(class_label.lower(), class_label)
        result     = self.ef.recognize(sam2_crop, top_k=3)
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
        gemini_report = self._gemini_report(sam2_crop, candidates, class_label, "v3")

        return {
            "pipeline_used": "v3", "fallback_from": None, "sam2_used": True,
            "final_crop": sam2_crop, "match": result["match"],
            "confidence": result["confidence"], "equipment_type": eq_type,
            "metadata": result["metadata"], "candidates": candidates,
            "gemini_report": gemini_report,
        }

    # =========================================================================
    # Gemini — type-specific report
    # =========================================================================

    def _call_gemini(self, prompt: str, image: Image.Image) -> str:
        from google.genai import types as gt
        image_part = gt.Part.from_bytes(
            data=_pil_to_bytes(image), mime_type="image/jpeg"
        )
        response = self.gemini.models.generate_content(
            model    = "gemini-2.5-flash",
            contents = [
                gt.Content(parts=[gt.Part(text=prompt), image_part])
            ],
        )
        return response.text

    def _gemini_report(self, crop, candidates, class_label, source) -> dict | None:
        if not candidates or self.gemini is None:
            return None

        eq_type         = YOLO_TO_TYPE.get(class_label.lower(), class_label)
        context_payload = self._build_context(candidates, eq_type, source)
        schema          = self._build_schema(eq_type)

        prompt = f"""You are an asset tracking system. Analyze the image against the candidates.
Respond ONLY with a valid JSON object — no markdown, no code fences, no explanation.

Pipeline: {source.upper()}
Equipment type: {eq_type}
Candidates:
{context_payload}

Return exactly this JSON structure (fill every field, use N/A if unknown):
{schema}"""

        try:
            raw = self._call_gemini(prompt, crop).strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                return {"_raw": raw, "_parse_error": True}
            except Exception:
                return {"_parse_error": True}
        except Exception as e:
            print(f"  [Gemini] Error: {e}")
            return None

    def _build_schema(self, eq_type: str) -> str:
        """
        Return a type-specific JSON schema string injected into the Gemini prompt.
        Each equipment type only receives fields relevant to it.
        """
        base = {
            "confirmed_model":          "full brand + model name",
            "confidence_level":         "High or Medium or Low",
            "estimated_value":          "price as string in TND",
            "equipment_type":           eq_type,
            "inventory_classification": "short classification label",
            "visual_anomalies":         "none or describe what looks off",
            "recommended_candidate":    1,
        }

        if eq_type == "laptop":
            base["specs"] = {
                "processor":       "CPU name and generation",
                "processor_brand": "Intel or AMD or Apple",
                "ram":             "amount and type e.g. 16GB DDR4",
                "storage":         "SSD capacity e.g. 512GB NVMe",
                "gpu":             "GPU model",
                "gpu_brand":       "NVIDIA or AMD or Intel or Integrated",
                "display_size":    "screen size in inches",
                "display_type":    "IPS / OLED / TN etc.",
                "battery":         "battery life or capacity",
            }

        elif eq_type == "refrigerator":
            base["specs"] = {
                "capacity_liters":  "total capacity in litres",
                "door_type":        "Single / Double / French Door / Side-by-Side",
                "defrost_system":   "NoFrost / LessFrost / Direct Cool",
                "compressor":       "compressor type",
                "energy_rating":    "energy star rating",
                "finish":           "color and material e.g. Inox Silver",
                "dimensions":       "H x W x D in cm if visible",
                "smart_wifi":       "Yes or No",
                "water_dispenser":  "Yes or No",
                "warranty_years":   "warranty in years",
            }

        elif eq_type in ("desktop computer", "computer"):
            base["specs"] = {
                "cpu":          "CPU model and generation",
                "ram":          "RAM amount e.g. 16GB",
                "gpu":          "GPU model or Integrated",
                "storage":      "storage e.g. 512GB SSD",
                "form_factor":  "Tower / Mini / All-in-One etc.",
                "os":           "operating system",
                "connectivity": "ports and wireless features",
                "warranty":     "warranty period",
            }

        elif eq_type == "printer":
            base["specs"] = {
                "print_technology": "Inkjet / Laser / Ink Tank",
                "functions":        "Print only / All-in-One / MFP",
                "color_printing":   "Color or Monochrome",
                "format":           "A4 / A3 etc.",
                "print_speed":      "pages per minute",
                "resolution":       "DPI e.g. 1200x1200",
                "duplex":           "Yes or No",
                "connectivity":     "USB / WiFi / Ethernet etc.",
                "warranty":         "warranty period",
            }

        elif eq_type == "air conditioner":
            base["specs"] = {
                "capacity_btu":   "BTU rating e.g. 12000",
                "coverage_m2":    "recommended room size in m²",
                "ac_type":        "Split / Portable / Window",
                "inverter":       "Yes or No",
                "energy_class":   "energy efficiency class e.g. A++",
                "color":          "unit color",
                "warranty_years": "warranty in years",
            }

        else:
            base["specs"] = {
                "key_spec_1": "value or N/A",
                "key_spec_2": "value or N/A",
                "key_spec_3": "value or N/A",
                "warranty":   "value or N/A",
            }

        return json.dumps(base, indent=4)

    # =========================================================================
    # Context payload builder
    # =========================================================================

    def _build_context(self, candidates, eq_type, source) -> str:
        lines = []
        for rank, c in enumerate(candidates, 1):
            m     = c.get("metadata", c)
            score = c.get("confidence",
                          round((1 - c.get("distance", 1)) * 100, 1))
            label = "EquipFace score" if source in ("v2", "v3") else "RAG score"

            block = [
                f"Candidate #{rank} ({label}: {score}%):",
                f"Brand: {m.get('brand', c.get('brand', 'N/A'))}",
                f"Model: {m.get('model', c.get('model', 'N/A'))}",
                f"Price: {m.get('price', c.get('price', 'N/A'))}",
            ]

            if eq_type == "laptop":
                block += [
                    f"Processor: {m.get('processor','N/A')} ({m.get('processor_brand','N/A')})",
                    f"RAM: {m.get('ram','N/A')} ({m.get('ram_type','N/A')})",
                    f"Storage: {m.get('ssd','N/A')}",
                    f"GPU: {m.get('gpu','N/A')} ({m.get('gpu_brand','N/A')})",
                    f"Display: {m.get('display','N/A')} inch {m.get('display_type','N/A')}",
                    f"Battery: {m.get('battery','N/A')}",
                ]
            elif eq_type in ("desktop computer", "computer"):
                block += [
                    f"CPU: {m.get('cpu','N/A')} @ {m.get('frequency','N/A')}",
                    f"RAM: {m.get('ram','N/A')}",
                    f"GPU: {m.get('gpu','N/A')}",
                    f"Storage: {m.get('storage','N/A')}",
                    f"Form Factor: {m.get('form_factor','N/A')}",
                    f"OS: {m.get('os','N/A')}",
                    f"Connectivity: {m.get('connectivity','N/A')}",
                    f"Warranty: {m.get('warranty','N/A')}",
                ]
            elif eq_type == "printer":
                block += [
                    f"Reference: {m.get('reference','N/A')}",
                    f"Technology: {m.get('print_technology','N/A')}",
                    f"Functions: {m.get('functions','N/A')}",
                    f"Print Speed: {m.get('print_speed','N/A')}",
                    f"Resolution: {m.get('resolution','N/A')}",
                    f"Connectivity: {m.get('connectivity','N/A')}",
                    f"Warranty: {m.get('warranty','N/A')}",
                ]
            elif eq_type == "air conditioner":
                block += [
                    f"Capacity: {m.get('capacity_btu','N/A')} BTU",
                    f"Coverage: {m.get('coverage_m2','N/A')} m²",
                    f"Type: {m.get('ac_type','N/A')}",
                    f"Inverter: {m.get('inverter','N/A')}",
                    f"Energy Class: {m.get('energy_class','N/A')}",
                    f"Warranty: {m.get('warranty_years','N/A')} years",
                ]
            else:  # refrigerator or unknown
                block += [
                    f"Capacity: {m.get('capacity','N/A')} L",
                    f"Door Type: {m.get('door_type','N/A')}",
                    f"Defrost: {m.get('defrost_system','N/A')}",
                    f"Energy Rating: {m.get('energy_rating','N/A')}",
                    f"Finish: {m.get('finish','N/A')}",
                    f"Smart WiFi: {m.get('smart_wifi','N/A')}",
                    f"Water Dispenser: {m.get('water_dispenser','N/A')}",
                    f"Warranty: {m.get('warranty_years','N/A')} years",
                ]

            lines.append("\n".join(block))
            lines.append("---")

        return "\n".join(lines)


# =============================================================================
# Pure-numpy mask shrink (no opencv needed)
# =============================================================================

def _shrink_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    """
    Shrink a boolean mask inward by `pixels` using 4-connectivity erosion.
    Pure numpy — no opencv required.
    Preserves edge details (logos, bezels) when applying SAM2 background mask.
    """
    result = mask.copy().astype(bool)
    for _ in range(pixels):
        up    = np.roll(result, -1, axis=0)
        down  = np.roll(result,  1, axis=0)
        left  = np.roll(result, -1, axis=1)
        right = np.roll(result,  1, axis=1)
        result = result & up & down & left & right
    return result