"""
EquipFace — Equipment Recognition System
=========================================
Works like OpenFace but for physical equipment.

Workflow:
  1. Enroll  : register an equipment item with 1–N reference images
  2. Recognize: identify an unknown equipment image against the registry
  3. Verify  : confirm whether an image matches a specific enrolled item

Built on OpenCLIP (same model your ChromaDB already uses).

Usage
-----
    from equipface import EquipFace

    ef = EquipFace()

    # Enroll
    ef.enroll("laptop_001", ["img1.jpg", "img2.jpg"], {"brand": "Acer", "model": "Aspire Lite"})

    # Recognize
    result = ef.recognize("unknown_device.jpg")
    # {"match": "laptop_001", "confidence": 87.4, "metadata": {...}, "all_scores": [...]}

    # Verify (1:1)
    ok = ef.verify("unknown_device.jpg", "laptop_001", threshold=60)
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, List, Optional
import open_clip
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH   = "./equipface_db"
DEFAULT_THRESHOLD = 55.0
MODEL_NAME        = "ViT-B-32"
PRETRAINED        = "laion2b_s34b_b79k"


# ---------------------------------------------------------------------------
# EquipFace Core
# ---------------------------------------------------------------------------
class EquipFace:
    """
    One-shot / few-shot equipment recognition using CLIP image embeddings.

    Parameters
    ----------
    db_path   : folder where embeddings + metadata are persisted
    threshold : minimum confidence (0-100) to accept a recognition hit
    device    : 'cuda' | 'cpu' | None (auto-detect)
    """

    def __init__(
        self,
        db_path:   str   = DEFAULT_DB_PATH,
        threshold: float = DEFAULT_THRESHOLD,
        device:    Optional[str] = None,
    ):
        self.db_path   = Path(db_path)
        self.threshold = threshold
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.db_path.mkdir(parents=True, exist_ok=True)

        # Load CLIP
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model.eval().to(self.device)

        # In-memory registry
        self._registry: dict = {}
        self._load_registry()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def enroll(
        self,
        equipment_id: str,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        metadata: Optional[dict] = None,
        overwrite: bool = False,
    ) -> dict:
        """
        Register an equipment item with one or more reference images.

        Parameters
        ----------
        equipment_id : unique identifier, e.g. "laptop_acer_aspire3"
        images       : path(s) or PIL Image(s)
        metadata     : any extra info (brand, model, price, specs …)
        overwrite    : if False, additional images are merged into existing entry

        Returns
        -------
        {"status": "enrolled", "id": ..., "num_embeddings": ...}
        """
        if isinstance(images, (str, Image.Image)):
            images = [images]

        embeddings = [self._embed(img) for img in images]
        stacked    = np.vstack(embeddings)

        if equipment_id in self._registry and not overwrite:
            existing = self._registry[equipment_id]["embeddings"]
            stacked  = np.vstack([existing, stacked])
            if len(stacked) > 8:
                stacked = self._prune_embeddings(stacked, k=8)

        self._registry[equipment_id] = {
            "embeddings": stacked,
            "metadata":   metadata or {},
        }
        self._save_entry(equipment_id)

        return {
            "status":         "enrolled",
            "id":             equipment_id,
            "num_embeddings": len(stacked),
        }

    def recognize(
        self,
        image: Union[str, Image.Image],
        top_k: int = 3,
    ) -> dict:
        """
        Identify an unknown equipment image against all enrolled items.

        Returns
        -------
        {
          "match":      str | None,
          "confidence": float,        # 0-100
          "metadata":   dict,
          "all_scores": list
        }
        """
        if not self._registry:
            return {"match": None, "confidence": 0.0, "metadata": {}, "all_scores": []}

        query_emb = self._embed(image)
        scores    = []

        for eq_id, entry in self._registry.items():
            ref_embs = entry["embeddings"]
            sims     = self._cosine_similarity(query_emb, ref_embs)
            score    = float(np.max(sims)) * 100
            scores.append({
                "id":         eq_id,
                "confidence": round(score, 2),
                "metadata":   entry["metadata"],
            })

        scores.sort(key=lambda x: x["confidence"], reverse=True)
        top   = scores[0]
        match = top["id"] if top["confidence"] >= self.threshold else None

        return {
            "match":      match,
            "confidence": top["confidence"],
            "metadata":   top["metadata"] if match else {},
            "all_scores": scores[:top_k],
        }

    def verify(
        self,
        image: Union[str, Image.Image],
        equipment_id: str,
        threshold: Optional[float] = None,
    ) -> dict:
        """
        1:1 check — does this image belong to a specific enrolled item?

        Returns
        -------
        {"verified": bool, "confidence": float, "equipment_id": str}
        """
        if equipment_id not in self._registry:
            raise KeyError(f"Equipment '{equipment_id}' is not enrolled.")

        cutoff     = threshold if threshold is not None else self.threshold
        query_emb  = self._embed(image)
        ref_embs   = self._registry[equipment_id]["embeddings"]
        sims       = self._cosine_similarity(query_emb, ref_embs)
        confidence = float(np.max(sims)) * 100

        return {
            "verified":     confidence >= cutoff,
            "confidence":   round(confidence, 2),
            "equipment_id": equipment_id,
        }

    def delete(self, equipment_id: str) -> bool:
        """Remove an enrolled item from the registry."""
        if equipment_id not in self._registry:
            return False
        del self._registry[equipment_id]
        entry_file = self.db_path / f"{equipment_id}.pkl"
        if entry_file.exists():
            entry_file.unlink()
        return True

    def list_enrolled(self) -> list:
        """Return all enrolled equipment IDs with metadata."""
        return [
            {
                "id":       eq_id,
                "num_refs": len(v["embeddings"]),
                "metadata": v["metadata"],
            }
            for eq_id, v in self._registry.items()
        ]

    def update_metadata(self, equipment_id: str, metadata: dict) -> bool:
        """Update metadata for an enrolled item without re-embedding."""
        if equipment_id not in self._registry:
            return False
        self._registry[equipment_id]["metadata"] = metadata
        self._save_entry(equipment_id)
        return True

    def count(self) -> int:
        return len(self._registry)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _embed(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Return a normalised L2 embedding vector (1, 512)."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    @staticmethod
    def _cosine_similarity(query: np.ndarray, refs: np.ndarray) -> np.ndarray:
        return (refs @ query.T).squeeze(-1)

    @staticmethod
    def _prune_embeddings(embs: np.ndarray, k: int) -> np.ndarray:
        """Keep the k most diverse embeddings via greedy max-min selection."""
        selected = [0]
        for _ in range(k - 1):
            remaining = [i for i in range(len(embs)) if i not in selected]
            if not remaining:
                break
            sel_embs = embs[selected]
            rem_embs = embs[remaining]
            sims     = sel_embs @ rem_embs.T
            min_sims = sims.max(axis=0)
            chosen   = remaining[int(np.argmin(min_sims))]
            selected.append(chosen)
        return embs[selected]

    def _save_entry(self, equipment_id: str):
        # Sanitise ID to be a safe filename
        safe_id = equipment_id.replace("/", "_").replace("\\", "_")
        path    = self.db_path / f"{safe_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "id":         equipment_id,
                "embeddings": self._registry[equipment_id]["embeddings"],
                "metadata":   self._registry[equipment_id]["metadata"],
            }, f)

    def _load_registry(self):
        for pkl_file in self.db_path.glob("*.pkl"):
            with open(pkl_file, "rb") as f:
                data  = pickle.load(f)
                eq_id = data.get("id", pkl_file.stem)
                self._registry[eq_id] = {
                    "embeddings": data["embeddings"],
                    "metadata":   data["metadata"],
                }
        if self._registry:
            print(f"[EquipFace] Loaded {len(self._registry)} enrolled items from '{self.db_path}'")


# ---------------------------------------------------------------------------
# CLI  (python equipface.py list / enroll / recognize / verify / delete)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    ef  = EquipFace(threshold=55.0)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "enroll" and len(sys.argv) >= 4:
        eq_id  = sys.argv[2]
        images = sys.argv[3:]
        result = ef.enroll(eq_id, images)
        print(json.dumps(result, indent=2))

    elif cmd == "recognize" and len(sys.argv) == 3:
        result = ef.recognize(sys.argv[2])
        print(json.dumps(result, indent=2))

    elif cmd == "verify" and len(sys.argv) == 4:
        result = ef.verify(sys.argv[2], sys.argv[3])
        print(json.dumps(result, indent=2))

    elif cmd == "list":
        enrolled = ef.list_enrolled()
        if enrolled:
            for item in enrolled:
                print(f"  • {item['id']}  ({item['num_refs']} refs)  {item['metadata']}")
        else:
            print("No equipment enrolled yet.")

    elif cmd == "delete" and len(sys.argv) == 3:
        ok = ef.delete(sys.argv[2])
        print("Deleted." if ok else "Not found.")

    else:
        print(__doc__)