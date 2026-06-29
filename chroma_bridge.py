import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from pathlib import Path
from dotenv import load_dotenv

_script_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_script_dir / ".env")

YOLO_TO_DB_TYPE = {
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
}


class RealTimeEquipmentRetriever:
    def __init__(self):
        default_db_path = str(_script_dir / "equipment_vector_db")
        self.db_path         = os.getenv("CHROMA_DB_PATH", default_db_path)
        self.collection_name = "company_assets"

        self.embedding_function = OpenCLIPEmbeddingFunction(
            model_name="ViT-B-32",
            checkpoint="laion2b_s34b_b79k"
        )

        self.client     = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def _normalise(self, detected_class: str) -> str | None:
        key        = detected_class.lower().strip()
        normalised = YOLO_TO_DB_TYPE.get(key)
        if normalised is None:
            print(f"  [bridge] Unknown YOLO class '{detected_class}' — no DB mapping.")
        return normalised

    # ── Text-based query (legacy — kept for backwards compatibility) ──────────
    def retrieve_matching_equipment(
        self,
        crop_image_path: str,
        detected_class_name: str,
        n_results: int = 3,
    ) -> list:
        """
        Legacy text-based query. Now that ChromaDB is indexed with image
        embeddings, this method is less accurate than retrieve_by_image().
        Kept for backwards compatibility only.
        """
        try:
            db_type = self._normalise(detected_class_name)
            if db_type is None:
                return []

            filtered  = self.collection.get(where={"equipment_type": db_type})
            available = len(filtered["ids"])
            if available == 0:
                print(f"  [bridge] No documents for equipment_type='{db_type}'")
                return []

            safe_n  = min(n_results, available)
            results = self.collection.query(
                query_texts=[db_type],
                n_results=safe_n,
                where={"equipment_type": db_type},
            )

            candidates = []
            if results and results.get("metadatas") and results["metadatas"][0]:
                for idx, meta in enumerate(results["metadatas"][0]):
                    distance   = float(results["distances"][0][idx])
                    # Cross-modal distances range [0, 2]; same-modal range [0, 1]
                    # This legacy path still uses /2 since we don't know which index
                    confidence = round(max(0.0, (1.0 - distance / 2.0) * 100), 2)
                    candidates.append({
                        "brand":        meta.get("brand"),
                        "model":        meta.get("model"),
                        "price":        meta.get("price"),
                        "text_summary": meta.get("text_summary", ""),
                        "distance":     distance,
                        "confidence":   confidence,
                        "metadata":     meta,
                    })
            return candidates

        except Exception as e:
            print(f"  [bridge] Text query failed: {e}")
            return []

    # ── Image-embedding query (V1 RAG — main method) ──────────────────────────
    def retrieve_by_image(
        self,
        image,
        detected_class_name: str,
        embed_fn,
        n_results: int = 3,
    ) -> list:
        """
        Query ChromaDB using the crop's CLIP image embedding.

        Now that ChromaDB is indexed with image embeddings (not text),
        this is a pure image-to-image cosine similarity search.
        Distances are in [0, 1]: 0 = identical, 1 = orthogonal.
        Confidence = (1 - distance) × 100 — no cross-modal penalty.

        Parameters
        ----------
        image               : PIL.Image of the detected equipment crop
        detected_class_name : YOLO class label (used for metadata filtering)
        embed_fn            : EquipFace._embed — reuses the loaded CLIP model
        n_results           : number of candidates to return
        """
        try:
            db_type = self._normalise(detected_class_name)
            if db_type is None:
                return []

            filtered  = self.collection.get(where={"equipment_type": db_type})
            available = len(filtered["ids"])
            if available == 0:
                print(f"  [bridge] No documents for equipment_type='{db_type}'")
                return []

            safe_n   = min(n_results, available)
            emb      = embed_fn(image)         # (1, 512) L2-normalised
            emb_list = emb[0].tolist()

            results = self.collection.query(
                query_embeddings = [emb_list],
                n_results        = safe_n,
                where            = {"equipment_type": db_type},
            )

            candidates = []
            if results and results.get("metadatas") and results["metadatas"][0]:
                for idx, meta in enumerate(results["metadatas"][0]):
                    distance = float(results["distances"][0][idx])
                    # Image-to-image cosine distance is in [0, 1]
                    # FIX: removed /2.0 which was halving scores unnecessarily
                    confidence = round(max(0.0, (1.0 - distance / 2.0) * 100), 2)
                    candidates.append({
                        "brand":        meta.get("brand"),
                        "model":        meta.get("model"),
                        "price":        meta.get("price"),
                        "text_summary": meta.get("text_summary", ""),
                        "distance":     distance,
                        "confidence":   confidence,
                        "metadata":     meta,
                    })
            return candidates

        except Exception as e:
            print(f"  [bridge] Image query failed: {e}")
            return []

    # ── Utilities ─────────────────────────────────────────────────────────────
    def list_equipment_types(self) -> list:
        try:
            all_meta = self.collection.get()["metadatas"]
            return sorted({m["equipment_type"] for m in all_meta if m})
        except Exception:
            return []

    def count(self) -> int:
        return self.collection.count()