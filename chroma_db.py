"""
chroma_db.py — rebuilt with image embeddings
=============================================
Key change from previous version:
  Instead of embedding text summaries (cross-modal at query time),
  each catalog item is now indexed by the MEAN CLIP image embedding
  of its reference photographs. Query time is pure image-to-image
  cosine similarity — no cross-modal penalty.

Run once to rebuild:
    python chroma_db.py
"""

import os
import logging
import numpy as np
import pandas as pd
import chromadb
import open_clip
import torch
from pathlib import Path
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger("VectorDBBuilder")

DB_PATH         = "./equipment_vector_db"
COLLECTION_NAME = "company_assets"
_BASE           = r"C:\Users\marye\Desktop\PFEMASTER\core_ai_setup\exploration"
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMGS        = 5    # max reference images per model to average

DATASET_MAP = {
    "laptop":           os.path.join(_BASE, "laptop_dataset_clean.csv"),
    "refrigerator":     os.path.join(_BASE, "refrigerator_dataset.csv"),
    "desktop computer": os.path.join(_BASE, "desktop_dataset.csv"),
    "printer":          os.path.join(_BASE, "printer_dataset.csv"),
    "air conditioner":  os.path.join(_BASE, "air_conditioner_dataset.csv"),
}


# ── Load CLIP once — same model as EquipFace ─────────────────────────────────
logger.info("Loading OpenCLIP ViT-B-32 (LAION-2B)...")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_model.eval()
logger.info("CLIP ready.")


def embed_image(pil_img) -> np.ndarray:
    """Return an L2-normalised 512-dim embedding for one PIL image."""
    tensor = _clip_preprocess(pil_img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        emb = _clip_model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].cpu().numpy()


def mean_image_embedding(image_paths: list) -> np.ndarray | None:
    """
    Compute the mean CLIP embedding across multiple reference images.
    Returns a normalised 512-dim vector, or None if no images load.
    """
    from PIL import Image as PILImage
    embeddings = []
    for p in image_paths[:MAX_IMGS]:
        try:
            pil = PILImage.open(str(p)).convert("RGB")
            embeddings.append(embed_image(pil))
        except Exception as e:
            logger.warning(f"    Could not load {p}: {e}")

    if not embeddings:
        return None

    mean = np.mean(embeddings, axis=0)
    # Re-normalise the mean so cosine similarity stays in [-1, 1]
    norm = np.linalg.norm(mean)
    return (mean / norm) if norm > 0 else None


def normalise_path(raw: str) -> str:
    return raw.replace("PFE MASTER", "PFEMASTER")


def find_images(folder_path: str) -> list:
    folder = Path(normalise_path(folder_path).replace("\\", os.sep).replace("/", os.sep))
    if not folder.is_absolute():
        folder = Path(_BASE) / folder
    if folder.is_file() and folder.suffix.lower() in IMG_EXTS:
        return [folder]
    if folder.is_dir():
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)[:MAX_IMGS]
    return []


def build_text_summary(row: pd.Series, class_name: str, df: pd.DataFrame) -> str:
    """Keep text summary as a searchable reference field (not used for embedding)."""
    brand = str(row.get("Brand", "Unknown"))
    model = str(row.get("Model_Name") or row.get("Name", "Unknown"))
    price = str(row.get("Price", "N/A"))
    skip  = {"Brand", "Model_Name", "Name", "Price", "text_summary", "Image_Path"}
    extra = [f"{c}: {row[c]}" for c in df.columns
             if c not in skip and pd.notna(row.get(c)) and str(row.get(c)).strip()]
    summary = f"{class_name} | Brand: {brand} | Model: {model} | Price: {price}"
    if extra:
        summary += " | " + " | ".join(extra)
    return summary


def build_metadata(row: pd.Series, class_name: str) -> dict:
    brand = str(row.get("Brand", "Unknown"))
    model = str(row.get("Model_Name") or row.get("Name", "Unknown"))
    price = str(row.get("Price", "N/A"))

    base = {"equipment_type": class_name, "brand": brand, "model": model, "price": price}

    if class_name == "laptop":
        base.update({
            "processor":       str(row.get("Processor_Name", "N/A")),
            "ram":             str(row.get("RAM", "N/A")),
            "ssd":             str(row.get("SSD", "N/A")),
            "gpu":             str(row.get("GPU", "N/A")),
            "display":         str(row.get("Display", "N/A")),
            "display_type":    str(row.get("Display_type", "N/A")),
            "processor_brand": str(row.get("Processor_Brand", "N/A")),
            "gpu_brand":       str(row.get("GPU_Brand", "N/A")),
            "ram_type":        str(row.get("RAM_TYPE", "N/A")),
        })
    elif class_name == "refrigerator":
        base.update({
            "capacity":       str(row.get("Capacity_Liters", "N/A")),
            "door_type":      str(row.get("Door_Type", "N/A")),
            "defrost_system": str(row.get("Defrost_System", "N/A")),
            "energy_rating":  str(row.get("Energy_Rating_Stars", "N/A")),
            "finish":         str(row.get("Material_Finish", "N/A")),
            "warranty_years": str(row.get("Warranty_Years", "N/A")),
        })
    elif class_name == "desktop computer":
        base.update({
            "cpu":          str(row.get("CPU", "N/A")),
            "ram":          str(row.get("RAM", "N/A")),
            "gpu":          str(row.get("GPU", "N/A")),
            "storage":      str(row.get("Storage", "N/A")),
            "form_factor":  str(row.get("Form_Factor", "N/A")),
            "os":           str(row.get("OS", "N/A")),
            "connectivity": str(row.get("Connectivity", "N/A")),
        })
    elif class_name == "printer":
        base.update({
            "reference":        str(row.get("Reference", "N/A")),
            "print_technology": str(row.get("Print_Technology", "N/A")),
            "functions":        str(row.get("Functions", "N/A")),
            "print_speed":      str(row.get("Print_Speed", "N/A")),
            "resolution":       str(row.get("Resolution", "N/A")),
            "connectivity":     str(row.get("Connectivity", "N/A")),
        })
    elif class_name == "air conditioner":
        base.update({
            "capacity_btu": str(row.get("Capacity_BTU", "N/A")),
            "ac_type":      str(row.get("AC_Type", "N/A")),
            "energy_class": str(row.get("Energy_Class", "N/A")),
            "coverage_m2":  str(row.get("Coverage_M2", "N/A")),
            "inverter":     str(row.get("Inverter", "N/A")),
        })
    return base


def build_database():
    logger.info("=" * 55)
    logger.info("  Building ChromaDB with IMAGE embeddings")
    logger.info("  (replaces text-embedding index — fixes cross-modal gap)")
    logger.info("=" * 55)

    # Use a dummy embedding function for collection creation;
    # we supply our own pre-computed embeddings on upsert.
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info("Deleted old collection.")
    except Exception:
        pass

    # The collection needs an embedding function only for text queries;
    # since we always pass pre-computed embeddings, we use a placeholder.
    # We still attach OpenCLIPEmbeddingFunction so ChromaDB stores the
    # model metadata correctly.
    ef_fn = OpenCLIPEmbeddingFunction(
        model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef_fn,
        metadata={"hnsw:space": "cosine"}
    )

    total_inserted = 0

    for class_name, csv_path in DATASET_MAP.items():
        if not os.path.exists(csv_path):
            logger.warning(f"CSV not found for '{class_name}': {csv_path} — skipping.")
            continue

        df = pd.read_csv(csv_path)
        logger.info(f"\nProcessing '{class_name}' — {len(df)} rows")

        batch_ids        = []
        batch_embeddings = []
        batch_documents  = []
        batch_metadatas  = []
        skipped_no_img   = 0

        for idx, row in df.iterrows():
            model_val = str(row.get("Model_Name") or row.get("Name", idx))
            import re
            safe      = re.sub(r"[^\w\-]", "_", model_val.strip())[:80]
            doc_id    = f"{class_name}_{idx}_{abs(hash(model_val)) % 10**8}"

            # Find reference images
            img_path_raw = str(row.get("Image_Path", "")).strip()
            if not img_path_raw or pd.isna(row.get("Image_Path", "")):
                skipped_no_img += 1
                continue

            image_paths = find_images(img_path_raw)
            if not image_paths:
                logger.warning(f"  No images for '{model_val}' — skipping.")
                skipped_no_img += 1
                continue

            # Compute mean image embedding
            mean_emb = mean_image_embedding(image_paths)
            if mean_emb is None:
                logger.warning(f"  All images failed for '{model_val}' — skipping.")
                skipped_no_img += 1
                continue

            summary  = build_text_summary(row, class_name, df)
            metadata = build_metadata(row, class_name)
            metadata["text_summary"]   = summary
            metadata["n_ref_images"]   = str(min(len(image_paths), MAX_IMGS))

            batch_ids.append(doc_id)
            batch_embeddings.append(mean_emb.tolist())
            batch_documents.append(summary)
            batch_metadatas.append(metadata)

        if batch_ids:
            logger.info(f"  Upserting {len(batch_ids)} items with image embeddings...")
            collection.upsert(
                ids        = batch_ids,
                embeddings = batch_embeddings,   # ← pre-computed image embeddings
                documents  = batch_documents,
                metadatas  = batch_metadatas,
            )
            total_inserted += len(batch_ids)
        if skipped_no_img:
            logger.info(f"  Skipped {skipped_no_img} rows (no images found).")

    logger.info(f"\n{'='*55}")
    logger.info(f"  Done. Total items indexed: {collection.count()}")
    logger.info(f"  Embedding type: CLIP image embeddings (ViT-B-32 LAION-2B)")
    logger.info(f"  Query mode: image → image (no cross-modal gap)")
    logger.info(f"{'='*55}")


if __name__ == "__main__":
    build_database()