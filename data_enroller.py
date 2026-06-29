"""
dataset_enrollor.py
====================
Bulk-enroll laptop, refrigerator, desktop, printer,
and air conditioner datasets into EquipFace.

Run once:
    python data_enroller.py

Already-enrolled items are skipped automatically — safe to re-run.
"""

import os
import re
import logging
import pandas as pd
from pathlib import Path
from equipface import EquipFace

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("DatasetEnrollor")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE = r"C:\Users\marye\Desktop\PFEMASTER\core_ai_setup\exploration"

LAPTOP_CSV       = os.path.join(_BASE, "laptop_dataset_clean.csv")
REFRIGERATOR_CSV = os.path.join(_BASE, "refrigerator_dataset.csv")
DESKTOP_CSV      = os.path.join(_BASE, "desktop_dataset.csv")
PRINTER_CSV      = os.path.join(_BASE, "printer_dataset.csv")
AC_CSV           = os.path.join(_BASE, "air_conditioner_dataset.csv")

# Absolute path so the registry is always found regardless of
# which directory the script is launched from
EQUIPFACE_DB        = os.path.join(_BASE, "equipface_db")
MAX_IMAGES_PER_ITEM = 5
IMG_EXTS            = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def normalise_image_path(raw_path: str) -> str:
    """
    Fix Image_Path values that still reference the old folder name
    'PFE MASTER' (with space) instead of 'PFEMASTER' (no space).
    Also handles purely relative paths by leaving them for
    find_images() to resolve against _BASE.
    """
    return raw_path.replace(
        r"C:\Users\marye\Desktop\PFE MASTER",
        r"C:\Users\marye\Desktop\PFEMASTER"
    )


def find_images(folder_path: str, max_images: int = MAX_IMAGES_PER_ITEM) -> list:
    """
    Return up to max_images image files from the given path.

    Handles three cases:
      1. Absolute path (updated or legacy) — used directly.
      2. Relative path (e.g. 'printer_dataset_images\\HP_A24J2C') —
         resolved against _BASE.
      3. Direct image file — returned as-is.
    """
    # Fix old folder name in path
    folder_path = normalise_image_path(folder_path)

    # Normalise separators for the current OS
    normalised = folder_path.replace("\\", os.sep).replace("/", os.sep)
    folder     = Path(normalised)

    # Resolve relative paths against the project base
    if not folder.is_absolute():
        folder = Path(_BASE) / folder

    if not folder.exists():
        return []

    if folder.is_file() and folder.suffix.lower() in IMG_EXTS:
        return [str(folder)]

    if folder.is_dir():
        images = [
            str(p) for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMG_EXTS
        ]
        return images[:max_images]

    return []


def safe_id(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", str(text).strip())[:80]


# ---------------------------------------------------------------------------
# Metadata builders
# ---------------------------------------------------------------------------

def build_laptop_metadata(row: pd.Series) -> dict:
    return {
        "equipment_type":  "laptop",
        "brand":           str(row.get("Brand", "Unknown")),
        "model":           str(row.get("Model_Name", "Unknown")),
        "price":           str(row.get("Price", "N/A")),
        "processor":       str(row.get("Processor_Name", "N/A")),
        "ram":             str(row.get("RAM", "N/A")),
        "ssd":             str(row.get("SSD", "N/A")),
        "gpu":             str(row.get("GPU", "N/A")),
        "display":         str(row.get("Display", "N/A")),
        "display_type":    str(row.get("Display_type", "N/A")),
        "battery":         str(row.get("Battery_Life", "N/A")),
        "processor_brand": str(row.get("Processor_Brand", "N/A")),
        "gpu_brand":       str(row.get("GPU_Brand", "N/A")),
        "ram_type":        str(row.get("RAM_TYPE", "N/A")),
        "adapter":         str(row.get("Adapter", "N/A")),
    }


def build_fridge_metadata(row: pd.Series) -> dict:
    return {
        "equipment_type":     "refrigerator",
        "brand":              str(row.get("Brand", "Unknown")),
        "model":              str(row.get("Name", "Unknown")),
        "price":              str(row.get("Price", "N/A")),
        "capacity":           str(row.get("Capacity_Liters", "N/A")),
        "door_type":          str(row.get("Door_Type", "N/A")),
        "defrost_system":     str(row.get("Defrost_System", "N/A")),
        "compressor":         str(row.get("Compressor_Type", "N/A")),
        "energy_rating":      str(row.get("Energy_Rating_Stars", "N/A")),
        "finish":             str(row.get("Material_Finish", "N/A")),
        "smart_wifi":         str(row.get("Smart_WiFi", "N/A")),
        "water_dispenser":    str(row.get("Water_Dispenser", "N/A")),
        "warranty_years":     str(row.get("Warranty_Years", "N/A")),
        "energy_consumption": str(row.get("Energy consumption", "N/A")),
        "dimensions":         str(row.get("Dimensions", "N/A")),
    }


def build_printer_metadata(row: pd.Series) -> dict:
    return {
        "equipment_type":   "printer",
        "brand":            str(row.get("Brand", "Unknown")),
        "model":            str(row.get("Model_Name", "Unknown")),
        "reference":        str(row.get("Reference", "N/A")),
        "price":            str(row.get("Price", "N/A")),
        "print_technology": str(row.get("Print_Technology", "N/A")),
        "functions":        str(row.get("Functions", "N/A")),
        "color":            str(row.get("Color", "N/A")),
        "format":           str(row.get("Format", "N/A")),
        "print_speed":      str(row.get("Print_Speed", "N/A")),
        "resolution":       str(row.get("Resolution", "N/A")),
        "duplex":           str(row.get("Duplex", "N/A")),
        "connectivity":     str(row.get("Connectivity", "N/A")),
        "warranty":         str(row.get("Warranty", "N/A")),
    }


def build_desktop_metadata(row: pd.Series) -> dict:
    return {
        "equipment_type": "desktop computer",
        "brand":          str(row.get("Brand", "Unknown")),
        "model":          str(row.get("Model_Name", "Unknown")),
        "price":          str(row.get("Price", "N/A")),
        "cpu":            str(row.get("CPU", "N/A")),
        "ram":            str(row.get("RAM", "N/A")),
        "gpu":            str(row.get("GPU", "N/A")),
        "storage":        str(row.get("Storage", "N/A")),
        "frequency":      str(row.get("Frequency", "N/A")),
        "form_factor":    str(row.get("Form_Factor", "N/A")),
        "os":             str(row.get("OS", "N/A")),
        "connectivity":   str(row.get("Connectivity", "N/A")),
        "warranty":       str(row.get("Warranty", "N/A")),
    }


def build_ac_metadata(row: pd.Series) -> dict:
    return {
        "equipment_type": "air conditioner",
        "brand":          str(row.get("Brand", "Unknown")),
        "model":          str(row.get("Model_Name", "Unknown")),
        "reference":      str(row.get("Reference", "N/A")),
        "price":          str(row.get("Price", "N/A")),
        "capacity_btu":   str(row.get("Capacity_BTU", "N/A")),
        "coverage_m2":    str(row.get("Coverage_M2", "N/A")),
        "ac_type":        str(row.get("AC_Type", "N/A")),
        "inverter":       str(row.get("Inverter", "N/A")),
        "energy_class":   str(row.get("Energy_Class", "N/A")),
        "color":          str(row.get("Color", "N/A")),
        "warranty_years": str(row.get("Warranty_Years", "N/A")),
    }


# ---------------------------------------------------------------------------
# Core enrollment function
# ---------------------------------------------------------------------------

def enroll_dataset(
    ef: EquipFace,
    csv_path: str,
    equipment_type: str,
    id_col: str,
    metadata_fn,
    enrolled_ids: set,
) -> dict:
    if not os.path.exists(csv_path):
        log.error(f"CSV not found: {csv_path}")
        return {"enrolled": 0, "skipped": 0, "no_images": 0}

    df = pd.read_csv(csv_path)
    log.info(f"[{equipment_type.upper()}] {len(df)} rows in CSV")

    enrolled  = 0
    skipped   = 0
    no_images = 0

    for idx, row in df.iterrows():
        raw_model = str(row.get(id_col, f"item_{idx}"))
        eq_id     = f"{equipment_type}_{safe_id(raw_model)}"

        if eq_id in enrolled_ids:
            log.info(f"   Skip (already enrolled): {eq_id}")
            skipped += 1
            continue

        img_path_raw = row.get("Image_Path", "")
        if pd.isna(img_path_raw) or not str(img_path_raw).strip():
            log.warning(f"   No Image_Path for {eq_id}")
            no_images += 1
            continue

        images = find_images(str(img_path_raw))
        if not images:
            log.warning(f"   No images found at '{img_path_raw}' for {eq_id}")
            no_images += 1
            continue

        metadata = metadata_fn(row)

        try:
            result = ef.enroll(eq_id, images, metadata=metadata)
            log.info(
                f"   Enrolled: {eq_id} "
                f"({result['num_embeddings']} embeddings from {len(images)} image(s))"
            )
            enrolled += 1
            enrolled_ids.add(eq_id)
        except Exception as e:
            log.error(f"   Failed to enroll {eq_id}: {e}")
            skipped += 1

    return {"enrolled": enrolled, "skipped": skipped, "no_images": no_images}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("  EquipFace Dataset Enrollor")
    log.info(f"  DB path: {EQUIPFACE_DB}")
    log.info("=" * 60)

    ef = EquipFace(db_path=EQUIPFACE_DB, threshold=55.0)
    log.info(f"Registry has {ef.count()} items before enrollment.\n")

    # Build the set once — shared across all datasets so each
    # newly enrolled item is immediately seen by the next dataset
    enrolled_ids = {item["id"] for item in ef.list_enrolled()}

    laptop_stats  = enroll_dataset(ef, LAPTOP_CSV,       "laptop",           "Model_Name", build_laptop_metadata,  enrolled_ids)
    fridge_stats  = enroll_dataset(ef, REFRIGERATOR_CSV, "refrigerator",     "Name",       build_fridge_metadata,  enrolled_ids)
    printer_stats = enroll_dataset(ef, PRINTER_CSV,      "printer",          "Model_Name", build_printer_metadata, enrolled_ids)
    desktop_stats = enroll_dataset(ef, DESKTOP_CSV,      "desktop computer", "Model_Name", build_desktop_metadata, enrolled_ids)
    ac_stats      = enroll_dataset(ef, AC_CSV,           "air conditioner",  "Model_Name", build_ac_metadata,      enrolled_ids)

    log.info("\n" + "=" * 60)
    log.info("  ENROLLMENT SUMMARY")
    log.info("=" * 60)
    for label, stats in [
        ("Laptops",           laptop_stats),
        ("Refrigerators",     fridge_stats),
        ("Printers",          printer_stats),
        ("Desktops",          desktop_stats),
        ("Air Conditioners",  ac_stats),
    ]:
        log.info(
            f"  {label:<20} → enrolled: {stats['enrolled']:3d}, "
            f"skipped: {stats['skipped']:3d}, "
            f"no images: {stats['no_images']:3d}"
        )
    log.info(f"\n  Total in registry: {ef.count()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()