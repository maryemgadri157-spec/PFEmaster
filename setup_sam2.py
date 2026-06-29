"""
setup_sam2.py
=============
Installs SAM2 and downloads the checkpoint.
Run once from the exploration/ folder with your venv active:

    python setup_sam2.py

What it does
------------
1. Installs the sam2 package from Meta's GitHub repo
2. Creates the sam2_checkpoints/ folder
3. Downloads sam2.1_hiera_small.pt (~180 MB) — the smallest/fastest checkpoint
4. Verifies the installation with a quick import test
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_URL  = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
    "sam2.1_hiera_small.pt"
)
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"
CHECKPOINT_DIR  = Path(__file__).resolve().parent / "sam2_checkpoints"

SAM2_REPO = "git+https://github.com/facebookresearch/sam2.git"


def run(cmd: list, desc: str):
    print(f"\n{'─'*50}")
    print(f"  {desc}")
    print(f"{'─'*50}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(1)


def download_checkpoint():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINT_DIR / CHECKPOINT_NAME

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"\n  Checkpoint already exists ({size_mb:.1f} MB) → {dest}")
        return

    print(f"\n  Downloading {CHECKPOINT_NAME} (~180 MB)...")
    print(f"  From: {CHECKPOINT_URL}")
    print(f"  To  : {dest}")
    print("  This may take a few minutes on a slow connection...")

    def _progress(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, dest, reporthook=_progress)
    print(f"\n  ✅ Downloaded: {dest}")


def verify_install():
    print("\n  Verifying SAM2 installation...")
    try:
        import sam2
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("  ✅ SAM2 imports successfully.")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def main():
    print("=" * 50)
    print("  SAM2 Setup Script")
    print("=" * 50)

    # Step 1: install sam2
    if not verify_install():
        print("\n  Installing SAM2 from GitHub...")
        run(
            [sys.executable, "-m", "pip", "install", SAM2_REPO],
            "pip install git+https://github.com/facebookresearch/sam2.git",
        )
        if not verify_install():
            print("\n[FAILED] SAM2 installation did not succeed.")
            print("Try manually: pip install git+https://github.com/facebookresearch/sam2.git")
            sys.exit(1)
    else:
        print("  SAM2 already installed — skipping pip install.")

    # Step 2: download checkpoint
    download_checkpoint()

    # Step 3: final check
    ckpt = CHECKPOINT_DIR / CHECKPOINT_NAME
    size = ckpt.stat().st_size / 1e6

    print(f"\n{'='*50}")
    print(f"  SAM2 Setup Complete!")
    print(f"{'='*50}")
    print(f"  Package  : installed ✅")
    print(f"  Checkpoint: {ckpt}")
    print(f"  Size     : {size:.1f} MB")
    print(f"\n  You can now select V3 in the Streamlit app.")
    print(f"  Run: python -m streamlit run streamlit_final.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()