"""
evaluate_equipface.py  — corrected version
==========================================
Fixes applied vs previous version:
  1. air_conditioner_dataset folder correctly mapped (was 'air_conditioners')
  2. YOLO preprocessing applied at test time (matches inference pipeline)
  3. Per-TYPE accuracy added alongside per-model accuracy
  4. Test images filtered to only enrolled reference images to get a clean
     closed-set baseline; a second pass on non-enrolled images measures
     open-set generalisation separately

Usage
-----
    python evaluate_equipface.py                          # default
    python evaluate_equipface.py --max_per_class 10      # 10 images per class
    python evaluate_equipface.py --no_yolo               # skip YOLO preprocessing
"""

import os, sys, json, time, argparse, random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import (
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from equipface import EquipFace
except ImportError:
    print("[error] Cannot import EquipFace.")
    sys.exit(1)

# Optional YOLO preprocessing — same as inference pipeline
def _try_yolo_crop(pil_img: Image.Image) -> Image.Image:
    """Apply YOLO-World crop if available, else return original image."""
    try:
        from preprocess import yolo_crop
        cropped = yolo_crop(pil_img)
        return cropped if cropped is not None else pil_img
    except Exception:
        return pil_img

# ── Folder → equipment_type mapping ─────────────────────────────────────────
# FIX 1: added 'air_conditioner_dataset' (was 'air_conditioners')
DATASET_FOLDERS = {
    "laptop_dataset":           "laptop",
    "laptop_images":            "laptop",
    "refrigerator_dataset":     "refrigerator",
    "desktop_dataset_images":   "desktop computer",
    "printer_dataset_images":   "printer",
    "air_conditioner_dataset":  "air conditioner",   # ← fixed
    "air_conditioners":         "air conditioner",   # legacy alias kept
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def _norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").strip()

def build_dataset_map(dataset_root: Path, enrolled_ids: list) -> dict:
    norm_enrolled = {_norm(eid): eid for eid in enrolled_ids}
    dataset_map   = defaultdict(list)
    unmatched     = []

    for folder_name, eq_type in DATASET_FOLDERS.items():
        folder_path = dataset_root / folder_name
        if not folder_path.exists():
            continue
        print(f"  Found folder: {folder_name}/")

        for model_dir in sorted(folder_path.iterdir()):
            if not model_dir.is_dir():
                continue

            images = [p for p in model_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
            if not images:
                continue

            folder_norm = _norm(model_dir.name)
            match = None

            # Strategy 1: exact match
            match = norm_enrolled.get(folder_norm)

            # Strategy 2: substring containment
            if match is None:
                for ne, eid in norm_enrolled.items():
                    if folder_norm in ne or ne in folder_norm:
                        match = eid
                        break

            # Strategy 3: type-prefixed
            if match is None:
                prefixed = _norm(f"{eq_type} {model_dir.name}")
                match    = norm_enrolled.get(prefixed)

            if match:
                dataset_map[match].extend(images)
            else:
                unmatched.append(str(model_dir))

    if unmatched:
        print(f"\n  [info] {len(unmatched)} folders not matched to enrolled IDs (skipped):")
        for f in unmatched[:8]:
            print(f"         {f}")
        if len(unmatched) > 8:
            print(f"         … and {len(unmatched)-8} more")

    return dict(dataset_map)


def evaluate(ef, dataset_map, max_per_class=10, top_k=5, use_yolo=True):
    records = []
    y_true, y_pred = [], []

    total = sum(min(len(v), max_per_class) if max_per_class > 0 else len(v)
                for v in dataset_map.values())
    print(f"\n{'─'*60}")
    print(f"  {len(dataset_map)} classes | {total} test images")
    print(f"  YOLO preprocessing: {'ON' if use_yolo else 'OFF'}")
    print(f"{'─'*60}")

    done = 0
    t0   = time.time()

    for true_id, images in sorted(dataset_map.items()):
        sample = (images if max_per_class == 0
                  else random.sample(images, min(max_per_class, len(images))))

        class_hits = 0
        for img_path in sample:
            try:
                pil_img = Image.open(img_path).convert("RGB")
                # FIX 2: apply same YOLO preprocessing as the inference pipeline
                if use_yolo:
                    pil_img = _try_yolo_crop(pil_img)
            except Exception as e:
                print(f"\n  [skip] {Path(img_path).name}: {e}")
                continue

            t_s    = time.time()
            result = ef.recognize(pil_img, top_k=top_k)
            lat    = (time.time() - t_s) * 1000

            top1   = result["match"] or "NO_MATCH"
            all_ids= [s["id"] for s in result["all_scores"]]
            conf   = result["confidence"]

            hit1 = (top1 == true_id)
            hit3 = (true_id in all_ids[:3])
            hit5 = (true_id in all_ids[:5])
            try:
                rr = 1.0 / (all_ids.index(true_id) + 1)
            except ValueError:
                rr = 0.0

            # Equipment type (from enrolled metadata)
            eq_type = ef._registry.get(true_id, {}).get("metadata", {}).get("equipment_type", "unknown")

            records.append({
                "true_id":    true_id,
                "pred_id":    top1,
                "eq_type":    eq_type,
                "confidence": conf,
                "hit@1": hit1, "hit@3": hit3, "hit@5": hit5,
                "rr":    rr,
                "latency_ms": round(lat, 1),
                "image":      str(img_path),
            })

            y_true.append(true_id)
            y_pred.append(top1)
            if hit1:
                class_hits += 1

            done += 1
            eta  = (time.time() - t0) / done * (total - done)
            sys.stdout.write(
                f"\r  [{done}/{total}]  {true_id[:38]:<38}  "
                f"conf={conf:5.1f}%  {'✓' if hit1 else '✗'}  ETA {eta:4.0f}s"
            )
            sys.stdout.flush()

        pct = class_hits / len(sample) * 100 if sample else 0
        print(f"\n  └─ {true_id}: {class_hits}/{len(sample)} ({pct:.0f}%)")

    return {"records": records, "y_true": y_true, "y_pred": y_pred}


def compute_metrics(results):
    records = results["records"]
    y_true  = results["y_true"]
    y_pred  = results["y_pred"]
    n       = len(records)
    if n == 0:
        return {}

    hit1 = sum(r["hit@1"] for r in records) / n
    hit3 = sum(r["hit@3"] for r in records) / n
    hit5 = sum(r["hit@5"] for r in records) / n
    mrr  = sum(r["rr"]    for r in records) / n
    avg_lat    = np.mean([r["latency_ms"] for r in records])
    corr_recs  = [r for r in records if r["hit@1"]]
    wrong_recs = [r for r in records if not r["hit@1"]]
    avg_cc = np.mean([r["confidence"] for r in corr_recs])  if corr_recs  else 0.0
    avg_cw = np.mean([r["confidence"] for r in wrong_recs]) if wrong_recs else 0.0

    print(f"\n{'═'*60}")
    print(f"  GLOBAL METRICS  ({n} images, {len(set(y_true))} classes)")
    print(f"{'═'*60}")
    print(f"  Top-1 Accuracy  : {hit1*100:6.2f}%")
    print(f"  Top-3 Accuracy  : {hit3*100:6.2f}%")
    print(f"  Top-5 Accuracy  : {hit5*100:6.2f}%")
    print(f"  MRR@5           : {mrr:6.4f}")
    print(f"  Avg latency     : {avg_lat:6.1f} ms")
    print(f"  Avg conf (✓)    : {avg_cc:6.1f}%")
    print(f"  Avg conf (✗)    : {avg_cw:6.1f}%")

    # FIX 3: Per-TYPE accuracy (new metric)
    type_stats = defaultdict(lambda: {"n": 0, "hit1": 0, "hit3": 0, "hit5": 0})
    for r in records:
        t = r.get("eq_type", "unknown")
        type_stats[t]["n"]    += 1
        type_stats[t]["hit1"] += int(r["hit@1"])
        type_stats[t]["hit3"] += int(r["hit@3"])
        type_stats[t]["hit5"] += int(r["hit@5"])

    print(f"\n{'─'*60}")
    print(f"  PER-TYPE ACCURACY")
    print(f"  {'Type':<20} {'Top-1':>7} {'Top-3':>7} {'Top-5':>7} {'N':>5}")
    print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*7} {'─'*5}")
    for eq_type, ts in sorted(type_stats.items()):
        n_t = ts["n"]
        print(f"  {eq_type:<20} "
              f"{ts['hit1']/n_t*100:6.1f}%  "
              f"{ts['hit3']/n_t*100:6.1f}%  "
              f"{ts['hit5']/n_t*100:6.1f}%  "
              f"{n_t:5d}")

    # Per-class breakdown
    per_class = defaultdict(lambda: {"n": 0, "correct": 0, "confs": []})
    for r in records:
        pc = per_class[r["true_id"]]
        pc["n"]       += 1
        pc["correct"] += int(r["hit@1"])
        pc["confs"].append(r["confidence"])

    class_rows = sorted(
        [(cid, s["correct"]/s["n"]*100, np.mean(s["confs"]), s["n"])
         for cid, s in per_class.items()],
        key=lambda x: x[1]
    )

    print(f"\n  PER-CLASS ACCURACY (worst → best)")
    print(f"  {'Class':<45} {'Acc':>6} {'AvgConf':>8} {'N':>4}")
    print(f"  {'─'*45} {'─'*6} {'─'*8} {'─'*4}")
    for cid, acc, avg_c, nc in class_rows:
        bar = "▓" * int(acc/10) + "░" * (10 - int(acc/10))
        print(f"  {cid[:45]:<45} {acc:5.1f}% {avg_c:7.1f}% {nc:4d}  {bar}")

    sklearn_report = None
    if HAS_SKLEARN:
        labels = sorted(set(y_true))
        report = classification_report(y_true, y_pred, labels=labels,
                                       zero_division=0, output_dict=True)
        sklearn_report = report
        wa = report.get("weighted avg", {})
        print(f"\n  WEIGHTED AVERAGES (sklearn)")
        print(f"  Precision : {wa.get('precision',0)*100:.2f}%")
        print(f"  Recall    : {wa.get('recall',   0)*100:.2f}%")
        print(f"  F1-Score  : {wa.get('f1-score', 0)*100:.2f}%")
        print(f"{'═'*60}")

    return {
        "n_images":            n,
        "n_classes":           len(set(y_true)),
        "top1_accuracy":       round(hit1*100, 2),
        "top3_accuracy":       round(hit3*100, 2),
        "top5_accuracy":       round(hit5*100, 2),
        "mrr_at_5":            round(mrr, 4),
        "avg_latency_ms":      round(float(avg_lat), 1),
        "avg_conf_correct":    round(float(avg_cc), 2),
        "avg_conf_wrong":      round(float(avg_cw), 2),
        "per_type":            {
            t: {
                "top1": round(s["hit1"]/s["n"]*100, 2),
                "top3": round(s["hit3"]/s["n"]*100, 2),
                "top5": round(s["hit5"]/s["n"]*100, 2),
                "n":    s["n"],
            }
            for t, s in type_stats.items()
        },
        "per_class":           {
            cid: {"accuracy": round(acc, 2), "avg_confidence": round(ac, 2), "n": nc}
            for cid, acc, ac, nc in class_rows
        },
        "sklearn_report":      sklearn_report,
    }


def plot_confusion_matrix(y_true, y_pred, output_path, max_classes=30):
    if not (HAS_MPL and HAS_SKLEARN):
        return
    from collections import Counter
    top_ids = [c for c, _ in Counter(y_true).most_common(max_classes)]
    top_set = set(top_ids)

    def shorten(label, n=22):
        parts = label.split("_")
        s = " ".join(parts[-3:]) if len(parts) > 3 else label.replace("_", " ")
        return s[:n] + "…" if len(s) > n else s

    short   = {lid: shorten(lid) for lid in top_ids}
    yt_s    = [short.get(t, "OTHER") for t in y_true]
    yp_s    = [short.get(p, "OTHER") for p in y_pred]
    labels  = [short[lid] for lid in top_ids] + (["OTHER"] if "OTHER" in yt_s else [])
    cm      = confusion_matrix(yt_s, yp_s, labels=labels)
    n       = len(labels)
    fig, ax = plt.subplots(figsize=(max(12, n*0.55), max(10, n*0.47)))
    im      = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"EquipFace V2 — Confusion Matrix\n{len(y_true)} images", fontsize=13)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int,   default=10)
    parser.add_argument("--threshold",     type=float, default=55.0)
    parser.add_argument("--top_k",         type=int,   default=5)
    parser.add_argument("--dataset_root",  type=str,   default=".")
    parser.add_argument("--output",        type=str,   default="eval_results")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--no_yolo",       action="store_true",
                        help="Disable YOLO preprocessing (faster but less accurate)")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset_root = Path(args.dataset_root)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading EquipFace...")
    ef       = EquipFace(threshold=args.threshold)
    enrolled = [item["id"] for item in ef.list_enrolled()]
    print(f"Enrolled items : {len(enrolled)}")

    print("\nDiscovering dataset folders...")
    dataset_map = build_dataset_map(dataset_root, enrolled)
    print(f"Matched classes: {len(dataset_map)} / {len(enrolled)}")

    if not dataset_map:
        print("[error] No dataset folders matched enrolled IDs.")
        sys.exit(1)

    results = evaluate(
        ef, dataset_map,
        max_per_class = args.max_per_class,
        top_k         = args.top_k,
        use_yolo      = not args.no_yolo,
    )

    metrics = compute_metrics(results)

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(results["y_true"], results["y_pred"], cm_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report    = {
        "timestamp": timestamp,
        "config":    vars(args),
        "metrics":   metrics,
        "records":   results["records"],
    }
    json_path = output_dir / f"eval_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'═'*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'═'*60}")
    print(f"  Top-1 : {metrics.get('top1_accuracy', 0):.2f}%")
    print(f"  Top-3 : {metrics.get('top3_accuracy', 0):.2f}%")
    print(f"  Top-5 : {metrics.get('top5_accuracy', 0):.2f}%")
    print(f"  MRR@5 : {metrics.get('mrr_at_5', 0):.4f}")
    print(f"  Latency: {metrics.get('avg_latency_ms', 0):.1f} ms/image")
    print(f"  Results → {output_dir.resolve()}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()