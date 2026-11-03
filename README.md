# EquipFace — Progressive Equipment Recognition with Computer Vision

End-of-studies project (PFE) at SFM Technologies, Tunis. An AI system that identifies office and consumer equipment from a single photo, returns its specifications, and generates a structured asset report. Built as three progressive versions, each engineered to solve a measured limitation of the previous one.

## Problem

SFM Technologies' asset auditing workflow was four manual steps: inspect on site, identify by eye or label, search inventory by reference, update the record by hand. The process was slow, operator-dependent, and failed completely when labels were worn. No image search existed, so a photo retrieved nothing.

The task: identify a device from one photo and retrieve its specifications. Constraints: CPU-only hardware, no labeled dataset, 4.5 months, 5 equipment categories, one developer.

## Three versions, one core

### V1 — Semantic baseline (RAG retrieval)

Four pretrained components, nothing trained, CPU-only:
- **YOLO-World** for detection (categories typed as text)
- **OpenCLIP** for 512-dim embeddings per crop
- **ChromaDB** for cosine similarity retrieval over the catalog
- **Gemini 2.5 Flash** for structured JSON report generation

Results on 500+ test images across 79 items:
- Top-1: 26.2%, Top-5: 52.9%, MRR@5: 0.357
- Latency: 116 ms per image on CPU

**Diagnosis**: image-to-text retrieval pays a cross-modal penalty. The correct item was found 53% of the time but ranked first only 26% of the time.

### V2 — EquipFace (image-to-image matching)

Borrowed the recipe from face recognition: detect → crop → embed → nearest reference wins. Each equipment model enrolled with multiple reference photos, OpenCLIP encoder frozen, open-set enrollment (new models added without retraining).

Built the registry from scratch: 79 → 248 models, 1,003 reference images across printers, desktops, laptops, refrigerators, and air conditioners, with 5 standard views per model.

Results: closed the retrieval gap but hit a new ceiling.
- Printer category: 70% Top-1 (distinctive shapes and panels)
- Refrigerators: 23.8% (white rectangles, little signal)
- Laptops: 19% (41 models sharing one chassis)

**Diagnosis**: the background competes with the device. Clutter amplifies failures from identical chassis, reflective finishes, partial occlusion, low resolution, and unseen viewing angles.

### V3 — SAM2 + EquipFace (segment, then recognize)

Added one stage: **SAM2** generates a background mask from YOLO's bounding box, then EquipFace matches on the clean device only. Soft masking with 3px erosion to preserve logos. Full fallback chain: SAM2 error falls back to V2, bad mask falls back to plain crop. Every fallback is visible in the UI.

Cost: 2.5-5 s per image on CPU. Benefit: noise-robust matching, especially for categories with weak signal.

## System architecture

Full pipeline (V3):
1. Input image or video frame
2. YOLO-World detection + bounding box
3. SAM2 segmentation + mask quality score (reject masks under 30% or over 97%)
4. Isolated feature extraction with soft erosion for logo preservation
5. EquipFace matching against the 248-item registry
6. Open-set registration for new items
7. Re-ID deduplication for video (similarity > 0.88 = same item)
8. Gemini contextual synthesis: model, price, specs, visual anomalies, tailored to equipment type

Deployed as a Streamlit demo with FastAPI backend. On a 300-frame video with 873 raw detections, the system returned 5 unique items in 120 seconds end-to-end on CPU.

## Reports that know the type

One JSON schema per equipment category: laptops get CPU, RAM, display fields; refrigerators get capacity and energy class. The system states its confidence, declines to confirm on weak evidence, and flags visual anomalies (e.g., detecting an iPad when the pipeline was called for a laptop).

## Method

Delivered in 6 Scrum sprints over 4.5 months (February to June 2026):
- Sprint 0: Research
- Sprint 1: V1 RAG pipeline
- Sprint 2: V2 EquipFace + registry
- Sprint 3: V3 SAM2 segmentation
- Sprint 4: Orchestrator, API, clients
- Sprint 5: Benchmarking and writing

12 user stories prioritized MoSCoW (6 must, 5 should, 1 could). Product Owner: company supervisor. Scrum Master: academic advisor.

## Three contributions

1. **EquipFace registry**: 248 models enrolled without retraining
2. **Safe segmentation**: soft masking with full fallback chain
3. **Evidence-driven method**: each version earned by measurement, not assumption

## Stack

- **Computer Vision**: YOLO-World, SAM2, OpenCLIP ViT-B-32
- **LLM**: Google Gemini 2.5 Flash (structured output)
- **Vector Search**: ChromaDB (cosine similarity)
- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Infrastructure**: CPU-only deployment

## Documentation

Full project presentation: [docs/presentation.pdf](docs/presentation.pdf)

## Limitations

- **Registry dependency**: not enrolled = not recognized
- **Reference depth**: one photo per model is one point, ten is a cloud
- **Dataset diversity**: consumer and office equipment only
- **Evaluation scope**: closed-set, one site, CPU-only

## What comes next

No architectural change required:
- Registry enrichment (more brands and views)
- Automatic enrollment triggered by procurement
- OCR as a second signal (reading labels and asset tags)
- Domain fine-tuning to widen confidence margins
- Cloud deployment to remove the CPU latency limit
- Tracking and lifecycle features for richer asset reporting

## What I owned

- End-to-end system architecture across three progressive versions
- Dataset collection and curation (1,003 images, 248 models, 5 categories)
- Multi-model pipeline integration (detection + segmentation + embedding + matching)
- Fallback chain engineering for production safety
- RAG-based report generation with per-category JSON schemas
- FastAPI backend and Streamlit demo interface
- Benchmarking methodology (Top-1, Top-3, Top-5, MRR@5, latency, confidence gap)
- Agile delivery across 6 sprints with weekly stakeholder reviews

---

**Project context**: Master's thesis (PFE), ESPRIT School of Business, defended July 2026
**Grade**: Très Bien (17.5/20)
**Supervisors**: Mariem Ben Noamen (company), Aymen Ben Brik (academic)
**Author**: Mariem Gadri — [LinkedIn](https://linkedin.com/in/mariemgadri)
