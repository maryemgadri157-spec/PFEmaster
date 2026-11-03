# # EquipFace — Multimodal Asset Recognition System

End-of-studies project (PFE) at SFM Technologies, Tunis. An AI system that identifies office equipment from photos and generates structured inventory reports for enterprise asset management.

## Problem

SFM Technologies manages large fleets of office equipment (printers, screens, workstations) across client sites. Manual asset auditing is slow, error-prone, and poorly standardized. A technician visiting a site may spend hours cataloguing equipment by hand, often misidentifying models or missing items entirely.

EquipFace replaces that manual process with a photo-based recognition system. A user uploads a photo of the equipment, and the system returns the exact model, generates a structured report, and populates the asset database.

## Approach

Multi-stage computer vision pipeline:

1. **Detection**: YOLO-World identifies and localizes objects in the frame (printer, screen, keyboard, etc.)
2. **Segmentation**: SAM2 (Segment Anything Model 2) extracts clean masks of each detected object
3. **Embedding**: OpenCLIP encodes each segmented object into a vector representation
4. **Classification**: Nearest-neighbor matching against a curated reference database of 248 equipment models stored in ChromaDB
5. **Report generation**: Gemini generates a formatted inventory report with model, condition, and metadata via structured prompting
6. **Delivery**: FastAPI backend serves predictions to a Streamlit web interface

## Dataset

- 1,003 images curated and labeled across 79 printer classes
- Multiple angles, lighting conditions, and backgrounds per model
- Train/validation split with stratified sampling
- Dataset collected independently (not SFM-proprietary)

## Results

- Top-1 accuracy: 70% on the printer class subset
- Top-5 accuracy: 92%
- Average inference time: ~2.3 seconds per image on consumer hardware
- Successfully deployed as a working prototype with a Streamlit web interface

## Stack

- **Computer Vision**: YOLO-World, SAM2, OpenCLIP
- **LLM**: Google Gemini (structured output)
- **Vector Search**: ChromaDB
- **Backend**: FastAPI, Python
- **Frontend**: Streamlit
- **Infrastructure**: Docker

## What I owned

- End-to-end system architecture and model selection
- Dataset collection, labeling, and augmentation
- Multi-model pipeline integration (detection + segmentation + embedding + classification)
- RAG-based report generation prompting
- FastAPI backend and Streamlit web interface
- Agile delivery across 6 sprints with weekly stakeholder reviews

---

**Project context**: Master's thesis (PFE), ESPRIT School of Business, 2026
**Grade**: Très Bien (17.5/20)
**Author**: Mariem Gadri — [LinkedIn](https://linkedin.com/in/mariemgadri)
