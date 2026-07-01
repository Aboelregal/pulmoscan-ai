# PulmoScan AI — Advanced AI Assistant for Chest CT Analysis

A full-stack, production-shaped Medical AI application for chest CT analysis: lung
segmentation, pulmonary nodule detection/segmentation, multi-label disease
classification, Grad-CAM explainability, and RAG-grounded structured report
generation — wrapped in a radiology-workstation-style React UI.

> **Status:** This is a portfolio/research project, not a medical device. It has
> not been trained or validated on clinical-grade data and must never be used
> for real patient care. Every AI-generated finding and report is explicitly
> labeled as requiring radiologist review.

---

## 1. What's actually implemented vs. what needs your GPU time

To be upfront, since this matters for how you present it in interviews:

| Component | Status |
|---|---|
| DICOM loading, resampling, HU windowing | ✅ Real, tested, production-quality |
| FastAPI backend, auth, DB schema, full API | ✅ Real, tested end-to-end |
| Lung segmentation (classical HU-threshold) | ✅ Real, runs out of the box |
| Lung segmentation (MONAI SegResNet) | ✅ Architecture wired; **needs trained weights** |
| Nodule detection (classical LoG blob candidate generation) | ✅ Real, runs out of the box |
| Nodule detection (trained CNN) | 🔶 Architecture stubbed; **needs LUNA16 training run** |
| Nodule segmentation (Swin UNETR) | ✅ Architecture wired; classical fallback works; **needs trained weights** |
| Disease classification (DenseNet3D) | ✅ Architecture wired; heuristic fallback works; **needs trained weights** |
| Grad-CAM | ✅ Real MONAI GradCAM wiring; saliency-proxy fallback without trained classifier |
| RAG (FAISS + BioBERT embeddings) | ✅ Real, works with the seed corpus included |
| Report generation | ✅ Real deterministic template engine; MedGemma path wired but needs the model downloaded/licensed |
| React CT viewer (slice nav, W/L, overlays, findings, report panel) | ✅ Real, functional; DICOM canvas rendering is a placeholder — swap in Cornerstone.js for pixel-accurate rendering |
| PDF export | ✅ Real, tested |
| Sample CT data to actually run a demo | ✅ Synthetic phantom DICOM series included (`sample_data/`), tested end-to-end through the real pipeline |

Everything runs and returns real (if less accurate) results without any trained
weights, using the classical/heuristic fallbacks. Dropping trained checkpoints
into `backend/weights/` automatically switches each module to the deep-learning
path — no code changes needed. See §5.

---

## 2. Architecture

```
React + TS Frontend (Vite, Tailwind)
        │  REST
FastAPI Gateway (auth, patients, studies, reports)
        │
  ┌─────┼─────────┬─────────────┬──────────────┐
Preprocessing  Segmentation   Detection    Classification + XAI
(pydicom,      (MONAI          (LoG blobs / (DenseNet3D +
 SimpleITK)     SegResNet/      nnU-Net       Grad-CAM)
                Swin UNETR)     stub)
        │                                        │
        └──────────────┬─────────────────────────┘
                   LLM + RAG (MedGemma / template + FAISS + BioBERT)
                        │
                   PostgreSQL (patients, studies, findings, reports)
```

## 3. Folder structure

```
ctvision-ai/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes (auth, patients, studies)
│   │   ├── core/            # config, auth/JWT
│   │   ├── db/               # SQLAlchemy models + session
│   │   ├── preprocessing/    # DICOM loading, HU windowing, resampling
│   │   ├── segmentation/     # lung + nodule segmentation
│   │   ├── detection/        # nodule detection
│   │   ├── classification/   # disease classifier
│   │   ├── xai/               # Grad-CAM
│   │   ├── llm/                # report generation
│   │   ├── rag/                 # retriever + literature corpus
│   │   ├── schemas/            # Pydantic I/O schemas
│   │   ├── utils/               # PDF export
│   │   └── main.py
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/         # typed API client
│   │   ├── components/  # AppShell, nav
│   │   └── pages/        # Dashboard, StudyBrowser, CTViewer
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 4. Database schema (summary)

- **users** — auth, roles (radiologist/admin/technician)
- **patients** — pseudonymized MRN, demographics
- **studies** — one per uploaded CT series; tracks preprocessing/analysis status
- **findings** — one row per detected nodule or positive disease-classification, with bbox, confidence, severity, segmentation/heatmap file paths
- **reports** — one per study: findings / impression / recommendations / patient-friendly summary / RAG references

Full SQLAlchemy models: `backend/app/db/models.py`.

## 5. Bringing your own trained weights

Each AI module checks `backend/weights/<name>.pt` at startup:

| File | Module |
|---|---|
| `lung_segmentation_segresnet.pt` | Lung segmentation |
| `nodule_segmentation_swin_unetr.pt` | Nodule segmentation |
| `disease_classifier_densenet3d.pt` | Disease classification |
| `nodule_detector_nnunet.pt` | Nodule detection (path stubbed — see `detection/nodule_detector.py`) |

If present, the module loads the checkpoint into the matching MONAI
architecture and runs the deep-learning path. If absent, it silently falls
back to the classical/heuristic implementation so the app never breaks.

Suggested training recipe:
- **Lung segmentation**: train `SegResNet` on Medical Segmentation Decathlon Task06_Lung or LUNA16 lung masks.
- **Nodule detection**: train on LUNA16 (888 scans, candidate + annotation CSVs).
- **Nodule segmentation**: fine-tune `SwinUNETR` on LIDC-IDRI nodule boundary annotations, cropped to 64³ ROIs.
- **Disease classifier**: multi-label `DenseNet121` (3D) across MosMedData (COVID/pneumonia), OSIC (fibrosis), and NSCLC-Radiomics (effusion/atelectasis annotations).

---

## 6. Running it

### Option A — Docker Compose (recommended)

```bash
cd ctvision-ai
docker compose up --build
```

- Backend: http://localhost:8000 (docs at http://localhost:8000/docs)
- Frontend: http://localhost:5173
- Postgres: localhost:5432 (user/pass/db: `ctvision`)

First boot creates tables automatically (`init_db()` on FastAPI startup).

### Option B — Run locally without Docker

**Backend:**
```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Quick local DB (no Postgres needed for a demo run):
export DATABASE_URL="sqlite:///./ctvision.db"

uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for interactive API docs.

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

### Try it end-to-end

1. In the Study Browser, create a patient (MRN + name).
2. Upload a Chest CT DICOM series as a `.zip` of `.dcm` files (e.g. a LUNA16 or LIDC-IDRI case — see §7 for where to get sample data).
3. Click **Run AI Analysis** — this runs preprocessing → lung segmentation → nodule detection → nodule segmentation → disease classification → Grad-CAM → RAG-grounded report generation as a background task.
4. Refresh the study list until status is `analyzed`, then click **Open Viewer** to see overlays, findings, and the generated report. Export as PDF from the report panel.

### Running tests

```bash
cd backend
pytest tests/ -v
```

## 7. Sample data — for demoing without downloading a real dataset

**This repo now includes `sample_data/demo_ct_series.zip`** — a synthetic
Chest CT DICOM series generated by `backend/scripts/generate_demo_ct.py`. It's
a simple geometric phantom (body outline, two lung fields, a dense nodule,
a patchy ground-glass-style region), not real anatomy or real patient data,
but it has valid DICOM tags and HU-realistic intensities, so it exercises the
**entire pipeline end-to-end**: upload → DICOM parsing → preprocessing →
lung segmentation → nodule detection → classification → report generation.

To use it: create a patient in the Study Browser, then upload
`sample_data/demo_ct_series.zip` directly — no other setup needed.

To regenerate it (or make a bigger/different one):
```bash
cd backend/scripts
python3 generate_demo_ct.py --out my_demo_series --slices 80
zip -r my_demo_series.zip my_demo_series
```

For real, clinically-sourced (de-identified) chest CT DICOM series to test
against actual anatomy, use one of these public datasets instead:
- **LUNA16**: https://luna16.grand-challenge.org/Data/ (derived from LIDC-IDRI)
- **LIDC-IDRI** via TCIA: https://www.cancerimagingarchive.net/collection/lidc-idri/
- **MosMedData** (COVID-19 CT): https://mosmed.ai/
- **Medical Segmentation Decathlon**: http://medicaldecathlon.com/

Download a case, zip the folder of `.dcm` files, and upload via the Study
Browser the same way.

## 8. Roadmap / future improvements

- Swap the placeholder CT canvas for **Cornerstone.js** for real pixel-accurate DICOM rendering with WebGL-accelerated window/level and true multi-planar reformatting.
- Move `_run_pipeline` from a FastAPI `BackgroundTasks` call to a **Celery/RQ worker queue** for real production throughput and retries.
- Wire the actual trained detection network into `NoduleDetector` (currently stubbed pending a LUNA16 training run).
- Add **history comparison** (prior vs. current study diffing) and longitudinal nodule growth tracking.
- Add **role-based access control** and audit logging for HIPAA-adjacent posture (this demo's auth is intentionally minimal).
- Expand the RAG corpus beyond the 10 seed entries via `scripts/build_rag_corpus.py` (not yet included) pulling PubMed E-utilities + Radiopaedia.
- 3D volume rendering (marching cubes / WebGL) as an optional viewer mode.

 
