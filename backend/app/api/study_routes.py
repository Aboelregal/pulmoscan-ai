"""
Study upload + AI analysis orchestration endpoints.

POST /studies/upload         - upload a DICOM series (zip or folder), preprocess, store
POST /studies/{id}/analyze   - run the full AI pipeline (segmentation -> detection ->
                                nodule segmentation -> classification -> Grad-CAM -> report)
GET  /studies/{id}           - study metadata
GET  /studies/{id}/findings  - list AI findings
GET  /studies/{id}/report    - structured report
GET  /studies/{id}/report/pdf- exported PDF report
"""
from __future__ import annotations
import shutil
import zipfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.db.models import Study, Patient, Finding, Report
from app.schemas.schemas import StudyOut, FindingOut, ReportOut

from app.preprocessing.dicom_loader import preprocess_ct_study, save_volume_npz
from app.segmentation.lung_segmentation import get_lung_segmenter
from app.segmentation.nodule_segmentation import get_nodule_segmenter
from app.detection.nodule_detector import get_nodule_detector
from app.classification.disease_classifier import get_disease_classifier
from app.xai.gradcam import generate_gradcam
from app.llm.report_generator import get_report_generator
from app.utils.pdf_export import export_report_pdf

import numpy as np

router = APIRouter(prefix="/studies", tags=["studies"])


@router.post("/upload", response_model=StudyOut)
async def upload_study(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")

    upload_id = str(uuid.uuid4())
    dest_dir = Path(settings.UPLOAD_DIR) / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    upload_path = dest_dir / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    dicom_dir = dest_dir
    if file.filename.lower().endswith(".zip"):
        with zipfile.ZipFile(upload_path) as zf:
            zf.extractall(dest_dir)
        upload_path.unlink()

    try:
        volume = preprocess_ct_study(dicom_dir)
    except Exception as e:
        raise HTTPException(400, f"Failed to process DICOM series: {e}")

    cache_path = Path(settings.CACHE_DIR) / f"{upload_id}.npz"
    save_volume_npz(volume, cache_path)

    study = Study(
        patient_id=patient.id,
        study_uid=volume.study_uid,
        modality="CT",
        series_count=1,
        storage_path=str(cache_path),
        status="uploaded",
    )
    db.add(study)
    db.commit()
    db.refresh(study)
    return study


@router.post("/{study_id}/analyze")
def analyze_study(study_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(404, "Study not found")

    study.status = "preprocessing"
    db.commit()

    background_tasks.add_task(_run_pipeline, study_id)
    return {"status": "analysis_started", "study_id": study_id}


def _run_pipeline(study_id: str):
    """Runs synchronously in a background task; in production this would be
    a Celery/RQ worker job so the API stays responsive under load."""
    from app.db.session import SessionLocal
    db = SessionLocal()
    try:
        study = db.query(Study).filter(Study.id == study_id).first()
        data = np.load(study.storage_path)
        volume, spacing = data["array"], tuple(data["spacing"])

        # 1. Lung segmentation
        lung_mask = get_lung_segmenter().segment(volume)

        # 2. Nodule detection
        candidates = get_nodule_detector().detect(volume, lung_mask, spacing)

        # 3. Per-candidate nodule segmentation + Grad-CAM handled at finding level
        nodule_segmenter = get_nodule_segmenter()
        findings_payload = []
        for c in candidates[:15]:  # cap findings surfaced per study
            mask, bounds = nodule_segmenter.segment_roi(volume, c.center_zyx)
            seg_path = Path(settings.CACHE_DIR) / f"{study_id}_{uuid.uuid4().hex[:8]}_mask.npz"
            np.savez_compressed(seg_path, mask=mask, bounds=bounds)

            finding = Finding(
                study_id=study.id,
                finding_type="nodule",
                confidence=c.confidence,
                bbox=c.bbox,
                diameter_mm=c.diameter_mm,
                slice_index=c.center_zyx[0],
                segmentation_path=str(seg_path),
            )
            db.add(finding)
            findings_payload.append({
                "finding_type": "nodule",
                "diameter_mm": c.diameter_mm,
                "confidence": c.confidence,
            })

        # 4. Disease classification (whole-volume)
        classification = get_disease_classifier().classify(volume, lung_mask)
        for label, result in classification.items():
            if result["probability"] >= 0.3:
                heatmap = generate_gradcam(volume, label)
                heatmap_path = Path(settings.CACHE_DIR) / f"{study_id}_{label}_heatmap.npz"
                np.savez_compressed(heatmap_path, heatmap=heatmap)

                finding = Finding(
                    study_id=study.id,
                    finding_type="disease_classification",
                    label=label,
                    confidence=result["probability"],
                    severity=result["severity"],
                    bbox={},
                    heatmap_path=str(heatmap_path),
                )
                db.add(finding)
                findings_payload.append({
                    "finding_type": "disease_classification",
                    "label": label,
                    "confidence": result["probability"],
                    "severity": result["severity"],
                })

        db.commit()

        # 5. Report generation (RAG-grounded)
        structured = get_report_generator().generate(findings_payload)
        report = Report(
            study_id=study.id,
            findings_text=structured.findings,
            impression_text=structured.impression,
            recommendations_text=structured.recommendations,
            patient_summary_text=structured.patient_summary,
            references_json=structured.references,
        )
        db.add(report)

        study.status = "analyzed"
        db.commit()
    except Exception as e:
        study.status = "failed"
        db.commit()
        raise e
    finally:
        db.close()


@router.get("/{study_id}", response_model=StudyOut)
def get_study(study_id: str, db: Session = Depends(get_db)):
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(404, "Study not found")
    return study


@router.get("/{study_id}/findings", response_model=list[FindingOut])
def get_findings(study_id: str, db: Session = Depends(get_db)):
    return db.query(Finding).filter(Finding.study_id == study_id).all()


@router.get("/{study_id}/report", response_model=ReportOut)
def get_report(study_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.study_id == study_id).first()
    if not report:
        raise HTTPException(404, "Report not yet generated")
    return report


@router.get("/{study_id}/report/pdf")
def get_report_pdf(study_id: str, db: Session = Depends(get_db)):
    report = db.query(Report).filter(Report.study_id == study_id).first()
    study = db.query(Study).filter(Study.id == study_id).first()
    if not report or not study:
        raise HTTPException(404, "Report not found")

    pdf_path = Path(settings.REPORTS_DIR) / f"{study_id}.pdf"
    export_report_pdf(report, study, pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"PulmoScan_Report_{study_id}.pdf")
