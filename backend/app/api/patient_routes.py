from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import Patient, Study
from app.schemas.schemas import PatientCreate, PatientOut, StudyOut

router = APIRouter(prefix="/patients", tags=["patients"])


@router.post("/", response_model=PatientOut)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    if db.query(Patient).filter(Patient.mrn == payload.mrn).first():
        raise HTTPException(400, "Patient with this MRN already exists")
    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("/", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).all()


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient


@router.get("/{patient_id}/studies", response_model=list[StudyOut])
def get_patient_studies(patient_id: str, db: Session = Depends(get_db)):
    return db.query(Study).filter(Study.patient_id == patient_id).all()
