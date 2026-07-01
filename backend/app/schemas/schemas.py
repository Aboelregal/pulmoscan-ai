from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class PatientCreate(BaseModel):
    mrn: str
    name: str
    birth_date: Optional[datetime] = None
    sex: Optional[str] = None


class PatientOut(PatientCreate):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class StudyOut(BaseModel):
    id: str
    patient_id: str
    study_uid: str
    status: str
    series_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class FindingOut(BaseModel):
    id: str
    finding_type: str
    label: Optional[str]
    confidence: float
    bbox: dict
    diameter_mm: Optional[float]
    severity: Optional[str]
    slice_index: Optional[int]

    class Config:
        from_attributes = True


class ReportOut(BaseModel):
    findings_text: str
    impression_text: str
    recommendations_text: str
    patient_summary_text: Optional[str]
    references_json: Optional[list]
    is_finalized: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserLogin(BaseModel):
    email: str
    password: str
