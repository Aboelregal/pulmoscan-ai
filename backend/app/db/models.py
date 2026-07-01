"""
SQLAlchemy ORM models -> mirrors the schema described in README/docs/schema.sql
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Float, JSON, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, default="radiologist")  # radiologist | admin | technician
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    mrn = Column(String, unique=True, index=True)  # medical record number (pseudonymized)
    name = Column(String)
    birth_date = Column(DateTime, nullable=True)
    sex = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    studies = relationship("Study", back_populates="patient", cascade="all, delete-orphan")


class Study(Base):
    __tablename__ = "studies"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    patient_id = Column(UUID(as_uuid=False), ForeignKey("patients.id"))
    study_uid = Column(String, unique=True, index=True)
    study_date = Column(DateTime, nullable=True)
    modality = Column(String, default="CT")
    description = Column(String, nullable=True)
    series_count = Column(Integer, default=0)
    storage_path = Column(String)  # path to preprocessed volume (.nii.gz / .npz)
    status = Column(String, default="uploaded")  # uploaded | preprocessing | analyzed | failed
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="studies")
    findings = relationship("Finding", back_populates="study", cascade="all, delete-orphan")
    report = relationship("Report", back_populates="study", uselist=False, cascade="all, delete-orphan")


class Finding(Base):
    """A single AI-detected finding: nodule, opacity, effusion region, etc."""
    __tablename__ = "findings"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    study_id = Column(UUID(as_uuid=False), ForeignKey("studies.id"))
    finding_type = Column(String)  # nodule | mass | effusion | opacity | atelectasis...
    label = Column(String, nullable=True)  # disease classification label if applicable
    confidence = Column(Float)
    bbox = Column(JSON)  # {x,y,z,w,h,d} in voxel space
    segmentation_path = Column(String, nullable=True)  # mask file path
    slice_index = Column(Integer, nullable=True)
    diameter_mm = Column(Float, nullable=True)
    severity = Column(String, nullable=True)  # mild | moderate | severe
    heatmap_path = Column(String, nullable=True)  # Grad-CAM overlay path
    created_at = Column(DateTime, default=datetime.utcnow)

    study = relationship("Study", back_populates="findings")


class Report(Base):
    __tablename__ = "reports"
    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    study_id = Column(UUID(as_uuid=False), ForeignKey("studies.id"), unique=True)
    findings_text = Column(Text)
    impression_text = Column(Text)
    recommendations_text = Column(Text)
    patient_summary_text = Column(Text, nullable=True)
    references_json = Column(JSON, nullable=True)  # RAG citations
    is_finalized = Column(Boolean, default=False)
    reviewed_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    study = relationship("Study", back_populates="report")
