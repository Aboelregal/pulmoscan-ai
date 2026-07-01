"""
Structured radiology report generation.

Two backends:
  1. Local HF model (MedGemma or similar instruct model) - used when
     settings.LLM_USE_REMOTE_API is False and the model can be loaded.
  2. Deterministic template engine - always available, zero dependencies,
     and used as the fallback so report generation never hard-fails. This
     is also genuinely useful to keep around: many real radiology report
     systems use structured templates + NLG hybrids rather than pure
     free-generation, precisely because it's auditable.

Either path is grounded with literature snippets from the RAG retriever
before generation, and both produce the same structured schema: Findings,
Impression, Recommendations, Patient Summary, References.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from app.rag.retriever import get_retriever, RetrievedDoc
from app.core.config import settings

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class StructuredReport:
    findings: str
    impression: str
    recommendations: str
    patient_summary: str
    references: list[dict] = field(default_factory=list)


class ReportGenerator:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm_model = None
        self.tokenizer = None
        self.llm_ready = False
        if _TRANSFORMERS_AVAILABLE and not settings.LLM_USE_REMOTE_API:
            self._try_load_local_llm()

    def _try_load_local_llm(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_NAME, torch_dtype="auto"
            )
            self.llm_ready = True
        except Exception:
            # Model gated/unavailable/no internet at build time -> template fallback.
            self.llm_ready = False

    def generate(self, findings: list[dict], patient_context: dict | None = None) -> StructuredReport:
        query = self._findings_to_query(findings)
        references = self.retriever.retrieve(query, top_k=5)

        if self.llm_ready:
            return self._generate_with_llm(findings, references, patient_context)
        return self._generate_with_template(findings, references, patient_context)

    # ---------- shared helpers ----------

    def _findings_to_query(self, findings: list[dict]) -> str:
        labels = [f["label"] for f in findings if f.get("label")]
        types = [f["finding_type"] for f in findings]
        return " ".join(set(labels + types)) or "normal chest CT"

    # ---------- LLM backend ----------

    def _generate_with_llm(self, findings, references: list[RetrievedDoc], patient_context) -> StructuredReport:
        context_block = "\n".join(f"- {r.title}: {r.snippet}" for r in references)
        findings_block = self._findings_to_text(findings)

        prompt = f"""You are assisting a radiologist by drafting a structured chest CT report.
Use only the findings provided below. Do not invent findings.

DETECTED FINDINGS:
{findings_block}

RELEVANT LITERATURE:
{context_block}

Write four sections: FINDINGS, IMPRESSION, RECOMMENDATIONS, PATIENT_SUMMARY (plain language, 6th-grade reading level).
"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.llm_model.generate(**inputs, max_new_tokens=512, do_sample=False)
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        sections = self._parse_llm_sections(text)
        return StructuredReport(
            findings=sections.get("FINDINGS", findings_block),
            impression=sections.get("IMPRESSION", "See findings."),
            recommendations=sections.get("RECOMMENDATIONS", "Radiologist review recommended."),
            patient_summary=sections.get("PATIENT_SUMMARY", ""),
            references=[r.__dict__ for r in references],
        )

    def _parse_llm_sections(self, text: str) -> dict:
        sections, current = {}, None
        for line in text.splitlines():
            stripped = line.strip()
            for key in ("FINDINGS", "IMPRESSION", "RECOMMENDATIONS", "PATIENT_SUMMARY"):
                if stripped.upper().startswith(key):
                    current = key
                    sections[current] = stripped.split(":", 1)[-1].strip()
                    break
            else:
                if current:
                    sections[current] += " " + stripped
        return sections

    # ---------- Deterministic template backend ----------

    def _findings_to_text(self, findings: list[dict]) -> str:
        if not findings:
            return "No significant abnormalities detected by automated analysis."
        lines = []
        for f in findings:
            desc = f"{f['finding_type'].replace('_', ' ').title()}"
            if f.get("label"):
                desc += f" ({f['label'].replace('_', ' ')})"
            if f.get("diameter_mm"):
                desc += f", diameter {f['diameter_mm']} mm"
            if f.get("confidence") is not None:
                desc += f", AI confidence {f['confidence']*100:.0f}%"
            if f.get("severity") and f["severity"] != "none":
                desc += f", severity: {f['severity']}"
            lines.append(f"- {desc}")
        return "\n".join(lines)

    def _generate_with_template(self, findings, references: list[RetrievedDoc], patient_context) -> StructuredReport:
        findings_text = self._findings_to_text(findings)

        significant = [f for f in findings if f.get("confidence", 0) >= 0.5 or f.get("severity") in ("moderate", "severe")]
        if significant:
            impression_lines = [
                f"{i+1}. {f['finding_type'].replace('_',' ').title()}"
                + (f" ({f['label'].replace('_',' ')})" if f.get("label") else "")
                + " - AI-assisted detection, radiologist correlation required."
                for i, f in enumerate(significant)
            ]
            impression = "\n".join(impression_lines)
        else:
            impression = "No findings meeting significance threshold on automated analysis. Radiologist review still required."

        recommendations = self._build_recommendations(findings)
        patient_summary = self._build_patient_summary(findings)

        return StructuredReport(
            findings=findings_text,
            impression=impression,
            recommendations=recommendations,
            patient_summary=patient_summary,
            references=[r.__dict__ for r in references],
        )

    def _build_recommendations(self, findings: list[dict]) -> str:
        recs = []
        for f in findings:
            if f["finding_type"] == "nodule" and f.get("diameter_mm"):
                d = f["diameter_mm"]
                if d < 6:
                    recs.append(f"Nodule ({d}mm): no routine follow-up needed per Fleischner low-risk criteria; clinical correlation advised.")
                elif d < 8:
                    recs.append(f"Nodule ({d}mm): consider follow-up CT at 6-12 months per Fleischner guidelines.")
                else:
                    recs.append(f"Nodule ({d}mm): consider PET-CT and/or short-interval follow-up (3 months) given size.")
        if not recs:
            recs.append("No specific follow-up indicated based on automated findings. Correlate clinically.")
        recs.append("This AI-generated report requires review and sign-off by a licensed radiologist before clinical use.")
        return "\n".join(f"- {r}" for r in recs)

    def _build_patient_summary(self, findings: list[dict]) -> str:
        if not findings:
            return "Your chest CT scan did not show any concerning findings on automated analysis. Your doctor will review the full results with you."
        return (
            "Your chest CT scan was analyzed by an AI system that flagged "
            f"{len(findings)} area(s) for your doctor to review closely. "
            "This doesn't necessarily mean something is wrong - it means these areas "
            "need a radiologist's expert look, which is a normal part of the process. "
            "Your care team will discuss the full results and any next steps with you."
        )


_generator_singleton: ReportGenerator | None = None


def get_report_generator() -> ReportGenerator:
    global _generator_singleton
    if _generator_singleton is None:
        _generator_singleton = ReportGenerator()
    return _generator_singleton
