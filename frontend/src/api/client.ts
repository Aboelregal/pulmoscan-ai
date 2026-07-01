import axios from "axios";

export const api = axios.create({
  baseURL: "/api/v1",
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("ctvision_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export interface Study {
  id: string;
  patient_id: string;
  study_uid: string;
  status: string;
  series_count: number;
  created_at: string;
}

export interface Finding {
  id: string;
  finding_type: string;
  label: string | null;
  confidence: number;
  bbox: Record<string, number>;
  diameter_mm: number | null;
  severity: string | null;
  slice_index: number | null;
}

export interface Report {
  findings_text: string;
  impression_text: string;
  recommendations_text: string;
  patient_summary_text: string | null;
  references_json: { title: string; source: string; url: string }[] | null;
  is_finalized: boolean;
}

export const StudyAPI = {
  list: (patientId: string) => api.get<Study[]>(`/patients/${patientId}/studies`),
  get: (studyId: string) => api.get<Study>(`/studies/${studyId}`),
  analyze: (studyId: string) => api.post(`/studies/${studyId}/analyze`),
  findings: (studyId: string) => api.get<Finding[]>(`/studies/${studyId}/findings`),
  report: (studyId: string) => api.get<Report>(`/studies/${studyId}/report`),
  reportPdfUrl: (studyId: string) => `/api/v1/studies/${studyId}/report/pdf`,
  upload: (patientId: string, file: File) => {
    const form = new FormData();
    form.append("patient_id", patientId);
    form.append("file", file);
    return api.post<Study>("/studies/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};
