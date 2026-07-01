import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, StudyAPI, Study } from "../api/client";

export default function StudyBrowser() {
  const [patientId, setPatientId] = useState("");
  const [mrn, setMrn] = useState("");
  const [name, setName] = useState("");
  const [studies, setStudies] = useState<Study[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");
  const navigate = useNavigate();

  const createPatient = async () => {
    const resp = await api.post("/patients/", { mrn, name });
    setPatientId(resp.data.id);
    setStatus(`Patient created: ${resp.data.id}`);
  };

  const loadStudies = async () => {
    if (!patientId) return;
    const resp = await StudyAPI.list(patientId);
    setStudies(resp.data);
  };

  const uploadFile = async () => {
    if (!patientId || !file) return;
    setStatus("Uploading + preprocessing DICOM series...");
    try {
      const resp = await StudyAPI.upload(patientId, file);
      setStatus(`Study uploaded: ${resp.data.id}`);
      loadStudies();
    } catch (e: any) {
      setStatus(`Upload failed: ${e?.response?.data?.detail ?? e.message}`);
    }
  };

  const runAnalysis = async (studyId: string) => {
    setStatus(`Starting AI analysis for ${studyId}...`);
    await StudyAPI.analyze(studyId);
    setStatus("Analysis running in background — refresh study list to check status.");
  };

  return (
    <div className="p-8 max-w-3xl">
      <h1 className="text-2xl font-semibold mb-6">Study Browser</h1>

      <div className="bg-panel2 rounded-lg p-4 border border-slate-800 mb-6">
        <h2 className="font-medium mb-3">1. Patient</h2>
        <div className="flex gap-2 mb-2">
          <input className="bg-panel border border-slate-700 rounded px-2 py-1 text-sm flex-1"
                 placeholder="MRN" value={mrn} onChange={(e) => setMrn(e.target.value)} />
          <input className="bg-panel border border-slate-700 rounded px-2 py-1 text-sm flex-1"
                 placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} />
          <button onClick={createPatient} className="bg-accent px-3 py-1 rounded text-sm">Create</button>
        </div>
        <input className="bg-panel border border-slate-700 rounded px-2 py-1 text-sm w-full"
               placeholder="Or paste existing patient_id" value={patientId}
               onChange={(e) => setPatientId(e.target.value)} />
      </div>

      <div className="bg-panel2 rounded-lg p-4 border border-slate-800 mb-6">
        <h2 className="font-medium mb-3">2. Upload Chest CT DICOM series (.zip)</h2>
        <input type="file" accept=".zip" onChange={(e) => setFile(e.target.files?.[0] ?? null)}
               className="text-sm mb-2 block" />
        <button onClick={uploadFile} className="bg-accent px-3 py-1 rounded text-sm">Upload</button>
      </div>

      <div className="bg-panel2 rounded-lg p-4 border border-slate-800 mb-6">
        <div className="flex justify-between items-center mb-3">
          <h2 className="font-medium">3. Studies</h2>
          <button onClick={loadStudies} className="text-xs text-blue-400 underline">Refresh</button>
        </div>
        <div className="space-y-2">
          {studies.map((s) => (
            <div key={s.id} className="flex justify-between items-center bg-panel rounded px-3 py-2 text-sm">
              <div>
                <div className="font-mono text-xs text-slate-400">{s.id}</div>
                <div>Status: <span className="text-blue-400">{s.status}</span></div>
              </div>
              <div className="flex gap-2">
                <button onClick={() => runAnalysis(s.id)} className="bg-slate-700 px-2 py-1 rounded text-xs">
                  Run AI Analysis
                </button>
                <button onClick={() => navigate(`/studies/${s.id}`)} className="bg-accent px-2 py-1 rounded text-xs">
                  Open Viewer
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {status && <div className="text-sm text-slate-400">{status}</div>}
    </div>
  );
}
