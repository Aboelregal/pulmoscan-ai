import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { StudyAPI, Finding, Report } from "../api/client";

type PlaneView = "axial" | "coronal" | "sagittal";

export default function CTViewer() {
  const { studyId } = useParams();
  const [findings, setFindings] = useState<Finding[]>([]);
  const [report, setReport] = useState<Report | null>(null);
  const [plane, setPlane] = useState<PlaneView>("axial");
  const [sliceIndex, setSliceIndex] = useState(50);
  const [windowLevel, setWindowLevel] = useState({ center: -600, width: 1500 });
  const [showOverlays, setShowOverlays] = useState(true);
  const [activeTab, setActiveTab] = useState<"findings" | "report">("findings");

  useEffect(() => {
    if (!studyId) return;
    StudyAPI.findings(studyId).then((r) => setFindings(r.data)).catch(() => {});
    StudyAPI.report(studyId).then((r) => setReport(r.data)).catch(() => setReport(null));
  }, [studyId]);

  return (
    <div className="flex h-screen">
      {/* Main viewer area */}
      <div className="flex-1 flex flex-col bg-black">
        <div className="flex items-center gap-4 px-4 py-2 bg-panel2 border-b border-slate-800 text-sm">
          <div className="flex gap-1">
            {(["axial", "coronal", "sagittal"] as PlaneView[]).map((p) => (
              <button
                key={p}
                onClick={() => setPlane(p)}
                className={`px-3 py-1 rounded ${plane === p ? "bg-accent" : "bg-slate-700"}`}
              >
                {p[0].toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <label className="text-slate-400">W:</label>
            <input type="number" value={windowLevel.width}
                   onChange={(e) => setWindowLevel((w) => ({ ...w, width: Number(e.target.value) }))}
                   className="w-16 bg-panel border border-slate-700 rounded px-1" />
            <label className="text-slate-400">L:</label>
            <input type="number" value={windowLevel.center}
                   onChange={(e) => setWindowLevel((w) => ({ ...w, center: Number(e.target.value) }))}
                   className="w-16 bg-panel border border-slate-700 rounded px-1" />
          </div>
          <label className="flex items-center gap-1 text-slate-400">
            <input type="checkbox" checked={showOverlays} onChange={(e) => setShowOverlays(e.target.checked)} />
            Overlays
          </label>
        </div>

        {/* Canvas placeholder: wire to Cornerstone.js / a WebGL volume renderer
            fed by /studies/{id}/slices/{plane}/{index} in a real deployment */}
        <div className="flex-1 flex items-center justify-center relative">
          <div className="w-[512px] h-[512px] bg-neutral-900 border border-slate-800 relative flex items-center justify-center">
            <span className="text-slate-600 text-sm">
              {plane} slice {sliceIndex} — DICOM render surface
              <br />(wire to Cornerstone.js in production)
            </span>
            {showOverlays &&
              findings
                .filter((f) => f.finding_type === "nodule" && f.slice_index === sliceIndex)
                .map((f) => (
                  <div
                    key={f.id}
                    className="absolute border-2 border-red-500"
                    style={{
                      left: `${(f.bbox.x ?? 100) % 400}px`,
                      top: `${(f.bbox.y ?? 100) % 400}px`,
                      width: `${(f.bbox.w ?? 30) + 10}px`,
                      height: `${(f.bbox.h ?? 30) + 10}px`,
                    }}
                  >
                    <span className="absolute -top-5 left-0 text-[10px] bg-red-500 px-1 rounded">
                      {(f.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
          </div>
        </div>

        <div className="px-4 py-3 bg-panel2 border-t border-slate-800">
          <input
            type="range"
            min={0}
            max={200}
            value={sliceIndex}
            onChange={(e) => setSliceIndex(Number(e.target.value))}
            className="w-full"
          />
          <div className="text-xs text-slate-400 text-center">Slice {sliceIndex} / 200</div>
        </div>
      </div>

      {/* Side panel: findings / report */}
      <aside className="w-96 bg-panel2 border-l border-slate-800 flex flex-col">
        <div className="flex border-b border-slate-800 text-sm">
          <button
            onClick={() => setActiveTab("findings")}
            className={`flex-1 py-2 ${activeTab === "findings" ? "bg-slate-800" : ""}`}
          >
            Findings ({findings.length})
          </button>
          <button
            onClick={() => setActiveTab("report")}
            className={`flex-1 py-2 ${activeTab === "report" ? "bg-slate-800" : ""}`}
          >
            Report
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 text-sm space-y-3">
          {activeTab === "findings" &&
            (findings.length === 0 ? (
              <p className="text-slate-500">No findings yet. Run AI analysis from Study Browser.</p>
            ) : (
              findings.map((f) => (
                <div key={f.id} className="bg-panel rounded p-3 border border-slate-800">
                  <div className="font-medium capitalize">{f.finding_type.replace("_", " ")}</div>
                  {f.label && <div className="text-slate-400 capitalize">{f.label.replace("_", " ")}</div>}
                  <div className="text-xs text-slate-500 mt-1">
                    Confidence: {(f.confidence * 100).toFixed(0)}%
                    {f.diameter_mm && ` · ${f.diameter_mm}mm`}
                    {f.severity && f.severity !== "none" && ` · ${f.severity}`}
                  </div>
                </div>
              ))
            ))}

          {activeTab === "report" &&
            (report ? (
              <div className="space-y-4">
                <section>
                  <h3 className="font-medium text-blue-400 mb-1">Findings</h3>
                  <p className="whitespace-pre-line text-slate-300">{report.findings_text}</p>
                </section>
                <section>
                  <h3 className="font-medium text-blue-400 mb-1">Impression</h3>
                  <p className="whitespace-pre-line text-slate-300">{report.impression_text}</p>
                </section>
                <section>
                  <h3 className="font-medium text-blue-400 mb-1">Recommendations</h3>
                  <p className="whitespace-pre-line text-slate-300">{report.recommendations_text}</p>
                </section>
                <section>
                  <h3 className="font-medium text-blue-400 mb-1">Patient Summary</h3>
                  <p className="text-slate-300">{report.patient_summary_text}</p>
                </section>
                {report.references_json && (
                  <section>
                    <h3 className="font-medium text-blue-400 mb-1">References</h3>
                    <ul className="text-xs text-slate-400 space-y-1">
                      {report.references_json.map((r, i) => (
                        <li key={i}>
                          <a href={r.url} target="_blank" rel="noreferrer" className="underline">
                            {r.title}
                          </a>{" "}
                          ({r.source})
                        </li>
                      ))}
                    </ul>
                  </section>
                )}
                {studyId && (
                  <a
                    href={StudyAPI.reportPdfUrl(studyId)}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-block mt-2 bg-accent px-3 py-1 rounded text-xs"
                  >
                    Export PDF
                  </a>
                )}
              </div>
            ) : (
              <p className="text-slate-500">No report yet. Run AI analysis first.</p>
            ))}
        </div>
      </aside>
    </div>
  );
}
