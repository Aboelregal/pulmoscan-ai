export default function Dashboard() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-semibold mb-1">Dashboard</h1>
      <p className="text-slate-400 mb-6">Overview of recent studies and AI analysis throughput.</p>

      <div className="grid grid-cols-4 gap-4 mb-8">
        {[
          { label: "Studies Today", value: "—" },
          { label: "Pending Analysis", value: "—" },
          { label: "Flagged Findings", value: "—" },
          { label: "Avg. Analysis Time", value: "—" },
        ].map((s) => (
          <div key={s.label} className="bg-panel2 rounded-lg p-4 border border-slate-800">
            <div className="text-xs text-slate-400">{s.label}</div>
            <div className="text-2xl font-semibold mt-1">{s.value}</div>
          </div>
        ))}
      </div>

      <div className="bg-panel2 rounded-lg p-6 border border-slate-800">
        <h2 className="font-medium mb-2">Getting started</h2>
        <ol className="list-decimal list-inside text-sm text-slate-300 space-y-1">
          <li>Go to Study Browser and create a patient</li>
          <li>Upload a Chest CT DICOM series (zip of .dcm files)</li>
          <li>Run AI analysis — segmentation, detection, classification, report generation</li>
          <li>Open the CT viewer to review overlays and the generated report</li>
        </ol>
      </div>
    </div>
  );
}
