import { Link } from "react-router-dom";
import { ReactNode } from "react";

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen flex">
      <aside className="w-56 bg-panel2 border-r border-slate-800 flex flex-col">
        <div className="p-4 border-b border-slate-800">
          <div className="text-lg font-semibold text-blue-400">PulmoScan AI</div>
          <div className="text-xs text-slate-400">Chest CT Analysis</div>
        </div>
        <nav className="flex-1 p-2 space-y-1 text-sm">
          <Link to="/" className="block px-3 py-2 rounded hover:bg-slate-800">Dashboard</Link>
          <Link to="/studies" className="block px-3 py-2 rounded hover:bg-slate-800">Study Browser</Link>
        </nav>
        <div className="p-4 text-xs text-slate-500 border-t border-slate-800">
          For research/portfolio use only.
          <br />Not a medical device.
        </div>
      </aside>
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
