import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import "./index.css";

import Dashboard from "./pages/Dashboard";
import StudyBrowser from "./pages/StudyBrowser";
import CTViewer from "./pages/CTViewer";
import AppShell from "./components/AppShell";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/studies" element={<StudyBrowser />} />
          <Route path="/studies/:studyId" element={<CTViewer />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  </React.StrictMode>
);
