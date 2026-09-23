import React from "react";
import ReactDOM from "react-dom/client";
import App from "./workspace/WorkspaceApp";
import { ToastProvider } from "./components/ui/Toast";
import "./index.css";
import "./workspace/styles.css";

// Keep shared notification support available to workspace components.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>
);
