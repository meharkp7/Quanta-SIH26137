import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { ToastProvider } from "./components/ui/Toast";
import "./index.css";

// ToastProvider must wrap the whole app: App calls useToast() at its top level
// (before the home/simulation early return), so the provider has to live above it.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>
);
