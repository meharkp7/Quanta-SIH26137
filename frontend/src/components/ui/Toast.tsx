import { useCallback, useReducer, useEffect, useRef } from "react";

export type ToastKind = "info" | "success" | "warning" | "error";

export interface Toast {
  id: string;
  kind: ToastKind;
  title: string;
  message?: string;
  action?: { label: string; onClick: () => void };
  duration?: number;
}

interface State {
  toasts: Toast[];
}

type Action =
  | { type: "ADD"; toast: Toast }
  | { type: "REMOVE"; id: string }
  | { type: "CLEAR" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "ADD":
      return { toasts: [...state.toasts, action.toast] };
    case "REMOVE":
      return { toasts: state.toasts.filter((t) => t.id !== action.id) };
    case "CLEAR":
      return { toasts: [] };
    default:
      return state;
  }
}

const ToastContext = React.createContext<{
  toasts: Toast[];
  push: (toast: Omit<Toast, "id">) => string;
  remove: (id: string) => void;
  clear: () => void;
} | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { toasts: [] });

  const push = useCallback((toast: Omit<Toast, "id">) => {
    const id = Math.random().toString(36).slice(2);
    dispatch({ type: "ADD", toast: { ...toast, id } });
    if (toast.duration !== 0) {
      setTimeout(() => dispatch({ type: "REMOVE", id }), toast.duration ?? 4000);
    }
    return id;
  }, []);

  const remove = useCallback((id: string) => dispatch({ type: "REMOVE", id }), []);
  const clear = useCallback(() => dispatch({ type: "CLEAR" }), []);

  return (
    <ToastContext.Provider value={{ toasts: state.toasts, push, remove, clear }}>
      {children}
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = React.useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}

import React from "react";

function ToastItem({ toast, onClose }: { toast: Toast; onClose: (id: string) => void }) {
  const icons: Record<ToastKind, React.ReactNode> = {
    info: <svg viewBox="0 0 24 24" width="16" height="16"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none"/><path d="M12 16v-4M12 8h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg>,
    success: <svg viewBox="0 0 24 24" width="16" height="16"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" stroke="currentColor" strokeWidth="2" fill="none"/><polyline points="22 4 12 14.01 9 11.01" stroke="currentColor" strokeWidth="2" fill="none"/></svg>,
    warning: <svg viewBox="0 0 24 24" width="16" height="16"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="currentColor" strokeWidth="2" fill="none"/><line x1="12" y1="9" x2="12" y2="13" stroke="currentColor" strokeWidth="2"/><line x1="12" y1="17" x2="12.01" y2="17" stroke="currentColor" strokeWidth="2"/></svg>,
    error: <svg viewBox="0 0 24 24" width="16" height="16"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" fill="none"/><line x1="15" y1="9" x2="9" y2="15" stroke="currentColor" strokeWidth="2"/><line x1="9" y1="9" x2="15" y2="15" stroke="currentColor" strokeWidth="2"/></svg>,
  };
  return (
    <div
      className={`toast-item toast-${toast.kind}`}
      style={{ "--toast-kind": toast.kind } as React.CSSProperties}
    >
      <div className="toast-icon">{icons[toast.kind]}</div>
      <div className="toast-content">
        <div className="toast-title">{toast.title}</div>
        {toast.message && <div className="toast-message">{toast.message}</div>}
      </div>
      {toast.action && (
        <button className="toast-action" onClick={toast.action.onClick}>
          {toast.action.label}
        </button>
      )}
      <button className="toast-close" onClick={() => onClose(toast.id)}>
        <svg viewBox="0 0 24 24" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2"/><line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2"/></svg>
      </button>
    </div>
  );
}

export function ToastContainer() {
  const { toasts, remove } = useToast();
  return (
    <div className="toast-container" aria-live="polite" aria-atomic="true">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onClose={remove} />
      ))}
    </div>
  );
}