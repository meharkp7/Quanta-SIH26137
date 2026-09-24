import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { postJson, SOLVE_TIMEOUT_MS } from "../api";
import {
  clone,
  Draft,
  Result,
  Run,
  requestBody,
  uid,
  validateDraft,
} from "./data";
import { useWorkspace } from "./Store";

type Job = {
  id: string;
  draft: Draft;
  status: "running" | "complete" | "error" | "cancelled";
  started: number;
  error?: string;
  run?: Run;
};
type Runner = {
  job: Job | null;
  start: (draft: Draft) => Promise<Run | undefined>;
  cancel: () => void;
};
const Context = createContext<Runner | null>(null);
export function RunProvider({ children }: { children: ReactNode }) {
  const store = useWorkspace();
  const [job, setJob] = useState<Job | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function start(source: Draft) {
    if (controller.current)
      throw new Error("A computation is already running.");
    if (!store.canEdit)
      throw new Error("Dispatcher access is required to run a scenario.");
    const invalid = validateDraft(source);
    if (invalid) throw new Error(invalid);
    const draft = clone(source),
      id = uid();
    draft.updatedAt = new Date().toISOString();
    const abort = new AbortController();
    controller.current = abort;
    setJob({ id, draft, status: "running", started: Date.now() });
    try {
      await store.saveDraft(draft);
      const result = await postJson<Result>(
        "/api/workspace/solve",
        requestBody(draft),
        SOLVE_TIMEOUT_MS,
        abort.signal,
      );
      if (abort.signal.aborted) return;
      const run: Run = {
        id,
        draft,
        result,
        createdAt: new Date().toISOString(),
        provenance: "computed",
      };
      let persistenceError = "";
      try {
        await store.saveRun(run);
      } catch (e) {
        persistenceError = `Computed successfully, but saving failed: ${(e as Error).message}. Export this result or retry saving.`;
      }
      setJob((j) =>
        j?.id === id
          ? { ...j, status: "complete", run, error: persistenceError }
          : j,
      );
      return run;
    } catch (e) {
      setJob((j) =>
        j?.id === id
          ? {
              ...j,
              status: abort.signal.aborted ? "cancelled" : "error",
              error: abort.signal.aborted
                ? "Request cancelled. The server may finish its current computation; no result will be saved."
                : (e as Error).message,
            }
          : j,
      );
      return undefined;
    } finally {
      controller.current = null;
    }
  }
  function cancel() {
    controller.current?.abort();
  }
  return (
    <Context.Provider value={{ job, start, cancel }}>
      {children}
    </Context.Provider>
  );
}
export function useRunner() {
  const ctx = useContext(Context);
  if (!ctx) throw new Error("RunProvider missing");
  return ctx;
}
