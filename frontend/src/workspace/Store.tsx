import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import type { ReactNode } from "react";
import type { Session } from "@supabase/supabase-js";
import { supabase, setRequestIdentity } from "./supabase";
import { Company, demoCompany, Draft, Run, seedDrafts } from "./data";

type Store = {
  session: Session | null;
  demo: boolean;
  ready: boolean;
  dataReady: boolean;
  company: Company | null;
  companies: Company[];
  drafts: Draft[];
  runs: Run[];
  error: string;
  canEdit: boolean;
  enterDemo: () => void;
  signOut: () => Promise<void>;
  reloadCompanies: () => Promise<void>;
  selectCompany: (id: string) => void;
  saveDraft: (draft: Draft) => Promise<void>;
  saveRun: (run: Run) => Promise<void>;
  deleteDraft: (id: string) => Promise<void>;
  resetDemo: () => void;
  setError: (value: string) => void;
};
const Context = createContext<Store | null>(null);
const DEMO_KEY = "quanta.demo.workspace.v1";

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [demo, setDemo] = useState(() => {
    if (location.pathname === "/auth/callback")
      sessionStorage.removeItem("quanta.demo");
    return sessionStorage.getItem("quanta.demo") === "true";
  });
  const [ready, setReady] = useState(!supabase);
  const [dataReady, setDataReady] = useState(false);
  const [companies, setCompanies] = useState<Company[]>([]);
  const [selected, setSelected] = useState("");
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");
  const company = demo
    ? demoCompany
    : companies.find((c) => c.id === selected) || companies[0] || null;
  const canEdit = company?.role !== "viewer";
  const epoch = useRef(0);
  const records = useRef({ drafts, runs });
  records.current = { drafts, runs };
  setRequestIdentity(
    demo
      ? { demo: true }
      : { token: session?.access_token, company: company?.id },
  );

  useEffect(() => {
    if (!supabase) return;
    supabase.auth.getSession().then(({ data, error }) => {
      setSession(data.session);
      if (error) setError(error.message);
      setReady(true);
    });
    const { data } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      setReady(true);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  const reloadCompanies = useCallback(async () => {
    if (!supabase || !session) {
      setCompanies([]);
      return;
    }
    const accepted = await supabase.rpc("accept_company_invites");
    if (accepted.error) throw accepted.error;
    const { data, error } = await supabase
      .from("company_members")
      .select("role, companies(id, name, industry)")
      .eq("user_id", session.user.id);
    if (error) throw error;
    setCompanies(
      (data || []).map((row: any) => ({ ...row.companies, role: row.role })),
    );
  }, [session?.user.id]);

  useEffect(() => {
    if (!demo) {
      setDataReady(false);
      reloadCompanies().catch((e) => {
        setError(e.message);
        setDataReady(true);
      });
    }
  }, [reloadCompanies, demo]);

  useEffect(() => {
    const generation = ++epoch.current;
    setDrafts([]);
    setRuns([]);
    setDataReady(false);
    if (demo) {
      try {
        const saved = localStorage.getItem(DEMO_KEY);
        const data = saved
          ? JSON.parse(saved)
          : { drafts: seedDrafts(), runs: [] };
        setDrafts(data.drafts);
        setRuns(data.runs);
        records.current = data;
        localStorage.setItem(DEMO_KEY, JSON.stringify(data));
      } catch {
        const data = { drafts: seedDrafts(), runs: [] };
        setDrafts(data.drafts);
        records.current = data;
      }
      setDataReady(true);
      return;
    }
    if (!company || !supabase) {
      setDataReady(true);
      return;
    }
    supabase
      .from("workspace_records")
      .select("kind,payload")
      .eq("company_id", company.id)
      .order("updated_at", { ascending: false })
      .then(({ data, error }) => {
        if (epoch.current !== generation) return;
        if (error) setError(error.message);
        else {
          setDrafts(
            (data || [])
              .filter((r) => r.kind === "scenario")
              .map((r) => r.payload as Draft),
          );
          setRuns(
            (data || [])
              .filter((r) => r.kind === "run")
              .map((r) => r.payload as Run),
          );
        }
        setDataReady(true);
      });
  }, [demo, company?.id, session?.user.id]);

  async function persist(kind: "scenario" | "run", payload: Draft | Run) {
    if (!company || !canEdit)
      throw new Error("A dispatcher or owner can save workspace changes.");
    const generation = epoch.current;
    if (!demo) {
      if (!supabase) throw new Error("Supabase is not configured.");
      const { error } = await supabase
        .from("workspace_records")
        .upsert({
          id: payload.id,
          company_id: company.id,
          kind,
          payload,
          updated_at: new Date().toISOString(),
        });
      if (error) throw error;
    }
    if (epoch.current !== generation) return;
    const next = { ...records.current };
    if (kind === "scenario")
      next.drafts = [
        payload as Draft,
        ...next.drafts.filter((d) => d.id !== payload.id),
      ];
    else
      next.runs = [
        payload as Run,
        ...next.runs.filter((r) => r.id !== payload.id),
      ];
    if (demo) localStorage.setItem(DEMO_KEY, JSON.stringify(next));
    records.current = next;
    setDrafts(next.drafts);
    setRuns(next.runs);
  }

  async function deleteDraft(id: string) {
    if (!canEdit || !company) throw new Error("Dispatcher access required.");
    const generation = epoch.current;
    if (!demo && supabase) {
      const { error } = await supabase
        .from("workspace_records")
        .delete()
        .eq("id", id)
        .eq("company_id", company.id)
        .eq("kind", "scenario");
      if (error) throw error;
    }
    if (epoch.current !== generation) return;
    const next = {
      ...records.current,
      drafts: records.current.drafts.filter((d) => d.id !== id),
    };
    if (demo) localStorage.setItem(DEMO_KEY, JSON.stringify(next));
    records.current = next;
    setDrafts(next.drafts);
  }
  function enterDemo() {
    setError("");
    sessionStorage.setItem("quanta.demo", "true");
    setDemo(true);
  }
  async function signOut() {
    if (!demo && supabase) {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
    }
    sessionStorage.removeItem("quanta.demo");
    setDemo(false);
    setCompanies([]);
    setSelected("");
    setError("");
  }
  function resetDemo() {
    if (!demo) return;
    const data = { drafts: seedDrafts(), runs: [] };
    localStorage.setItem(DEMO_KEY, JSON.stringify(data));
    records.current = data;
    setDrafts(data.drafts);
    setRuns([]);
  }
  return (
    <Context.Provider
      value={{
        session,
        demo,
        ready,
        dataReady,
        company,
        companies,
        drafts,
        runs,
        error,
        canEdit,
        enterDemo,
        signOut,
        reloadCompanies,
        selectCompany: setSelected,
        saveDraft: (d) => persist("scenario", d),
        saveRun: (r) => persist("run", r),
        deleteDraft,
        resetDemo,
        setError,
      }}
    >
      {children}
    </Context.Provider>
  );
}
export function useWorkspace() {
  const value = useContext(Context);
  if (!value) throw new Error("WorkspaceProvider missing");
  return value;
}
