import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { ChevronDown, MessageCircleQuestion, Send, Sparkles } from "lucide-react";
import { postJson, SOLVE_TIMEOUT_MS } from "@/api";
import { useQuanta } from "./store";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  text: string;
  source?: "groq" | "local";
};

type AssistantStatus = { configured: boolean; model: string; reason?: string };

type AskResponse = { answer: string; source: "groq" | "local"; facts?: Record<string, unknown> };

// Route → the `section` key app/assistant.py understands.
function sectionFromPath(pathname: string): string {
  if (pathname.startsWith("/app/what-if")) return "what-if";
  if (pathname.startsWith("/app/route-lab")) return "route-lab";
  if (pathname.startsWith("/app/forecasting")) return "forecasting";
  if (pathname.startsWith("/app")) return "dashboard";
  if (pathname.startsWith("/login")) return "login";
  return "dashboard";
}

const GREETING: Message = {
  role: "assistant",
  text: "Hi — I read the numbers your experiments actually produced and explain them in plain English. Ask me anything you just ran.",
};

/**
 * Floating experiment assistant.
 *
 * The page never invents prose: it ships the live context (solve, benchmark,
 * what-if delta, forecast metrics, fleet) to `POST /api/assistant/ask`, which
 * computes grounded facts locally and lets Groq turn those exact numbers into
 * 2–5 plain sentences. When Groq is unreachable the same facts are rendered
 * locally, so the panel keeps working offline.
 */
export function Chatbot() {
  const location = useLocation();
  const { graph, solve, config, scenarioId, results } = useQuanta();

  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<AssistantStatus | null>(null);
  const [error, setError] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Which backend path will answer — checked once when the panel opens.
  useEffect(() => {
    if (!open || status) return;
    fetch("/api/assistant/status")
      .then((r) => (r.ok ? r.json() : null))
      .then((body: AssistantStatus | null) => body && setStatus(body))
      .catch(() => setStatus(null));
  }, [open, status]);

  const section = sectionFromPath(location.pathname);

  /**
   * Everything the analyser can use, in the exact shape documented in
   * `app/assistant.py`. Only real, already-computed numbers are included.
   */
  const context = useMemo(() => {
    const ctx: Record<string, unknown> = { section, scenario_id: scenarioId, method: config.method };

    if (solve) {
      ctx.solve = {
        method: solve.method,
        status: solve.status,
        elapsed_s: solve.elapsed_s,
        evaluations: solve.evaluations,
        budget_note: solve.budget_note,
        trace: solve.trace ? { best: solve.trace.best, mean: solve.trace.mean } : undefined,
        evaluation: {
          method: solve.method,
          objective: solve.evaluation.objective,
          feasible: solve.evaluation.feasible,
          distance_m: solve.evaluation.distance_m,
          time_s: solve.evaluation.time_s,
          violations: solve.evaluation.violations,
          vehicles: solve.evaluation.vehicles.length,
        },
      };
    }

    if (results.compare) ctx.compare = results.compare;
    if (results.whatif) ctx.whatif = results.whatif;
    if (results.forecast) ctx.forecast = results.forecast;

    const vehicles = solve
      ? solve.evaluation.vehicles.map((vehicle) => ({
          id: vehicle.id,
          status: !vehicle.edge_ids?.length ? "idle" : vehicle.feasible ? "moving" : "delayed",
          stops: vehicle.order.length,
          distance_m: vehicle.distance_m ?? 0,
          elapsed_s: vehicle.elapsed_s,
        }))
      : (graph?.fleet || []).map((vehicle) => ({
          id: vehicle.id,
          status: "idle",
          stops: 0,
          distance_m: 0,
          elapsed_s: 0,
        }));
    if (vehicles.length) ctx.fleet = { vehicles };

    return ctx;
  }, [section, scenarioId, config.method, solve, graph, results]);

  // Suggestions reflect what is actually in context, so a chip never points at
  // data the page does not have yet.
  const suggestions = useMemo(() => {
    const list: string[] = ["What is this page doing?", "Summarise my latest run"];
    if (results.compare) {
      list.unshift("What difference came between ALNS and QPSO?");
      list.push("Which algorithm performed best?");
    }
    if (results.whatif) list.push("What did this disruption cost us?");
    if (results.forecast) list.push("How accurate is the forecast?");
    if (solve) list.push("Is my plan feasible?");
    return list.slice(0, 4);
  }, [results.compare, results.whatif, results.forecast, solve]);

  // Keep the newest message in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy, open]);

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || busy) return;
      setError("");
      setInput("");
      setMessages((current) => [...current, { role: "user", text: message }]);
      setBusy(true);
      try {
        const response = await postJson<AskResponse>(
          "/api/assistant/ask",
          { message, context },
          SOLVE_TIMEOUT_MS + 15000,
        );
        setMessages((current) => [
          ...current,
          { role: "assistant", text: response.answer, source: response.source },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [busy, context],
  );

  return (
    <>
      <button
        type="button"
        className="q-chat-fab"
        aria-label={open ? "Close experiment assistant" : "Open experiment assistant"}
        title="Experiment assistant — plain-English answers about what you just ran"
        onClick={() => {
          setOpen((value) => !value);
          setTimeout(() => inputRef.current?.focus(), 60);
        }}
      >
        {open ? <ChevronDown size={20} /> : <MessageCircleQuestion size={20} />}
      </button>

      {open && (
        <aside className="q-chat-panel" aria-label="Experiment assistant">
          <header className="q-chat-head">
            <span className="q-chat-mark"><Sparkles size={14} /></span>
            <div className="q-chat-title">
              <strong>Experiment assistant</strong>
              <small>
                {status?.configured
                  ? `Groq · ${status.model}`
                  : status
                    ? "local answers (no API key)"
                    : "checking connection…"}
              </small>
            </div>
            <button type="button" className="q-icon-btn" aria-label="Collapse assistant" onClick={() => setOpen(false)}>
              <ChevronDown size={16} />
            </button>
          </header>

          <div className="q-chat-log" ref={scrollRef} aria-live="polite">
            {messages.map((message, index) => (
              <div key={index} className={`q-chat-msg ${message.role}`}>
                <p>{message.text}</p>
                {message.role === "assistant" && index > 0 && (
                  <small>
                    {message.source === "groq"
                      ? "Groq, grounded on the numbers on this page"
                      : message.source === "local"
                        ? "Computed locally from your data"
                        : ""}
                  </small>
                )}
              </div>
            ))}
            {busy && (
              <div className="q-chat-msg assistant">
                <p className="q-chat-typing"><i /><i /><i /></p>
              </div>
            )}
          </div>

          <div className="q-chat-sugs">
            {suggestions.map((suggestion) => (
              <button key={suggestion} type="button" disabled={busy} onClick={() => void send(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>

          {error ? (
            <p className="q-chat-note error">{error}</p>
          ) : (
            <p className="q-chat-note">
              Answers are built from live page data — never from a fixed script.
            </p>
          )}

          <form
            className="q-chat-form"
            onSubmit={(event) => {
              event.preventDefault();
              void send(input);
            }}
          >
            <input
              ref={inputRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={`Ask about ${section === "dashboard" ? "your run" : section}…`}
              aria-label="Ask the experiment assistant"
              maxLength={400}
            />
            <button type="submit" className="q-chat-send" aria-label="Send question" disabled={busy || !input.trim()}>
              <Send size={15} />
            </button>
          </form>
        </aside>
      )}
    </>
  );
}
