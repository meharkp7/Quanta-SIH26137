// Node's fetch requires absolute URLs — it has no document base like a
// browser. Resolve relative `/api/...` calls against the jsdom origin
// (configured to the live backend in vitest.config.ts).
const origFetch = globalThis.fetch.bind(globalThis);

globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const base = globalThis.location?.href ?? "http://127.0.0.1:8765/";
  const resolved = typeof input === "string" && input.startsWith("/") ? new URL(input, base).href : input;
  return origFetch(resolved as RequestInfo, init);
}) as typeof fetch;

export {};
