import { useCallback, useEffect, useState } from "react";

// Theme preference: "system" (default) follows the OS, otherwise an explicit
// light/dark choice persisted in localStorage. The resolved value is written
// to <html data-theme>, which is what quanta.css keys its tokens on; an inline
// script in index.html applies the same rule before first paint so there is no
// flash of the wrong theme.
export type ThemeChoice = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

export const THEME_KEY = "quanta.theme";
const ORDER: ThemeChoice[] = ["system", "light", "dark"];

const media = () => (typeof window === "undefined" ? null : window.matchMedia?.("(prefers-color-scheme: dark)") ?? null);

export function resolveTheme(choice: ThemeChoice): ResolvedTheme {
  if (choice === "system") return media()?.matches ? "dark" : "light";
  return choice;
}

export function applyTheme(choice: ThemeChoice): ResolvedTheme {
  const resolved = resolveTheme(choice);
  document.documentElement.dataset.theme = resolved;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", resolved === "dark" ? "#060b13" : "#eef4f0");
  return resolved;
}

export function readTheme(): ThemeChoice {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
  } catch {
    /* private mode — fall through to the system preference */
  }
  return "system";
}

/** Current choice + a cycler for the topbar button (system → light → dark). */
export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readTheme);

  useEffect(() => {
    applyTheme(choice);
  }, [choice]);

  // Follow the OS while the choice is "system".
  useEffect(() => {
    const mq = media();
    if (!mq?.addEventListener) return;
    const onChange = () => {
      if (readTheme() === "system") applyTheme("system");
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  const set = useCallback((next: ThemeChoice) => {
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      /* ignore quota / private-mode errors */
    }
    setChoice(next);
  }, []);

  const cycle = useCallback(() => {
    const index = ORDER.indexOf(readTheme());
    set(ORDER[(index + 1) % ORDER.length]);
  }, [set]);

  return { choice, set, cycle, resolved: resolveTheme(choice) };
}
