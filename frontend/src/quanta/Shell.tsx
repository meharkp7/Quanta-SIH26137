import { useEffect, useState } from "react";
import { Navigate, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  CircleAlert,
  FlaskConical,
  LayoutDashboard,
  LineChart,
  LogOut,
  Monitor,
  Moon,
  Route as RouteIcon,
  Sun,
  X,
} from "lucide-react";
import { useWorkspace } from "@/workspace/Store";
import { ToastContainer } from "@/components/ui/Toast";
import { useQuanta } from "./store";
import { useTheme } from "./theme";
import { Chatbot } from "./Chatbot";
import { fmtClockTime } from "./ui";

// The four application sections from the image's sidebar (panel 2).
export const SECTIONS = [
  { to: "/app", label: "Dashboard", n: 2, icon: LayoutDashboard, end: true },
  { to: "/app/what-if", label: "What-If Scenarios", n: 3, icon: FlaskConical, end: false },
  { to: "/app/route-lab", label: "Route Lab", n: 4, icon: RouteIcon, end: false },
  { to: "/app/forecasting", label: "Forecasting", n: 5, icon: LineChart, end: false },
] as const;

// Bottom workflow rail — Login → Configure → Optimize → Simulate → Replan →
// Forecast. Steps link to the section that owns them.
const STEPS = [
  { key: "login", label: "Login", hint: "Access your workspace", to: "/login", icon: "🔒" },
  { key: "configure", label: "Configure", hint: "Set inputs and preferences", to: "/app", icon: "⚙" },
  { key: "optimize", label: "Optimize", hint: "Generate best routes", to: "/app/route-lab", icon: "📊" },
  { key: "simulate", label: "Simulate", hint: "Test what-if scenarios", to: "/app/what-if", icon: "🧪" },
  { key: "replan", label: "Replan", hint: "Adapt and improve", to: "/app/what-if", icon: "🔄" },
  { key: "forecast", label: "Forecast", hint: "Predict and stay ahead", to: "/app/forecasting", icon: "📈" },
] as const;

function activeStep(pathname: string, replanned: boolean) {
  if (pathname === "/login" || pathname === "/") return "login";
  if (pathname.startsWith("/app/route-lab")) return "optimize";
  if (pathname.startsWith("/app/what-if")) return replanned ? "replan" : "simulate";
  if (pathname.startsWith("/app/forecasting")) return "forecast";
  return "configure";
}

// Topbar theme control — cycles system → light → dark. The resolved choice is
// stored by src/quanta/theme.ts and reflected on <html data-theme>.
const THEME_META = {
  system: { icon: Monitor, label: "Theme: follow system" },
  light: { icon: Sun, label: "Theme: light" },
  dark: { icon: Moon, label: "Theme: dark" },
} as const;

export function TopBar() {
  const { choice, cycle } = useTheme();
  const meta = THEME_META[choice];
  const ThemeIcon = meta.icon;
  return (
    <header className="q-topbar">
      <div className="q-brand">
        <span className="q-logo-mark small">
          <RouteIcon size={15} strokeWidth={2.6} />
        </span>
        <span className="q-brand-word">QUANTA</span>
        <i className="q-brand-divider" />
        <span className="q-brand-sub">Fleet Routing Intelligence</span>
      </div>
      <p className="q-tagline">Smarter Routes. Greener Cities. Greater Impact.</p>
      <button
        type="button"
        className="q-icon-btn q-theme-btn"
        onClick={cycle}
        title={`${meta.label} — click to switch`}
        aria-label={`${meta.label}, click to switch theme`}
      >
        <ThemeIcon size={16} />
      </button>
    </header>
  );
}

function StepRail({ current, signedIn }: { current: string; signedIn: boolean }) {
  return (
    <footer className="q-rail">
      <ol className="q-rail-steps">
        {STEPS.map((step) => {
          const done = signedIn && step.key === "login";
          const active = step.key === current;
          return (
            <li key={step.key}>
              <NavLink to={step.to} className={`q-rail-step ${active ? "active" : ""} ${done ? "done" : ""}`}>
                <span className="q-rail-icon">{step.icon}</span>
                <span>
                  <strong>{step.label}</strong>
                  <small>{step.hint}</small>
                </span>
              </NavLink>
            </li>
          );
        })}
      </ol>
      <div className="q-rail-brand">
        <strong>QUANTA</strong>
        <small>A CLEANER, MORE SUSTAINABLE TOMORROW</small>
      </div>
    </footer>
  );
}

export function Shell() {
  const store = useWorkspace();
  const { error, setError, scenarioId, replanned } = useQuanta();
  const location = useLocation();
  const navigate = useNavigate();
  const [now, setNow] = useState(fmtClockTime());

  useEffect(() => {
    const id = window.setInterval(() => setNow(fmtClockTime()), 30000);
    return () => window.clearInterval(id);
  }, []);

  if (!store.ready) return <div className="q-loading">Checking your session…</div>;
  if (!store.demo && !store.session) return <Navigate to="/login" replace />;
  if (!store.demo && !store.company) return <Navigate to="/onboarding" replace />;

  const company = store.company;
  const current = activeStep(location.pathname, replanned);

  return (
    <div className="q-shell">
      <ToastContainer />

      <TopBar />

      <div className="q-body">
        <aside className="q-sidebar">
          <div className="q-company">
            <span className="q-company-logo">{(company?.name || "Demo Logistics").slice(0, 1)}</span>
            <span>
              <strong>{company?.name || "Acme Logistics Pvt. Ltd."}</strong>
              <small className="q-live"><i /> Live Operations</small>
            </span>
          </div>

          <nav className="q-nav" aria-label="Sections">
            {SECTIONS.map((section) => {
              const Icon = section.icon;
              return (
                <NavLink key={section.to} to={section.to} end={section.end}>
                  <span className="q-nav-num">{section.n}</span>
                  <Icon size={17} />
                  <span>{section.label}</span>
                </NavLink>
              );
            })}
          </nav>

          <div className="q-side-meta">
            <small>{scenarioId || "no scenario"}</small>
            <small>{now}</small>
          </div>

          <button
            className="q-signout"
            onClick={async () => {
              try {
                await store.signOut();
                navigate("/login", { replace: true });
              } catch (e) {
                store.setError((e as Error).message);
              }
            }}
          >
            <LogOut size={16} /> {store.demo ? "Leave demo" : "Sign out"}
          </button>
        </aside>

        <main className="q-main">
          {(error || store.error) && (
            <div className="q-error-bar">
              <CircleAlert size={15} />
              {error || store.error}
              <button aria-label="Dismiss error" onClick={() => { setError(""); store.setError(""); }}>
                <X size={15} />
              </button>
            </div>
          )}
          <Outlet />
        </main>
      </div>

      <StepRail current={current} signedIn />
      <Chatbot />
    </div>
  );
}

// Public layout around the LOGIN section: same header + workflow rail as the
// application, exactly as the image shows across all five panels.
export function LoginLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  return (
    <div className="q-shell public">
      <ToastContainer />
      <TopBar />
      <div className="q-public-body">{children}</div>
      <StepRail current={activeStep(location.pathname, false)} signedIn={false} />
    </div>
  );
}
