import { useEffect, useState } from "react";
import {
  BrowserRouter,
  Link,
  NavLink,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  Activity,
  BarChart3,
  Bell,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Database,
  FlaskConical,
  LayoutDashboard,
  LogOut,
  Menu,
  Plus,
  Route as RouteIcon,
  Search,
  Settings,
  ShieldCheck,
  Truck,
  Waypoints,
  X,
} from "lucide-react";
import { WorkspaceProvider, useWorkspace } from "./Store";
import { RunProvider, useRunner } from "./Runner";
import { Landing } from "./Landing";
import { AuthPage, Onboarding } from "./Auth";
import { Brand, Button, Notice, Tag } from "./ui";
import {
  Dashboard,
  Results,
  Comparisons,
  DataModels,
  RunPage,
  History,
  Documentation,
  SettingsPage,
} from "./pages";
import { ScenarioList, ScenarioWizard, FleetPage } from "./scenarios";
import { Operations, ShortestPath } from "./operations";
import { dateLabel } from "./data";

const navigation = [
  {
    label: "Overview",
    items: [
      ["/app", "Dashboard", LayoutDashboard],
      ["/app/scenarios", "Scenarios", Waypoints],
      ["/app/fleet", "Fleet & deliveries", Truck],
    ],
  },
  {
    label: "Operations",
    items: [
      ["/app/live", "Live operations", Activity],
      ["/app/shortest-path", "Shortest path", RouteIcon],
      ["/app/simulations", "Simulations", FlaskConical],
    ],
  },
  {
    label: "Intelligence",
    items: [
      ["/app/results", "Results", BarChart3],
      ["/app/comparisons", "Comparisons", LayoutDashboard],
      ["/app/data", "Data & models", Database],
    ],
  },
] as const;

function Shell() {
  const store = useWorkspace(),
    navigate = useNavigate(),
    location = useLocation(),
    runner = useRunner();
  const [query, setQuery] = useState(""),
    [mobile, setMobile] = useState(false),
    [activity, setActivity] = useState(false);
  if (!store.ready)
    return <div className="w-loading">Loading your workspace…</div>;
  if (!store.demo && !store.session) return <Navigate to="/login" replace />;
  if (!store.demo && !store.company)
    return <Navigate to="/onboarding" replace />;
  const company = store.company!;
  const current =
    navigation
      .flatMap((n) => [...n.items])
      .find(([url]) => url === location.pathname || (url !== "/app" && location.pathname.startsWith(`${url}/`)))?.[1] ||
    (location.pathname.includes("settings")
      ? "Settings"
      : location.pathname.includes("run")
        ? "Optimisation run"
        : "Scenario workspace");
  const matches = store.drafts
    .filter((d) => d.name.toLowerCase().includes(query.toLowerCase()))
    .slice(0, 6);
  return (
    <div className="w-app">
      <aside className={`w-sidebar ${mobile ? "open" : ""}`}>
        <Link to="/" className="w-sidebar-brand">
          <Brand light />
        </Link>
        <button
          className="w-company-card"
          onClick={() => navigate("/app/settings")}
        >
          <span className="w-company-logo">{company.name.slice(0, 1)}</span>
          <span>
            <strong>{company.name}</strong>
            <small>{store.demo ? "Demo workspace" : "Company workspace"}</small>
          </span>
          <ChevronRight size={15} />
        </button>
        <nav>
          {navigation.map((section) => (
            <div className="w-nav-section" key={section.label}>
              <small>{section.label}</small>
              {section.items.map(([url, label, Icon]) => (
                <NavLink
                  key={url}
                  to={url}
                  end={url === "/app"}
                  onClick={() => setMobile(false)}
                >
                  <Icon size={18} />
                  <span>{label}</span>
                  {label === "Live operations" && <i className="w-nav-dot" />}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>
        <div className="w-sidebar-bottom">
          <div className="w-demo-card">
            <ShieldCheck size={19} />
            <strong>
              {store.demo
                ? "Explore the full workflow"
                : "Your workspace is private"}
            </strong>
            <p>
              {store.demo
                ? "Sample scenarios. Real routing computations."
                : `${company.role} access · Company-scoped data`}
            </p>
            <Link to="/app/documentation">
              Platform guide <ArrowGuide />
            </Link>
          </div>
          <NavLink to="/app/settings">
            <Settings size={18} />
            Settings
          </NavLink>
          <button
            onClick={async () => {
              try {
                await store.signOut();
                navigate("/");
              } catch (e) {
                store.setError((e as Error).message);
              }
            }}
          >
            <LogOut size={18} />
            {store.demo ? "Leave demo" : "Sign out"}
          </button>
        </div>
      </aside>
      {mobile && (
        <button
          className="w-mobile-overlay"
          aria-label="Close navigation"
          onClick={() => setMobile(false)}
        />
      )}
      <div className="w-workspace">
        <header className="w-topbar">
          <button
            className="w-mobile-menu"
            aria-label="Open navigation"
            onClick={() => setMobile(!mobile)}
          >
            <Menu size={20} />
          </button>
          <div className="w-breadcrumb">
            Workspace <ChevronRight size={14} />
            <strong>{current}</strong>
          </div>
          <div className="w-global-search">
            <Search size={16} />
            <input
              aria-label="Search scenarios"
              placeholder="Search scenarios…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            {query && (
              <div className="w-search-results">
                {matches.length ? (
                  matches.map((d) => (
                    <button
                      key={d.id}
                      onClick={() => {
                        setQuery("");
                        navigate(`/app/scenarios/${d.id}`);
                      }}
                    >
                      <Waypoints size={16} />
                      <span>{d.name}</span>
                      <ChevronRight size={14} />
                    </button>
                  ))
                ) : (
                  <p>No matching scenarios</p>
                )}
              </div>
            )}
          </div>
          <Tag tone={store.demo ? "amber" : "green"}>
            {store.demo ? "Demo workspace" : "Company workspace"}
          </Tag>
          <button
            className={`w-icon-button ${activity ? "active" : ""}`}
            aria-label="Show recent activity"
            onClick={() => setActivity(!activity)}
          >
            <Bell size={19} />
          </button>
          <button className="w-user" onClick={() => navigate("/app/settings")}>
            <span className="w-avatar">
              {store.demo
                ? "D"
                : (store.session?.user.user_metadata.full_name ||
                    store.session?.user.email ||
                    "U")[0].toUpperCase()}
            </span>
            <span>
              {store.demo
                ? "Demo user"
                : store.session?.user.user_metadata.full_name || "My account"}
            </span>
            <ChevronDown size={14} />
          </button>
        </header>
        {activity && (
          <div className="w-activity">
            <h3>
              Recent activity
              <button
                aria-label="Close activity"
                onClick={() => setActivity(false)}
              >
                <X size={16} />
              </button>
            </h3>
            {store.runs.length ? (
              store.runs.slice(0, 5).map((r) => (
                <Link
                  key={r.id}
                  to={`/app/results/${r.id}`}
                  onClick={() => setActivity(false)}
                >
                  <CheckActivity />
                  <span>
                    {r.draft.name}
                    <small>
                      {r.result.method} completed · {dateLabel(r.createdAt)}
                    </small>
                  </span>
                </Link>
              ))
            ) : (
              <p>Your completed runs will appear here.</p>
            )}
          </div>
        )}
        {runner.job?.status === "running" &&
          location.pathname !== "/app/run" && (
            <div className="w-running-banner">
              <span className="w-spinner" />
              {runner.job.draft.name} is computing{" "}
              <Link to="/app/run">View run →</Link>
            </div>
          )}
        <main className="w-content">
          {store.error && (
            <Notice tone="red">
              {store.error}
              <button
                className="w-notice-close"
                onClick={() => store.setError("")}
                aria-label="Dismiss error"
              >
                <X size={15} />
              </button>
            </Notice>
          )}
          {store.dataReady ? (
            <Outlet />
          ) : (
            <div className="w-loading">Loading workspace data…</div>
          )}
        </main>
        <footer className="w-app-footer">
          <span>Quanta · Adaptive fleet intelligence</span>
          <span>
            {store.demo
              ? "Sample data stays in this browser"
              : "Company workspace"}
            <i />
            {company.industry}
          </span>
        </footer>
      </div>
    </div>
  );
}
function ArrowGuide() {
  return <ChevronRight size={14} />;
}
function CheckActivity() {
  return (
    <span className="w-activity-icon">
      <Activity size={17} />
    </span>
  );
}
function AppRoutes() {
  const store = useWorkspace();
  return (
    <RunProvider
      key={`${store.demo ? "demo" : store.session?.user.id}-${store.company?.id}`}
    >
      <Routes>
        <Route path="/" element={<Landing />} />
        {["login", "signup", "forgot-password", "reset-password"].map(
          (path) => (
            <Route
              key={path}
              path={`/${path}`}
              element={<AuthPage key={path} />}
            />
          ),
        )}
        <Route path="/auth/callback" element={<Onboarding />} />
        <Route path="/onboarding" element={<Onboarding />} />
        <Route path="/documentation" element={<Documentation publicPage />} />
        <Route path="/app" element={<Shell />}>
          <Route index element={<Dashboard />} />
          <Route path="scenarios" element={<ScenarioList />} />
          <Route path="scenarios/new" element={<ScenarioWizard />} />
          <Route path="scenarios/:id" element={<ScenarioWizard />} />
          <Route path="fleet" element={<FleetPage />} />
          <Route path="live" element={<Operations />} />
          <Route path="shortest-path" element={<ShortestPath />} />
          <Route path="simulations" element={<History />} />
          <Route path="run" element={<RunPage />} />
          <Route path="results" element={<Results />} />
          <Route path="results/:id" element={<Results />} />
          <Route path="comparisons" element={<Comparisons />} />
          <Route path="data" element={<DataModels />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="documentation" element={<Documentation />} />
        </Route>
        <Route
          path="*"
          element={
            <div className="w-not-found">
              <Brand />
              <h1>This page isn’t on the route.</h1>
              <Link className="w-button primary" to="/app">
                Go to dashboard
              </Link>
            </div>
          }
        />
      </Routes>
    </RunProvider>
  );
}
export default function WorkspaceApp() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <WorkspaceProvider>
        <AppRoutes />
      </WorkspaceProvider>
    </BrowserRouter>
  );
}

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}
