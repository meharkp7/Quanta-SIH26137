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
  MonitorPlay,
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
import { AuthPage, Onboarding } from "./Auth";
import { Brand, Notice } from "./ui";
import { dateLabel } from "./data";
// ── Quanta redesign: the five numbered sections from the design image ──────
import { QuantaProvider } from "@/quanta/store";
import { LoginLayout, Shell as QuantaShell } from "@/quanta/Shell";
import { Login } from "@/quanta/Login";
import { Dashboard } from "@/quanta/Dashboard";
import { WhatIf } from "@/quanta/WhatIf";
import { RouteLab } from "@/quanta/RouteLab";
import { Forecasting } from "@/quanta/Forecasting";

function AppRoutes() {
  const store = useWorkspace();
  return (
    <RunProvider
      key={`${store.demo ? "demo" : store.session?.user.id}-${store.company?.id}`}
    >
      <QuantaProvider>
        <Routes>
          {/* 1. LOGIN — the public entry, exactly as the image's first panel. */}
          <Route
            path="/"
            element={
              <LoginLayout>
                <Login />
              </LoginLayout>
            }
          />
          <Route
            path="/login"
            element={
              <LoginLayout>
                <Login />
              </LoginLayout>
            }
          />
          {["signup", "forgot-password", "reset-password"].map((path) => (
            <Route
              key={path}
              path={`/${path}`}
              element={
                <LoginLayout>
                  <AuthPage key={path} />
                </LoginLayout>
              }
            />
          ))}
          <Route path="/auth/callback" element={<Onboarding />} />
          <Route path="/onboarding" element={<Onboarding />} />

          {/* 2–5. The numbered application sections in one shell. */}
          <Route path="/app" element={<QuantaShell />}>
            <Route index element={<Dashboard />} />
            <Route path="what-if" element={<WhatIf />} />
            <Route path="route-lab" element={<RouteLab />} />
            <Route path="forecasting" element={<Forecasting />} />
            <Route path="*" element={<Navigate to="/app" replace />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </QuantaProvider>
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
