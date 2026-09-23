import { useState } from "react";
import type { FormEvent } from "react";
import { Link, Navigate, useNavigate, useLocation } from "react-router-dom";
import {
  ArrowRight,
  CheckCircle2,
  Building2,
  ShieldCheck,
  Route,
} from "lucide-react";
import { supabase } from "./supabase";
import { useWorkspace } from "./Store";
import { Brand, Button, Field, Notice } from "./ui";

export function AuthPage() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const store = useWorkspace();
  const signup = pathname === "/signup",
    forgot = pathname === "/forgot-password",
    reset = pathname === "/reset-password";
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const [email, setEmail] = useState(""),
    [password, setPassword] = useState(""),
    [name, setName] = useState("");
  if (
    store.session &&
    !reset &&
    !forgot &&
    !message &&
    store.company &&
    !store.demo
  )
    return <Navigate to="/app" replace />;
  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setMessage("");
    if (!supabase) {
      setError(
        "Company sign-in is not configured yet. You can explore the demo workspace.",
      );
      return;
    }
    setBusy(true);
    try {
      if (forgot) {
        const { error } = await supabase.auth.resetPasswordForEmail(email, {
          redirectTo: `${location.origin}/reset-password`,
        });
        if (error) throw error;
        setMessage(
          "If this email has an account, a password reset link is on its way.",
        );
      } else if (reset) {
        const { error } = await supabase.auth.updateUser({ password });
        if (error) throw error;
        setMessage(
          "Your password has been updated. You can return to your workspace.",
        );
      } else if (signup) {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: { full_name: name },
            emailRedirectTo: `${location.origin}/auth/callback`,
          },
        });
        if (error) throw error;
        sessionStorage.removeItem("quanta.demo");
        if (data.session) location.assign("/onboarding");
        else
          setMessage(
            "Check your email to verify your account. Then we’ll set up your company workspace.",
          );
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) throw error;
        sessionStorage.removeItem("quanta.demo");
        location.assign("/onboarding");
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="w-auth">
      <aside className="w-auth-story">
        <Link to="/">
          <Brand light />
        </Link>
        <div>
          <span className="w-eyebrow">BUILT FOR THE ROAD AHEAD</span>
          <h1>
            Every delivery.
            <br />A better decision.
          </h1>
          <p>
            Plan your fleet, respond to disruption, and make every route
            accountable.
          </p>
          <div className="w-auth-points">
            <span>
              <Route /> Adaptive route planning
            </span>
            <span>
              <ShieldCheck /> Independently validated results
            </span>
            <span>
              <Building2 /> Your company. Your workspace.
            </span>
          </div>
        </div>
        <small>QUANTA · ADAPTIVE FLEET INTELLIGENCE</small>
      </aside>
      <main className="w-auth-form">
        <Link to="/" className="w-back">
          ← Back to Quanta
        </Link>
        <div>
          <span className="w-eyebrow">YOUR OPERATIONS START HERE</span>
          <h1>
            {signup
              ? "Create your account"
              : forgot
                ? "Reset your password"
                : reset
                  ? "Choose a new password"
                  : "Welcome back"}
          </h1>
          <p>
            {signup
              ? "Start with your account. Set up your company after verification."
              : forgot
                ? "We’ll send a secure recovery link to your email."
                : reset
                  ? "Use at least 8 characters for your new password."
                  : "Sign in to your company workspace."}
          </p>
          {!supabase && (
            <Notice tone="amber">
              Company access will be available after Supabase setup. The demo
              workspace is ready to explore.
            </Notice>
          )}
          {error && <Notice tone="red">{error}</Notice>}
          {message && <Notice tone="green">{message}</Notice>}
          <form onSubmit={submit}>
            {signup && (
              <Field label="Full name">
                <input
                  required
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your full name"
                />
              </Field>
            )}
            {!reset && (
              <Field label="Work email">
                <input
                  required
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="you@company.com"
                />
              </Field>
            )}
            {!forgot && (
              <Field label="Password">
                <input
                  required
                  type="password"
                  minLength={8}
                  autoComplete={
                    signup || reset ? "new-password" : "current-password"
                  }
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="At least 8 characters"
                />
              </Field>
            )}
            {!signup && !forgot && !reset && (
              <Link className="w-forgot" to="/forgot-password">
                Forgot password?
              </Link>
            )}
            <Button disabled={busy || !supabase}>
              {busy
                ? "Please wait…"
                : signup
                  ? "Create account"
                  : forgot
                    ? "Send reset link"
                    : reset
                      ? "Update password"
                      : "Sign in"}
              <ArrowRight size={16} />
            </Button>
          </form>
          <p className="w-auth-switch">
            {signup ? "Already have an account?" : "New to Quanta?"}{" "}
            <Link to={signup ? "/login" : "/signup"}>
              {signup ? "Sign in" : "Create an account"}
            </Link>
          </p>
          <div className="w-auth-divider">
            <span>or explore the platform</span>
          </div>
          <Button
            variant="secondary"
            onClick={() => {
              store.enterDemo();
              navigate("/app");
            }}
          >
            Explore demo workspace <ArrowRight size={16} />
          </Button>
        </div>
        <small>
          <ShieldCheck size={14} /> Company data is isolated by workspace
          membership.
        </small>
      </main>
    </div>
  );
}

export function Onboarding() {
  const store = useWorkspace();
  const [name, setName] = useState(""),
    [industry, setIndustry] = useState("Last-mile delivery");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const navigate = useNavigate();
  if (!store.ready)
    return <div className="w-loading">Checking your session…</div>;
  if (!store.session) return <Navigate to="/login" replace />;
  if (store.company && !store.demo) return <Navigate to="/app" replace />;
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!supabase) return;
    setBusy(true);
    setError("");
    try {
      const { error } = await supabase.rpc("create_company", {
        company_name: name,
        company_industry: industry,
      });
      if (error) throw error;
      await store.reloadCompanies();
      navigate("/app");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <main className="w-onboarding">
      <Link to="/">
        <Brand />
      </Link>
      <div className="w-onboard-card">
        <span className="w-onboard-icon">
          <Building2 size={28} />
        </span>
        <h1>Make room for your team.</h1>
        <p>
          Create a company workspace for your scenarios, fleet and routing
          results. You’ll be its owner.
        </p>
        {(error || store.error) && (
          <Notice tone="red">{error || store.error}</Notice>
        )}
        <form onSubmit={submit}>
          <Field label="Company name">
            <input
              required
              minLength={2}
              maxLength={120}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Meridian Logistics"
            />
          </Field>
          <Field label="Industry">
            <select
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
            >
              {[
                "Last-mile delivery",
                "Retail & distribution",
                "Manufacturing",
                "Research & education",
                "Other",
              ].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </Field>
          <Button disabled={busy}>
            {busy ? "Creating workspace…" : "Create company workspace"}
            <ArrowRight size={16} />
          </Button>
        </form>
        <small>
          <CheckCircle2 size={14} /> You can add team members in Settings.
        </small>
      </div>
    </main>
  );
}
