import { Link, useNavigate } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  ChevronRight,
  Route,
  ShieldCheck,
  TrafficCone,
  BarChart3,
  Truck,
  Network,
  Cpu,
} from "lucide-react";
import { useWorkspace } from "./Store";
import { Brand, Button, NetworkView, Tag } from "./ui";
import { generateNetwork } from "./data";
const preview = generateNetwork();
export function Landing() {
  const store = useWorkspace(),
    navigate = useNavigate();
  function demo() {
    store.enterDemo();
    navigate("/app");
  }
  return (
    <div className="w-landing">
      <header className="w-marketing-nav">
        <Link to="/">
          <Brand />
        </Link>
        <nav>
          <a href="#platform">Platform</a>
          <a href="#solutions">Solutions</a>
          <a href="#how-it-works">How it works</a>
          <Link to="/documentation">Documentation</Link>
        </nav>
        <div>
          <Link className="w-login-link" to="/login">
            Log in
          </Link>
          <Link className="w-button primary" to="/signup">
            Get started <ArrowUpRight size={16} />
          </Link>
        </div>
      </header>
      <main>
        <section className="w-hero">
          <div className="w-hero-copy">
            <span className="w-eyebrow">
              <span className="w-small-dot" /> ADAPTIVE FLEET INTELLIGENCE
            </span>
            <h1>
              Better routes.
              <br />
              More resilient
              <br />
              <em>operations.</em>
            </h1>
            <p>
              Turn complex delivery networks into clear decisions. Plan your
              fleet, adapt to traffic disruptions, and see why every route
              works.
            </p>
            <div className="w-hero-actions">
              <Button onClick={demo}>
                Explore the platform <ArrowRight size={17} />
              </Button>
              <Link className="w-button secondary" to="/signup">
                Create a workspace
              </Link>
            </div>
            <div className="w-hero-proof">
              <span>
                <CheckCircle2 size={15} /> Constraint-aware routing
              </span>
              <span>
                <CheckCircle2 size={15} /> Explainable results
              </span>
            </div>
          </div>
          <div className="w-hero-product">
            <div className="w-product-top">
              <Brand />
              <span>
                <i /> OPERATIONS OVERVIEW
              </span>
              <span className="w-avatar">M</span>
            </div>
            <div className="w-product-heading">
              <div>
                <small>MERIDIAN LOGISTICS · DEMO</small>
                <h3>One network. A clearer view.</h3>
              </div>
              <Tag tone="green">Ready to plan</Tag>
            </div>
            <NetworkView graph={preview} compact />
            <div className="w-product-stats">
              <div>
                <Truck size={17} />
                <strong>4</strong>
                <span>Sample vehicles</span>
              </div>
              <div>
                <Route size={17} />
                <strong>12</strong>
                <span>Delivery stops</span>
              </div>
              <div>
                <Network size={17} />
                <strong>24</strong>
                <span>Road nodes</span>
              </div>
            </div>
            <div className="w-hero-callout">
              <ShieldCheck size={22} />
              <div>
                <strong>Every route checked.</strong>
                <span>Capacity, time windows and road connectivity.</span>
              </div>
            </div>
          </div>
        </section>
        <section className="w-value-strip">
          <span>BUILT AROUND YOUR OPERATION</span>
          <div>Fleet planning</div>
          <i /> <div>Dynamic replanning</div>
          <i />
          <div>Traffic simulation</div>
          <i />
          <div>Algorithm benchmarking</div>
        </section>
        <section id="platform" className="w-marketing-section">
          <div className="w-section-intro">
            <div>
              <span className="w-eyebrow">ONE CONNECTED PLATFORM</span>
              <h2>
                From the first stop
                <br />
                to the final decision.
              </h2>
            </div>
            <p>
              A shared workspace connects your network, vehicles and delivery
              constraints to the results your team needs.
            </p>
          </div>
          <div className="w-feature-grid">
            {[
              [
                Route,
                "Plan with confidence",
                "Configure your network, fleet and service windows. Compare quantum-inspired and classical routing methods.",
              ],
              [
                TrafficCone,
                "Respond to disruption",
                "Close a road, inspect the impact and compute a new route under the updated conditions.",
              ],
              [
                BarChart3,
                "Make the result explainable",
                "Inspect vehicle routes, convergence, constraint checks and comparable algorithm results.",
              ],
            ].map(([Icon, title, text]: any, i) => (
              <article key={title}>
                <span className="w-feature-number">0{i + 1}</span>
                <Icon size={27} />
                <h3>{title}</h3>
                <p>{text}</p>
                <button onClick={demo}>
                  Explore capability <ArrowUpRight size={16} />
                </button>
              </article>
            ))}
          </div>
        </section>
        <section id="solutions" className="w-solutions">
          <div>
            <span className="w-eyebrow">DESIGNED FOR REAL QUESTIONS</span>
            <h2>
              What happens when
              <br />
              the city changes?
            </h2>
            <p>
              Explore delivery planning, rush-hour conditions and incident
              response in a repeatable demo workspace. Existing solvers
              calculate the results; simulation makes them visible.
            </p>
            <Button onClick={demo}>
              Open the demo <ArrowRight size={16} />
            </Button>
          </div>
          <div className="w-solution-list">
            {[
              [
                "01",
                "Last-mile distribution",
                "Balance delivery demand against available fleet capacity.",
              ],
              [
                "02",
                "Incident response",
                "Find legal alternatives when a directed road is closed.",
              ],
              [
                "03",
                "Research & evaluation",
                "Compare methods on the same network and constraints.",
              ],
            ].map(([n, t, d]) => (
              <div key={n}>
                <span>{n}</span>
                <div>
                  <h3>{t}</h3>
                  <p>{d}</p>
                </div>
                <ArrowUpRight size={19} />
              </div>
            ))}
          </div>
        </section>
        <section id="how-it-works" className="w-marketing-section">
          <span className="w-eyebrow">A TRACEABLE WORKFLOW</span>
          <h2>Configure. Compute. Understand.</h2>
          <div className="w-workflow">
            {[
              [
                "01",
                "Build your scenario",
                "Choose a network and define the fleet, demand and traffic conditions.",
              ],
              [
                "02",
                "Run an optimiser",
                "Search for routes and check each plan against routing constraints.",
              ],
              [
                "03",
                "Explore the outcome",
                "Replay movement, inspect results and compare alternative algorithms.",
              ],
            ].map(([n, t, d]) => (
              <div key={n}>
                <span>{n}</span>
                <h3>{t}</h3>
                <p>{d}</p>
              </div>
            ))}
          </div>
        </section>
        <section className="w-cta">
          <div>
            <span className="w-eyebrow">SEE THE VISION IN ACTION</span>
            <h2>Your next route starts here.</h2>
          </div>
          <Button onClick={demo}>
            Explore Quanta <ArrowRight size={18} />
          </Button>
        </section>
      </main>
      <footer className="w-footer">
        <Brand />
        <span>Adaptive quantum-inspired vehicle routing · SIH26137</span>
        <Link to="/documentation">
          Platform documentation <ArrowUpRight size={14} />
        </Link>
      </footer>
    </div>
  );
}
