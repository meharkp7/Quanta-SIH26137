import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

/** True when the user prefers reduced motion (also used to skip rAF smoothing). */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(
    () => typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    if (!window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

/**
 * Eased count-up toward `target`. Animates from the previously settled value
 * with an ease-out cubic over `duration` ms. Non-finite targets render as-is
 * (caller shows the "—" empty state). CPU-light: single rAF, no libraries.
 */
export function useCountUp(target: number, duration = 900): number {
  const reduced = usePrefersReducedMotion();
  const [value, setValue] = useState<number>(target);
  const settledRef = useRef<number>(target);
  useEffect(() => {
    if (!Number.isFinite(target)) return;
    if (reduced || settledRef.current === target) {
      setValue(target);
      settledRef.current = target;
      return;
    }
    const from = settledRef.current;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(from + (target - from) * eased);
      if (t < 1) {
        raf = requestAnimationFrame(tick);
      } else {
        settledRef.current = target;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      settledRef.current = target;
    };
  }, [target, duration, reduced]);
  return value;
}

/** One-shot IntersectionObserver reveal for scroll-triggered entrance. */
export function useInView<T extends HTMLElement = HTMLDivElement>(threshold = 0.2): {
  ref: RefObject<T>;
  inView: boolean;
} {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [threshold]);
  return { ref, inView };
}
