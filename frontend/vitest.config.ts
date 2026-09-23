import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

// jsdom origin = the live backend so relative `/api/...` fetches hit it
// directly (Node fetch has no CORS layer).
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "jsdom",
      environmentOptions: { jsdom: { url: "http://127.0.0.1:8765" } },
      include: ["src/**/*.test.tsx"],
      setupFiles: ["src/test-setup.ts"],
      testTimeout: 300000,
      hookTimeout: 60000,
      css: false,
    },
  }),
);
