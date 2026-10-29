import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// npm run dev → http://localhost:5173, API проксируется на uvicorn :8200.
const api = "http://127.0.0.1:8200";
const proxied = ["/search", "/ask", "/harness", "/runs", "/dashboard", "/stats", "/health", "/etl"];

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
  server: { proxy: Object.fromEntries(proxied.map((p) => [p, api])) },
});
