import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: the dashboard talks to /admin/* on the same origin, and Vite
// forwards to the gateway. No hard-coded host in the browser code, so the
// same build works behind the nginx proxy in docker-compose.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/admin": { target: "http://localhost:8000", changeOrigin: true },
      "/healthz": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
