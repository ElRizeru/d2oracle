import { defaultClientConditions, defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    conditions: ["onnxruntime-web-use-extern-wasm", ...defaultClientConditions],
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "onnxruntime-web/wasm"],
  },
  build: {
    outDir: "dist",
    assetsDir: "assets",
  },
});
