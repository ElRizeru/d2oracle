import { copyFileSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB_ROOT = resolve(HERE, "..");
const SRC_DIST = join(WEB_ROOT, "node_modules", "onnxruntime-web", "dist");
const OUT_ORT = join(WEB_ROOT, "public", "ort");

// Only the standard SIMD WASM runtime is needed when importing onnxruntime-web/wasm.
const REQUIRED_ORT_FILES = new Set([
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]);

mkdirSync(OUT_ORT, { recursive: true });
for (const existing of readdirSync(OUT_ORT)) {
  if (!REQUIRED_ORT_FILES.has(existing)) {
    rmSync(join(OUT_ORT, existing), { force: true });
  }
}

const copied = [];
for (const entry of REQUIRED_ORT_FILES) {
  copyFileSync(join(SRC_DIST, entry), join(OUT_ORT, entry));
  copied.push(entry);
}

console.log(`Synced ${copied.length} onnxruntime-web WASM assets to ${OUT_ORT}`);
