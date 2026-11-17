import type { Env } from "onnxruntime-web";

export function normalizeModelBaseUrl(input: string): string {
  const trimmed = input.trim();
  const base = trimmed.length > 0 ? trimmed : "./models";
  if (typeof window !== "undefined" && !/^https?:\/\//i.test(base)) {
    const clean = base.replace(/\/+$/, "") + "/";
    return new URL(clean, window.location.href).toString().replace(/\/+$/, "");
  }
  return base.replace(/\/+$/, "");
}

export function resolveRuntimeDirUrl(relativeDir: string): string {
  const clean = relativeDir.replace(/^\/+|\/+$/g, "") + "/";
  if (typeof window !== "undefined") {
    return new URL(clean, window.location.href).toString();
  }
  return `./${clean}`;
}

export interface OrtEnvLike {
  wasm: Pick<Env.WebAssemblyFlags, "numThreads" | "proxy" | "wasmPaths">;
}

export function configureOrtEnvironment(ortModule: { env: OrtEnvLike }): void {
  const isolated =
    typeof window !== "undefined" && Boolean(window.crossOriginIsolated);
  const hw =
    typeof navigator !== "undefined" && navigator.hardwareConcurrency
      ? navigator.hardwareConcurrency
      : 1;

  // GitHub Pages serves static HTML without COOP/COEP headers, so SharedArrayBuffer is unavailable.
  ortModule.env.wasm.numThreads = isolated ? Math.min(4, hw) : 1;
  ortModule.env.wasm.proxy = false;
  if (typeof window !== "undefined") {
    ortModule.env.wasm.wasmPaths = resolveRuntimeDirUrl("ort");
  }
}
