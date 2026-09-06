import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const sourceDir = join(root, "node_modules", "onnxruntime-web", "dist");
const targetDir = join(root, "public", "ort");

if (!existsSync(sourceDir)) {
  process.exit(0);
}

mkdirSync(targetDir, { recursive: true });

const requiredFiles = new Set(["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]);

for (const file of readdirSync(sourceDir)) {
  if (requiredFiles.has(file)) {
    copyFileSync(join(sourceDir, file), join(targetDir, file));
  }
}
