import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const sourceDir = join(root, "public", "models");
const targetDir = join(root, "public", "model-chunks");
const chunkSize = 15 * 1024 * 1024;
const models = ["model1", "model2"];

mkdirSync(targetDir, { recursive: true });

for (const file of readdirSync(targetDir)) {
  if (file.includes(".onnx.gz.") && file.endsWith(".part")) {
    rmSync(join(targetDir, file));
  }
}

for (const model of models) {
  const compressed = gzipSync(readFileSync(join(sourceDir, `${model}.onnx`)), { level: 9 });
  const chunkCount = Math.ceil(compressed.length / chunkSize);

  for (let index = 0; index < chunkCount; index += 1) {
    const start = index * chunkSize;
    const end = Math.min(start + chunkSize, compressed.length);
    writeFileSync(join(targetDir, `${model}.onnx.gz.${index}.part`), compressed.subarray(start, end));
  }

  console.log(`${model}: ${compressed.length} bytes across ${chunkCount} chunks`);
}
