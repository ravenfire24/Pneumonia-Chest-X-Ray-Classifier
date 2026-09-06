"use client";

import { Activity, AlertTriangle, FileImage, ShieldAlert, UploadCloud, X } from "lucide-react";
import { ChangeEvent, useMemo, useRef, useState } from "react";
import type { InferenceSession, Tensor } from "onnxruntime-web/wasm";

const M1_CLASSES = ["Normal", "Pneumonia", "Tuberculosis"] as const;
const M2_CLASSES = ["Bacterial", "Normal", "Viral"] as const;
const PNEUMONIA_THRESHOLD = 0.8;
const MODEL_INPUT_SIZE = 224;
const MODEL_CHUNKS = {
  model1: 2,
  model2: 2
} as const;

type Probability = {
  label: string;
  value: number;
};

type ModelResult = {
  label: string;
  probabilities: Probability[];
};

type FinalResult = {
  label: string;
  tone: "good" | "warn" | "bad";
  detail: string;
};

let model1Session: InferenceSession | null = null;
let model2Session: InferenceSession | null = null;
let ortModulePromise: Promise<typeof import("onnxruntime-web/wasm")> | null = null;

function getOrt() {
  if (!ortModulePromise) {
    ortModulePromise = import("onnxruntime-web/wasm").then((ort) => {
      ort.env.wasm.wasmPaths = {
        mjs: "/ort/ort-wasm-simd-threaded.mjs",
        wasm: "/ort/ort-wasm-simd-threaded.wasm"
      };
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      return ort;
    });
  }

  return ortModulePromise;
}

async function fetchModelBytes(slot: "model1" | "model2") {
  const chunks = await Promise.all(
    Array.from({ length: MODEL_CHUNKS[slot] }, async (_, index) => {
      const response = await fetch(`/model-chunks/${slot}.onnx.gz.${index}.part`);
      if (!response.ok) {
        throw new Error(`Unable to load ${slot} model chunk ${index + 1}.`);
      }

      return new Uint8Array(await response.arrayBuffer());
    })
  );
  const compressedLength = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const compressed = new Uint8Array(compressedLength);
  let offset = 0;

  for (const chunk of chunks) {
    compressed.set(chunk, offset);
    offset += chunk.byteLength;
  }

  if (!("DecompressionStream" in window)) {
    throw new Error("This browser cannot decompress the model files. Please use a current version of Chrome, Edge, or Firefox.");
  }

  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function getSession(slot: "model1" | "model2") {
  const existing = slot === "model1" ? model1Session : model2Session;
  if (existing) {
    return existing;
  }

  const ort = await getOrt();
  const modelBytes = await fetchModelBytes(slot);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all"
  });

  if (slot === "model1") {
    model1Session = session;
  } else {
    model2Session = session;
  }

  return session;
}

async function imageToTensor(file: File): Promise<Tensor> {
  const ort = await getOrt();
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_INPUT_SIZE;
  canvas.height = MODEL_INPUT_SIZE;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Unable to prepare image canvas.");
  }

  context.drawImage(bitmap, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  bitmap.close();

  const { data } = context.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const input = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

  for (let i = 0; i < planeSize; i += 1) {
    const pixel = i * 4;
    input[i] = (data[pixel] / 255 - mean[0]) / std[0];
    input[planeSize + i] = (data[pixel + 1] / 255 - mean[1]) / std[1];
    input[planeSize * 2 + i] = (data[pixel + 2] / 255 - mean[2]) / std[2];
  }

  return new ort.Tensor("float32", input, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
}

function softmax(logits: Float32Array | number[]) {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

async function runModel(slot: "model1" | "model2", tensor: Tensor, labels: readonly string[]) {
  const session = await getSession(slot);
  const feeds = { [session.inputNames[0]]: tensor };
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]].data as Float32Array;
  const probabilities = softmax(logits);
  const topIndex = probabilities.indexOf(Math.max(...probabilities));

  return {
    label: labels[topIndex],
    probabilities: labels.map((label, index) => ({
      label,
      value: probabilities[index]
    }))
  };
}

function getFinalResult(model1: ModelResult, model2: ModelResult | null): FinalResult {
  if (model1.label === "Normal") {
    return {
      label: "Normal",
      tone: "good",
      detail: "No abnormality was detected by the triage model."
    };
  }

  if (model1.label === "Tuberculosis") {
    return {
      label: "Tuberculosis",
      tone: "bad",
      detail: "The triage model detected a tuberculosis pattern."
    };
  }

  if (!model2) {
    return {
      label: "Pneumonia",
      tone: "warn",
      detail: "The triage model detected pneumonia."
    };
  }

  const pneumoniaConfidence = model1.probabilities.find((item) => item.label === "Pneumonia")?.value ?? 0;

  if (pneumoniaConfidence > PNEUMONIA_THRESHOLD) {
    const subtype = model2.probabilities
      .filter((item) => item.label === "Bacterial" || item.label === "Viral")
      .sort((a, b) => b.value - a.value)[0];

    return {
      label: `${subtype.label} Pneumonia`,
      tone: "bad",
      detail: `Pneumonia confidence was ${(pneumoniaConfidence * 100).toFixed(1)}%, so the subtype model's Normal output was not used.`
    };
  }

  return {
    label: model2.label === "Normal" ? "Normal" : `${model2.label} Pneumonia`,
    tone: model2.label === "Normal" ? "good" : "bad",
    detail: `Pneumonia confidence was ${(pneumoniaConfidence * 100).toFixed(1)}%, so the subtype model result was used directly.`
  };
}

function ProbabilityBars({ title, result }: { title: string; result: ModelResult }) {
  const topValue = Math.max(...result.probabilities.map((item) => item.value));

  return (
    <section className="panel">
      <div className="panel-title">
        <Activity size={18} aria-hidden="true" />
        <h2>{title}</h2>
      </div>
      <div className="bars">
        {result.probabilities.map((item) => (
          <div className="bar-row" key={item.label}>
            <div className="bar-label">
              <span>{item.label}</span>
              <strong>{(item.value * 100).toFixed(1)}%</strong>
            </div>
            <div className="bar-track" aria-hidden="true">
              <div
                className={item.value === topValue ? "bar-fill top" : "bar-fill"}
                style={{ width: `${Math.max(2, item.value * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const analysisRunId = useRef(0);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [model1, setModel1] = useState<ModelResult | null>(null);
  const [model2, setModel2] = useState<ModelResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const finalResult = useMemo(() => {
    if (!model1) {
      return null;
    }

    return getFinalResult(model1, model2);
  }, [model1, model2]);
  const hasAnalysisState = Boolean(imageUrl || fileName || model1 || model2 || error || isRunning);

  async function analyze(file: File) {
    const runId = analysisRunId.current + 1;
    analysisRunId.current = runId;
    setIsRunning(true);
    setError(null);
    setModel1(null);
    setModel2(null);
    setFileName(file.name);
    setImageUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      return URL.createObjectURL(file);
    });

    try {
      const tensor = await imageToTensor(file);
      if (analysisRunId.current !== runId) {
        return;
      }

      const firstModel = await runModel("model1", tensor, M1_CLASSES);
      if (analysisRunId.current !== runId) {
        return;
      }

      setModel1(firstModel);

      if (firstModel.label === "Pneumonia") {
        const secondModel = await runModel("model2", tensor, M2_CLASSES);
        if (analysisRunId.current !== runId) {
          return;
        }

        setModel2(secondModel);
      }
    } catch (caught) {
      if (analysisRunId.current === runId) {
        setError(caught instanceof Error ? caught.message : "The image could not be analyzed.");
      }
    } finally {
      if (analysisRunId.current === runId) {
        setIsRunning(false);
      }
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      void analyze(file);
    }
  }

  function reset() {
    analysisRunId.current += 1;
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    setImageUrl(null);
    setFileName(null);
    setModel1(null);
    setModel2(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <p className="eyebrow">ONNX browser inference</p>
          <h1>Chest X-Ray Classifier</h1>
        </div>
        <button
          className="remove-analysis-button"
          type="button"
          onClick={reset}
          disabled={!hasAnalysisState}
          aria-label="Remove uploaded X-ray"
          title="Remove uploaded X-ray"
        >
          Remove
        </button>
      </header>

      <section className="workspace">
        <div className="image-pane">
          <label className={imageUrl ? "dropzone has-image" : "dropzone"}>
            <input ref={inputRef} type="file" accept="image/png,image/jpeg" onChange={handleFileChange} />
            {imageUrl ? (
              <img src={imageUrl} alt={fileName ?? "Uploaded chest X-ray"} />
            ) : (
              <span className="empty-state">
                <UploadCloud size={34} aria-hidden="true" />
                <strong>Upload chest X-ray</strong>
                <small>PNG or JPEG</small>
              </span>
            )}
          </label>
          {fileName ? (
            <div className="file-chip">
              <FileImage size={16} aria-hidden="true" />
              <span>{fileName}</span>
              <button
                className="remove-file-button"
                type="button"
                onClick={reset}
                aria-label="Remove uploaded X-ray"
                title="Remove uploaded X-ray"
              >
                <X size={16} aria-hidden="true" />
              </button>
            </div>
          ) : null}
        </div>

        <div className="results-pane">
          {isRunning ? (
            <section className="panel status-panel">
              <div className="loader" />
              <p>Running inference...</p>
            </section>
          ) : null}

          {error ? (
            <section className="panel error-panel">
              <AlertTriangle size={20} aria-hidden="true" />
              <p>{error}</p>
            </section>
          ) : null}

          {finalResult ? (
            <section className={`verdict ${finalResult.tone}`}>
              <div>
                <p className="eyebrow">Final verdict</p>
                <h2>{finalResult.label}</h2>
                <p>{finalResult.detail}</p>
              </div>
              <ShieldAlert size={26} aria-hidden="true" />
            </section>
          ) : (
            <section className="panel idle-panel">
              <p>Select an X-ray image to start inference.</p>
            </section>
          )}

          {model1 ? <ProbabilityBars title="Model 1 Triage" result={model1} /> : null}
          {model2 ? <ProbabilityBars title="Model 2 Pneumonia Subtype" result={model2} /> : null}

          <p className="disclaimer">
            Research prototype only. Do not use this output as a medical diagnosis.
          </p>
        </div>
      </section>
    </main>
  );
}
