# Pneumonia Chest X-Ray Classifier

Browser-based chest X-ray classification app for detecting **Normal**, **Pneumonia**, and **Tuberculosis** patterns, with pneumonia subtype classification for **Bacterial** and **Viral** results.

The original Streamlit/PyTorch app has been migrated to **Next.js on Vercel**. Inference now runs in the browser with **ONNX Runtime Web**, avoiding heavyweight PyTorch serverless functions.

## Features

- Upload PNG or JPEG chest X-ray images
- Run a two-stage DenseNet121 classifier pipeline
- View confidence scores for each model
- Get a final triage verdict
- Keep medical images client-side during inference

## Tech Stack

- Next.js
- React
- TypeScript
- ONNX Runtime Web
- Vercel

## Model Flow

1. Model 1 predicts `Normal`, `Pneumonia`, or `Tuberculosis`.
2. Model 2 runs only when Model 1 predicts `Pneumonia`.
3. If Model 1 pneumonia confidence is above 80%, Model 2's `Normal` class is ignored and the higher pneumonia subtype is used.
4. Otherwise, Model 2's result is used directly.

## Local Development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Build

```bash
npm run build
```

## Model Export

The deployed app uses ONNX models in `public/models/`.

To regenerate them from the original PyTorch checkpoints:

```bash
pip install -r requirements.txt
python scripts/export_models_to_onnx.py
```

## Medical Notice

This project is a research prototype and must not be used as a medical diagnosis tool.
