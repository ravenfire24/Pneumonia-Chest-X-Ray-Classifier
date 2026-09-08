# Pneumonia Chest X-Ray Classifier

Browser-based chest X-ray classification app for detecting **Normal**, **Pneumonia**, and **Tuberculosis** patterns, with pneumonia subtype classification for **Bacterial** and **Viral** results.

Live App: https://pneumonia-chest-x-ray-classifier.vercel.app

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

##  Dataset

The dataset contains:

| Class | Description |
|---|---|
| Normal | Healthy chest X-rays |
| Pneumonia | Pneumonia infected lungs |
| Tuberculosis | Tuberculosis infected lungs |

Dataset size: 15,000+ images

https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset

https://www.kaggle.com/datasets/shreyanmohanty/chest-x-ray-dataset-for-pneumonia-classification

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
pip install -r scripts/requirements-export.txt
python scripts/export_models_to_onnx.py
npm run chunk-models
```

![alt text](https://github.com/ravenfire24/Pneumonia-Chest-X-Ray-Classifier/blob/main/result.JPG)
