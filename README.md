
## Pneumonia Chest X-Ray Classifier


  Deep learning-based medical image classification system for detecting **Pneumonia**, **Tuberculosis**, and **Normal** chest X-rays using **DenseNet121** and **PyTorch**.
The model was trained on over **15,000+ chest X-ray images** with data augmentation and class-weighted loss to improve performance on imbalanced medical datasets.

---

## Live Demo
https://pneumonia-chest-x-ray-classifier-zdegfsfuiszm2qrnxmu5py.streamlit.app/

### Features in the Demo
- Upload chest X-ray images
- Predict:
  - Normal
  - Pneumonia
  - Tuberculosis
- View model confidence scores
- Test using sample X-ray images

##  Features

- Multi-class chest X-ray classification
- Transfer learning with DenseNet121
- Image augmentation pipeline
- Confusion matrix visualization
- Model checkpoint saving/loading
- PyTorch implementation
- Training and evaluation scripts
---
## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

##  Model Architecture

This project uses:

- **DenseNet121**
- Pretrained ImageNet weights
- Fine-tuning for medical imaging tasks
- Weighted CrossEntropyLoss for class imbalance handling

---

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

---

##  Results

| Metric | Score |
|---|---|
| Validation Accuracy | 85%+ |
| Framework | PyTorch |
| Model | DenseNet121 |

 ![alt text](https://github.com/ravenfire24/Pneumonia-Chest-X-Ray-Classifier/blob/main/result.JPG)



