# 🌱 Sugarcane Breed Classification using CNNs

A deep learning project to classify sugarcane breeds from images using Convolutional Neural Networks (CNNs) and transfer learning. Trained on a small dataset of 70 labeled images across 6 unique sugarcane breeds using ResNet18, ResNet50, and EfficientNet-B0 with ensemble voting for enhanced accuracy.

---

## 📌 Problem Statement

Accurately identifying sugarcane breeds from leaf/stalk images is essential for agricultural research and supply chain tracking. Manual identification is time-consuming and error-prone. This project automates the classification process using deep learning.

---

## 📂 Dataset

- Total Images: **70**
- Classes: **6 distinct sugarcane breeds**
- Format: Images organized in folders by breed name
- Sample Breeds:
  - `CoLK-14201`
  - `CoLK-16466`
  - `CoLK-94184`
  - `Colk-12209`
  - `Colk-15466`
  - `Colk-16470`

---

## ✅ Project Goals

- Classify sugarcane breed from image input
- Improve accuracy using **transfer learning**
- Analyze misclassifications and errors
- Combine multiple models using ensemble for better performance

---

## 🛠️ Techniques Used

- **Data Augmentation**: Rotation, brightness jitter, horizontal flips
- **Transfer Learning**: Pretrained models (ResNet18, ResNet50, EfficientNet-B0)
- **Model Ensembling**: Majority voting among top-3 models
- **Error Analysis**: Confusion matrix, misclassified image visualization
- **Test-Time Augmentation (TTA)**: Boosted final predictions
- **PyTorch & torchvision**: Model development
- **Google Colab**: Training environment

---

## 🔍 Performance Metrics

| Model          | Accuracy | F1-Score (Weighted) |
|----------------|----------|---------------------|
| ResNet18       | 94%      | 0.94                |
| ResNet50       | 96%      | 0.95                |
| EfficientNet-B0| 93%      | 0.92                |
| **Ensemble**   | **97%**  | **0.96**            |

---

## 📊 Evaluation

- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Misclassified image visualization
- Priority-based prediction logic for ensemble (e.g., trust ResNet50 > EfficientNet > ResNet18)

---

## 🧠 Models Used

- [x] **ResNet18** (lightweight & fast)
- [x] **EfficientNet-B0** (balanced accuracy and efficiency)
- [x] **ResNet50** (deeper network for refined features)

---

## 📁 Project Structure
sugarcane_classifier/
├── data/
│ └── train/val/test (folders by breed)
├── models/
│ └── resnet18.pth, efficientnet.pth, resnet50.pth
├── notebook/
│ └── sugarcane_classification.ipynb
├── utils/
│ └── visualize.py, metrics.py
├── ensemble_predict.py
└── README.md


---

## 🖼️ Sample Predictions

| Image | Predicted Breed | Actual Breed | Correct |
|-------|------------------|--------------|---------|
| ![](example1.jpg) | Colk-15466 | Colk-15466 | ✅ |
| ![](example2.jpg) | CoLK-16466 | Colk-12209 | ❌ |

---

## 🚀 How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/sugarcane-classifier.git
   cd sugarcane-classifier


   💡 Future Improvements
Collect more labeled sugarcane images

Use crop-specific domain adaptation

Deploy as a mobile/web application

Integrate attention-based models (e.g., ViT)

👨‍💻 Author
Rajat garg
Machine Learning & Computer Vision Enthusiast
