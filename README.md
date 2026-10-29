# AI-Driven Pottery Classification

A computer vision project that uses deep learning to classify pottery images as **handmade or machine-made**. The project combines image dataset preparation, a custom Convolutional Neural Network (CNN), and MobileNetV2 transfer learning to explore image classification using TensorFlow/Keras.

An individual academic project.

**Tech Stack:** Python · TensorFlow/Keras · NumPy · scikit-learn · Hugging Face Datasets · Matplotlib · Seaborn · Google Colab

## Project Overview

| Component | Description |
|---|---|
| Task | Binary image classification |
| Classes | Handmade and machine-made pottery |
| Image Input | RGB images resized to 224 × 224 |
| Dataset | 4,598 images assembled from public datasets and supplementary image collections |
| Models | Custom CNN, MobileNetV2 transfer learning |
| Evaluation | Accuracy, classification reports, confusion matrices, external image predictions |
| Notebook | [`AI-Driven-Pottery-Classification.ipynb`](AI-Driven-Pottery-Classification.ipynb) |

## Dataset

The image dataset combines public image collections from Hugging Face with supplementary images.

| Source | Category | Images |
|---|---|---:|
| [Pottery-Like](https://huggingface.co/datasets/eddie-cui/Pottery-Like) | Handmade | 3,851 |
| [Kitchenware 5000 Split](https://huggingface.co/datasets/groove-net/kitchenware_5000_split) | Machine-made | 522 |
| Supplementary image collections | Handmade | 50 |
| Supplementary image collections | Machine-made | 175 |
| **Total** | **Two classes** | **4,598** |

The merged dataset contains **3,901 handmade** and **697 machine-made** images.

## Methodology

### 1. Dataset Preparation
- Load image datasets from Hugging Face.
- Select relevant kitchenware categories for machine-made pottery.
- Convert images to RGB and resize them to 224 × 224 pixels.
- Combine public and supplementary image collections.

### 2. Custom CNN
Build a Convolutional Neural Network from scratch with four convolutional blocks, Batch Normalization, Max Pooling, and a dense classification head.

The training pipeline includes:
- Image augmentation using rotation, shifts, shear, zoom, and horizontal flips.
- Binary Focal Cross-Entropy loss.
- Adam optimizer.
- Best-validation-accuracy checkpoint selection.

### 3. MobileNetV2 Transfer Learning
Use a pretrained MobileNetV2 backbone with ImageNet weights and frozen feature-extraction layers. Add a global average pooling layer and a sigmoid classification output for binary prediction.

### 4. Model Evaluation
Evaluate the models using accuracy, confusion matrices, classification reports, and predictions on four external pottery images.

## Results

The following results are reported in the notebook's saved outputs.

| Model | Evaluation Set | Accuracy |
|---|---|---:|
| Custom CNN | 920-image validation set | 99.00% |
| MobileNetV2 | 340-image test set | 99.12% |

### Confusion Matrices

**Custom CNN**

![CNN Confusion Matrix](assets/cnn_confusion_matrix.png)

**MobileNetV2**

![MobileNetV2 Confusion Matrix](assets/mobilenetv2_confusion_matrix.png)

### CNN Training Performance

![CNN Training Curves](assets/cnn_training_curves.png)

### External Image Predictions

The trained CNN was also applied to four external pottery images:

| Image | Predicted Class | Model Confidence |
|---|---|---:|
| `handmade.jpg` | Handmade | 61.35% |
| `handmade1.jpg` | Handmade | 69.20% |
| `Machinmade.jpg` | Machine-made | 99.63% |
| `Machinmade1.jpg` | Machine-made | 78.81% |

## Skills Demonstrated

- Computer vision and image classification
- Image preprocessing and augmentation
- Convolutional Neural Networks (CNNs)
- Transfer learning with MobileNetV2
- TensorFlow/Keras model development
- Dataset integration and class filtering
- Model evaluation and performance visualization
- Python-based deep learning workflows

## Getting Started

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Install the packages listed in `requirements.txt` when running locally.
3. Run the notebook cells in sequence to prepare the images, train the models, and review the evaluation results.

The notebook uses public Hugging Face datasets and supplementary image collections for dataset preparation.

## Repository Structure

```text
AI-Driven-Pottery-Classification/
├── assets/
│   ├── cnn_confusion_matrix.png
│   ├── mobilenetv2_confusion_matrix.png
│   └── cnn_training_curves.png
├── AI-Driven-Pottery-Classification.ipynb
├── requirements.txt
└── README.md
```
