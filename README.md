# Vasudha_Annam
Annam.AI Hackathon 2025    "Fueling Innovation in Food &amp; Sustainable Agriculture with AI"


Challenge 1:
# Soil Image Classification: Challenge 1

This repository contains the solution for the Soil Image Classification Challenge hosted on Kaggle:  
https://www.kaggle.com/competitions/soil-classification

---

## Project Overview

The goal of this challenge is to classify soil images into different soil types based on a given dataset. This project uses deep learning with transfer learning (ResNet18) and PyTorch framework to build an image classification model. The dataset contains images of various formats and resolutions, which are analyzed and preprocessed before training.

---

## Dataset

- **Training Images:** Located in `train/` directory.
- **Test Images:** Located in `test/` directory.
- **Labels:** Provided in `train_labels.csv`.
- **Test IDs:** Provided in `test_ids.csv`.

---

## Key Features

- Data exploration and analysis of image formats, resolutions, and aspect ratios.
- Handling of multiple image file types (jpg, png, gif, webp).
- Weighted sampling to handle class imbalance.
- Data augmentation with torchvision transforms.
- Transfer learning using pretrained ResNet18.
- Training and evaluation with F1-score metric.

---

## Setup Instructions

## Setup and Run Instructions

- Clone the repository to your local machine.
- Install the required Python packages (PyTorch, torchvision, pandas, numpy, matplotlib, scikit-learn, tqdm, pillow).
- Place the dataset folders (`train/` and `test/`) and CSV files (`train_labels.csv` and `test_ids.csv`) in the project directory.
- Run the training script to train the model.
- Run the prediction script to generate predictions on the test data.

