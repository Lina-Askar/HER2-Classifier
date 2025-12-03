🎯 HER2-Classifier — Deep Learning–Based Algorithm for Automated HER2 Scoring

This repository contains a complete AI pipeline for automated HER2 scoring from breast cancer histopathology images.

The system integrates:

Virtual Staining (H&E → IHC) using a PSPStain ResNet-based generator

HER2 multi-class classification using a modified DenseNet201 (IHCNet)

Quality filtering for synthetic IHC

Grad-CAM & Pseudo-Color Visualizations

FastAPI backend

Flutter mobile application

🚀 Project Workflow

📁 Repository Structure
HER2-Classifier/
│
├── backend/                # FastAPI backend, inference pipelines (IHCNet + PSPStain)
│
├── frontend/               # Flutter mobile application
│
├── UI-Screens/             # App interface & workflow images (PNG)
│   ├── Login.png
│   ├── Upload image.png
│   ├── Processing page.png
│   ├── Classification Result Page (Synthetic IHC).png
│   ├── History page.png
│   ├── Admin Settings Page.png
│   ├── Logout.png
│   └── System Workflow.png
│
└── README.md

🧠 Models
1️⃣ IHCNet — HER2 Classifier

Backbone: DenseNet201

Custom classifier head: 512 → 256 → 4 classes

Swish activation, BatchNorm, Dropout

Trained on real IHC → then retrained using real + high-quality synthetic IHC

🔗 Original Implementation:
https://github.com/Sakib-Hossain-Shovon/IHCNet

Our enhancements:

Rebuilt the network in PyTorch

Loaded our own trained weights

Added Grad-CAM

Added pseudo-color visualization

Integrated with FastAPI

2️⃣ PSPStain — Virtual IHC Generator

Used to convert H&E patches → synthetic IHC.

Model structure:

ResNet-based generator

6 residual blocks

Instance Normalization

Spectral Normalization

🔗 Original Implementation:
https://github.com/ccitachi/PSPStain

Our pipeline:
H&E Patch (256×256)
        ↓
PSPStain ResNet Generator
        ↓
Synthetic IHC Patch
        ↓
IHCNet → HER2 Scoring

📊 Dataset (Kaggle)

We used one main dataset inside Google Colab:

📌 HER2 IHC Patch Dataset (Main Training Data)

Used for:

Baseline training

Evaluation

Retraining after merging synthetic patches

Kaggle path inside code:

DATASET_ROOT = "/kaggle/input/ihc-dataset"


Dataset link:
👉 https://www.kaggle.com/datasets/linaaskar/ihc-dataset

Labels included: 0, 1+, 2+, 3+

🧹 Synthetic IHC Quality Filtering

To ensure the retraining dataset is reliable, we applied:

🔍 1️⃣ Blur Check

Rejects images with low Laplacian variance.

🔍 2️⃣ Brightness & Contrast Check

Rejects synthetic images that are:

Too bright

Too dark

Very low contrast

🔍 3️⃣ Confidence Check

Pass synthetic IHC → pretrained IHCNet
If confidence < 0.55 → reject

🔍 4️⃣ Label Mismatch

If predicted HER2 class ≠ original class → reject

📉 Final Result

Rejected: 85.4%

Accepted: 14.6% (high-quality synthetic images)

These were merged with real IHC patches for retraining.

📈 Performance Summary
📌 Baseline IHCNet (Real IHC Only)

Accuracy: 93.85%

Very strong performance on 0, 2+, 3+

1+ class remains the most challenging

📌 Retrained IHCNet (Real + Synthetic IHC)

Accuracy: > 94%

Improved recall for 1+ and 2+

Overfitting significantly reduced

Better generalization across staining variations

📌 PSPStain Evaluation (Synthetic Only)

Accuracy ≈ 71.9%

Confirms synthetic images alone are unreliable

But AFTER filtering → synthetic data becomes powerful augmentation

📱 Flutter App – UI Preview
🔐 Login Page

⬆️ Upload Image Page

⚙️ Processing Page

📊 Classification Result Page

📜 History Page

🔧 Admin Settings

🚪 Logout

🔧 Implementation Requirements
Software

Python 3.x

Colab / Jupyter Notebook

PyTorch

OpenCV

NumPy & SciPy

Grad-CAM Toolkit

FastAPI

Flutter

Hardware

GPU-enabled environment

Large storage

⚙️ Backend (FastAPI)

Install:

pip install fastapi uvicorn torch torchvision opencv-python numpy pillow


Run server:

uvicorn main:app --host 0.0.0.0 --port 8000


Endpoint:

POST /predict-her2


Returns:

HER2 score

Confidence

Grad-CAM

Pseudo-color

Synthetic IHC

👩‍💻 Project Team
Name	
Lina Askar
Farah Basmaih
Najla Almaghlouth
Lama Alghofaili	
Kholoud Alkenani
Supervisor: Dr. Najah Alsubaie	
🔮 Future Work

Vision Transformers (ViTs)

Whole-slide image inference

Clinical deployment

Multi-biomarker digital pathology

🔒 License

For academic use only.
Please cite IHCNet, PSPStain, and this repository.
