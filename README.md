🎯 HER2-Classifier — Deep Learning–Based Algorithm for Automated HER2 Scoring

This repository contains a complete AI pipeline for automated HER2 scoring from breast cancer histopathology images.

The system integrates:

Virtual Staining (H&E → IHC) using PSPStain

HER2 multi-class classification using DenseNet201 (IHCNet)

Quality filtering for synthetic IHC

Grad-CAM & Pseudo-color visualizations

FastAPI backend

Flutter mobile application

🚀 Project Workflow
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/system_workflow.png" width="750">
📁 Repository Structure
HER2-Classifier/
│
├── backend/                # FastAPI backend — IHCNet + PSPStain inference
├── frontend/               # Flutter mobile application
├── UI-Screens/             # App UI images used in this README
│   ├── login.png
│   ├── upload_image.png
│   ├── processing_page.png
│   ├── classification_result.png
│   ├── history_page.png
│   ├── admin_settings.png
│   ├── logout.png
│   └── system_workflow.png
└── README.md

🧠 Models
1️⃣ IHCNet — HER2 Classifier

Backbone: DenseNet201

Custom classifier: 512 → 256 → 4 classes

Swish activation, BatchNorm, Dropout

Trained on:

Real IHC patches

Real + high-quality synthetic IHC patches (after filtering)

🔗 Original Implementation:
https://github.com/Sakib-Hossain-Shovon/IHCNet

✔️ Our Enhancements

Rebuilt the network in PyTorch

Loaded our own trained .pth weights

Added Grad-CAM

Added pseudo-color visualization

Integrated the classifier into FastAPI backend

2️⃣ PSPStain — Virtual IHC Generator

Used to convert H&E → synthetic IHC.

Model architecture:

ResNet-based generator

6 residual blocks

InstanceNorm

SpectralNorm

🔗 Original Implementation:
https://github.com/ccitachi/PSPStain

✔️ Pipeline
H&E patch (256×256)
        ↓
PSPStain ResNet Generator
        ↓
Synthetic IHC Patch
        ↓
IHCNet → HER2 Score

📊 Dataset (Kaggle)

We used one main dataset stored on Kaggle.

📌 HER2 IHC Patch Dataset (Main Training Dataset)

Used for:

Baseline IHCNet training

Evaluation

Retraining after merging high-quality synthetic IHC

Path in code:

DATASET_ROOT = "/kaggle/input/ihc-dataset"


Dataset link:
👉 https://www.kaggle.com/datasets/linaaskar/ihc-dataset

Labels included: 0, 1+, 2+, 3+

🧹 Synthetic IHC Quality Filtering

Before retraining IHCNet, a strict QC pipeline removed poor synthetic patches.

Checks applied:
1️⃣ Blur Check

Rejects images with low Laplacian variance.

2️⃣ Brightness & Contrast Check

Rejects images that are:

Too bright

Too dark

Very low contrast

3️⃣ Confidence Check

If IHCNet confidence < 0.55 → reject.

4️⃣ Label Mismatch

If predicted HER2 class ≠ original H&E class → reject.

📉 Filtering Result
Category	Percentage
Rejected	85.4%
Accepted	14.6%

Only high-quality synthetic images were merged with real IHC for retraining.

📈 Performance Summary
Baseline IHCNet (Real IHC Only)

Accuracy: 93.85%

Strong at 0, 2+, 3+

Class 1+ remains hardest

Retrained IHCNet (Real + Synthetic IHC)

Accuracy: >94%

Major improvement for 1+ and 2+

Reduced overfitting

Better generalization

PSPStain Evaluation (Synthetic Only)

Accuracy: ≈71.9%
Synthetic images alone are not reliable
→ but after filtering they become useful augmentation.

📱 Flutter App — UI Preview
🔐 Login Page
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/login.png" width="450">
⬆️ Upload Image
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/upload_image.png" width="450">
⚙️ Processing Page
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/processing_page.png" width="450">
📊 Classification Result
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/classification_result.png" width="450">
📜 History Page
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/history_page.png" width="450">
🔧 Admin Settings
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/admin_settings.png" width="450">
🚪 Logout
<img src="https://github.com/Lina-Askar/HER2-Classifier/blob/main/UI-Screens/logout.png" width="450">
🔧 Backend (FastAPI)

Install:

pip install fastapi uvicorn torch torchvision opencv-python numpy pillow


Run server:

uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint

POST /predict-her2

Returns:

HER2 score

Confidence

Grad-CAM

Pseudo-color

Synthetic IHC (if input was H&E)

👩‍💻 Project Team
Member
Lina Askar
Farah Basmaih
Najla Almaghlouth
Lama Alghofaili
Kholoud Alkenani

Supervisor: Dr. Najah Alsubaie

🔮 Future Work

Vision Transformers (ViTs)

Whole-slide image inference (WSI)

Clinical deployment

Multi-biomarker digital pathology

🔒 License

For academic and research use only.
Please cite IHCNet, PSPStain, and this repository if used.
