🎯 HER2-Classifier — Deep Learning–Based Algorithm for Automated HER2 Scoring

This repository contains a complete AI pipeline for automated HER2 scoring from breast cancer histopathology images.

The system integrates:

Virtual Staining (H&E → IHC) using PSPStain

HER2 multi-class classification using a modified DenseNet201 (IHCNet)

Quality filtering for synthetic IHC

Grad-CAM & Pseudo-Color Visualizations

FastAPI backend

Flutter mobile application

🚀 Project Workflow

Upload H&E or IHC image

Virtual IHC generation (if input is H&E)

IHCNet → HER2 prediction (0, 1+, 2+, 3+)

Grad-CAM + Pseudo-color visualization

Results displayed in Flutter app

📁 Repository Structure
HER2-Classifier/
│
├── backend/                # FastAPI backend: IHCNet + PSPStain inference pipeline
│
├── frontend/               # Flutter app (login → upload → results → history)
│
├── UI-Screens/             # App screenshots used in README
│   ├── login.png
│   ├── upload_image.png
│   ├── processing_page.png
│   ├── classification_result.png
│   ├── history_page.png
│   ├── admin_settings.png
│   ├── logout.png
│   └── system_workflow.png
│
└── README.md

🧠 Models
1️⃣ IHCNet — HER2 Classifier

Backbone: DenseNet201

Custom classifier head: 512 → 256 → 4

Swish activation + BatchNorm + Dropout

Trained on real IHC patches

Retrained using real + high-quality synthetic IHC

🔗 Original Implementation:
https://github.com/Sakib-Hossain-Shovon/IHCNet

✔ Our Enhancements:

Rebuilt network in PyTorch

Loaded our own weights

Added Grad-CAM

Added pseudo-color visualization

Integrated with FastAPI

2️⃣ PSPStain — Virtual IHC Generator

Used to convert H&E → synthetic IHC.

Model characteristics:

ResNet generator

6 residual blocks

Instance normalization

Spectral normalization

🔗 Original Implementation:
https://github.com/ccitachi/PSPStain

Our pipeline:
H&E Patch (256×256)
↓
PSPStain Generator (ResNet)
↓
Synthetic IHC
↓
IHCNet → HER2 Score

📊 Dataset (Kaggle)
Main dataset used in all experiments:

📌 HER2 IHC Patch Dataset
Kaggle Path:

/kaggle/input/ihc-dataset


Dataset Link:
👉 https://www.kaggle.com/datasets/linaaskar/ihc-dataset

Labels included: 0, 1+, 2+, 3+

🧹 Synthetic IHC Quality Filtering

Before retraining, we filtered all synthetic IHC patches using:

🔍 1️⃣ Blur Detection

Using Laplacian variance → reject blurry samples.

🔍 2️⃣ Brightness & Contrast

Reject:

too bright

too dark

low contrast images

🔍 3️⃣ IHCNet Confidence Check

If confidence < 0.55 → reject.

🔍 4️⃣ Label Mismatch

If predicted HER2 ≠ original H&E label → reject.

📉 Final Filtering Result
Type	Percentage
Rejected	85.4%
Accepted	14.6%

Only clean synthetic images were merged into training.

📈 Performance Summary
Baseline IHCNet (Real IHC Only)

Accuracy: 93.85%

Strong on classes 0, 2+, 3+

Class 1+ remains the hardest

Retrained IHCNet (Real + Synthetic IHC)

Accuracy: > 94%

Better recall for 1+ and 2+

Overfitting reduced

Much stronger generalization

PSPStain Evaluation (Synthetic Only)

Accuracy: ≈ 71.9%

Synthetic images alone aren't perfect,
but after filtering they significantly improved the classifier.

📱 Flutter App — UI Preview
🔐 Login Page

📤 Upload Image

⚙️ Processing

📊 Classification Result (HER2 score + Grad-CAM + Pseudo-color)

📜 History Page

🛠 Admin Settings

🚪 Logout Confirmation

🔧 Implementation Requirements
Software

Python 3.x

Google Colab / Jupyter Notebook

PyTorch

OpenCV

NumPy & SciPy

Matplotlib / Seaborn

FastAPI

Flutter

Hardware

GPU-enabled environment

High storage capacity

⚙️ Backend (FastAPI)

Install:

pip install fastapi uvicorn torch torchvision opencv-python numpy pillow


Run server:

uvicorn main:app --host 0.0.0.0 --port 8000

API Endpoint:

POST /predict-her2

Returns:

HER2 score

Confidence

Grad-CAM visualization

Pseudo-color map

Synthetic IHC (if input = H&E)

👩‍💻 Project Team
Name
Lina Askar
Farah Basmaih
Najla Almaghlouth
Lama Alghofaili
Kholoud Alkenani

Supervisor:
Dr. Najah Alsubaie

🔮 Future Work

Integrate Vision Transformers (ViTs)

Whole-slide image inference

Fully deployed clinical API

Multi-biomarker digital pathology

🔒 License

This project is for academic and research purposes only.
Please cite IHCNet, PSPStain, and this repository if used.
