# 🎯 HER2-Classifier — Deep Learning–Based Algorithm for Automated HER2 Scoring

This repository contains a complete AI pipeline for **automated HER2 scoring** from breast cancer histopathology images.

The system integrates:

- **Virtual Staining (H&E → IHC)** using a PSPStain ResNet-based generator  
- **HER2 multi-class classification** using a modified DenseNet201 (IHCNet)  
- **Quality filtering for synthetic IHC**  
- **Grad-CAM & Pseudo-Color Visualizations**  
- **FastAPI backend**  
- **Flutter mobile application**

---

## 🚀 Project Workflow

1. **Input Image (H&E or IHC)**  
2. **Virtual IHC Generation** using PSPStain (if the input is H&E)  
3. **HER2 Classification** using IHCNet → {0, 1+, 2+, 3+}  
4. **Confidence Score + Grad-CAM + Pseudo-color map**  
5. **Results displayed in Flutter application**  

---

## 📁 Repository Structure

```text
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
└── README.md               # Project documentation
🧠 Models
1️⃣ IHCNet — HER2 Classifier
Backbone: DenseNet201, pretrained on ImageNet

Custom classifier head: 512 → 256 → 4 classes with Swish activation, BatchNorm, Dropout

Trained first on real IHC patches, then retrained using real + high-quality synthetic IHC

Original implementation (we adapted from this repository):
IHCNet original GitHub repo

In our code we:

Reimplemented the network in PyTorch

Loaded our own trained weights (.pth)

Integrated Grad-CAM and pseudo-color visualization for interpretability

2️⃣ PSPStain — Virtual IHC Generator
ResNet-based generator with:

n_blocks = 6

n_downsampling = 2

instance normalization

spectral weight normalization

Used to generate synthetic IHC from H&E patches before feeding them into IHCNet

Part of our figures and reference images follow the architecture and examples from the original PSPStain work.

Original implementation (we adapted & reused generator weights):
PSPStain original GitHub repo

In our backend we:

Load the PSPStain generator as ResnetGenerator

Pass 256×256 H&E patches through it → generate synthetic IHC

Then send the result to IHCNet for HER2 scoring

Some visual illustrations in this repository (e.g., synthetic H&E → IHC examples) are based on / inspired by the figures in the PSPStain repository and our own generated outputs.

📊 Datasets (Kaggle)
All datasets were accessed through Kaggle inside Google Colab.

1️⃣ Real IHC Patch Dataset
Used to train the baseline IHCNet (real IHC only).

Kaggle path in code:

python

DATASET_ROOT = "/kaggle/input/ihc-dataset"
Kaggle dataset :
HER2 IHC Patch Dataset

This dataset provides HER2-stained IHC patches labeled as 0, 1+, 2+, 3+ and was used for the initial training and evaluation of IHCNet.

2️⃣ Paired H&E / IHC Dataset (BCI)
Used for:

Training the PSPStain virtual staining model

Generating synthetic IHC patches for retraining IHCNet

Kaggle path in code:

python

DATASET_ROOT_BCI = "/kaggle/input/paired-labeled/BCI_dataset/IHC"
Kaggle dataset (replace with your exact link):
BCI Paired H&E–IHC Dataset

This dataset contains paired H&E and IHC images, which we cropped into 224×224 / 256×256 patches. Synthetic IHC generated from these H&E patches was later filtered and merged with the real IHC dataset for retraining.

🧹 Synthetic IHC Quality Filtering
Before retraining IHCNet with synthetic data, we applied a strict quality assessment pipeline:

Blur check — Laplacian variance on the grayscale image

Brightness / contrast check — reject too dark / too bright / low-contrast images

IHCNet confidence check

Pass the synthetic IHC through a pretrained IHCNet

If softmax confidence < 0.55 → reject

Label mismatch check

If predicted label ≠ original H&E label → reject

As a result:

85.4% of synthetic images were rejected

Only the top-quality 14.6% were used for retraining IHCNet

📈 Performance Summary
Baseline IHCNet (Real IHC only)
Test accuracy: 93.85%

High precision/recall for 0, 2+, 3+

1+ remains the most challenging class

Retrained IHCNet (Real + Synthetic IHC)
Test accuracy: > 95%

Improved recall for 1+ and 2+

Overfitting gap reduced (train vs val accuracy closer)

More robust generalization across heterogeneous staining conditions

PSPStain Evaluation (Synthetic IHC only)
Directly classifying PSPStain images with IHCNet yielded ≈ 71.9% accuracy,
confirming that synthetic images are useful but less reliable than real IHC.

After filtering and retraining, synthetic samples acted as data augmentation rather than direct substitutes for real IHC.

🔧 Implementation Requirements
Software
Python 3.x

Google Colab / Jupyter Notebook

PyTorch

OpenCV

NumPy & SciPy

Matplotlib & Seaborn

Grad-CAM Toolkit

FastAPI (API integration)

Flutter (application development)

Hardware
GPU-enabled environment (Colab / local GPU)

Sufficient storage for large histopathology datasets

⚙️ Backend (FastAPI)
The backend exposes a REST API around the PSPStain generator + IHCNet classifier.

1️⃣ Install requirements
bash
pip install fastapi uvicorn torch torchvision opencv-python numpy pillow
2️⃣ Run server
bash
uvicorn main:app --host 0.0.0.0 --port 8000
3️⃣ Example endpoint
POST /predict-her2

Input: image (H&E or IHC)

Output:

predicted HER2 score (0/1+/2+/3+)

confidence

Grad-CAM & pseudo-color maps (Base64)

synthetic IHC if input was H&E

Flutter calls this endpoint using the configured API link in the Admin Settings screen.

📱 Frontend (Flutter App)
Preview images are available in UI-Screens/:

Login.png — secure login for doctors / admins

Upload image.png — upload H&E or IHC (patch or WSI crop)

Processing page.png — shows analysis in progress

Classification Result Page (Synthetic IHC).png — HER2 score + confidence + visualizations

History page.png — previous reports and analysis history

Admin Settings Page.png — API URL, model (.pth) path, user management

Logout.png — safe logout confirmation

System Workflow.png — end-to-end overview of the pipeline

👩‍💻 Project Team
Name	
Lina Askar
Farah Basmaih	
Najla Almaghlouth	
Lama Alghofaili
Kholoud Alkenani	
Supervisor: Dr. Najah Alsubaie	

🔮 Future Work
Integrate Vision Transformers (ViTs) / hybrid CNN-Transformer models to better capture fine-grained membrane patterns, especially in borderline classes (1+ / 2+).

Extend from patch-based inference to whole-slide analysis with tiling and slide-level aggregation.

Deploy a full clinical API for real-time HER2 scoring and visual explanation.

Combine additional biomarkers to build a comprehensive multi-modal digital pathology platform.

🔒 License
This project is provided for academic and research purposes only.
Please cite the original IHCNet and PSPStain works, as well as this repository, if you use or extend this code.
