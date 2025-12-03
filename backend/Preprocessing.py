import os
import cv2

base_dir = "/content/drive/MyDrive/BCI_HER2_Project/data/H&E/+3"
patch_size = 224
def save_patch(patch, folder, base_name, count):
    out_path = os.path.join(folder, f"{base_name}_patch{count}.png")
    cv2.imwrite(out_path, patch)
for root, dirs, files in os.walk(base_dir):
    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape
            base_name = os.path.splitext(filename)[0]
            if h <= patch_size and w <= patch_size:
                continue
            count = 0
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    ph, pw, _ = patch.shape
                    if ph < patch_size * 0.5 or pw < patch_size * 0.5:
                        continue
                    if ph < patch_size or pw < patch_size:
                        patch = cv2.copyMakeBorder(
                            patch,
                            0, patch_size - ph,
                            0, patch_size - pw,
                            cv2.BORDER_CONSTANT, value=[0,0,0]
                        )
                    save_patch(patch, root, base_name, count)
                    count += 1
# Augmentation
import os, random, shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
target_class = "+3"
base_path = f"/content/drive/MyDrive/BCI_HER2_Project/data/H&E/{target_class}"
train_folder = os.path.join(base_path, f"H&E_{target_class}_TRAIN")
val_folder   = os.path.join(base_path, f"H&E_{target_class}_VAL")
test_folder  = os.path.join(base_path, f"H&E_{target_class}_TEST")
temp_dir = "/content/aug_temp5"
os.makedirs(temp_dir, exist_ok=True)
target_total = 55900

train_count = len([f for f in os.scandir(train_folder) if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))])
val_count   = len([f for f in os.scandir(val_folder) if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))])
test_count  = len([f for f in os.scandir(test_folder) if f.name.lower().endswith(('.png', '.jpg', '.jpeg'))])

current_total = train_count + val_count + test_count
needed = target_total - current_total

print(f"Current total images: {current_total} (Train: {train_count}, Val: {val_count}, Test: {test_count})")
print(f" Need to generate: {needed}")

if needed <= 0:
    print(" No augmentation needed. Dataset is already balanced.")
else:
    print(f" Generating {needed} new images into {temp_dir} ...")

    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    source_images = [f for f in os.listdir(train_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for i in range(needed):
        img_path = os.path.join(train_folder, random.choice(source_images))
        img = load_img(img_path)
        x = img_to_array(img).reshape((1,) + img.size[::-1] + (3,))

        aug_img = next(datagen.flow(x, batch_size=1))[0]
        aug_img = array_to_img(aug_img)

        aug_img.save(os.path.join(temp_dir, f"aug_{i}.png"))

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{needed} images locally...")

    print("Augmentation finished locally!")
    print(" Moving augmented images to TRAIN folder...")
    for filename in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, filename), os.path.join(train_folder, filename))

    print("All augmented images have been moved to TRAIN!")
    final_total = len([f for f in os.listdir(train_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) + val_count + test_count
    print(f"Final total images in class {target_class}: {final_total}")

# ============================================================
# PSPStain Fake IHC Generator + Quality Filter for /H&E Dataset
# ============================================================

import os, sys, csv, random
from glob import glob

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models

# ============================================================
# DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ============================================================
# PATHS
# ============================================================

PSP_CODE_ROOT = "/kaggle/input/pspstain/PSPStain-main"
if PSP_CODE_ROOT not in sys.path:
    sys.path.append(PSP_CODE_ROOT)

from models.networks import ResnetGenerator, get_norm_layer

PSP_WEIGHT = "/kaggle/input/pspstain-weights/PSPStain/BCI_net_G.pth"
IHCNET_WEIGHTS_PATH = "/kaggle/input/densenet201/ihcnet_best (3).pth"

HE_ROOT = "/kaggle/input/paired-label"

FAKE_ROOT = "/kaggle/working/IHC_fake_only"
os.makedirs(FAKE_ROOT, exist_ok=True)

ACCEPT_CSV = os.path.join(FAKE_ROOT, "fake_accepted.csv")
REJECT_CSV = os.path.join(FAKE_ROOT, "fake_rejected.csv")

CLASSES = ["0", "1", "2", "3"]
print("FAKE_ROOT:", FAKE_ROOT)

# ============================================================
# IHCNet
# ============================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class IHCNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        base = models.densenet201(
            weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = base.classifier.in_features
        base.classifier = nn.Identity()
        self.backbone = base.features

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Swish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

IMG_SIZE_IHC = 224
ihc_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_IHC, IMG_SIZE_IHC)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

ihcnet = IHCNet().to(DEVICE)
state_ihc = torch.load(IHCNET_WEIGHTS_PATH, map_location=DEVICE)
ihcnet.load_state_dict(state_ihc, strict=True)
ihcnet.eval()
print("Loaded IHCNet weights:", IHCNET_WEIGHTS_PATH)

# ============================================================
# PSPStain Generator Loader
# ============================================================

IMG_SIZE_GAN = 256

def build_psp_generator(weight_path):
    print("\n===== PSPStain GENERATOR LOAD =====")
    print("Weight file:", weight_path)

    class Opt: 
        pass

    opt = Opt()
    opt.weight_norm = 'spectral'
    opt.n_downsampling = 2

    norm_layer = get_norm_layer('instance')

    netG = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=6,
        padding_type='reflect',
        no_antialias=False,
        no_antialias_up=False,
        opt=opt,
    ).to(DEVICE)

    state = torch.load(weight_path, map_location=DEVICE)
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    missing, unexpected = netG.load_state_dict(clean_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    netG.eval()
    return netG

G = build_psp_generator(PSP_WEIGHT)

he_pre = transforms.Compose([
    transforms.Resize((IMG_SIZE_GAN, IMG_SIZE_GAN)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

# ============================================================
# GLOBAL reject_reasons COLLECTOR
# ============================================================

reject_reasons = {}

# ============================================================
# Quality Check Function
# ============================================================

def check_quality(fake_pil, label_idx):
    reasons = []

    # --- 1) blur check ---
    np_img = np.array(fake_pil.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 30:
        reasons.append(f"too_blurry(var={lap_var:.1f})")

    # --- 2) brightness / contrast ---
    img_float = np_img.astype(np.float32) / 255.0
    brightness = img_float.mean()
    contrast   = img_float.std()

    if brightness < 0.15 or brightness > 0.85:
        reasons.append(f"bad_brightness({brightness:.2f})")
    if contrast < 0.03 or contrast > 0.40:
        reasons.append(f"bad_contrast({contrast:.2f})")

    # --- 3) IHCNet prediction ---
    with torch.no_grad():
        ihc_t = ihc_transform(fake_pil).unsqueeze(0).to(DEVICE)
        logits = ihcnet(ihc_t)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())
        conf   = float(probs[pred])

    if conf < 0.55:
        reasons.append(f"low_confidence({conf:.2f})")

    if pred != label_idx:
        reasons.append(f"label_mismatch(he={label_idx}, ihcnet={pred})")

    # Collect global counts
    for r in reasons:
        reject_reasons[r] = reject_reasons.get(r, 0) + 1

    good = (len(reasons) == 0)
    return good, reasons, pred, conf

# ============================================================
# Collect all H&E recursively (READ ONLY)
# ============================================================

def collect_he_paths():
    items = []
    for cls in CLASSES:
        class_idx = int(cls)
        root = os.path.join(HE_ROOT, cls)

        if not os.path.isdir(root):
            print("WARNING: folder not found:", root)
            continue

        patterns = [
            os.path.join(root, "**", "*.png"),
            os.path.join(root, "**", "*.jpg"),
            os.path.join(root, "**", "*.jpeg"),
        ]

        paths = []
        for pat in patterns:
            paths.extend(glob(pat, recursive=True))

        print(f"Class {cls}: found {len(paths)} images")
        for p in paths:
            items.append((p, class_idx))

    random.shuffle(items)
    print("Total H&E patches:", len(items))
    return items

he_items = collect_he_paths()

# ============================================================
# Prepare fake folders
# ============================================================

for cls in CLASSES:
    os.makedirs(os.path.join(FAKE_ROOT, cls), exist_ok=True)

# CSV
acc_f = open(ACCEPT_CSV, "w", newline="")
rej_f = open(REJECT_CSV, "w", newline="")
acc_w = csv.writer(acc_f)
rej_w = csv.writer(rej_f)

acc_w.writerow(["he_path", "fake_path", "class", "pred_class", "confidence"])
rej_w.writerow(["he_path", "class", "pred_class", "confidence", "reasons"])

# ============================================================
# GENERATION LOOP
# ============================================================

print("\n===== START GENERATION & FILTERING =====")

total = len(he_items)
accepted = 0
rejected_count = 0

for idx, (he_path, label_idx) in enumerate(he_items, 1):

    try:
        he_img = Image.open(he_path).convert("RGB")
    except Exception as e:
        print("Failed to open:", he_path, "|", e)
        continue

    # 1) H&E -> tensor
    he_t = he_pre(he_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_t = G(he_t)
        fake_t = (fake_t + 1.0) / 2.0  # [0,1]

    fake_pil = transforms.ToPILImage()(torch.clamp(fake_t[0].cpu(), 0, 1))

    # 2) quality check
    good, reasons, pred_cls, conf = check_quality(fake_pil, label_idx)

    # 3) output path
    cls_name = str(label_idx)
    base_name = os.path.basename(he_path)
    fake_name = os.path.splitext(base_name)[0] + "_fake.png"
    fake_dir  = os.path.join(FAKE_ROOT, cls_name)
    fake_path = os.path.join(fake_dir, fake_name)

    if good:
        fake_pil.save(fake_path)
        acc_w.writerow([he_path, fake_path, cls_name, pred_cls, f"{conf:.4f}"])
        accepted += 1
    else:
        rej_w.writerow([he_path, cls_name, pred_cls, f"{conf:.4f}", ";".join(reasons)])
        rejected_count += 1

    if idx % 200 == 0 or idx == total:
        print(f"[{idx}/{total}] accepted={accepted} rejected={rejected_count}", end="\r")

acc_f.close()
rej_f.close()

# ============================================================
# REPORT SECTION
# ============================================================

print("\n================= FILTER REPORT =================")
print(f"Total generated images : {total}")
print(f"Accepted              : {accepted}")
print(f"Rejected              : {rejected_count}")

if total > 0:
    print(f"Rejection rate        : {rejected_count/total*100:.2f}%")
    print(f"Acceptance rate       : {accepted/total*100:.2f}%")

print("\nReasons for rejection:")
for reason, count in reject_reasons.items():
    pct = (count / rejected_count * 100) if rejected_count > 0 else 0
    print(f" - {reason}: {count} ({pct:.2f}%)")

print("=================================================\n")

print("\n===== DONE =====")
print("Accepted CSV :", ACCEPT_CSV)
print("Rejected CSV :", REJECT_CSV)
print("Fake dataset :", FAKE_ROOT)
