# ============================================================
# PSPStain (Evaluation Only) + IHCNet Evaluation Pipeline
# ============================================================

import os
import sys
import csv
import random
from glob import glob

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# DEVICE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ============================================================
# PATHS
# ============================================================

PSP_CODE_ROOT = "/kaggle/input/pspstain/PSPStain-main"

PSP_WEIGHT_BCI  = "/kaggle/input/pspstain-weights/PSPStain/BCI_net_G.pth"

IHCNET_WEIGHTS_PATH = "/kaggle/input/ihcnet-weight/ihcnet_best (6).pth"

BCI_HE_ROOT  = "/kaggle/input/bci-he/bci-he"
BCI_IHC_ROOT = "/kaggle/input/bci-ihc/bci"

OUTPUT_DIR  = "/kaggle/working/pspstain_eval"
HTML_DIR    = os.path.join(OUTPUT_DIR, "html")
SAMPLES_DIR = os.path.join(HTML_DIR, "samples")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

print("OUTPUT_DIR:", OUTPUT_DIR)

if PSP_CODE_ROOT not in sys.path:
    sys.path.append(PSP_CODE_ROOT)
print("Added to PYTHONPATH:", PSP_CODE_ROOT)

from models.networks import ResnetGenerator, get_norm_layer


IDX_TO_CLASS = {0: "0", 1: "1+", 2: "2+", 3: "3+"}

IMG_SIZE_GAN = 256   #  PSPStain
IMG_SIZE_IHC = 224   #  IHCNet

# ============================================================
# IHCNet
# ============================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class IHCNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        base = models.densene201(
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

ihc_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_IHC, IMG_SIZE_IHC)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

ihcnet = IHCNet().to(DEVICE)
state_ihc = torch.load(IHCNET_WEIGHTS_PATH, map_location=DEVICE)
ihcnet.load_state_dict(state_ihc, strict=True)
ihcnet.eval()
print("Loaded IHCNet weights:", IHCNET_WEIGHTS_PATH)

# ============================================================
# PSPStain Generator Loader  
# ============================================================

def build_psp_generator(weight_path):
    """
    PSPStain:
    - netG = resnet_6blocks
    - normG = instance
    - weight_norm = spectral
    - n_downsampling = 2
    """
    print("\n===== PSPStain EVAL =====")
    print("Weight file:", weight_path)

    class Opt:
        pass

    opt = Opt()
    opt.weight_norm    = 'spectral'
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
        opt=opt
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

# ============================================================
# 2. DATA HELPERS (BCI Pairs)
# ============================================================

def list_img(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(("png","jpg","jpeg"))]

def collect_bci_pairs():
    pairs = []
    for cls in ["0", "1+", "2+", "3+"]:
        he_dir  = os.path.join(BCI_HE_ROOT, cls)
        ihc_dir = os.path.join(BCI_IHC_ROOT, cls)
        if (not os.path.isdir(he_dir)) or (not os.path.isdir(ihc_dir)):
            continue
        common = set(list_img(he_dir)) & set(list_img(ihc_dir))
        for f in common:
            pairs.append((cls, os.path.join(he_dir, f), os.path.join(ihc_dir, f)))
    random.shuffle(pairs)
    return pairs

# ============================================================
# 3. Datasets: Real IHC + Fake (PSPStain)
# ============================================================

class BCIRealIHCTest(Dataset):
    def __init__(self, pairs):
        self.items = []
        for cls, _, ihc in pairs:
            label = ["0", "1+", "2+", "3+"].index(cls)
            self.items.append((ihc, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, l = self.items[i]
        img = ihc_transform(Image.open(path).convert("RGB"))
        return img, l, path

he_pre = transforms.Compose([
    transforms.Resize((IMG_SIZE_GAN, IMG_SIZE_GAN)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  
])

class BCIFakeIHCTest(Dataset):
    def __init__(self, pairs, G):
        self.pairs = pairs
        self.G = G

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        cls, he_path, ihc_path = self.pairs[i]
        label = ["0","1+","2+","3+"].index(cls)

        he = Image.open(he_path).convert("RGB")
        he_t = he_pre(he).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            fake = self.G(he_t)
            
            fake = (fake + 1.0) / 2.0
          
            fake = F.interpolate(fake, size=(IMG_SIZE_IHC, IMG_SIZE_IHC),
                                 mode="bilinear", align_corners=False)
            fake = fake.squeeze(0).cpu()

        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        fake = (fake - mean) / std

        return fake, label, ihc_path + "_fake"

# ============================================================
# 4. TEST + SAVE
# ============================================================

@torch.no_grad()
def test_and_save(model, loader, csv_filename):
    model.eval()
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    for imgs, labels, paths in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_paths.extend(paths)

    all_probs  = np.vstack(all_probs)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image","true_label","pred_label","confidence","p0","p1+","p2+","p3+"])
        chosen = all_probs[np.arange(len(all_preds)), all_preds]
        for path, t, p, c, prob in zip(all_paths, all_labels, all_preds, chosen, all_probs):
            w.writerow([
                os.path.basename(str(path)),
                IDX_TO_CLASS.get(int(t), str(int(t))),
                IDX_TO_CLASS.get(int(p), str(int(p))),
                f"{c:.6f}"
            ] + [f"{x:.6f}" for x in prob])

    print(f"Saved predictions -> {csv_path}")
    return all_labels, all_preds, all_probs

def console_analysis(model, test_loader, device, idx_to_class,
                     train_acc_for_check=None, val_acc_for_check=None):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(1).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())

    all_probs = np.asarray(all_probs)
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    total = len(all_labels)

    correct = int(np.sum(np.array(all_preds) == np.array(all_labels)))
    acc_total = 100.0 * correct / max(1, total)
    print("\nOverall Test Accuracy: %.2f%%  (%d/%d)" % (acc_total, correct, total))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "      " + "  ".join([f"{c:>5}" for c in classes])
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{v:>5d}" for v in row])
        print(f"{classes[i]:>5} {row_str}")

    # Heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix Heatmap")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_pspstain.png")
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()
    print(f"Saved confusion matrix -> {cm_path}")

    support = cm.sum(axis=1); correct_diag = cm.diagonal()
    per_cls_acc = np.divide(correct_diag, np.maximum(1, support))
    print("\nPer-Class Accuracy:")
    for c, a, s in zip(classes, per_cls_acc, support):
        print(f"  - {c}: {a*100:.2f}% (support={int(s)})")

    print("\nTop confusing class pairs (true -> predicted):")
    pairs=[]
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i==j: continue
            cnt = cm[i,j]
            if cnt>0:
                rate = cnt/max(1,support[i])
                pairs.append(((classes[i],classes[j]),cnt,rate))
    pairs.sort(key=lambda x:(x[1],x[2]), reverse=True)
    for (t,p),cnt,rate in pairs[:8]:
        print(f"  - {t} -> {p}: {cnt} ({rate*100:.2f}% of true {t})")

    pred_counts = np.bincount(np.array(all_preds), minlength=len(classes))
    true_counts = support
    print("\nDistribution (true vs predicted):")
    for i,c in enumerate(classes):
        t_pct = 100.0*true_counts[i]/max(1,total)
        p_pct = 100.0*pred_counts[i]/max(1,total)
        diff = p_pct - t_pct
        tag = "over-predicted" if diff>0 else ("under-predicted" if diff<0 else "balanced")
        print(f"  - {c}: true={true_counts[i]} ({t_pct:.1f}%) | pred={pred_counts[i]} ({p_pct:.1f}%) -> {tag} by {abs(diff):.1f} pts")

    correct_mask = (np.array(all_preds)==np.array(all_labels))
    chosen_probs = all_probs[np.arange(total), np.array(all_preds)]
    conf_correct = chosen_probs[correct_mask]
    conf_wrong   = chosen_probs[~correct_mask]
    print("\nConfidence (softmax):")
    if conf_correct.size>0:
        print(f"  - Mean confidence (correct): {conf_correct.mean():.3f}")
    if conf_wrong.size>0:
        print(f"  - Mean confidence (wrong):   {conf_wrong.mean():.3f}")

    if (train_acc_for_check is not None) and (val_acc_for_check is not None):
        gap = train_acc_for_check - val_acc_for_check
        print("\nOverfitting Check:")
        print("   Train Acc: %.2f%% | Val Acc: %.2f%% | Gap: %.2f%%" %
              (train_acc_for_check, val_acc_for_check, gap))
        if gap > 10:
            print("   Possible overfitting -> regularization.")
        else:
            print("   No severe overfitting detected.")
    else:
        print("\n Overfitting check skipped (no train/val accuracy available).")

    print("\n Evaluation complete.\n")

# ============================================================
# 5. HTML triplets (H&E, Fake, Real)
# ============================================================

@torch.no_grad()
def save_triplet_for_html(G, pairs, max_samples=50, tag="BCI"):
    print(f"\n========== SAVING HTML SAMPLES ({tag}) ==========")
    G.eval()
    rows = []
    count = 0

    he_pre_html = transforms.Compose([
        transforms.Resize((IMG_SIZE_GAN, IMG_SIZE_GAN), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    for src, he_path, ihc_path in pairs:
        if count >= max_samples:
            break
        he_img  = Image.open(he_path).convert("RGB")
        ihc_img = Image.open(ihc_path).convert("RGB")

        he_t = he_pre_html(he_img).unsqueeze(0).to(DEVICE)

        fake_ihc = G(he_t)
        fake_ihc = (fake_ihc + 1.0) / 2.0

        def to_pil(t):
            t = torch.clamp(t, 0, 1)
            return transforms.ToPILImage()(t)

        he_out   = to_pil((he_t.cpu().squeeze(0) * 0.5 + 0.5))
        fake_out = to_pil(fake_ihc.cpu().squeeze(0))
        real_out = ihc_img.resize((IMG_SIZE_GAN, IMG_SIZE_GAN), Image.BICUBIC)

        he_name   = f"{tag}_{count:03d}_he.png"
        fake_name = f"{tag}_{count:03d}_fake.png"
        real_name = f"{tag}_{count:03d}_real.png"

        he_out.save(os.path.join(SAMPLES_DIR, he_name))
        fake_out.save(os.path.join(SAMPLES_DIR, fake_name))
        real_out.save(os.path.join(SAMPLES_DIR, real_name))

        rows.append((he_name, fake_name, real_name, src, os.path.basename(ihc_path)))
        count += 1

    html_path = os.path.join(HTML_DIR, f"index_{tag}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>PSPStain HER2 Samples</title></head><body>\n")
        f.write(f"<h1>PSPStain ({tag}): H&amp;E → IHC (Real vs Fake)</h1>\n")
        f.write("<table border='1' cellspacing='0' cellpadding='4'>\n")
        f.write("<tr><th>#</th><th>H&amp;E (Input)</th><th>Fake IHC</th><th>Real IHC</th><th>Source</th><th>Filename</th></tr>\n")
        for i,(he_name,fake_name,real_name,src,base) in enumerate(rows):
            f.write("<tr>")
            f.write(f"<td>{i}</td>")
            f.write(f"<td><img src='samples/{he_name}' width='128'></td>")
            f.write(f"<td><img src='samples/{fake_name}' width='128'></td>")
            f.write(f"<td><img src='samples/{real_name}' width='128'></td>")
            f.write(f"<td>{src}</td>")
            f.write(f"<td>{base}</td>")
            f.write("</tr>\n")
        f.write("</table></body></html>\n")

    print("HTML saved at:", html_path)

# ============================================================
# 6. Evaluation wrapper  (BCI)
# ============================================================

def eval_with_generator(tag, weight_path):
    print(f"\n===== PSPStain EVAL ({tag}) =====")

    G = build_psp_generator(weight_path)

    # 1) BCI pairs
    pairs = collect_bci_pairs()
    print("Total BCI pairs:", len(pairs))
    if len(pairs) == 0:
        print("No BCI pairs found! Check BCI_HE_ROOT / BCI_IHC_ROOT paths.")
        return

    # 2) FAKE (PSPStain)
    print("\n=== EVAL: FAKE (PSPStain GENERATED) ===")
    fake_ds = BCIFakeIHCTest(pairs, G)
    fake_ld = DataLoader(fake_ds, batch_size=16, shuffle=False)
    test_and_save(ihcnet, fake_ld, f"fake_bci_{tag.lower()}.csv")
    console_analysis(ihcnet, fake_ld, DEVICE, IDX_TO_CLASS)

    # 3) REAL IHC
    print("\n=== EVAL: REAL BCI IHC ===")
    real_ds = BCIRealIHCTest(pairs)
    real_ld = DataLoader(real_ds, batch_size=32, shuffle=False)
    test_and_save(ihcnet, real_ld, f"real_bci_{tag.lower()}.csv")
    console_analysis(ihcnet, real_ld, DEVICE, IDX_TO_CLASS)

    # 4) HTML triplets
    print("\n=== SAVING HTML ===")
    save_triplet_for_html(G, pairs[:50], max_samples=50, tag=tag)

    print(f"\nDONE EVAL ({tag})! (Check: {OUTPUT_DIR})")

# ============================================================
# 7. MAIN
# ============================================================

def main():
    # BCI weights
    eval_with_generator("BCI",  PSP_WEIGHT_BCI)
    # MIST weights
    eval_with_generator("MIST", PSP_WEIGHT_MIST)

if __name__ == "__main__":
    main()
