# ============================================================
# IHCNet + Full Detailed Analysis  
# ============================================================

import os, time, random, warnings, csv, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# =====================[ CONFIG ]=====================
DATASET_ROOT = "/kaggle/input/ihc-dataset"          
BCI_ROOT     = "/kaggle/input/bci-final/bci"          

PAIRED_ROOT  = "/kaggle/input/gan-ihc/paired-label-fake"       

OUTPUT_DIR   = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
EPOCHS = 17
BATCH_TRAIN = 128
BATCH_EVAL  = 128
INIT_LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
SEED = 42

CLASS_DIRS = ["0","1","2","3"]

CLASS_TO_IDX = {"0":0,"1":1,"2":2,"3":3}
IDX_TO_CLASS = {0:"0",1:"1+",2:"2+",3:"3+"}

BCI_CLASS_DIRS = {
    "0": "0",
    "1": "1+",
    "2": "2+",
    "3": "3+"
}

PAIRED_CLASS_DIRS = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3"
}

TEST_RATIO = 0.15              
VAL_WITHIN_TRAIN = 0.15       

# =====================[ UTILS ]=====================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(SEED)
torch.backends.cudnn.benchmark = True

VALID_EXT = (".png",".jpg",".jpeg",".tif",".tiff")

def list_images_recursively(root_dir):
 
    items = []
    if not os.path.isdir(root_dir):
        return items
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                items.append(os.path.join(r, f))
    return items

def list_subset(class_root, subset_name):
    sub = os.path.join(class_root, subset_name)
    return list_images_recursively(sub) if os.path.isdir(sub) else []

def list_images_in_class(class_name):
  
    class_root = os.path.join(DATASET_ROOT, class_name)
    if not os.path.isdir(class_root):
        return [], [], [], []

    img_test  = list_subset(class_root, "test")
    img_val   = list_subset(class_root, "val")
    img_train = list_subset(class_root, "train")

    all_imgs = list_images_recursively(class_root)

    used = set(img_test) | set(img_val) | set(img_train)
    remaining = [p for p in all_imgs if p not in used]

    return img_train, img_val, img_test, remaining

def list_paired_in_class(class_name):
  
    if class_name not in PAIRED_CLASS_DIRS:
        return []
    folder_name = PAIRED_CLASS_DIRS[class_name]
    class_root = os.path.join(PAIRED_ROOT, folder_name)
    if not os.path.isdir(class_root):
        print(f"[Synthetic IHC] Folder for class {class_name} not found at {class_root}")
        return []
    imgs = list_images_recursively(class_root)
    print(f"[Synthetic IHC] Class {class_name} ({folder_name}): {len(imgs)} images")
    return imgs

def sample_up_to(seq, k, rng):
    if k <= 0 or len(seq) == 0:
        return []
    if len(seq) <= k:
        return list(seq)
    idx = rng.choice(len(seq), size=k, replace=False)
    return [seq[i] for i in idx]

def strict_split_items():

    rng_global = np.random.default_rng(SEED)
    train_items, val_items, test_items = [], [], []

    print("Aggregating & STRICT splitting per-class ...")
    for c in CLASS_DIRS:
        idx = CLASS_TO_IDX[c]
        rng = np.random.default_rng(SEED + idx)

        img_train, img_val, img_test, img_rest = list_images_in_class(c)

        paired_imgs = list_paired_in_class(c)  

        img_rest_extended = img_rest + paired_imgs

        all_unique = list(
            dict.fromkeys(
                img_train + img_val + img_test + img_rest_extended
            )
        )
        n_total = len(all_unique)
        if n_total == 0:
            print(f"  • Class {c}: no images found.")
            continue

        n_test_target = int(math.floor(n_total * TEST_RATIO))
        n_trainval_after_test = n_total - n_test_target
        n_val_target = int(math.floor(n_trainval_after_test * VAL_WITHIN_TRAIN))

        # ---------  test set ---------
        chosen_test = []
        pool_test_pref = [
            list(dict.fromkeys(img_test)),          
            list(dict.fromkeys(img_val)),           
            list(dict.fromkeys(img_train)),        
            list(dict.fromkeys(img_rest_extended)), 
        ]
        already = set()

        for pool in pool_test_pref:
            candidates = [p for p in pool if p not in already]
            need = n_test_target - len(chosen_test)
            part = sample_up_to(candidates, need, rng)
            chosen_test.extend(part)
            already.update(part)
            if len(chosen_test) >= n_test_target:
                break

        # ---------  val set ---------
        allowed_for_val = [p for p in all_unique if p not in set(chosen_test)]

        chosen_val = []
        pref_val = [
            [p for p in img_val   if p in allowed_for_val],
            [p for p in img_train if p in allowed_for_val],
            [p for p in img_rest_extended if p in allowed_for_val],  
        ]
        already_val = set(chosen_test)
        for pool in pref_val:
            candidates = [p for p in pool if p not in already_val]
            need = n_val_target - len(chosen_val)
            part = sample_up_to(candidates, need, rng)
            chosen_val.extend(part)
            already_val.update(part)
            if len(chosen_val) >= n_val_target:
                break

        # ---------  train set ---------
        chosen_test_set = set(chosen_test)
        chosen_val_set  = set(chosen_val)
        chosen_train = [p for p in all_unique
                        if (p not in chosen_test_set and p not in chosen_val_set)]

        rng.shuffle(chosen_train)
        rng.shuffle(chosen_val)
        rng.shuffle(chosen_test)

        train_items += [(p, idx) for p in chosen_train]
        val_items   += [(p, idx) for p in chosen_val]
        test_items  += [(p, idx) for p in chosen_test]

        print(
            f"  • Class {c}: total={n_total} → "
            f"test={len(chosen_test)} (~{100*len(chosen_test)/max(1,n_total):.1f}%), "
            f"val={len(chosen_val)} (~{100*len(chosen_val)/max(1,n_total):.1f}%), "
            f"train={len(chosen_train)} (~{100*len(chosen_train)/max(1,n_total):.1f}%)"
        )

    rng_global.shuffle(train_items)
    rng_global.shuffle(val_items)
    rng_global.shuffle(test_items)

    print(f"Final split → TRAIN={len(train_items)} | VAL={len(val_items)} | TEST={len(test_items)}")
    return train_items, val_items, test_items

# =====================[ DATASET ]===================
class IHCDataset(Dataset):
    def __init__(self, items, transform=None, return_path=False):
        self.items=items; self.transform=transform; self.return_path=return_path
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        for _ in range(5):
            path,label=self.items[idx]
            try:
                img=Image.open(path).convert("RGB")
                if self.transform: img=self.transform(img)
                if self.return_path: return img,label,path
                return img,label
            except (UnidentifiedImageError, OSError):
                idx=(idx+1)%len(self.items)
        dummy=torch.zeros(3,224,224)
        return (dummy,0,"corrupted") if self.return_path else (dummy,0)

train_tfms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
eval_tfms=train_tfms

def load_bci_dataset():
    
    bci_items = []
    for label_str, folder_name in BCI_CLASS_DIRS.items():
        idx = int(label_str)
        class_path = os.path.join(BCI_ROOT, folder_name)
        if not os.path.isdir(class_path):
            print(f"Folder {folder_name} not found, skipping.")
            continue
        imgs = list_images_recursively(class_path)
       
        for p in imgs:
            bci_items.append((p, idx))
    
    return bci_items

def build_loaders():

    train_items, val_items, test_items = strict_split_items()

    bci_items = load_bci_dataset()
    train_items += bci_items


    train_loader = DataLoader(IHCDataset(train_items, train_tfms),
                              batch_size=BATCH_TRAIN, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(IHCDataset(val_items, eval_tfms),
                            batch_size=BATCH_EVAL, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(IHCDataset(test_items, eval_tfms, return_path=True),
                             batch_size=BATCH_EVAL, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader

# =====================[ MODEL ]========
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class IHCNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        base = models.densenet201(
            weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        )
        num_ftrs = base.classifier.in_features
        base.classifier = nn.Identity()
        self.backbone = base.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512), nn.BatchNorm1d(512), Swish(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), Swish(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.classifier(x)

# =====================[ TRAIN / VAL ]================
def train_one_epoch(model,loader,optimizer,scaler,criterion):
    model.train(); total=0; correct=0; loss_sum=0.0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            logits=model(imgs)
            loss=criterion(logits,labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        loss_sum += loss.item()*imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds==labels).sum().item()
        total   += imgs.size(0)
    return loss_sum/total, 100*correct/total

@torch.no_grad()
def validate(model,loader,criterion):
    model.eval(); total=0; correct=0; loss_sum=0.0
    for imgs,labels in loader:
        imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
        logits=model(imgs)
        loss=criterion(logits,labels)
        loss_sum += loss.item()*imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds==labels).sum().item()
        total   += imgs.size(0)
    return loss_sum/total, 100*correct/total

def train_model(train_loader,val_loader):
    model=IHCNet().to(DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler=torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
    BEST_MODEL_PATH=os.path.join(OUTPUT_DIR,"ihcnet_best.pth")
    LOG_CSV=os.path.join(OUTPUT_DIR,"ihcnet_log.csv")
    with open(LOG_CSV,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_acc","val_acc","train_loss","val_loss","minutes"])
    best_val=1e9
    global train_acc_global, val_acc_global
    for epoch in range(1,EPOCHS+1):
        t0=time.time()
        tr_loss,tr_acc = train_one_epoch(model,train_loader,optimizer,scaler,criterion)
        val_loss,val_acc = validate(model,val_loader,criterion)
        scheduler.step()
        minutes=(time.time()-t0)/60
        train_acc_global, val_acc_global = tr_acc, val_acc
        print(f"[{epoch:02d}/{EPOCHS}] Train {tr_acc:.2f}% | Val {val_acc:.2f}% | ValLoss {val_loss:.4f} | {minutes:.1f}m")
        with open(LOG_CSV,"a",newline="") as f:
            csv.writer(f).writerow([epoch,tr_acc,val_acc,tr_loss,val_loss,minutes])
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best model saved ({val_acc:.2f}%)")
    FINAL_PATH=os.path.join(OUTPUT_DIR,"ihcnet_final.pth")
    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Training finished → {FINAL_PATH}")
    return model, BEST_MODEL_PATH

# =====================[ TEST + DETAILED ANALYSIS ]============
@torch.no_grad()
def test_and_save(model, loader):
    model.eval()
    all_labels, all_preds, all_probs, all_paths = [], [], [], []
    for imgs,labels,paths in loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_paths.extend(paths)
    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels= np.concatenate(all_labels)
    csv_path = os.path.join(OUTPUT_DIR,"test_predictions.csv")
    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["image","true_label","pred_label","confidence","p0","p1+","p2+","p3+"])
        chosen = all_probs[np.arange(len(all_preds)), all_preds]
        for path,t,p,c,prob in zip(all_paths,all_labels,all_preds,chosen,all_probs):
            w.writerow([os.path.basename(path), IDX_TO_CLASS[int(t)], IDX_TO_CLASS[int(p)], f"{c:.6f}"]+[f"{x:.6f}" for x in prob])
    print(f"Saved predictions → {csv_path}")
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
            all_probs.extend(probs); all_preds.extend(preds); all_labels.extend(labels.numpy().tolist())

    all_probs = np.asarray(all_probs)
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    total = len(all_labels)

    correct = int(np.sum(np.array(all_preds) == np.array(all_labels)))
    acc_total = 100.0 * correct / max(1, total)
    print(f"\nOverall Test Accuracy: {acc_total:.2f}%  ({correct}/{total})")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "      " + "  ".join([f"{c:>5}" for c in classes]); print(header)
    for i, row in enumerate(cm):
        row_str = " ".join([f"{v:>5d}" for v in row]); print(f"{classes[i]:>5} {row_str}")

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix Heatmap")
    cm_path = os.path.join(OUTPUT_DIR,"confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight"); plt.close()
    print(f"Saved confusion matrix → {cm_path}")

    support = cm.sum(axis=1); correct_diag = cm.diagonal()
    per_cls_acc = np.divide(correct_diag, np.maximum(1, support))
    print("\nPer-Class Accuracy:")
    for c, a, s in zip(classes, per_cls_acc, support):
        print(f"  - {c}: {a*100:.2f}% (support={int(s)})")

    print("\nTop confusing class pairs (true → predicted):")
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
        print(f"  - {t} → {p}: {cnt} ({rate*100:.2f}% of true {t})")

    pred_counts = np.bincount(np.array(all_preds), minlength=len(classes))
    true_counts = support
    print("\nDistribution (true vs predicted):")
    for i,c in enumerate(classes):
        t_pct = 100.0*true_counts[i]/max(1,total)
        p_pct = 100.0*pred_counts[i]/max(1,total)
        diff = p_pct - t_pct
        tag = "over-predicted" if diff>0 else ("balanced" if diff==0 else "under-predicted")
        print(f"  - {c}: true={true_counts[i]} ({t_pct:.1f}%) | pred={pred_counts[i]} ({p_pct:.1f}%) → {tag} by {abs(diff):.1f} pts")

    correct_mask = (np.array(all_preds)==np.array(all_labels))
    chosen_probs = all_probs[np.arange(total), np.array(all_preds)]
    conf_correct = chosen_probs[correct_mask]
    conf_wrong   = chosen_probs[~correct_mask]
    print("\nConfidence (softmax):")
    if conf_correct.size>0: print(f"  - Mean confidence (correct): {conf_correct.mean():.3f}")
    if conf_wrong.size>0:   print(f"  - Mean confidence (wrong):   {conf_wrong.mean():.3f}")

    if (train_acc_for_check is not None) and (val_acc_for_check is not None):
        gap = train_acc_for_check - val_acc_for_check
        print("\nOverfitting Check:")
        print(f"   Train Acc: {train_acc_for_check:.2f}% | Val Acc: {val_acc_for_check:.2f}% | Gap: {gap:.2f}%")
        if gap > 10: print("   Possible overfitting → try stronger dropout or lower LR.")
        else:        print("   No severe overfitting detected.")
    else:
        print("\n Overfitting check skipped (no train/val accuracy available).")

    print("\n Evaluation complete.\n")

# =====================[ RUN ]========================
train_loader, val_loader, test_loader = build_loaders()
model, BEST_MODEL_PATH = train_model(train_loader, val_loader)
best = IHCNet().to(DEVICE)
best.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
y_true, y_pred, _ = test_and_save(best, test_loader)
console_analysis(best, test_loader, DEVICE, IDX_TO_CLASS,
                 train_acc_for_check=globals().get("train_acc_global", None),
                 val_acc_for_check=globals().get("val_acc_global", None))
print("\n Done.")
