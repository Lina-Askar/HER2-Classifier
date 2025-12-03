import os
import sys
PSP_CODE_ROOT = os.path.join(os.path.dirname(__file__), "PSPStain-main")
if PSP_CODE_ROOT not in sys.path:
    sys.path.append(PSP_CODE_ROOT)
print("Added to PYTHONPATH:", PSP_CODE_ROOT)
from models.networks import ResnetGenerator as PSPResnetGenerator, get_norm_layer
import torchvision.transforms as transforms
import torch
from PIL import Image
import os, io, base64
from datetime import datetime
import functools
import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# =========================================
# CONFIG
# =========================================

PRIMARY_LAYER = "denseblock3"
IDX_TO_CLASS = {0: "0", 1: "1+", 2: "2+", 3: "3+"}
NUM_CLASSES = 4
IMG_SIZE_GAN = 256  # PSPStain
IMG_SIZE_IHC = 224  # IHCNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_LAYERS = [
    "conv0","denseblock1","denseblock2","denseblock3","denseblock4"
]

WEIGHTS_IHCNET = os.getenv("IHCNET_WEIGHTS", "ihcnet_best (3).pth")
WEIGHTS_PSPStain = os.getenv("PSPStain_WEIGHTS", "BCI_net_G.pth")

# WSI patching
WSI_LONG_EDGE = 1400
PATCH_SIZE = 400
STRIDE = 300
MIN_VALID_SIDE = 120
MIN_NONBLANK_PIX = 200

# transforms
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE_IHC, IMG_SIZE_IHC)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
he_pre = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
pix_tfm_in = transforms.Compose([
    transforms.Resize((IMG_SIZE_GAN,IMG_SIZE_GAN), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
])

# =========================================
# MODEL DEFINITIONS
# =========================================

class Swish(nn.Module):
    def forward(self, x): return x*torch.sigmoid(x)

class IHCNet(nn.Module):
    def __init__(self, num_classes=4, pretrained=False):
        super().__init__()
        base = models.densenet201(
            weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = base.classifier.in_features
        base.classifier = nn.Identity()
        self.backbone = base.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features,512), nn.BatchNorm1d(512), Swish(), nn.Dropout(0.3),
            nn.Linear(512,256), nn.BatchNorm1d(256), Swish(), nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.backbone(x)
        x = self.gap(x).view(x.size(0),-1)
        return self.classifier(x)

def build_and_load_ihcnet(path, device):
    m = IHCNet().to(device)
    s = torch.load(path, map_location=device)
    if "model_state" in s:
        m.load_state_dict(s["model_state"], strict=True)
    else:
        new={}
        for k,v in s.items():
            new[k.replace("module.","")] = v
        m.load_state_dict(new, strict=True)
    m.eval()
    return m

# ======================================================
# EXTRA SCORES
# ======================================================

def dab_score(cv):
    hsv=cv2.cvtColor(cv,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    return int(np.sum((h>5)&(h<32)&(s>40)&(v>40)))

def texture_score(cv):
    g=cv2.cvtColor(cv,cv2.COLOR_BGR2GRAY)
    lap=cv2.Laplacian(g,cv2.CV_64F)
    return float(np.mean(np.abs(lap)))

def tissue_area(cv):
    g=cv2.cvtColor(cv,cv2.COLOR_BGR2GRAY)
    return int(np.sum(g<240))

# ======================================================
# UTILS
# ======================================================

def pil_to_cv(p): return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
def cv_to_pil(c): return Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB))

def to_b64(p):
    b=io.BytesIO()
    p.save(b,format="PNG")
    return base64.b64encode(b.getvalue()).decode()

def draw_patch_box(full, box):
    x0,y0,x1,y1 = map(int, box)
    cv2.rectangle(full,(x0,y0),(x1,y1),(255,0,0),8)
    return full

# ======================================================
# PATCHING
# ======================================================

def remove_blank(cv):
    g=cv2.cvtColor(cv,cv2.COLOR_BGR2GRAY)
    mask=(g<240)&(g>15)
    if np.sum(mask)<MIN_NONBLANK_PIX:
        return cv,(0,0)
    ys,xs=np.where(mask)
    pad=5
    y0=max(0,ys.min()-pad)
    y1=min(cv.shape[0],ys.max()+pad)
    x0=max(0,xs.min()-pad)
    x1=min(cv.shape[1],xs.max()+pad)
    return cv[y0:y1,x0:x1],(x0,y0)

def patch_if_needed(pil):
    w,h=pil.size
    full=pil_to_cv(pil)
    if max(w,h)<=WSI_LONG_EDGE:
        return [pil],[(0,0,w,h)], full
    cropped,(ox,oy)=remove_blank(full)
    H,W,_=cropped.shape
    patches=[]; boxes=[]

    for y in range(0,H,STRIDE):
        for x in range(0,W,STRIDE):
            p=cropped[y:y+PATCH_SIZE,x:x+PATCH_SIZE]
            if p.shape[0]<MIN_VALID_SIDE or p.shape[1]<MIN_VALID_SIDE:
                continue
            g=cv2.cvtColor(p,cv2.COLOR_BGR2GRAY)
            if np.sum(g<250)<MIN_NONBLANK_PIX:
                continue
            patches.append(cv_to_pil(p))
            boxes.append((x+ox,y+oy,x+ox+p.shape[1],y+oy+p.shape[0]))

    if not patches:
        return [pil],[(0,0,w,h)], full

    return patches, boxes, full

# ======================================================
# GRAD CAM
# ======================================================

class GradCam:
    def __init__(self,model,layer):
        self.model=model
        self.layer=layer
        self.grad=None; self.act=None
        self.register()

    def register(self):
        def fwd(m,i,o): self.act=o.detach()
        def bwd(m,gi,go): self.grad=go[0].detach()
        for n,m in self.model.backbone.named_modules():
            if n==self.layer:
                m.register_forward_hook(fwd)
                m.register_backward_hook(bwd)

    def generate(self,x,ci):
        self.model.zero_grad()
        with torch.enable_grad():
            out=self.model(x)
            out[0,ci].backward(retain_graph=True)
        w=self.grad.mean(dim=(2,3),keepdim=True)
        cam=torch.sum(w*self.act,dim=1).squeeze().cpu().numpy()
        cam=np.maximum(cam,0)
        cam/=cam.max()+1e-12
        cam=cv2.GaussianBlur(cam,(5,5),0.5)
        return cam

# =====================[ WSI HELPERS ]=====================

def make_gradcam_wsi(full_cv, patch_cv, box, cam):
    x0, y0, x1, y1 = map(int, box)
    ph, pw = patch_cv.shape[:2]

    cam_r = cv2.resize(cam, (pw, ph))
    cam_u = (cam_r * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam_u, cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(patch_cv, 0.45, heat, 0.55, 0)

    final = full_cv.copy()
    final[y0:y1, x0:x1] = overlay

    rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    h = rgb.shape[0]

    grad = np.linspace(0, 1, h).reshape(h, 1)
    bar = (cm.viridis_r(grad)[:, :, :3] * 255).astype(np.uint8)
    bar = np.repeat(bar, 35, axis=1)

    spacer = np.ones((h, 6, 3), dtype=np.uint8) * 230
    pad = np.ones((h, 20, 3), dtype=np.uint8) * 255

    text_panel = np.ones((h, 150, 3), dtype=np.uint8) * 255
    cv2.putText(text_panel, "High", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(text_panel, "Medium", (5, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(text_panel, "Low", (5, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)

    out = np.hstack((rgb, spacer, bar, pad, text_panel))
    return Image.fromarray(out)


def make_pseudo_wsi(full_cv, patch_cv, box):
    x0, y0, x1, y1 = map(int, box)

    g = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2GRAY)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    pseudo = cv2.applyColorMap((255 - g).astype(np.uint8), cv2.COLORMAP_JET)

    final = full_cv.copy()
    final[y0:y1, x0:x1] = pseudo

    rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    h = rgb.shape[0]

    grad = np.linspace(0, 1, h).reshape(h, 1)
    bar = (cm.jet_r(grad)[:, :, :3] * 255).astype(np.uint8)
    bar = np.repeat(bar, 35, axis=1)

    spacer = np.ones((h, 6, 3), dtype=np.uint8) * 230
    pad = np.ones((h, 20, 3), dtype=np.uint8) * 255

    text_panel = np.ones((h, 150, 3), dtype=np.uint8) * 255
    cv2.putText(text_panel, "High", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(text_panel, "Medium", (5, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(text_panel, "Low", (5, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)

    out = np.hstack((rgb, spacer, bar, pad, text_panel))
    return Image.fromarray(out)

def probs_chart(probs_dict):
    labels = list(probs_dict.keys())
    values = [v*100 for v in probs_dict.values()]
    fig, ax = plt.subplots(figsize=(6,4))
    base_color = "#4F7BFF"
    ax.set_ylim(0,100)
    bars = ax.bar(labels, values, color=base_color, width=0.55,
                  edgecolor="#1E3A8A", alpha=0.9)
    for b,v in zip(bars,values):
        ax.text(b.get_x()+b.get_width()/2, v+2, f"{int(v)}%",
                ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#1E3A8A")
    ax.set_ylabel("Probability (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("HER2 Score", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)



# ======================================================
# CORE INFERENCE
# ======================================================

@torch.no_grad()
def _predict_core(pil, file_name="img.png"):
    pil = pil.convert("RGB")

    # Save original image before resizing for model input
    orig_b64 = to_b64(pil.copy())

    patches, boxes, full_cv = patch_if_needed(pil)

    scores = []
    patch_cvs = []
    patch_probs = []

    # patch eval
    for p in patches:
        cvp=pil_to_cv(p)
        patch_cvs.append(cvp)

        x=tfm(p).unsqueeze(0).to(DEVICE)
        logits=_MODEL(x)
        pr=torch.softmax(logits,1).cpu().numpy()[0]
        patch_probs.append(pr)

        dab=dab_score(cvp)
        txt=texture_score(cvp)
        ta=tissue_area(cvp)

        # rank with model-based weighting
        predicted = int(np.argmax(pr))
        rank = (0.55 * pr[predicted] +
                0.25 * (dab/50000) +
                0.15 * (txt/40) +
                0.05 * (ta/200000))
        scores.append(rank)

    scores=np.array(scores)
    patch_probs=np.array(patch_probs)

    best_idx=int(np.argmax(scores))
    best_probs=patch_probs[best_idx]
    pred_idx=int(np.argmax(best_probs))
    pred_label=IDX_TO_CLASS[pred_idx]
    confidence=float(best_probs[pred_idx])*100

    best_cv = patch_cvs[best_idx]
    box = boxes[best_idx]

    # GradCAM on best patch → overlay on WSI full + per-layer tiles
    x_best = tfm(cv_to_pil(best_cv)).unsqueeze(0).to(DEVICE)

    gradcam_layers = []
    primary_gradcam_b64 = None
    wsi_grad = None

    for lname in TARGET_LAYERS:
        gc = GradCam(_MODEL, lname)
        cam = gc.generate(x_best, pred_idx)

        ph, pw = best_cv.shape[:2]
        cam_r = cv2.resize(cam, (pw, ph))
        cam_u = (cam_r * 255).astype(np.uint8)
        heat = cv2.applyColorMap(cam_u, cv2.COLORMAP_VIRIDIS)
        overlay_small = cv2.addWeighted(best_cv, 0.45, heat, 0.55, 0)
        overlay_small_pil = cv_to_pil(overlay_small)

        gradcam_layers.append({
            "layer_name": lname,
            "gradcam_b64": to_b64(overlay_small_pil)
        })

        if lname == PRIMARY_LAYER:
            wsi_grad_img = make_gradcam_wsi(full_cv, best_cv, box, cam)
            wsi_grad = wsi_grad_img
            primary_gradcam_b64 = to_b64(wsi_grad_img)

    if primary_gradcam_b64 is None and len(gradcam_layers):
        primary_gradcam_b64 = gradcam_layers[-1]["gradcam_b64"]

    wsi_pseudo = make_pseudo_wsi(full_cv, best_cv, box)

    probs_dict={IDX_TO_CLASS[i]:float(best_probs[i]) for i in range(4)}
    chart_img=probs_chart(probs_dict)

# Resize for frontend display (grad-cam)
    gradcam_width, gradcam_height = 420, 180
    orig_resized = pil.copy().resize((gradcam_width, gradcam_height))
    orig_small = pil.resize((gradcam_width, gradcam_height))

    # Resize to original dimensions if width exceeds 1400
    if pil.width > WSI_LONG_EDGE:
        pil = pil.resize((WSI_LONG_EDGE, int(pil.height * WSI_LONG_EDGE / pil.width)))

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": probs_dict,
        "probs_chart_b64": to_b64(chart_img),
        "orig_b64": orig_b64,
        "origResizedB64": to_b64(orig_resized),
        "gradcam_layers": gradcam_layers,
        "primary_gradcam_layer": PRIMARY_LAYER,
        "primary_gradcam_b64": to_b64(wsi_grad),
        "pseudo_b64": to_b64(wsi_pseudo),
        "wsi_box": {
            "x0": int(box[0]), "y0": int(box[1]),
            "x1": int(box[2]), "y1": int(box[3]),
        }
    }
@torch.no_grad()
def predict_with_visuals(pil, file_name="img.png"):
    return _predict_core(pil,file_name)

# ======================================================
# FASTAPI
# ======================================================

app=FastAPI(title="HER2 IHCNet WSI", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

def read_image(b):
    try:
        img=Image.open(io.BytesIO(b))
        img.verify()
        return Image.open(io.BytesIO(b)).convert("RGB")
    except:
        raise HTTPException(status_code=400,detail="Invalid image")

@app.post("/predict")
def predict(file:UploadFile=File(...)):
    b=file.file.read()
    if not b: raise HTTPException(status_code=400,detail="Empty file")
    img=read_image(b)
    return JSONResponse(predict_with_visuals(img,file.filename))
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "ihcnet": _MODEL is not None,
        "psp": _PSP is not None,
    }

# ======================================================
# MODEL LOADING
# ======================================================

_MODEL = None
_PSP = None  # PSPStain generator (BCI_net_G.pth)

try:
    _MODEL = build_and_load_ihcnet(WEIGHTS_IHCNET, DEVICE)
except Exception as e:
    print("[ERR] Load IHCNet:", e)

# ============================================================
# PSPStain Generator Loader
# ============================================================

def build_psp_generator(weight_path):
    class Opt: pass
    opt = Opt()
    opt.weight_norm = "spectral"
    opt.n_downsampling = 2

    norm_layer = get_norm_layer("instance")

    netG = PSPResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
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

    netG.load_state_dict(clean_state, strict=False)
    netG.eval()
    return netG

# ============================================================
# GAN + WSI + IHCNet Pipeline 
# ============================================================

@torch.no_grad()
def _ganpredict_core(pil_he, file_name="img.png"):
    global _PSP

    pil_he = pil_he.convert("RGB")
    orig_b64 = to_b64(pil_he.copy())  

    # ---------------------------
    # Ensure models are loaded
    # ---------------------------
    if _MODEL is None:
        raise HTTPException(500, "IHCNet not loaded")

    if _PSP is None:
        try:
            _PSP = build_psp_generator("BCI_net_G.pth")
        except Exception as e:
            raise HTTPException(500, f"PSPStain not loaded: {e}")

    # ====================================================
    # 1 — Generate IHC from H&E
    # ====================================================
    img_tensor = he_pre(pil_he).unsqueeze(0).to(DEVICE)
    gen_tensor = _PSP(img_tensor)
    gen_tensor = (gen_tensor + 1) / 2
    pil_gen = transforms.ToPILImage()(gen_tensor.squeeze(0).cpu())

    # ====================================================
    # 2 — Treat Generated IHC as WSI → Patch it
    # ====================================================
    patches, boxes, full_cv = patch_if_needed(pil_gen)

    scores = []
    patch_cvs = []
    patch_probs = []

    for p in patches:
        cvp = pil_to_cv(p)
        patch_cvs.append(cvp)

        x = tfm(p).unsqueeze(0).to(DEVICE)
        logits = _MODEL(x)
        pr = torch.softmax(logits, 1).cpu().numpy()[0]
        patch_probs.append(pr)

        dab = dab_score(cvp)
        txt = texture_score(cvp)
        ta = tissue_area(cvp)

        predicted = int(np.argmax(pr))
        rank = (0.55 * pr[predicted] +
                0.25 * (dab/50000) +
                0.15 * (txt/40) +
                0.05 * (ta/200000))
        scores.append(rank)

    scores = np.array(scores)
    patch_probs = np.array(patch_probs)

    best_idx = int(np.argmax(scores))
    best_cv = patch_cvs[best_idx]
    best_probs = patch_probs[best_idx]
    box = boxes[best_idx]

    pred_idx = int(np.argmax(best_probs))
    pred_label = IDX_TO_CLASS[pred_idx]
    confidence = float(best_probs[pred_idx]) * 100.0

    probs_dict = {IDX_TO_CLASS[i]: float(best_probs[i]) for i in range(NUM_CLASSES)}
    chart_img = probs_chart(probs_dict)

    # ====================================================
    # 3 — Grad-CAM on Generated Patch
    # ====================================================
    ihc_tensor = tfm(cv_to_pil(best_cv)).unsqueeze(0).to(DEVICE)

    gradcam_layers = []
    wsi_grad = None

    def make_gradcam_wsi_smallbar(full_cv, patch_cv, box, cam):
        x0, y0, x1, y1 = map(int, box)
        ph, pw = patch_cv.shape[:2]

        cam_r = cv2.resize(cam, (pw, ph))
        cam_u = (cam_r * 255).astype(np.uint8)
        heat = cv2.applyColorMap(cam_u, cv2.COLORMAP_VIRIDIS)
        overlay = cv2.addWeighted(patch_cv, 0.45, heat, 0.55, 0)

        final = full_cv.copy()
        final[y0:y1, x0:x1] = overlay

        rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        h = rgb.shape[0]

        grad = np.linspace(0, 1, h).reshape(h, 1)
        bar = (cm.viridis_r(grad)[:, :, :3] * 255).astype(np.uint8)
        bar = np.repeat(bar, 12, axis=1)  # thinner bar

        spacer = np.ones((h, 3, 3), dtype=np.uint8) * 230
        pad = np.ones((h, 8, 3), dtype=np.uint8) * 255

        text_panel = np.ones((h, 90, 3), dtype=np.uint8) * 255
        cv2.putText(text_panel, "High", (2, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "Medium", (2, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "Low", (2, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        out = np.hstack((rgb, spacer, bar, pad, text_panel))
        return Image.fromarray(out)

    def make_pseudo_wsi_smallbar(full_cv, patch_cv, box):
        x0, y0, x1, y1 = map(int, box)

        g = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2GRAY)
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        pseudo = cv2.applyColorMap((255 - g).astype(np.uint8), cv2.COLORMAP_JET)

        final = full_cv.copy()
        final[y0:y1, x0:x1] = pseudo

        rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
        h = rgb.shape[0]

        grad = np.linspace(0, 1, h).reshape(h, 1)
        bar = (cm.jet_r(grad)[:, :, :3] * 255).astype(np.uint8)
        bar = np.repeat(bar, 12, axis=1)  # thinner bar

        spacer = np.ones((h, 3, 3), dtype=np.uint8) * 230
        pad = np.ones((h, 8, 3), dtype=np.uint8) * 255

        text_panel = np.ones((h, 90, 3), dtype=np.uint8) * 255
        cv2.putText(text_panel, "High", (2, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "Medium", (2, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(text_panel, "Low", (2, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        out = np.hstack((rgb, spacer, bar, pad, text_panel))
        return Image.fromarray(out)

    for lname in TARGET_LAYERS:
        gc = GradCam(_MODEL, lname)
        cam = gc.generate(ihc_tensor, pred_idx)

        # small tile
        ph, pw = best_cv.shape[:2]
        cam_r = cv2.resize(cam, (pw, ph))
        cam_u = (cam_r * 255).astype(np.uint8)
        heat = cv2.applyColorMap(cam_u, cv2.COLORMAP_VIRIDIS)
        overlay = cv2.addWeighted(best_cv, 0.45, heat, 0.55, 0)
        overlay_pil = cv_to_pil(overlay)

        gradcam_layers.append({
            "layer_name": lname,
            "gradcam_b64": to_b64(overlay_pil)
        })

        # WSI + Color bar
        if lname == PRIMARY_LAYER:
            wsi_grad_img = make_gradcam_wsi_smallbar(full_cv, best_cv, box, cam)
            wsi_grad = wsi_grad_img

    primary_gradcam_b64 = to_b64(wsi_grad) if wsi_grad else gradcam_layers[-1]["gradcam_b64"]

    # ====================================================
    # 4 — Pseudo Color WSI on Generated IHC
    # ====================================================
    wsi_pseudo = make_pseudo_wsi_smallbar(full_cv, best_cv, box)

    gradcam_width, gradcam_height = 420, 180
    orig_resized = pil_he.copy().resize((gradcam_width, gradcam_height))

    # ====================================================
    # 5 — Return Final JSON
    # ====================================================
    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": probs_dict,
        "probs_chart_b64": to_b64(chart_img),

        "orig_b64": orig_b64,                # H&E original
        "generated_b64": to_b64(pil_gen),    # Generated IHC

        "origResizedB64": to_b64(orig_resized),
        "gradcam_layers": gradcam_layers,
        "primary_gradcam_layer": PRIMARY_LAYER,
        "primary_gradcam_b64": primary_gradcam_b64,
        "pseudo_b64": to_b64(wsi_pseudo),
        "wsi_box": {
            "x0": int(box[0]),
            "y0": int(box[1]),
            "x1": int(box[2]),
            "y1": int(box[3]),
        }
    }

# ============================================================
# FastAPI Endpoint
# ============================================================

@app.post("/GANpredict")
def gan_predict(file: UploadFile = File(...)):
    b = file.file.read()
    if not b:
        raise HTTPException(status_code=400, detail="Empty file")
    pil = read_image(b)
    return JSONResponse(_ganpredict_core(pil, file.filename))
