"""
MDGTnet Training Script - Adapted for Houston13/18 .mat Data
Bypasses the original .npy preprocessing pipeline and loads
data directly from the Houston .mat files used by our framework.

Usage:
    python train_mdgtnet_houston.py --seed 678
"""

import argparse
import gc
import json
import os
import random
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from networks.MDGTnet import MDGTnet
from loss import SimiValue, DiffLoss, SDPloss
from utils.cls_weight_calculation import weight_calc_HSI
from utils.normHSI import normHSI_smp_s
from utils.lr_adjust import lr_adj

# ─────────────────────── CLI ───────────────────────
parser = argparse.ArgumentParser(description='MDGTnet for Houston DA')
parser.add_argument('--seed',        type=int, default=678)
parser.add_argument('--source_name', type=str, default='Houston13')
parser.add_argument('--target_name', type=str, default='Houston18')
parser.add_argument('--data_dir',    type=str, default='../../Houston/')
parser.add_argument('--epochs',      type=int, default=50)
parser.add_argument('--batch_size',  type=int, default=512)
parser.add_argument('--lr',          type=float, default=0.006)
parser.add_argument('--patch_size',  type=int, default=3)
parser.add_argument('--gpu',         type=str, default='0')
args = parser.parse_args()

# ─────────────────────── Seed ───────────────────────
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

os.makedirs('./logs', exist_ok=True)
os.makedirs('./models/MDGTnet_H1318', exist_ok=True)
os.makedirs('../../ablation_results', exist_ok=True)

NUM_CLASSES = 7   # 7 shared classes between Houston13 and Houston18
SLICE_SIZE  = args.patch_size

# ─────────────────────── Data Loading ───────────────────────
def load_mat_h5py(img_path, lbl_path):
    """Load image and label from separate Houston v7.3 .mat files via h5py."""
    with h5py.File(img_path, 'r') as f:
        img = np.array(f['ori_data']).astype(np.float32)  # [C, W, H] in h5py
        # h5py transposes MATLAB arrays → restore spatial dims: [C,W,H] → [H,W,C]
        img = img.transpose(2, 1, 0)
    with h5py.File(lbl_path, 'r') as f:
        lbl = np.array(f['map']).astype(np.int64)         # [W, H] in h5py → [H, W]
        if lbl.shape[0] < lbl.shape[1]:
            lbl = lbl.T
    return img, lbl

def normalize_image(img):
    """Per-band min-max normalization → [0, 1]."""
    c = img.shape[0]
    out = np.zeros_like(img)
    for i in range(c):
        mn, mx = img[i].min(), img[i].max()
        if mx > mn:
            out[i] = (img[i] - mn) / (mx - mn)
        else:
            out[i] = 0.0
    return out

def extract_patches(img, label, patch_size):
    """
    Extract all labeled patches from image.
    img   : [C, H, W]  float32
    label : [H, W]     int64  (0 = background, 1..7 = classes)
    Returns:
        patches : [N, C, P, P]  float32
        labels  : [N]           int64  (1-indexed, 0 filtered out)
    """
    C, H, W = img.shape
    pad = patch_size // 2
    img_pad = np.pad(img, ((0,0),(pad,pad),(pad,pad)), mode='reflect')

    patches, labels = [], []
    for h in range(H):
        for w in range(W):
            cls = label[h, w]
            if cls == 0:
                continue
            patch = img_pad[:, h:h+patch_size, w:w+patch_size]
            patches.append(patch)
            labels.append(cls)

    patches = np.stack(patches, axis=0).astype(np.float32)  # [N, C, P, P]
    labels  = np.array(labels, dtype=np.int64)               # [N]
    return patches, labels

def one_hot(labels, num_classes):
    """Convert 1-indexed labels [N] to one-hot [N, C]."""
    N = len(labels)
    oh = np.zeros((N, num_classes), dtype=np.float32)
    for i, lbl in enumerate(labels):
        if 1 <= lbl <= num_classes:
            oh[i, lbl - 1] = 1.0
    return oh


# ─────────────────────── Load Houston data ───────────────────────
src_name = args.source_name
tgt_name = args.target_name
data_dir = args.data_dir

print(f"Loading {src_name} (source)...")
src_img_raw, src_lbl7 = load_mat_h5py(
    os.path.join(data_dir, f'{src_name}.mat'),
    os.path.join(data_dir, f'{src_name}_7gt.mat')
)

print(f"Loading {tgt_name} (target)...")
tgt_img_raw, tgt_lbl7 = load_mat_h5py(
    os.path.join(data_dir, f'{tgt_name}.mat'),
    os.path.join(data_dir, f'{tgt_name}_7gt.mat')
)

# Data is now [H,W,C] for images, [H,W] for labels
# Rearrange to [C,H,W] for processing
def to_chw(arr):
    """Convert [H,W,C] → [C,H,W]"""
    return arr.transpose(2, 0, 1)

src_img_raw = to_chw(src_img_raw)
tgt_img_raw = to_chw(tgt_img_raw)

# Ensure labels are [H, W] to match image [C, H, W]
_, H_src, W_src = src_img_raw.shape
if src_lbl7.shape != (H_src, W_src):
    src_lbl7 = src_lbl7.T
_, H_tgt, W_tgt = tgt_img_raw.shape
if tgt_lbl7.shape != (H_tgt, W_tgt):
    tgt_lbl7 = tgt_lbl7.T

print(f"Source image: {src_img_raw.shape}, label: {src_lbl7.shape}")
print(f"Target image: {tgt_img_raw.shape}, label: {tgt_lbl7.shape}")

# Normalize
src_img = normalize_image(src_img_raw)
tgt_img = normalize_image(tgt_img_raw)
del src_img_raw, tgt_img_raw; gc.collect()

# Align spectral dims: MDGTnet expects same C for both domains
# Pad the smaller one to match the larger
C_src, C_tgt = src_img.shape[0], tgt_img.shape[0]
C_max = max(C_src, C_tgt)
def pad_channels(img, C_out):
    C, H, W = img.shape
    if C == C_out:
        return img
    out = np.zeros((C_out, H, W), dtype=np.float32)
    out[:C] = img
    return out

src_img = pad_channels(src_img, C_max)
tgt_img = pad_channels(tgt_img, C_max)
IN_CHANNELS = C_max
print(f"Aligned spectral channels: {IN_CHANNELS}")

# Extract patches
print("Extracting source patches...")
src_patches, src_labels = extract_patches(src_img, src_lbl7, SLICE_SIZE)
print("Extracting target patches...")
tgt_patches, tgt_labels = extract_patches(tgt_img, tgt_lbl7, SLICE_SIZE)

src_oh = one_hot(src_labels, NUM_CLASSES)
tgt_oh = one_hot(tgt_labels, NUM_CLASSES)

print(f"Source: {src_patches.shape[0]} labeled patches")
print(f"Target: {tgt_patches.shape[0]} labeled patches")

# ─────────────────────── Dataset ───────────────────────
class HoustonPairDataset(Dataset):
    """
    Yields matched/mismatched (src, tgt) patch pairs for MDGTnet training.
    25% same-class pairs, 75% different-class pairs (following original design).
    """
    def __init__(self, src_patches, src_oh, tgt_patches, tgt_oh, length=50000):
        self.src_p  = torch.from_numpy(src_patches).float()
        self.src_oh = torch.from_numpy(src_oh).float()
        self.tgt_p  = torch.from_numpy(tgt_patches).float()
        self.tgt_oh = torch.from_numpy(tgt_oh).float()
        self.length = length
        self.N_src  = len(src_patches)
        self.N_tgt  = len(tgt_patches)

        # Build per-class index lists for fast sampling
        self.src_by_cls = {}
        self.tgt_by_cls = {}
        for c in range(NUM_CLASSES):
            self.src_by_cls[c] = (src_oh[:, c] == 1).nonzero()[0]
            self.tgt_by_cls[c] = (tgt_oh[:, c] == 1).nonzero()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        # Pick random source sample
        i_s = random.randint(0, self.N_src - 1)
        cls_s = int(self.src_oh[i_s].argmax().item())

        if random.random() < 0.25 and len(self.tgt_by_cls.get(cls_s, [])) > 0:
            # Same class pair
            pool = self.tgt_by_cls[cls_s]
            i_t = int(pool[random.randint(0, len(pool) - 1)])
        else:
            # Different class pair
            other_cls = random.choice([c for c in range(NUM_CLASSES) if c != cls_s and len(self.tgt_by_cls.get(c, [])) > 0])
            pool = self.tgt_by_cls[other_cls]
            i_t = int(pool[random.randint(0, len(pool) - 1)])

        return [self.src_p[i_s], self.src_p[i_s], self.src_oh[i_s],
                self.tgt_p[i_t], self.tgt_p[i_t], self.tgt_oh[i_t]]


# ─────────────────────── Model ───────────────────────
# MDGTnet model params - adapted for our channel count and 7 classes
out_ch     = [256, 384, 256, 256, 256, 256, 128, 64]
spec_range = [0, IN_CHANNELS]
padding    = 0
class_num  = NUM_CLASSES

model = MDGTnet(
    in_ch=IN_CHANNELS,
    out_ch=out_ch,
    padding=padding,
    slice_size=SLICE_SIZE,
    spec_range=spec_range,
    class_num=class_num
).to(device)

# ─────────────────────── Dataset & Loader ───────────────────────
train_set    = HoustonPairDataset(src_patches, src_oh, tgt_patches, tgt_oh, length=60000)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

# ─────────────────────── Loss & training ───────────────────────
# Class weights from label distribution
all_labels = np.concatenate([src_labels, tgt_labels])
weight_cls = weight_calc_HSI(all_labels, cls_id=list(range(1, NUM_CLASSES + 1)))

loss_classify = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weight_cls.to(device))
simi_cal  = SimiValue()
loss_diff = DiffLoss()
loss_prog = SDPloss()
w = [1, 3, 10]

learning_rate = args.lr
num_epoch     = args.epochs
train_loss_list = []

print(f"\nStarting MDGTnet training ({src_name} → {tgt_name}), {num_epoch} epochs...")
t_start = time.time()

for epoch in range(num_epoch):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    loop = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f'Epoch [{epoch+1}/{num_epoch}]')
    for i, data in loop:
        optimizer, learning_rate = lr_adj(epoch * len(train_loader) + i + 1,
                                          learning_rate, model)
        optimizer.zero_grad()

        y_out_1, s1_1, s1_2, s1_3, s1_4 = model(data[1].to(device), data[0].to(device))
        y_out_2, s2_1, s2_2, s2_3, s2_4 = model(data[4].to(device), data[3].to(device))

        cons_flag = torch.full([data[0].shape[0]], -1).to(device)
        for b in range(data[0].shape[0]):
            if data[2][b].equal(data[5][b]):
                cons_flag[b] = 1

        bce_w1 = torch.full(data[2].size(), 1, device=device)
        bce_w1[data[2] > 0.5] = 2
        bce_w2 = torch.full(data[5].size(), 1, device=device)
        bce_w2[data[5] > 0.5] = 2

        loss1 = (torch.mean(bce_w1 * loss_classify(y_out_1, data[2].float().to(device))) +
                 torch.mean(bce_w2 * loss_classify(y_out_2, data[5].float().to(device))))

        sims = [simi_cal(s1_1, s2_1), simi_cal(s1_2, s2_2),
                simi_cal(s1_3, s2_3), simi_cal(s1_4, s2_4)]
        loss2 = sum(loss_diff(s, cons_flag) for s in sims)
        loss3 = loss_prog(sims, cons_flag)

        batch_loss = w[0]*loss1 + w[1]*loss2 + w[2]*loss3
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += batch_loss.item()

        with torch.no_grad():
            oa_1 = (y_out_1.argmax(1) == data[2].argmax(1).to(device)).float().mean().item()
            oa_2 = (y_out_2.argmax(1) == data[5].argmax(1).to(device)).float().mean().item()

        loop.set_postfix(loss=f'{batch_loss.item():.4f}', oa_src=f'{oa_1:.3f}', oa_tgt=f'{oa_2:.3f}')

    train_loss_list.append(train_loss)
    os.makedirs('./models/MDGTnet_H1318', exist_ok=True)
    torch.save(model.state_dict(), f'./models/MDGTnet_H1318/model{epoch}.pth')
    print(f'[{epoch+1:03d}/{num_epoch}] {time.time()-epoch_start:.1f}s  Loss: {train_loss:.4f}')

print(f"\nTraining done in {time.time()-t_start:.1f}s")

# ─────────────────────── Evaluation on Target ───────────────────────
print("\nEvaluating on target domain...")
model.eval()

class EvalDataset(Dataset):
    def __init__(self, patches):
        self.data = torch.from_numpy(patches).float()
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

eval_loader = DataLoader(EvalDataset(tgt_patches), batch_size=256, shuffle=False)

all_preds, all_gts = [], []
with torch.no_grad():
    for batch in eval_loader:
        out, *_ = model(batch.to(device), batch.to(device))
        preds = out.argmax(dim=1).cpu().numpy()   # 0-indexed
        all_preds.extend(preds.tolist())

all_gts = (tgt_labels - 1).tolist()   # convert to 0-indexed

all_preds = np.array(all_preds)
all_gts   = np.array(all_gts)

# Per-class recall (TPR)
tpr = []
for c in range(NUM_CLASSES):
    mask = all_gts == c
    if mask.sum() == 0:
        tpr.append(0.0)
    else:
        tpr.append((all_preds[mask] == c).sum() / mask.sum() * 100)

OA    = (all_preds == all_gts).mean() * 100
AA    = float(np.mean(tpr))
Kappa = cohen_kappa_score(all_gts, all_preds) * 100

print(f"\n{'='*45}")
print(classification_report(all_gts, all_preds, digits=4))
print(f"OA: {OA:.4f}")
print(f"AA: {AA:.4f}")
print(f"Kappa: {Kappa:.4f}")

# ─────────────────────── Save JSON ───────────────────────
res_dict = {
    'OA':      round(OA, 4),
    'AA':      round(AA, 4),
    'Kappa':   round(Kappa, 4),
    'classes': {str(c+1): round(tpr[c], 4) for c in range(NUM_CLASSES)}
}
json_path = os.path.join('../../ablation_results', f'MDGTnet_results_seed_{seed}.json')
with open(json_path, 'w') as f:
    json.dump(res_dict, f, indent=4)
print(f"Results saved to {json_path}")
