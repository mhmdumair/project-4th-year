"""
train.py  —  Violence Detector Training
=========================================
Trains EfficientNet-B0 + BiLSTM with K-Fold cross validation.

Resume safety:
  - Re-run at any time after a crash / OOM kill
  - Completed folds are never retrained
  - Interrupted folds resume from their last completed epoch
  - Data split is locked to disk on first run

Usage:
  python train.py
"""

import os, cv2, json, time, random, shutil, gc
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from datetime import timedelta
import numpy as np
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ══════════════════════════════════════════════════════════════
#  CONFIG  — edit these before training
# ══════════════════════════════════════════════════════════════
DATASET_DIR     = "dataset"
TEST_DIR        = "dataset/test"
CHECKPOINT_DIR  = "checkpoints"
LOG_FILE        = "training_log.json"
SPLIT_FILE      = os.path.join(CHECKPOINT_DIR, "data_split.json")
FOLD_STATE_FILE = os.path.join(CHECKPOINT_DIR, "fold_state.json")

# Data
NUM_TEST        = 5        # per class → 60 total held-out test videos
NUM_FOLDS       = 2         # K-fold value

# Model
N_FRAMES        = 16
IMG_SIZE        = 224
HIDDEN_SIZE     = 256
NUM_LAYERS      = 2
DROPOUT         = 0.5

# Training
BATCH_SIZE      = 4         # low to avoid OOM on CPU
EPOCHS          = 2
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
THRESHOLD       = 0.5
NUM_WORKERS     = 0         # 0 = single process, avoids RAM spike from workers

# Memory / gradient accumulation
# Set ACCUMULATION_STEPS > 1 if you get OOM with BATCH_SIZE=4.
# e.g. BATCH_SIZE=2 + ACCUMULATION_STEPS=2 behaves like BATCH_SIZE=4
# but uses half the RAM at peak.
ACCUMULATION_STEPS = 1      # 1 = disabled (normal training)

# Overfitting controls
LABEL_SMOOTHING = 0.1       # softens hard 0/1 labels → reduces overconfidence
MIXUP_ALPHA     = 0.2       # mixup blending strength (0 = disabled)

# Early stopping
WARMUP_EPOCHS   = 5         # don't check patience before this epoch
ES_PATIENCE     = 7         # stop fold if no improvement for this many epochs
LR_PATIENCE     = 3         # halve LR after this many epochs without improvement

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY      = DEVICE.type == "cuda"


# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════
W = 72

def banner(t):   print("\n" + "═"*W + f"\n  {t}\n" + "═"*W)
def section(t):  print(f"\n{'─'*W}\n  {t}\n{'─'*W}")
def info(m):     print(f"  [INFO]  {m}")
def ok(m):       print(f"  [ OK ]  {m}")
def warn(m):     print(f"  [WARN]  {m}")
def step(n, m):  print(f"\n  ── STEP {n}: {m}")

def fmt_time(s): return str(timedelta(seconds=int(s)))

def ram_usage():
    """Returns current RAM usage, e.g. 3.2 / 7.6 GB  (42%)"""
    if not HAS_PSUTIL:
        return "install psutil for RAM info"
    vm    = psutil.virtual_memory()
    used  = vm.used  / 1024**3
    total = vm.total / 1024**3
    return f"{used:.1f} / {total:.1f} GB  ({vm.percent:.0f}%)"

def epoch_row(ep, total, tr_loss, tr_acc, vl_loss, vl_acc,
              elapsed, tag, lr, no_imp):
    flag = ("  ◀ BEST" if tag == "best"
            else f"  (no improve {no_imp}/{ES_PATIENCE})" if no_imp > 0
            else "")
    print(f"  Ep {ep:02d}/{total} │"
          f" Train loss={tr_loss:.4f} acc={tr_acc*100:5.1f}% │"
          f" Val loss={vl_loss:.4f} acc={vl_acc*100:5.1f}% │"
          f" lr={lr:.1e} [{elapsed}]{flag}")


# ══════════════════════════════════════════════════════════════
#  TEST SET PREPARATION
# ══════════════════════════════════════════════════════════════
def prepare_test_set():
    step(1, f"Carving out held-out test set ({NUM_TEST} per class = {NUM_TEST*2} total)")
    for cls in ["Violence", "NonViolence"]:
        src = os.path.join(DATASET_DIR, cls)
        dst = os.path.join(TEST_DIR, cls)
        os.makedirs(dst, exist_ok=True)
        already = [f for f in os.listdir(dst) if f.lower().endswith((".mp4",".avi"))]
        if len(already) >= NUM_TEST:
            ok(f"{cls}: {len(already)} test videos already set aside — skipping.")
            continue
        videos = [f for f in os.listdir(src) if f.lower().endswith((".mp4",".avi"))]
        chosen = random.sample(videos, min(NUM_TEST, len(videos)))
        for f in chosen:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        ok(f"{cls}: moved {len(chosen)} videos → {dst}")


# ══════════════════════════════════════════════════════════════
#  DATA SPLIT  — locked to disk after first run
# ══════════════════════════════════════════════════════════════
def scan_paths(root_dir):
    paths, labels = [], []
    for lbl, folder in enumerate(["NonViolence", "Violence"]):
        fp = os.path.join(root_dir, folder)
        if not os.path.exists(fp): continue
        for f in sorted(os.listdir(fp)):
            if f.lower().endswith((".mp4",".avi")):
                paths.append(os.path.join(fp, f))
                labels.append(lbl)
    return paths, labels


def get_or_create_split():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            s = json.load(f)
        paths  = s["paths"]
        labels = s["labels"]

        # Safety check: if any saved path points inside TEST_DIR,
        # the split was created before test videos were moved out.
        # Delete and recreate so test videos never appear in training.
        leaked = [p for p in paths if TEST_DIR.replace("/", os.sep) in p
                  or TEST_DIR in p]
        if leaked:
            warn(f"Split contains {len(leaked)} test videos -- recreating.")
            os.remove(SPLIT_FILE)
            return get_or_create_split()

        ok(f"Loading locked data split from {SPLIT_FILE}")
        info(f"Total training videos: {len(paths)}")
        return paths, labels

    info("No saved split found -- scanning dataset ...")
    paths, labels = scan_paths(DATASET_DIR)
    with open(SPLIT_FILE, "w") as f:
        json.dump({"paths": paths, "labels": labels}, f, indent=2)
    ok(f"Split locked -> {SPLIT_FILE}  ({len(paths)} videos)")
    return paths, labels

# ══════════════════════════════════════════════════════════════
#  FOLD STATE  — tracks which folds are done/in-progress
# ══════════════════════════════════════════════════════════════
def load_fold_state():
    if os.path.exists(FOLD_STATE_FILE):
        with open(FOLD_STATE_FILE) as f:
            return json.load(f)
    return {"fold_status": {}, "fold_results": {}}


def save_fold_state(state):
    with open(FOLD_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════
class ViolenceDataset(Dataset):
    def __init__(self, paths, labels, transform, augment=False):
        self.paths   = paths
        self.labels  = labels
        self.augment = augment
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]) if augment else transform

    def _frames(self, path):
        try:
            cap   = cv2.VideoCapture(path)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            want  = set(torch.linspace(0, total-1, N_FRAMES).long().tolist())
            out, i = [], 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if i in want:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out.append(self.tf(frame))
                i += 1
            cap.release()
        except Exception as e:
            warn(f"Corrupted video skipped ({os.path.basename(path)}): {e}")
            return torch.zeros(N_FRAMES, 3, IMG_SIZE, IMG_SIZE)
        while len(out) < N_FRAMES:
            out.append(out[-1] if out else torch.zeros(3, IMG_SIZE, IMG_SIZE))
        return torch.stack(out[:N_FRAMES])          # [T, C, H, W]

    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        return self._frames(self.paths[idx]), torch.tensor(float(self.labels[idx]))


# ══════════════════════════════════════════════════════════════
#  MIXUP
# ══════════════════════════════════════════════════════════════
def mixup_batch(frames, labels, alpha=MIXUP_ALPHA):
    """Blends pairs of training samples. Returns mixed frames + soft labels."""
    if alpha <= 0:
        return frames, labels
    lam = np.random.beta(alpha, alpha)
    B   = frames.size(0)
    idx = torch.randperm(B, device=frames.device)
    mixed_frames = lam * frames + (1 - lam) * frames[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_frames, mixed_labels


# ══════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=True)
        feat_dim = self.cnn.classifier.in_features          # 1280
        self.cnn.classifier = nn.Identity()

        # Freeze entire CNN; only last 2 blocks trainable initially
        for p in self.cnn.parameters():          p.requires_grad = False
        for p in self.cnn.blocks[-2:].parameters(): p.requires_grad = True

        self.lstm = nn.LSTM(feat_dim, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, bidirectional=True,
                            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0)
        self.attn = nn.Linear(HIDDEN_SIZE * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _pool(self, h):
        w = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
        return (h * w).sum(dim=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.cnn(x.view(B*T, C, H, W)).view(B, T, -1)
        h, _ = self.lstm(f)
        return self.head(self._pool(h)).squeeze(-1)

    def unfreeze_all(self):
        for p in self.cnn.parameters(): p.requires_grad = True
        ok("Full CNN backbone unfrozen.")


def build_model():
    return CNN_BiLSTM().to(DEVICE)


# Label-smoothed BCE loss
class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Smooth: 0 → smoothing/2,  1 → 1 - smoothing/2
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


# ══════════════════════════════════════════════════════════════
#  CHECKPOINT (per fold)
# ══════════════════════════════════════════════════════════════
def fold_ckpt_path(fold):
    return os.path.join(CHECKPOINT_DIR, f"fold{fold}_last.pth")

def fold_best_path(fold):
    return os.path.join(CHECKPOINT_DIR, f"fold{fold}_best.pth")

def save_ckpt(state, path):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(state, path)

def load_ckpt(path, model, optimizer, scheduler):
    """Returns (start_epoch, best_val_acc, no_improve). Defaults if missing."""
    if not os.path.exists(path):
        return 0, 0.0, 0
    ok(f"  Resuming from {path}")
    c = torch.load(path, map_location=DEVICE)
    model.load_state_dict(c["model"])
    optimizer.load_state_dict(c["optimizer"])
    scheduler.load_state_dict(c["scheduler"])
    return c["epoch"], c["best_val_acc"], c.get("no_improve", 0)


# ══════════════════════════════════════════════════════════════
#  ONE EPOCH
# ══════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion,
              optimizer=None, train=True, label=""):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    tag  = "Train" if train else "Val  "
    pbar = tqdm(loader, desc=f"    {tag} {label}", ncols=80, unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")

    acc_steps = ACCUMULATION_STEPS if train else 1
    if train:
        optimizer.zero_grad()          # zero once before the loop

    with torch.set_grad_enabled(train):
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            if train and MIXUP_ALPHA > 0:
                frames, labels = mixup_batch(frames, labels)

            logits = model(frames)
            # Divide loss by accumulation steps so gradients scale correctly
            loss   = criterion(logits, labels) / acc_steps

            if train:
                loss.backward()
                # Only update weights every acc_steps batches
                if (batch_idx + 1) % acc_steps == 0 or                         (batch_idx + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # Undo the division for logging (show real loss value)
            total_loss += loss.item() * acc_steps
            hard_labels = (labels > 0.5).float()
            preds    = (torch.sigmoid(logits) > THRESHOLD).float()
            correct += (preds == hard_labels).sum().item()
            total   += labels.size(0)
            current_avg_loss = total_loss / max(pbar.n, 1)
            current_acc = (correct / total * 100) if total > 0 else 0
            pbar.set_postfix(loss=f"{current_avg_loss:.4f}",
            acc=f"{current_acc:.1f}%")

    return total_loss / len(loader), correct / total


# ══════════════════════════════════════════════════════════════
#  TRAIN ONE FOLD
# ══════════════════════════════════════════════════════════════
def train_fold(fold, tr_paths, tr_labels, vl_paths, vl_labels,
               fold_state, base_tf):

    section(f"FOLD {fold+1}/{NUM_FOLDS}")
    info(f"Train: {len(tr_paths)}  |  Val: {len(vl_paths)}")

    # Dataloaders
    train_loader = DataLoader(
        ViolenceDataset(tr_paths, tr_labels, transform=base_tf, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        ViolenceDataset(vl_paths, vl_labels, transform=base_tf, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model + optimizer + scheduler
    model     = build_model()
    criterion = SmoothBCELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5,
    patience=LR_PATIENCE
    )

    # Resume if this fold was interrupted
    ckpt_path = fold_ckpt_path(fold)
    start_epoch, best_val_acc, no_improve = load_ckpt(
        ckpt_path, model, optimizer, scheduler
    )
    if start_epoch == 0:
        info(f"Starting fold {fold+1} from scratch.")
    else:
        info(f"Resuming fold {fold+1} from epoch {start_epoch+1}  "
             f"(best val acc: {best_val_acc*100:.1f}%  "
             f"no-improve: {no_improve})")

    history = []

    for epoch in range(start_epoch, EPOCHS):
        ep_label = f"[F{fold+1} {epoch+1:02d}/{EPOCHS}]"
        t0 = time.time()
        info(f"RAM: {ram_usage()}  — Fold {fold+1}  Epoch {epoch+1}/{EPOCHS}")

        # Phase 2: unfreeze full CNN at epoch 10
        if epoch == 10:
            section(f"  Fold {fold+1} — Phase 2: unfreezing full CNN (lr → 1e-5)")
            model.unfreeze_all()
            for g in optimizer.param_groups: g["lr"] = 1e-5

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, train=True,  label=ep_label)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    train=False, label=ep_label)

        # Scheduler steps on val accuracy (mode=max)
        scheduler.step(vl_acc)

        elapsed = fmt_time(time.time() - t0)
        cur_lr  = optimizer.param_groups[0]["lr"]
        is_best = vl_acc > best_val_acc

        if is_best:
            best_val_acc = vl_acc
            no_improve   = 0
            save_ckpt({"model": model.state_dict()}, fold_best_path(fold))
            ok(f"  Fold {fold+1} best saved "
               f"(val acc = {vl_acc*100:.1f}%  epoch {epoch+1})")
        else:
            no_improve += 1

        epoch_row(epoch+1, EPOCHS, tr_loss, tr_acc, vl_loss, vl_acc,
                  elapsed, "best" if is_best else "", cur_lr, no_improve)

        # Always save last checkpoint (enables resume)
        save_ckpt({
            "epoch":        epoch + 1,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "no_improve":   no_improve,
        }, ckpt_path)

        history.append({
            "fold": fold+1, "epoch": epoch+1,
            "tr_loss": round(tr_loss, 4), "tr_acc":  round(tr_acc, 4),
            "vl_loss": round(vl_loss, 4), "vl_acc":  round(vl_acc, 4),
            "lr": cur_lr, "time": elapsed
        })

        # ── Early stopping (only after warmup) ──────────────
        if epoch + 1 > WARMUP_EPOCHS and no_improve >= ES_PATIENCE:
            section(f"  Early stopping — fold {fold+1} "
                    f"(no improve for {ES_PATIENCE} epochs)")
            info(f"  Best val acc: {best_val_acc*100:.1f}%  "
                 f"at epoch {epoch+1 - no_improve}")
            break

    # Mark fold complete
    fold_state["fold_status"][str(fold)]  = "complete"
    fold_state["fold_results"][str(fold)] = {
        "best_val_acc": round(best_val_acc, 4),
        "best_model":   fold_best_path(fold)
    }
    save_fold_state(fold_state)

    # Append history to log
    try:
        with open(LOG_FILE) as f: log = json.load(f)
    except FileNotFoundError:
        log = []
    log.extend(history)
    with open(LOG_FILE, "w") as f: json.dump(log, f, indent=2)

    ok(f"Fold {fold+1} complete — best val acc: {best_val_acc*100:.1f}%")
    return best_val_acc


# ══════════════════════════════════════════════════════════════
#  FINAL TEST EVALUATION
# ══════════════════════════════════════════════════════════════
def evaluate_test_set(best_fold, base_tf):
    banner("FINAL TEST SET EVALUATION")

    test_paths, test_labels = scan_paths(TEST_DIR)
    if not test_paths:
        warn("No test videos found — skipping.")
        return

    info(f"Test videos: {len(test_paths)}")
    info(f"Using weights from fold {best_fold+1}: {fold_best_path(best_fold)}")

    model = build_model()
    model.load_state_dict(
        torch.load(fold_best_path(best_fold), map_location=DEVICE)["model"]
    )
    model.eval()

    # Copy best model to a clean final path
    final_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    torch.save(
        torch.load(fold_best_path(best_fold), map_location=DEVICE),
        final_path
    )
    ok(f"Final model saved → {final_path}  (use this in predict.py)")

    ds     = ViolenceDataset(test_paths, test_labels, transform=base_tf)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    criterion = SmoothBCELoss()
    t_loss, t_acc = run_epoch(model, loader, criterion,
                              train=False, label="[test]")

    section(f"Test  →  Loss: {t_loss:.4f}   Accuracy: {t_acc*100:.1f}%")

    # Per-video table
    col = [42, 13, 13, 7]
    print(f"  {'File':<{col[0]}} {'True':<{col[1]}} "
          f"{'Predicted':<{col[2]}} {'Conf':>{col[3]}}")
    print("  " + "─" * (sum(col)+4))

    # Eval transform (no augmentation)
    eval_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    ok_cnt = 0
    for path, label in zip(test_paths, test_labels):
        # FIX: even sampling across full video (same as ViolenceDataset._frames)
        try:
            cap   = cv2.VideoCapture(path)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            want  = set(torch.linspace(0, total-1, N_FRAMES).long().tolist())
            raw, fi = [], 0
            while True:
                ret, frm = cap.read()
                if not ret: break
                if fi in want: raw.append(frm)
                fi += 1
            cap.release()
        except Exception as e:
            warn(f"Skipping corrupted video {os.path.basename(path)}: {e}")
            continue

        frames = [eval_tf(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) for frm in raw]
        while len(frames) < N_FRAMES:
            frames.append(frames[-1] if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE))
        tensor = torch.stack(frames[:N_FRAMES]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            conf = torch.sigmoid(model(tensor)).item()

        true = "Violence"    if label == 1 else "NonViolence"
        pred = "Violence"    if conf > THRESHOLD else "NonViolence"
        mark = "OK" if true == pred else "WRONG"
        if true == pred: ok_cnt += 1
        print(f"  {os.path.basename(path):<{col[0]}} "
              f"{true:<{col[1]}} {pred:<{col[2]}} "
              f"{conf:>{col[3]}.3f}  {mark}")

    print(f"\n  Result: {ok_cnt}/{len(test_paths)} correct  "
          f"({ok_cnt/len(test_paths)*100:.1f}%)\n")

    result = {
        "test_accuracy":  round(t_acc, 4),
        "test_loss":      round(t_loss, 4),
        "correct":        ok_cnt,
        "total":          len(test_paths),
        "best_fold":      best_fold + 1,
        "final_model":    final_path
    }
    rpath = os.path.join(CHECKPOINT_DIR, "test_results.json")
    with open(rpath, "w") as f: json.dump(result, f, indent=2)
    ok(f"Test results saved → {rpath}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    banner("VIOLENCE DETECTOR — TRAINING")
    info(f"Device        : {DEVICE}  (pin_memory={PIN_MEMORY})")
    info(f"K-Fold        : K={NUM_FOLDS}")
    info(f"Epochs/fold   : {EPOCHS}  (warmup={WARMUP_EPOCHS}, "
         f"patience={ES_PATIENCE})")
    info(f"Batch size    : {BATCH_SIZE}  |  Frames/clip: {N_FRAMES}  |  Grad accum steps: {ACCUMULATION_STEPS}")
    info(f"Overfitting   : label_smooth={LABEL_SMOOTHING}  "
         f"mixup_alpha={MIXUP_ALPHA}  dropout={DROPOUT}")
    info(f"Checkpoints   : {CHECKPOINT_DIR}/")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── STEP 1: carve out test set ───────────────────────────
    prepare_test_set()

    # ── STEP 2: load / lock data split ──────────────────────
    step(2, "Loading / locking data split")
    all_paths, all_labels = get_or_create_split()

    # ── STEP 3: load fold state ──────────────────────────────
    step(3, "Checking fold progress")
    fold_state = load_fold_state()
    completed  = [k for k,v in fold_state["fold_status"].items()
                  if v == "complete"]
    info(f"Completed folds: {[int(k)+1 for k in completed] or 'none'}")

    # ── STEP 4: base transform (no augment) ─────────────────
    base_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ── STEP 5: K-Fold loop ──────────────────────────────────
    banner(f"K-FOLD TRAINING  (K={NUM_FOLDS})")
    info("Re-run at any time to resume — completed folds are skipped.\n")

    labels_arr = np.array(all_labels)
    paths_arr  = np.array(all_paths)
    skf        = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True,
                                 random_state=42)

    fold_accs = {}
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(paths_arr, labels_arr)):

        # Skip already-completed folds
        if str(fold) in fold_state["fold_status"] and \
           fold_state["fold_status"][str(fold)] == "complete":
            res = fold_state["fold_results"][str(fold)]
            ok(f"Fold {fold+1} already complete "
               f"(val acc={res['best_val_acc']*100:.1f}%) — skipping.")
            fold_accs[fold] = res["best_val_acc"]
            continue

        # Mark fold as in-progress
        fold_state["fold_status"][str(fold)] = "in_progress"
        save_fold_state(fold_state)

        tr_paths  = paths_arr[tr_idx].tolist()
        tr_labels = labels_arr[tr_idx].tolist()
        vl_paths  = paths_arr[vl_idx].tolist()
        vl_labels = labels_arr[vl_idx].tolist()

        best_acc = train_fold(fold, tr_paths, tr_labels,
                              vl_paths, vl_labels, fold_state, base_tf)
        fold_accs[fold] = best_acc

        # Free all fold memory before starting the next fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info(f"RAM after fold {fold+1} cleanup: {ram_usage()}")

    # ── STEP 6: pick best fold ───────────────────────────────
    section("K-FOLD RESULTS SUMMARY")
    for fold, acc in sorted(fold_accs.items()):
        marker = ""
        print(f"  Fold {fold+1}: val acc = {acc*100:.2f}%{marker}")

    best_fold = max(fold_accs, key=fold_accs.get)
    ok(f"\n  Best fold: Fold {best_fold+1}  "
       f"(val acc = {fold_accs[best_fold]*100:.2f}%)")
    avg = sum(fold_accs.values()) / len(fold_accs)
    info(f"  Average val acc across folds: {avg*100:.2f}%")
    info(f"  (If folds are consistent, your model is genuinely good)")

    # ── STEP 7: final test evaluation ───────────────────────
    evaluate_test_set(best_fold, base_tf)

    banner("TRAINING COMPLETE")
    ok(f"Use  checkpoints/best_model.pth  in predict.py")
    ok(f"Run:  python predict.py --video path/to/video.mp4")