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

import os, cv2, json, time, random, shutil, gc, sys, multiprocessing
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
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
NUM_TEST        = 50        # per class → 100 total held-out test videos
NUM_FOLDS       = 1         # 1 = single training run (recommended)
                            # 3 = full K-fold (3x longer, use to verify results)

# Model
N_FRAMES        = 16
IMG_SIZE        = 224
HIDDEN_SIZE     = 256
NUM_LAYERS      = 2
DROPOUT         = 0.5

# Training
BATCH_SIZE      = 4         # low to avoid OOM on CPU
EPOCHS          = 30        # total epochs per fold (across all sessions)
EPOCHS_PER_SESSION = 30     # set to EPOCHS value to run full fold in one session
                            # early stopping will cut it short automatically
                            # set lower (e.g. 5) if you need to pause between sessions
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

# ══════════════════════════════════════════════════════════════
#  CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════
def validate_config():
    """Catches common config mistakes before wasting training time."""
    assert BATCH_SIZE >= 1,         "BATCH_SIZE must be >= 1"
    assert ACCUMULATION_STEPS >= 1, "ACCUMULATION_STEPS must be >= 1"
    assert N_FRAMES <= 32,          "N_FRAMES > 32 will likely cause OOM"
    assert EPOCHS >= 1,             "EPOCHS must be >= 1"
    assert NUM_TEST >= 5,           "NUM_TEST too small for reliable evaluation"
    assert 0.0 <= LABEL_SMOOTHING < 0.5, "LABEL_SMOOTHING should be between 0 and 0.5"

    if DEVICE.type == "cpu" and BATCH_SIZE > 8:
        warn(f"BATCH_SIZE={BATCH_SIZE} on CPU is very slow — recommend 4 or lower")
    if NUM_WORKERS > 0 and DEVICE.type == "cpu":
        warn(f"NUM_WORKERS={NUM_WORKERS} on CPU can cause OOM — recommend 0")
    if EPOCHS_PER_SESSION and EPOCHS_PER_SESSION < WARMUP_EPOCHS:
        warn(f"EPOCHS_PER_SESSION={EPOCHS_PER_SESSION} is less than "
             f"WARMUP_EPOCHS={WARMUP_EPOCHS} — early stopping will never trigger")

    ok("Config validated.")


# ══════════════════════════════════════════════════════════════
#  TEST SET PREPARATION
# ══════════════════════════════════════════════════════════════
def prepare_test_set():
    """
    Carves out a balanced test set that includes BOTH old and new videos.

    Old naming:  V_1.mp4  / NV_1.mp4
    New naming:  V_new_1.mp4 / NV_new_1.mp4

    Strategy: fill half the test quota from new videos, half from old.
    This ensures the test set represents both data sources.
    If not enough new videos exist, fills remainder from old ones.
    """
    step(1, f"Carving out held-out test set ({NUM_TEST} per class = {NUM_TEST*2} total)")

    for cls in ["Violence", "NonViolence"]:
        src = os.path.join(DATASET_DIR, cls)
        dst = os.path.join(TEST_DIR, cls)
        os.makedirs(dst, exist_ok=True)

        already = [f for f in os.listdir(dst) if f.lower().endswith((".mp4",".avi"))]
        if len(already) >= NUM_TEST:
            ok(f"{cls}: {len(already)} test videos already set aside — skipping.")
            continue

        need = NUM_TEST - len(already)

        # Split available videos by naming pattern
        all_vids = [f for f in os.listdir(src) if f.lower().endswith((".mp4",".avi"))]
        new_vids  = [f for f in all_vids if "_new_" in f.lower()]
        old_vids  = [f for f in all_vids if "_new_" not in f.lower()]

        # Aim for 50/50 split between old and new in the test set
        want_new = min(len(new_vids), need // 2)
        want_old = min(len(old_vids), need - want_new)
        # If not enough of one type, fill with the other
        if want_new + want_old < need:
            extra = need - want_new - want_old
            if len(new_vids) - want_new >= extra:
                want_new += extra
            else:
                want_old += extra

        chosen_new = random.sample(new_vids, want_new) if want_new > 0 else []
        chosen_old = random.sample(old_vids, want_old) if want_old > 0 else []
        chosen     = chosen_new + chosen_old

        for f in chosen:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))

        ok(f"{cls}: moved {len(chosen)} videos → {dst}  "
           f"({len(chosen_new)} new + {len(chosen_old)} old)")


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
#  VIDEO VALIDATION
# ══════════════════════════════════════════════════════════════
def validate_videos(paths, labels):
    """
    Pre-filters corrupted or too-short videos before training starts.
    One upfront cost instead of scattered errors during training.
    """
    valid_paths, valid_labels = [], []
    skipped = 0
    for path, label in zip(paths, labels):
        try:
            cap = cv2.VideoCapture(path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count >= 1:
                valid_paths.append(path)
                valid_labels.append(label)
            else:
                warn(f"Skipping empty video: {os.path.basename(path)}")
                skipped += 1
        except Exception as e:
            warn(f"Skipping corrupted video: {os.path.basename(path)} ({e})")
            skipped += 1
    if skipped:
        warn(f"Excluded {skipped} invalid videos from training.")
    else:
        ok(f"All {len(valid_paths)} videos validated.")
    return valid_paths, valid_labels


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
            if self.augment and total > N_FRAMES:
                # Temporal jitter — slight random offset so model sees
                # different frame positions each epoch (not same 16 frames)
                jitter = random.randint(-2, 2)
                indices = torch.linspace(0, total-1, N_FRAMES).long() + jitter
                indices = torch.clamp(indices, 0, total - 1)
                want = set(indices.tolist())
            else:
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
# Mixed precision scaler — active on GPU only, no-op on CPU
_AMP_SCALER = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None


def run_epoch(model, loader, criterion,
              optimizer=None, train=True,
              fold=0, epoch=0, total_epochs=0, phase="Train"):
    """Single epoch with inline \r progress. No new lines created."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    acc_steps = ACCUMULATION_STEPS if train else 1
    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        for batch_idx, (frames, labels) in enumerate(loader):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            if train and MIXUP_ALPHA > 0:
                frames, labels = mixup_batch(frames, labels)

            use_amp = _AMP_SCALER is not None
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(frames)
                loss   = criterion(logits, labels) / acc_steps

            if train:
                if use_amp:
                    _AMP_SCALER.scale(loss).backward()
                    if (batch_idx + 1) % acc_steps == 0 or                             (batch_idx + 1) == n_batches:
                        _AMP_SCALER.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        _AMP_SCALER.step(optimizer)
                        _AMP_SCALER.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % acc_steps == 0 or                             (batch_idx + 1) == n_batches:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

            total_loss += loss.item() * acc_steps
            hard_labels = (labels > 0.5).float()
            preds    = (torch.sigmoid(logits) > THRESHOLD).float()
            correct += (preds == hard_labels).sum().item()
            total   += labels.size(0)

            # ── Inline progress: overwrites same line with \r ──────
            done     = batch_idx + 1
            pct      = done / n_batches
            loss_now = total_loss / done
            acc_now  = correct / total * 100

            # Get terminal width — truncate line to fit, preventing wrap
            try:
                term_w = os.get_terminal_size().columns
            except OSError:
                term_w = 80

            # Build fixed-width prefix and suffix first
            prefix = f"  F{fold+1} Ep{epoch+1:02d}/{total_epochs} {phase} "
            suffix = f" {done}/{n_batches} loss={loss_now:.4f} acc={acc_now:.1f}%"
            # Bar fills whatever space is left, minimum 8 chars
            bar_w  = max(8, term_w - len(prefix) - len(suffix) - 4)
            filled = int(pct * bar_w)
            bar_str = "█" * filled + "░" * (bar_w - filled)
            line    = f"{prefix}[{bar_str}]{suffix}"
            # Pad to terminal width so previous longer lines are fully erased
            line    = line[:term_w].ljust(term_w)
            sys.stdout.write("\r" + line)
            sys.stdout.flush()

    return total_loss / n_batches, correct / total


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

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        # Phase 2: unfreeze full CNN at epoch 10
        if epoch == 10:
            print(f"\n  Fold {fold+1} — Phase 2: full CNN unfrozen (lr -> 1e-5)")
            model.unfreeze_all()
            for g in optimizer.param_groups: g["lr"] = 1e-5

        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion,
            optimizer, train=True,
            fold=fold, epoch=epoch, total_epochs=EPOCHS, phase="Tr"
        )
        # Val runs on the same line — no \n between train and val
        vl_loss, vl_acc = run_epoch(
            model, val_loader, criterion,
            train=False,
            fold=fold, epoch=epoch, total_epochs=EPOCHS, phase="Val"
        )
        # After val: clear the line fully, then print permanent summary
        try:
            term_w = os.get_terminal_size().columns
        except OSError:
            term_w = 120
        sys.stdout.write("\r" + " " * term_w + "\r")
        sys.stdout.flush()

        # ── Print clean epoch summary (permanent line) ───────────────
        scheduler.step(vl_acc)
        elapsed = fmt_time(time.time() - t0)
        cur_lr  = optimizer.param_groups[0]["lr"]
        is_best = vl_acc > best_val_acc

        if is_best:
            best_val_acc = vl_acc
            no_improve   = 0
            save_ckpt({"model": model.state_dict()}, fold_best_path(fold))

        else:
            no_improve += 1

        flag = " <-- BEST" if is_best else (
            f" (no improve {no_improve}/{ES_PATIENCE})" if no_improve > 0 else ""
        )
        print(
            f"  F{fold+1} Ep{epoch+1:02d}/{EPOCHS} | "
            f"Tr loss={tr_loss:.4f} acc={tr_acc*100:5.1f}% | "
            f"Val loss={vl_loss:.4f} acc={vl_acc*100:5.1f}% | "
            f"lr={cur_lr:.1e} [{elapsed}]{flag}"
        )

        # ── Save checkpoint ──────────────────────────────────────────
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

        # ── RAM safety check ─────────────────────────────────────────
        if HAS_PSUTIL and psutil.virtual_memory().percent > 90:
            print(
                f"  [WARN] RAM critically high: {ram_usage()} — "
                f"forcing gc.collect(). Consider BATCH_SIZE=2, ACCUMULATION_STEPS=2"
            )
            gc.collect()

        # ── Session epoch limit ──────────────────────────────────────
        session_epochs_done = epoch + 1 - start_epoch
        if EPOCHS_PER_SESSION and session_epochs_done >= EPOCHS_PER_SESSION:
            print(
                f"\n  Session limit: {EPOCHS_PER_SESSION} epochs done "
                f"(total {epoch+1}/{EPOCHS}). "
                f"Re-run train.py to continue from epoch {epoch+2}."
            )
            # Save history log before pausing
            try:
                with open(LOG_FILE) as f: log = json.load(f)
            except FileNotFoundError:
                log = []
            log.extend(history)
            with open(LOG_FILE, "w") as f: json.dump(log, f, indent=2)
            return best_val_acc, "session_limit"

        # ── Early stopping (only after warmup) ───────────────────────
        if epoch + 1 > WARMUP_EPOCHS and no_improve >= ES_PATIENCE:
            print(
                f"\n  Early stopping fold {fold+1} "
                f"(no improve {ES_PATIENCE} epochs). "
                f"Best val acc: {best_val_acc*100:.1f}%"
            )
            break

    # ── Fold complete ─────────────────────────────────────────────────────
    # Cleanup happens in finally block below
    fold_state["fold_status"][str(fold)]  = "complete"
    fold_state["fold_results"][str(fold)] = {
        "best_val_acc": round(best_val_acc, 4),
        "best_model":   fold_best_path(fold)
    }
    save_fold_state(fold_state)

    try:
        with open(LOG_FILE) as f: log = json.load(f)
    except FileNotFoundError:
        log = []
    log.extend(history)
    with open(LOG_FILE, "w") as f: json.dump(log, f, indent=2)

    print(f"\n  Fold {fold+1} complete — best val acc: {best_val_acc*100:.1f}%")
    return best_val_acc, "complete"


def _cleanup_fold(train_loader, val_loader, model, optimizer, scheduler):
    """Guaranteed memory cleanup after each fold — runs even on exceptions."""
    try:
        del train_loader, val_loader, model, optimizer, scheduler
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



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
                              train=False,
                              fold=0, epoch=0, total_epochs=1, phase="Test")

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
    validate_config()
    info(f"Device        : {DEVICE}  (pin_memory={PIN_MEMORY})")
    info(f"K-Fold        : K={NUM_FOLDS}")
    info(f"Epochs/fold   : {EPOCHS} total  |  This session: {EPOCHS_PER_SESSION or EPOCHS} epochs then stop")
    info(f"Early stopping: warmup={WARMUP_EPOCHS}, patience={ES_PATIENCE}")
    info(f"Batch size    : {BATCH_SIZE}  |  Frames/clip: {N_FRAMES}  |  Grad accum steps: {ACCUMULATION_STEPS}")
    info(f"Overfitting   : label_smooth={LABEL_SMOOTHING}  "
         f"mixup_alpha={MIXUP_ALPHA}  dropout={DROPOUT}")
    info(f"Checkpoints   : {CHECKPOINT_DIR}/")
    sess = f"{EPOCHS_PER_SESSION} epochs then pause" if EPOCHS_PER_SESSION else "run to completion"
    info(f"Session mode  : {sess}  (change EPOCHS_PER_SESSION in CONFIG)")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── STEP 1: carve out test set ───────────────────────────
    prepare_test_set()

    # ── STEP 2: load / lock data split ──────────────────────
    step(2, "Loading / locking data split")
    all_paths, all_labels = get_or_create_split()

    # ── Validate all videos upfront ──────────────────────────
    info("Validating dataset videos ...")
    all_paths, all_labels = validate_videos(all_paths, all_labels)

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

    # Generate fold splits
    # NUM_FOLDS=1 uses a simple 85/15 train/val split (StratifiedKFold needs >=2)
    # NUM_FOLDS>=2 uses full K-fold cross validation
    if NUM_FOLDS == 1:
        from sklearn.model_selection import train_test_split as _tts
        tr_idx, vl_idx = _tts(
            np.arange(len(paths_arr)), test_size=0.15,
            stratify=labels_arr, random_state=42
        )
        fold_splits = [(tr_idx, vl_idx)]
    else:
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        fold_splits = list(skf.split(paths_arr, labels_arr))

    fold_accs = {}
    for fold, (tr_idx, vl_idx) in enumerate(fold_splits):

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

        best_acc, fold_outcome = train_fold(
            fold, tr_paths, tr_labels,
            vl_paths, vl_labels, fold_state, base_tf
        )
        fold_accs[fold] = best_acc

        # Free all fold memory before starting the next fold
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        info(f"RAM after fold {fold+1} cleanup: {ram_usage()}")

        # Session limit hit inside this fold — stop here, don't start next fold
        if fold_outcome == "session_limit":
            banner("SESSION PAUSED")
            info(f"Completed {EPOCHS_PER_SESSION} epochs this session.")
            info(f"Fold {fold+1} is still IN PROGRESS (not marked complete).")
            info(f"Re-run train.py to continue from epoch where it stopped.")
            info(f"Best val acc so far: {fold_accs[fold]*100:.1f}%")
            break

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
    # Runs regardless of how training ended (early stop, session limit, or full run)
    # Uses the best weights saved so far — even if training isn't 100% complete
    if os.path.exists(fold_best_path(best_fold)):
        evaluate_test_set(best_fold, base_tf)
    else:
        warn("No best model saved yet — test evaluation skipped.")
        warn("This means no epoch completed successfully. Check for errors above.")

    banner("TRAINING COMPLETE")
    ok(f"Use  checkpoints/best_model.pth  in predict.py")
    ok(f"Run:  python predict.py --video path/to/video.mp4")

# ──────────────────────────────────────────────────────────────────────────
#  QUICK TEST MODE
#  Before running the full 30-hour K-fold, verify the pipeline works
#  end-to-end by temporarily overriding config values:
#
#    EPOCHS    = 2      # just 2 epochs per fold
#    NUM_FOLDS = 1      # only 1 fold
#    NUM_TEST  = 5      # smaller test set carve-out
#
#  Steps:
#    1. Edit the CONFIG section above — change those 3 values
#    2. Delete checkpoints/ folder so it starts fresh
#    3. Run:  python train.py
#    4. Verify it completes without OOM or crash
#    5. Restore original values (EPOCHS=30, NUM_FOLDS=3, NUM_TEST=30)
#    6. Delete checkpoints/ again and run for real
#
#  The whole quick test should finish in ~10-20 minutes on CPU.
# ──────────────────────────────────────────────────────────────────────────