"""
Violence Detector — Training Script
Architecture: Video → EfficientNet-B0 (CNN) → BiLSTM → Attention Pool → Binary label
Batch size: 4  |  Plots saved to: plots/
"""

# ── 1. Imports ─────────────────────────────────────────────────────────────
import os, cv2, json, time, random, shutil, gc, sys
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)
from datetime import timedelta
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print('⚠️  psutil not found — RAM guard disabled.')

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
print('✅ Imports OK')
print(f'📁 Plots → {os.path.abspath(PLOTS_DIR)}')


# ── 2. Configuration ────────────────────────────────────────────────────────
# PATHS
DATASET_DIR     = 'dataset'
TEST_DIR        = 'dataset/test'
CHECKPOINT_DIR  = 'checkpoints'
LOG_FILE        = 'training_log.json'
SPLIT_FILE      = os.path.join(CHECKPOINT_DIR, 'data_split.json')
FOLD_STATE_FILE = os.path.join(CHECKPOINT_DIR, 'fold_state.json')

# DATA
NUM_TEST  = 100
NUM_FOLDS = 1

# MODEL
N_FRAMES    = 16
IMG_SIZE    = 224
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT     = 0.5

# TRAINING  ← batch size 4 (direct, no accumulation trick)
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 1    # effective batch = BATCH_SIZE * 1 = 4
EPOCHS             = 30
EPOCHS_PER_SESSION = 30
LR                 = 1e-4
WEIGHT_DECAY       = 1e-4
THRESHOLD          = 0.5
NUM_WORKERS        = 0

# REGULARISATION
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA     = 0.2

# EARLY STOPPING
WARMUP_EPOCHS = 5
ES_PATIENCE   = 7
LR_PATIENCE   = 3

# DEVICE
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = DEVICE.type == 'cuda'

# RAM GUARD
RAM_WARN_PCT = 85

frame_bytes = BATCH_SIZE * N_FRAMES * 3 * IMG_SIZE * IMG_SIZE * 4
frame_MB    = frame_bytes / 1024**2
print(f'🖥️  Device          : {DEVICE}  (pin_memory={PIN_MEMORY})')
print(f'📦  Batch size      : {BATCH_SIZE}  (accumulation steps={ACCUMULATION_STEPS})')
print(f'🎞️  Frames/clip     : {N_FRAMES}  |  IMG_SIZE: {IMG_SIZE}')
print(f'💾  Peak frame tensor : ~{frame_MB:.0f} MB  (model weights on top)')


# ── 3. Helpers ──────────────────────────────────────────────────────────────
def ram_pct():
    return psutil.virtual_memory().percent if HAS_PSUTIL else 0

def ram_str():
    if not HAS_PSUTIL: return 'n/a'
    vm = psutil.virtual_memory()
    return f'{vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB ({vm.percent:.0f}%)'

def ram_guard():
    if HAS_PSUTIL and psutil.virtual_memory().percent > RAM_WARN_PCT:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def fmt_time(s):
    return str(timedelta(seconds=int(s)))

def close_plots():
    plt.close('all')
    gc.collect()

def validate_config():
    assert BATCH_SIZE >= 1
    assert ACCUMULATION_STEPS >= 1
    assert N_FRAMES <= 32
    assert EPOCHS >= 1
    assert NUM_TEST >= 5
    assert 0.0 <= LABEL_SMOOTHING < 0.5
    print('✅ Config valid')

validate_config()
print(f'🧠 RAM now: {ram_str()}')


# ── 4. Dataset Preparation ──────────────────────────────────────────────────
def prepare_test_set():
    print(f'📦 Carving test set ({NUM_TEST}/class)')
    for cls in ['Violence', 'NonViolence']:
        src = os.path.join(DATASET_DIR, cls)
        dst = os.path.join(TEST_DIR, cls)
        os.makedirs(dst, exist_ok=True)
        already = [f for f in os.listdir(dst) if f.lower().endswith(('.mp4','.avi'))]
        if len(already) >= NUM_TEST:
            print(f'  ✅ {cls}: {len(already)} already set aside')
            continue
        need     = NUM_TEST - len(already)
        all_vids = [f for f in os.listdir(src) if f.lower().endswith(('.mp4','.avi'))]
        new_vids = [f for f in all_vids if '_new_' in f.lower()]
        old_vids = [f for f in all_vids if '_new_' not in f.lower()]
        want_new = min(len(new_vids), need // 2)
        want_old = min(len(old_vids), need - want_new)
        chosen   = (random.sample(new_vids, want_new) if want_new else []) + \
                   (random.sample(old_vids, want_old) if want_old else [])
        for f in chosen:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        print(f'  ✅ {cls}: moved {len(chosen)} videos')

prepare_test_set()


def scan_paths(root_dir):
    paths, labels = [], []
    for lbl, folder in enumerate(['NonViolence', 'Violence']):
        fp = os.path.join(root_dir, folder)
        if not os.path.exists(fp): continue
        for f in sorted(os.listdir(fp)):
            if f.lower().endswith(('.mp4','.avi')):
                paths.append(os.path.join(fp, f))
                labels.append(lbl)
    return paths, labels


def get_or_create_split():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f: s = json.load(f)
        paths, labels = s['paths'], s['labels']
        leaked = [p for p in paths if TEST_DIR in p]
        if leaked:
            print(f'⚠️  Split leaked {len(leaked)} test videos — recreating')
            os.remove(SPLIT_FILE)
            return get_or_create_split()
        print(f'✅ Loaded split: {len(paths)} videos')
        return paths, labels
    paths, labels = scan_paths(DATASET_DIR)
    with open(SPLIT_FILE, 'w') as f:
        json.dump({'paths': paths, 'labels': labels}, f, indent=2)
    print(f'✅ Split created: {len(paths)} videos')
    return paths, labels


def validate_videos(paths, labels):
    valid_p, valid_l, skipped = [], [], 0
    for p, l in zip(paths, labels):
        try:
            cap = cv2.VideoCapture(p)
            ok  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 1
            cap.release()
            if ok: valid_p.append(p); valid_l.append(l)
            else:  skipped += 1
        except: skipped += 1
    print(f'✅ {len(valid_p)} valid  |  ⚠️ {skipped} skipped')
    return valid_p, valid_l


all_paths, all_labels = get_or_create_split()
all_paths, all_labels = validate_videos(all_paths, all_labels)
nv = sum(all_labels)
print(f'Violence: {nv}  NonViolence: {len(all_labels)-nv}')


# ── 5. Dataset Class ────────────────────────────────────────────────────────
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
                jitter  = random.randint(-2, 2)
                indices = torch.linspace(0, total-1, N_FRAMES).long() + jitter
                indices = torch.clamp(indices, 0, total-1)
                want    = set(indices.tolist())
            else:
                want = set(torch.linspace(0, total-1, N_FRAMES).long().tolist())

            out, i = [], 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                if i in want:
                    out.append(self.tf(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    del frame
                else:
                    del frame
                i += 1
            cap.release()
        except Exception:
            return torch.zeros(N_FRAMES, 3, IMG_SIZE, IMG_SIZE)

        while len(out) < N_FRAMES:
            out.append(out[-1] if out else torch.zeros(3, IMG_SIZE, IMG_SIZE))
        return torch.stack(out[:N_FRAMES])

    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        return self._frames(self.paths[idx]), torch.tensor(float(self.labels[idx]))

print('✅ ViolenceDataset defined')


# ── 6. Model ────────────────────────────────────────────────────────────────
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
        feat_dim = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()

        for p in self.cnn.parameters():              p.requires_grad = False
        for p in self.cnn.blocks[-2:].parameters():  p.requires_grad = True

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
        print('  🔓 Full CNN unfrozen')


class SmoothBCELoss(nn.Module):
    def __init__(self, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


def build_model():
    return CNN_BiLSTM().to(DEVICE)


_m = build_model()
total  = sum(p.numel() for p in _m.parameters())
train_ = sum(p.numel() for p in _m.parameters() if p.requires_grad)
print(f'Total params    : {total:,}')
print(f'Trainable Ph.1  : {train_:,}  ({train_/total:.1%})')
del _m; gc.collect()


# ── 7. Mixup ────────────────────────────────────────────────────────────────
def mixup_batch(frames, labels, alpha=MIXUP_ALPHA):
    if alpha <= 0: return frames, labels
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(frames.size(0), device=frames.device)
    frames.mul_(lam).add_(frames[idx] * (1 - lam))
    return frames, lam * labels + (1 - lam) * labels[idx]

print('✅ Mixup defined')


# ── 8. Checkpoint Utilities ─────────────────────────────────────────────────
def fold_ckpt_path(fold): return os.path.join(CHECKPOINT_DIR, f'fold{fold}_last.pth')
def fold_best_path(fold): return os.path.join(CHECKPOINT_DIR, f'fold{fold}_best.pth')

def save_ckpt(state, path):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(state, path)

def load_ckpt(path, model, optimizer, scheduler):
    if not os.path.exists(path): return 0, 0.0, 0
    print(f'  ↩️  Resuming {path}')
    c = torch.load(path, map_location=DEVICE)
    model.load_state_dict(c['model'])
    optimizer.load_state_dict(c['optimizer'])
    scheduler.load_state_dict(c['scheduler'])
    del c; gc.collect()
    meta = torch.load(path, map_location='cpu')
    epoch, best, ni = meta['epoch'], meta['best_val_acc'], meta.get('no_improve', 0)
    del meta; gc.collect()
    return epoch, best, ni

def load_fold_state():
    if os.path.exists(FOLD_STATE_FILE):
        with open(FOLD_STATE_FILE) as f: return json.load(f)
    return {'fold_status': {}, 'fold_results': {}}

def save_fold_state(state):
    with open(FOLD_STATE_FILE, 'w') as f: json.dump(state, f, indent=2)

def append_log(history):
    try:
        with open(LOG_FILE) as f: log = json.load(f)
    except FileNotFoundError:
        log = []
    log.extend(history)
    with open(LOG_FILE, 'w') as f: json.dump(log, f, indent=2)

print('✅ Checkpoint utilities defined')


# ── 9. Training Loop ────────────────────────────────────────────────────────
_AMP_SCALER = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None

def run_epoch(model, loader, criterion,
              optimizer=None, train=True, collect_preds=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_targets = [], []
    n_batches = len(loader)
    acc_steps = ACCUMULATION_STEPS if train else 1

    if train: optimizer.zero_grad()

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
                    if (batch_idx + 1) % acc_steps == 0 or (batch_idx + 1) == n_batches:
                        _AMP_SCALER.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        _AMP_SCALER.step(optimizer)
                        _AMP_SCALER.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % acc_steps == 0 or (batch_idx + 1) == n_batches:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

            loss_val    = loss.item() * acc_steps
            total_loss += loss_val

            with torch.no_grad():
                probs_cpu   = torch.sigmoid(logits).detach().cpu()
                hard_labels = (labels > 0.5).float().cpu()
                preds       = (probs_cpu > THRESHOLD).float()
                correct    += (preds == hard_labels).sum().item()
                total      += labels.size(0)

                if collect_preds:
                    all_probs.extend(probs_cpu.tolist())
                    all_targets.extend(hard_labels.tolist())

            del frames, labels, logits, loss, probs_cpu, hard_labels, preds
            ram_guard()

            print(f'  Batch {batch_idx+1}/{n_batches} '
                  f'loss={total_loss/(batch_idx+1):.4f} '
                  f'acc={correct/max(total,1)*100:.1f}% '
                  f'RAM={ram_str()}', end='\r')

    print(' ' * 100, end='\r')
    return total_loss / n_batches, correct / max(total, 1), all_probs, all_targets

print('✅ run_epoch defined')


# ── 10. Train One Fold ──────────────────────────────────────────────────────
def train_fold(fold, tr_paths, tr_labels, vl_paths, vl_labels, fold_state, base_tf):
    print(f'\n{"-"*60}')
    print(f'  FOLD {fold+1}/{NUM_FOLDS}  Train:{len(tr_paths)}  Val:{len(vl_paths)}')
    print(f'{"-"*60}')

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

    model     = build_model()
    criterion = SmoothBCELoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=LR_PATIENCE
    )

    ckpt_path = fold_ckpt_path(fold)
    start_epoch, best_val_acc, no_improve = load_ckpt(
        ckpt_path, model, optimizer, scheduler
    )

    history      = []
    fold_outcome = 'complete'

    try:
        for epoch in range(start_epoch, EPOCHS):
            t0 = time.time()

            if epoch == 10:
                print(f'  🔓 Phase 2: full CNN unfrozen (lr → 1e-5)')
                model.unfreeze_all()
                for g in optimizer.param_groups: g['lr'] = 1e-5

            tr_loss, tr_acc, _, _ = run_epoch(
                model, train_loader, criterion,
                optimizer, train=True, collect_preds=False
            )
            vl_loss, vl_acc, _, _ = run_epoch(
                model, val_loader, criterion,
                train=False, collect_preds=False
            )

            scheduler.step(vl_acc)
            elapsed = fmt_time(time.time() - t0)
            cur_lr  = optimizer.param_groups[0]['lr']
            is_best = vl_acc > best_val_acc

            if is_best:
                best_val_acc = vl_acc; no_improve = 0
                save_ckpt({'model': model.state_dict()}, fold_best_path(fold))
            else:
                no_improve += 1

            flag = ' ⭐ BEST' if is_best else f' (no impr {no_improve}/{ES_PATIENCE})'
            print(
                f'  Ep{epoch+1:02d}/{EPOCHS} | '
                f'Tr loss={tr_loss:.4f} acc={tr_acc*100:5.1f}% | '
                f'Val loss={vl_loss:.4f} acc={vl_acc*100:5.1f}% | '
                f'lr={cur_lr:.1e} [{elapsed}] RAM={ram_str()}{flag}'
            )

            ckpt = {
                'epoch': epoch+1, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_acc': best_val_acc, 'no_improve': no_improve
            }
            save_ckpt(ckpt, ckpt_path)
            del ckpt; gc.collect()

            history.append({
                'fold': fold+1, 'epoch': epoch+1,
                'tr_loss': round(tr_loss,4), 'tr_acc': round(tr_acc,4),
                'vl_loss': round(vl_loss,4), 'vl_acc': round(vl_acc,4),
                'lr': cur_lr, 'time': elapsed
            })

            if EPOCHS_PER_SESSION and (epoch + 1 - start_epoch) >= EPOCHS_PER_SESSION:
                print(f'  ⏸️  Session limit. Resume from epoch {epoch+2}.')
                fold_outcome = 'session_limit'
                break

            if epoch + 1 > WARMUP_EPOCHS and no_improve >= ES_PATIENCE:
                print(f'  🛑 Early stopping. Best val acc: {best_val_acc*100:.1f}%')
                break

    finally:
        append_log(history)
        del train_loader, val_loader, model, optimizer, scheduler, criterion
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f'  🧹 Fold cleanup done. RAM: {ram_str()}')

    if fold_outcome == 'complete':
        fold_state['fold_status'][str(fold)]  = 'complete'
        fold_state['fold_results'][str(fold)] = {
            'best_val_acc': round(best_val_acc, 4),
            'best_model':   fold_best_path(fold)
        }
        save_fold_state(fold_state)

    return best_val_acc, fold_outcome, history

print('✅ train_fold defined')


# ── 11. Run Training ────────────────────────────────────────────────────────
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

base_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

labels_arr = np.array(all_labels)
paths_arr  = np.array(all_paths)
fold_state = load_fold_state()
fold_accs  = {}

if NUM_FOLDS == 1:
    tr_idx, vl_idx = train_test_split(
        np.arange(len(paths_arr)), test_size=0.15,
        stratify=labels_arr, random_state=42
    )
    fold_splits = [(tr_idx, vl_idx)]
else:
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(skf.split(paths_arr, labels_arr))

for fold, (tr_idx, vl_idx) in enumerate(fold_splits):
    if fold_state['fold_status'].get(str(fold)) == 'complete':
        res = fold_state['fold_results'][str(fold)]
        print(f'✅ Fold {fold+1} done (acc={res["best_val_acc"]*100:.1f}%) — skipping')
        fold_accs[fold] = res['best_val_acc']
        continue

    fold_state['fold_status'][str(fold)] = 'in_progress'
    save_fold_state(fold_state)

    best_acc, fold_outcome, _ = train_fold(
        fold,
        paths_arr[tr_idx].tolist(), labels_arr[tr_idx].tolist(),
        paths_arr[vl_idx].tolist(), labels_arr[vl_idx].tolist(),
        fold_state, base_tf
    )
    fold_accs[fold] = best_acc

    if fold_outcome == 'session_limit':
        print('\n⏸️  Session paused. Re-run to continue.'); break

print('\n' + '='*60)
best_fold = max(fold_accs, key=fold_accs.get)
for f, a in sorted(fold_accs.items()):
    marker = '  ← BEST' if f == best_fold else ''
    print(f'  Fold {f+1}: {a*100:.2f}%{marker}')
print(f'  Avg: {sum(fold_accs.values())/len(fold_accs)*100:.2f}%')


# ── 12. Final Test Set Evaluation ───────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

true_labels = pred_labels = pred_probs = None

if not os.path.exists(fold_best_path(best_fold)):
    print('⚠️  No best model saved — skipping test eval')
else:
    print('\n' + '='*60)
    print('  FINAL TEST SET EVALUATION')
    print('='*60)

    test_paths, test_labels_raw = scan_paths(TEST_DIR)
    if not test_paths:
        print('⚠️  No test videos found')
    else:
        model = build_model()
        ckpt  = torch.load(fold_best_path(best_fold), map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        del ckpt; gc.collect()
        model.eval()

        final_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        torch.save({'model': model.state_dict()}, final_path)
        print(f'  ✅ Final model → {final_path}')

        all_probs_test, all_preds_test, all_true_test = [], [], []

        for path, label in zip(test_paths, test_labels_raw):
            try:
                cap   = cv2.VideoCapture(path)
                total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
                want  = set(torch.linspace(0, total-1, N_FRAMES).long().tolist())
                raw, fi = [], 0
                while True:
                    ret, frm = cap.read()
                    if not ret: break
                    if fi in want:
                        raw.append(eval_tf(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))); del frm
                    else:
                        del frm
                    fi += 1
                cap.release()
            except: continue

            while len(raw) < N_FRAMES:
                raw.append(raw[-1] if raw else torch.zeros(3, IMG_SIZE, IMG_SIZE))
            tensor = torch.stack(raw[:N_FRAMES]).unsqueeze(0).to(DEVICE)
            del raw

            with torch.no_grad():
                conf = torch.sigmoid(model(tensor)).item()
            del tensor

            true = 'Violence' if label == 1 else 'NonViolence'
            pred = 'Violence' if conf > THRESHOLD else 'NonViolence'
            mark = '✅' if true == pred else '❌'
            all_probs_test.append(conf)
            all_preds_test.append(1 if conf > THRESHOLD else 0)
            all_true_test.append(label)
            print(f'  {os.path.basename(path)[:40]:<40} {true:<12} {pred:<12} {conf:.3f} {mark}')
            ram_guard()

        del model; gc.collect()

        ok_cnt = sum(p == t for p, t in zip(all_preds_test, all_true_test))
        print(f'\n  Result: {ok_cnt}/{len(all_true_test)} ({ok_cnt/len(all_true_test)*100:.1f}%)')

        true_labels = all_true_test
        pred_labels = all_preds_test
        pred_probs  = all_probs_test

        with open(os.path.join(CHECKPOINT_DIR, 'test_results.json'), 'w') as f:
            json.dump({'accuracy': round(ok_cnt/len(all_true_test),4),
                       'correct': ok_cnt, 'total': len(all_true_test)}, f, indent=2)


# ── 13. Plots ───────────────────────────────────────────────────────────────
# Load history from disk
fold_history_for_plot = {}
if os.path.exists(LOG_FILE):
    with open(LOG_FILE) as f: log = json.load(f)
    for row in log:
        fk = row['fold'] - 1
        if fk not in fold_history_for_plot: fold_history_for_plot[fk] = []
        fold_history_for_plot[fk].append(row)
    print(f'✅ Loaded history for {len(fold_history_for_plot)} fold(s)')
else:
    print('⚠️  No training_log.json found — training curves unavailable')


# ── 13.1  Training Loss (separate) ──────────────────────────────────────────
if fold_history_for_plot:
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Training & Validation Loss', fontsize=13, fontweight='bold')

    for fi, history in fold_history_for_plot.items():
        ep  = [r['epoch']   for r in history]
        c   = colors[fi % len(colors)]
        lbl = f'Fold {fi+1}'
        ax.plot(ep, [r['tr_loss'] for r in history], '--', color=c, alpha=0.7,
                label=f'{lbl} Train')
        ax.plot(ep, [r['vl_loss'] for r in history], '-',  color=c,
                label=f'{lbl} Val')

    ax.set(xlabel='Epoch', ylabel='Loss')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curve.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print('📊 Saved loss_curve.png')


# ── 13.2  Training Accuracy (separate) ──────────────────────────────────────
if fold_history_for_plot:
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle('Training & Validation Accuracy', fontsize=13, fontweight='bold')

    for fi, history in fold_history_for_plot.items():
        ep  = [r['epoch']   for r in history]
        c   = colors[fi % len(colors)]
        lbl = f'Fold {fi+1}'
        ax.plot(ep, [r['tr_acc']*100 for r in history], '--', color=c, alpha=0.7,
                label=f'{lbl} Train')
        ax.plot(ep, [r['vl_acc']*100 for r in history], '-',  color=c,
                label=f'{lbl} Val')

    ax.set(xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 105))
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_curve.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print('📊 Saved accuracy_curve.png')


# ── 13.3  Combined Training Curves (overview) ────────────────────────────────
if fold_history_for_plot:
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Training Overview — Loss & Accuracy', fontsize=13, fontweight='bold')

    for fi, history in fold_history_for_plot.items():
        ep  = [r['epoch']   for r in history]
        c   = colors[fi % len(colors)]
        lbl = f'Fold {fi+1}'
        axes[0].plot(ep, [r['tr_loss'] for r in history], '--', color=c, alpha=0.6)
        axes[0].plot(ep, [r['vl_loss'] for r in history], '-',  color=c, label=lbl)
        axes[1].plot(ep, [r['tr_acc']*100 for r in history], '--', color=c, alpha=0.6)
        axes[1].plot(ep, [r['vl_acc']*100 for r in history], '-',  color=c, label=lbl)

    axes[0].set(title='Loss (solid=val, dash=train)', xlabel='Epoch', ylabel='Loss')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set(title='Accuracy (solid=val, dash=train)', xlabel='Epoch',
                ylabel='Acc %', ylim=(0, 105))
    axes[1].axhline(50, color='gray', linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print('📊 Saved training_curves.png')
else:
    print('⚠️  Skipped training curves — no history')


# ── 13.4  K-Fold Summary ────────────────────────────────────────────────────
if fold_accs:
    folds_x = [f'Fold {k+1}' for k in sorted(fold_accs)]
    accs_y  = [fold_accs[k]*100 for k in sorted(fold_accs)]
    avg     = np.mean(accs_y)

    fig, ax = plt.subplots(figsize=(max(5, len(folds_x)*2), 5))
    bar_c = ['darkorange' if a == max(accs_y) else 'steelblue' for a in accs_y]
    bars  = ax.bar(folds_x, accs_y, color=bar_c, edgecolor='white')
    ax.axhline(avg, color='red', linestyle='--', label=f'Avg {avg:.1f}%')
    for bar, acc in zip(bars, accs_y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                f'{acc:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.set(ylim=(0, 110), ylabel='Val Accuracy (%)', title='K-Fold Results')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'fold_summary.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print('📊 Saved fold_summary.png')


# ── 13.5  Confusion Matrix ───────────────────────────────────────────────────
if true_labels is not None:
    class_names = ['NonViolence', 'Violence']
    cm = confusion_matrix(true_labels, pred_labels)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Confusion Matrix — Test Set', fontsize=13, fontweight='bold')

    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[0], colorbar=False, cmap='Blues')
    axes[0].set_title('Counts')

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
        ax=axes[1], colorbar=False, cmap='Blues', values_format='.2%')
    axes[1].set_title('Normalised')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print('📊 Saved confusion_matrix.png')
    print(classification_report(true_labels, pred_labels, target_names=class_names))
else:
    print('⚠️  Confusion matrix skipped — no test results')


# ── 13.6  ROC Curve ──────────────────────────────────────────────────────────
if true_labels is not None:
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
    roc_auc  = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ROC Curve & Confidence Distribution', fontsize=13, fontweight='bold')

    # ROC
    axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0,1],[0,1],'k--',lw=1)
    axes[0].plot(fpr[best_idx], tpr[best_idx], 'ro', ms=8,
                 label=f'Best thr={thresholds[best_idx]:.2f}')
    axes[0].set(title='ROC Curve', xlabel='False Positive Rate',
                ylabel='True Positive Rate', xlim=[0,1], ylim=[0,1.02])
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Confidence distribution
    probs_arr = np.array(pred_probs)
    labs_arr  = np.array(true_labels)
    bins = np.linspace(0, 1, 21)
    axes[1].hist(probs_arr[labs_arr==0], bins=bins, alpha=0.6,
                 color='royalblue', label='NonViolence')
    axes[1].hist(probs_arr[labs_arr==1], bins=bins, alpha=0.6,
                 color='tomato',    label='Violence')
    axes[1].axvline(THRESHOLD, color='black', linestyle='--',
                    label=f'Threshold={THRESHOLD}')
    axes[1].set(title='Confidence Distribution', xlabel='Predicted Probability',
                ylabel='Count')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    del probs_arr, labs_arr

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print(f'📊 Saved roc_curve.png  (AUC={roc_auc:.4f})')
    print(f'   Optimal threshold: {thresholds[best_idx]:.3f}')
else:
    print('⚠️  ROC curve skipped — no test results')


# ── 13.7  Precision-Recall Curve ────────────────────────────────────────────
if true_labels is not None:
    precision, recall, pr_thresh = precision_recall_curve(true_labels, pred_probs)
    ap = average_precision_score(true_labels, pred_probs)
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_f1_idx = np.argmax(f1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Precision-Recall Analysis', fontsize=13, fontweight='bold')

    axes[0].plot(recall, precision, color='darkorange', lw=2, label=f'AP={ap:.3f}')
    axes[0].axhline(sum(true_labels)/len(true_labels), color='gray',
                    linestyle='--', label='Baseline')
    axes[0].set(title='PR Curve', xlabel='Recall', ylabel='Precision',
                xlim=[0,1], ylim=[0,1.05])
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(pr_thresh, f1, color='purple', lw=2)
    axes[1].axvline(THRESHOLD, color='black', linestyle='--',
                    label=f'Current thr={THRESHOLD}')
    axes[1].axvline(pr_thresh[best_f1_idx], color='green', linestyle='--',
                    label=f'Best F1 thr={pr_thresh[best_f1_idx]:.2f} ({f1[best_f1_idx]:.3f})')
    axes[1].set(title='F1 vs Threshold', xlabel='Threshold', ylabel='F1')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'precision_recall.png'), dpi=120, bbox_inches='tight')
    close_plots()
    print(f'📊 Saved precision_recall.png  (AP={ap:.4f})')
else:
    print('⚠️  PR curve skipped — no test results')


# ── 14. Inference ───────────────────────────────────────────────────────────
def predict_video(video_path, model_path='checkpoints/best_model.pth'):
    assert os.path.exists(model_path), f'Model not found: {model_path}'
    assert os.path.exists(video_path), f'Video not found: {video_path}'

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    model = build_model()
    ckpt  = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model'])
    del ckpt; gc.collect()
    model.eval()

    cap   = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    want  = set(torch.linspace(0, total-1, N_FRAMES).long().tolist())
    raw, fi = [], 0
    while True:
        ret, frm = cap.read()
        if not ret: break
        if fi in want:
            raw.append(tf(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))); del frm
        else:
            del frm
        fi += 1
    cap.release()

    while len(raw) < N_FRAMES:
        raw.append(raw[-1] if raw else torch.zeros(3, IMG_SIZE, IMG_SIZE))
    tensor = torch.stack(raw[:N_FRAMES]).unsqueeze(0).to(DEVICE)
    del raw

    with torch.no_grad():
        conf = torch.sigmoid(model(tensor)).item()
    del tensor, model; gc.collect()

    label = 'Violence' if conf > THRESHOLD else 'NonViolence'
    print(f'  📹 {os.path.basename(video_path)}')
    print(f'  🏷️  {label}  ({conf:.3f})')
    return label, conf

print('✅ predict_video() ready')


# ── 15. Output Summary ──────────────────────────────────────────────────────
print('\n' + '='*60)
print('  OUTPUTS')
print('='*60)
items = [
    ('plots/loss_curve.png',           'Training & validation loss (separate)'),
    ('plots/accuracy_curve.png',       'Training & validation accuracy (separate)'),
    ('plots/training_curves.png',      'Loss & accuracy overview'),
    ('plots/fold_summary.png',         'K-fold bar chart'),
    ('plots/confusion_matrix.png',     'Confusion matrix (counts + normalised)'),
    ('plots/roc_curve.png',            'ROC + confidence histogram'),
    ('plots/precision_recall.png',     'PR curve + F1 vs threshold'),
    ('checkpoints/best_model.pth',     'Final model weights'),
    ('checkpoints/test_results.json',  'Test accuracy & counts'),
    ('training_log.json',              'Epoch metrics (all folds)'),
]
for path, desc in items:
    mark = '✅' if os.path.exists(path) else '⏳'
    print(f'  {mark} {path:<45} {desc}')
print(f'\n🧠 Final RAM: {ram_str()}')
print('\n  🎉 Done!  Use predict_video() for new videos.')
