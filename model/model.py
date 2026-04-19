import os
import cv2
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import shutil

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR   = "dataset"
TEST_DIR      = "dataset/test"       # test samples will be copied here
N_FRAMES      = 16
IMG_SIZE      = 224
BATCH_SIZE    = 8
EPOCHS        = 30
LR            = 1e-4
HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
DROPOUT       = 0.5
THRESHOLD     = 0.5
NUM_TEST      = 10                   # videos to pull per class for test set
NUM_VAL_SPLIT = 0.15                 # 15% of remaining for validation
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH     = "best_model.pth"


# ─────────────────────────────────────────────
# STEP 1 — SPLIT OUT TEST SET
# Physically moves N videos per class into dataset/test/
# so they are never seen during training/validation
# ─────────────────────────────────────────────
def prepare_test_set(dataset_dir, test_dir, num_test=10):
    for cls in ["Violence", "NonViolence"]:
        src = os.path.join(dataset_dir, cls)
        dst = os.path.join(test_dir, cls)
        os.makedirs(dst, exist_ok=True)

        already_moved = os.listdir(dst)
        if len(already_moved) >= num_test:
            print(f"[TEST SET] {cls}: {len(already_moved)} files already in test dir, skipping.")
            continue

        videos = [f for f in os.listdir(src) if f.endswith((".mp4", ".avi"))]
        chosen = random.sample(videos, min(num_test, len(videos)))
        for f in chosen:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        print(f"[TEST SET] Moved {len(chosen)} '{cls}' videos → {dst}")


# ─────────────────────────────────────────────
# STEP 2 — LOAD PATHS & LABELS
# ─────────────────────────────────────────────
def load_dataset(root_dir):
    paths, labels = [], []
    for label, folder in enumerate(["NonViolence", "Violence"]):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".mp4", ".avi")):
                paths.append(os.path.join(folder_path, fname))
                labels.append(float(label))  # 0 = NonViolence, 1 = Violence
    print(f"[DATASET] Loaded {len(paths)} videos from {root_dir}")
    return paths, labels


# ─────────────────────────────────────────────
# STEP 3 — DATASET CLASS
# ─────────────────────────────────────────────
class ViolenceDataset(Dataset):
    def __init__(self, video_paths, labels, n_frames=N_FRAMES, transform=None, augment=False):
        self.video_paths = video_paths
        self.labels      = labels
        self.n_frames    = n_frames
        self.transform   = transform
        self.augment     = augment

        self.aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]) if augment else None

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = max(total, 1)

        indices = set(torch.linspace(0, total - 1, self.n_frames).long().tolist())
        frames, i = [], 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.augment and self.aug:
                    frame = self.aug(frame)
                elif self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            i += 1

        cap.release()

        # Pad if video too short
        while len(frames) < self.n_frames:
            frames.append(frames[-1] if frames else torch.zeros(3, IMG_SIZE, IMG_SIZE))

        return torch.stack(frames[:self.n_frames])  # [n_frames, C, H, W]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frames = self.extract_frames(self.video_paths[idx])
        return frames, torch.tensor(self.labels[idx], dtype=torch.float32)


# ─────────────────────────────────────────────
# STEP 4 — MODEL
# ─────────────────────────────────────────────
class CNN_BiLSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        # Pretrained EfficientNet-B0 backbone
        self.cnn = timm.create_model("efficientnet_b0", pretrained=True)
        cnn_out  = self.cnn.classifier.in_features  # 1280
        self.cnn.classifier = nn.Identity()

        # Freeze all, unfreeze last 2 blocks
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.blocks[-2:].parameters():
            param.requires_grad = True

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention over time steps
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _attend(self, lstm_out):
        scores  = self.attention(lstm_out).squeeze(-1)       # [B, T]
        weights = torch.softmax(scores, dim=1).unsqueeze(-1) # [B, T, 1]
        return (lstm_out * weights).sum(dim=1)               # [B, H*2]

    def forward(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.cnn(x).view(B, T, -1)      # [B, T, 1280]
        lstm_out, _ = self.lstm(features)           # [B, T, H*2]
        pooled   = self._attend(lstm_out)           # [B, H*2]
        return self.classifier(pooled).squeeze(-1)  # [B]

    def unfreeze_full(self):
        """Call after initial warmup to fine-tune entire CNN"""
        for param in self.cnn.parameters():
            param.requires_grad = True
        print("[MODEL] Full CNN unfrozen for fine-tuning.")


# ─────────────────────────────────────────────
# STEP 5 — TRAIN / EVAL HELPERS
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for frames, labels in tqdm(loader, desc="Train" if train else "Eval ", leave=False):
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)

            logits = model(frames)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            preds   = (torch.sigmoid(logits) > THRESHOLD).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), correct / total


# ─────────────────────────────────────────────
# STEP 6 — INFERENCE (for your pipeline)
# ─────────────────────────────────────────────
def predict_segment(frame_list, model, transform, threshold=THRESHOLD):
    """
    frame_list : list of BGR numpy frames around a suspicious point
    Returns    : (is_violent: bool, confidence: float)
    """
    model.eval()
    frames = []
    for frame in frame_list:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    frames = frames[:N_FRAMES]

    tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # [1, T, C, H, W]
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    return prob > threshold, round(prob, 4)


# ─────────────────────────────────────────────
# STEP 7 — TEST SET EVALUATION
# ─────────────────────────────────────────────
def evaluate_test_set(model, test_dir, transform):
    print("\n" + "="*50)
    print("FINAL TEST SET EVALUATION")
    print("="*50)

    test_paths, test_labels = load_dataset(test_dir)
    if not test_paths:
        print("[TEST] No test videos found.")
        return

    test_ds     = ViolenceDataset(test_paths, test_labels, transform=transform, augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)
    criterion   = nn.BCEWithLogitsLoss()

    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, train=False)

    print(f"Test Loss : {test_loss:.4f}")
    print(f"Test Acc  : {test_acc:.4f}  ({test_acc*100:.1f}%)")

    # Per-video results
    print("\nPer-video predictions:")
    print(f"{'File':<40} {'Label':<12} {'Pred':<12} {'Conf'}")
    print("-"*75)
    for path, label in zip(test_paths, test_labels):
        cap    = cv2.VideoCapture(path)
        frames_raw = []
        while len(frames_raw) < N_FRAMES:
            ret, f = cap.read()
            if not ret: break
            frames_raw.append(f)
        cap.release()

        is_violent, conf = predict_segment(frames_raw, model, transform)
        true_lbl  = "Violence" if label == 1 else "NonViolence"
        pred_lbl  = "Violence" if is_violent else "NonViolence"
        match     = "✓" if true_lbl == pred_lbl else "✗"
        print(f"{os.path.basename(path):<40} {true_lbl:<12} {pred_lbl:<12} {conf}  {match}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Carve out test set (run once, safe to re-run)
    prepare_test_set(DATASET_DIR, TEST_DIR, num_test=NUM_TEST)

    # 2. Load remaining data → train/val split
    paths, labels = load_dataset(DATASET_DIR)
    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels,
        test_size=NUM_VAL_SPLIT,
        stratify=labels,
        random_state=42
    )
    print(f"[SPLIT] Train: {len(X_train)} | Val: {len(X_val)}")

    # 3. Transforms
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = ViolenceDataset(X_train, y_train, transform=base_transform, augment=True)
    val_ds   = ViolenceDataset(X_val,   y_val,   transform=base_transform, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model, optimizer, scheduler
    model     = CNN_BiLSTM().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # 5. Training loop
    best_val_acc = 0.0
    print(f"\n[TRAINING] Device: {DEVICE}")
    print("="*50)

    for epoch in range(EPOCHS):

        # Phase 2: unfreeze full CNN after epoch 10
        if epoch == 10:
            model.unfreeze_full()
            for g in optimizer.param_groups:
                g["lr"] = 1e-5

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, train=False)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Best model saved (val_acc={val_acc:.4f})")

    # 6. Final test set evaluation
    evaluate_test_set(model, TEST_DIR, base_transform)