"""
predict.py  —  Violence Detector Inference
===========================================
Loads trained weights and predicts on new videos or raw frames.
No training happens here — weights are loaded once and reused.

Usage examples:
  # Single video file
  python predict.py --video path/to/video.mp4

  # Multiple videos
  python predict.py --video v1.mp4 v2.mp4 v3.mp4

  # Adjust detection threshold (default 0.5)
  python predict.py --video clip.mp4 --threshold 0.65
"""

import os, cv2, time, argparse, gc
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# ══════════════════════════════════════════════════════════════
#  CONFIG  — must match what was used in train.py
# ══════════════════════════════════════════════════════════════
MODEL_PATH   = "checkpoints/best_model.pth"
N_FRAMES     = 16
IMG_SIZE     = 224
HIDDEN_SIZE  = 256
NUM_LAYERS   = 2
DROPOUT      = 0.5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
#  MODEL DEFINITION  (must be identical to train.py)
# ══════════════════════════════════════════════════════════════
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=False)
        feat_dim = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()

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


# ══════════════════════════════════════════════════════════════
#  LOAD MODEL 
# ══════════════════════════════════════════════════════════════
def load_model(model_path=MODEL_PATH):
    """
    Loads architecture + trained weights from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'.\n"
            f"Run train.py first to generate the weights file."
        )

    model = CNN_BiLSTM().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()                          
    
    del ckpt; gc.collect()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return model, transform


# ══════════════════════════════════════════════════════════════
#  PREDICT FROM FRAMES
# ══════════════════════════════════════════════════════════════
def predict_frames(frame_list, model, transform, threshold=0.5):
    """
    Predict whether a sequence of frames is violent.
    """
    if not frame_list:
        return "NonViolence", 0.0

    frames = _sample_frames(frame_list, N_FRAMES)

    processed = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        processed.append(transform(rgb))

    tensor = torch.stack(processed).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        conf  = torch.sigmoid(logit).item()

    label = "Violence" if conf > threshold else "NonViolence"
    return label, round(conf, 4)


def _sample_frames(frame_list, n):
    """Evenly sample n frames from frame_list, padding if too short."""
    total = len(frame_list)
    if total == 0:
        return [frame_list[0]] * n if frame_list else []
    if total <= n:
        return frame_list + [frame_list[-1]] * (n - total)
    step = total / n
    indices = [int(i * step) for i in range(n)]
    return [frame_list[i] for i in indices]


# ══════════════════════════════════════════════════════════════
#  PREDICT FROM VIDEO FILE
# ══════════════════════════════════════════════════════════════
def predict_video(video_path, model_path=MODEL_PATH, threshold=0.5):
    """
    Predict violence from a video file path. Instantiates and disposes 
    its own model structures to protect memory layouts cleanly.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model, transform = load_model(model_path)

    cap    = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        del frame
    cap.release()

    if not frames:
        print(f"[predict] WARNING: could not read frames from {video_path}")
        del model; gc.collect()
        return "NonViolence", 0.0

    label, confidence = predict_frames(
        frames, model, transform, threshold
    )
    
    del model; gc.collect()
    return label, confidence


# ══════════════════════════════════════════════════════════════
#  DISPLAY HELPER
# ══════════════════════════════════════════════════════════════
def _print_result(path, label, confidence, elapsed_ms):
    bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
    print(f"\n  File       : {os.path.basename(path)}")
    print(f"  Confidence : [{bar}] {confidence:.3f}")
    print(f"  Result     : {label:<11}")
    print(f"  Time       : {elapsed_ms:.0f}ms")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Violence Detector — Inference"
    )
    parser.add_argument(
        "--video", nargs="+", required=True,
        help="Path(s) to video file(s) to analyse"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Detection threshold 0.0–1.0 (default 0.5)."
    )
    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Path to model weights (default: {MODEL_PATH})"
    )
    args = parser.parse_args()

    print(f"\n  Threshold : {args.threshold}")
    print(f"  Videos    : {len(args.video)}")
    print("  " + "─" * 50)

    violent_count = 0
    for video_path in args.video:
        t0 = time.time()
        try:
            label, conf = predict_video(
                video_path, model_path=args.model, threshold=args.threshold
            )
        except FileNotFoundError as e:
            print(f"\n  ERROR: {e}")
            continue

        elapsed_ms = (time.time() - t0) * 1000
        _print_result(video_path, label, conf, elapsed_ms)
        if label == "Violence":
            violent_count += 1

    if len(args.video) > 1:
        print(f"\n  ─────────────────────────────────────────────────")
        print(f"  Summary: {violent_count}/{len(args.video)} videos flagged as violent")