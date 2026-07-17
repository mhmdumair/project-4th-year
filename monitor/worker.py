import subprocess
import json
import numpy as np
import os
import sys
import time
import hashlib
import tempfile
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ATTACKED_DIR        = "dataset"
GROUND_TRUTH_JSON   = "attack.json"
WORKER_DATA_DIR     = "worker_data"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE  = "ffmpeg"
FFPROBE_EXE = "ffprobe"

CHUNK_DURATION_SEC  = 5.0
NUM_WORKERS         = 10

# --- Audio Constraints ---
AUDIO_SAMPLE_RATE       = 44100
AUDIO_MICRO_WINDOW_SEC  = 0.5
FLUX_SAMPLE_RATE        = 22050
FLUX_WINDOW_SEC         = 0.05

def progress_bar(current, total, width=40, extra=""):
    if total <= 0: return
    frac = min(current / total, 1.0)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%  {extra}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

# ==============================================================================
# THE CHUNKER (Strict GOP segmentation, preserving real timestamps)
# ==============================================================================

def chunk_video(video_path: str, chunk_dir: str) -> list:
    """
    Uses the segment muxer to perfectly split the video at GOP boundaries.
    -reset_timestamps 0 ensures the chunks retain their absolute original time.
    """
    chunk_pattern = os.path.join(chunk_dir, "chunk_%04d.mp4")
    cmd = [
        FFMPEG_EXE, "-y", "-i", video_path,
        "-c", "copy", 
        "-f", "segment", 
        "-segment_time", str(CHUNK_DURATION_SEC),
        "-reset_timestamps", "0",  # Crucial: DO NOT RESET CLOCK TO 0
        chunk_pattern
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return sorted(glob.glob(os.path.join(chunk_dir, "chunk_*.mp4")))

# ==============================================================================
# THE WORKER (Runs strictly on the pre-sliced GOP segment)
# ==============================================================================

def worker_extract_chunk(chunk_path: str, seq_id: int, video_id: str) -> dict:
    packet = {
        "video_id": video_id,
        "sequence_id": seq_id,
        "audio_window_sec": AUDIO_MICRO_WINDOW_SEC,
        "flux_window_sec": FLUX_WINDOW_SEC,
        "frames": [],
        "audio_rms": [],
        "spectral_flux": []
    }

    # 1. EXTRACT FRAMES & REAL TIMESTAMPS
    frame_cmd = [
        FFPROBE_EXE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pkt_size,pict_type,key_frame,pkt_pts_time,best_effort_timestamp_time",
        "-of", "compact=p=0", chunk_path
    ]
    proc = subprocess.Popen(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    first_pts = None
    
    for line in proc.stdout:
        line = line.strip()
        if not line: continue
        parts = {k: v for k, v in (kv.split("=", 1) for kv in line.split("|") if "=" in kv)}
        try:
            size = float(parts.get("pkt_size", 0))
            if size == 0: continue
            ptype = parts.get("pict_type", "U")
            if parts.get("key_frame", "0") == "1": ptype = "I"
            
            # Use real time from stream! Fallback to best_effort if pkt_pts_time is missing
            pts_str = parts.get("pkt_pts_time")
            if not pts_str or pts_str == "N/A":
                pts_str = parts.get("best_effort_timestamp_time", 0.0)
            
            try:
                pts = float(pts_str)
            except ValueError:
                pts = 0.0

            if first_pts is None: first_pts = pts
            
            packet["frames"].append({"time": pts, "type": ptype, "size": int(size)})
        except Exception:
            continue
    proc.wait()

    if first_pts is None:
        first_pts = 0.0
    packet["chunk_start_time"] = first_pts

    # 2. AUDIO RMS (Native to chunk)
    rms_cmd = [
        FFMPEG_EXE, "-v", "error", "-i", chunk_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "pipe:1"
    ]
    rms_proc = subprocess.Popen(rms_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    win_bytes = int(AUDIO_MICRO_WINDOW_SEC * AUDIO_SAMPLE_RATE) * 2
    leftover = b""
    audio_offset = first_pts
    
    while True:
        raw = rms_proc.stdout.read(win_bytes)
        if not raw: break
        leftover += raw
        while len(leftover) >= win_bytes:
            window_raw = leftover[:win_bytes]
            leftover = leftover[win_bytes:]
            samples = np.frombuffer(window_raw, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2))) if np.any(samples) else 0.0
            packet["audio_rms"].append({
                "time": round(audio_offset + AUDIO_MICRO_WINDOW_SEC / 2, 6),
                "rms": round(rms, 4)
            })
            audio_offset += AUDIO_MICRO_WINDOW_SEC
    rms_proc.wait()

    # 3. SPECTRAL FLUX (Native to chunk, accepts 50ms loss at chunk boundary)
    flux_cmd = [
        FFMPEG_EXE, "-v", "error", "-i", chunk_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(FLUX_SAMPLE_RATE), "-ac", "1",
        "-f", "s16le", "pipe:1"
    ]
    flux_proc = subprocess.Popen(flux_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    hop_samples = max(1, int(FLUX_WINDOW_SEC * FLUX_SAMPLE_RATE))
    n_fft = hop_samples * 2
    hann_win = np.hanning(n_fft)
    hop_bytes = n_fft * 2
    leftover2 = b""
    prev_spec = None
    flux_offset = first_pts
    
    while True:
        raw = flux_proc.stdout.read(hop_bytes)
        if not raw: break
        leftover2 += raw
        while len(leftover2) >= hop_bytes:
            window_raw = leftover2[:hop_bytes]
            leftover2 = leftover2[hop_bytes:]
            samples = np.frombuffer(window_raw, dtype=np.int16).astype(np.float64)
            spectrum = np.abs(np.fft.rfft(samples * hann_win))
            norm = np.linalg.norm(spectrum)
            if norm > 1e-9: spectrum /= norm
            
            fv = 0.0
            if prev_spec is not None:
                fv = float(np.sum(np.maximum(spectrum - prev_spec, 0.0)))
                
            packet["spectral_flux"].append({
                "time": round(flux_offset + FLUX_WINDOW_SEC / 2, 6),
                "flux": round(fv, 6)
            })
            prev_spec = spectrum
            flux_offset += FLUX_WINDOW_SEC
    flux_proc.wait()

    packet["perceptual_chunk_hash"] = hashlib.md5(f"{video_id}|{first_pts:.3f}|{seq_id}".encode()).hexdigest()
    return packet

# ==============================================================================
# PIPELINE COORDINATOR
# ==============================================================================

def process_video(video_path: str):
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    print(f"\n  Video: {video_name}")

    with tempfile.TemporaryDirectory() as chunk_dir:
        print(f"  [CHUNKER] Slicing into native GOP segments...")
        chunk_files = chunk_video(video_path, chunk_dir)
        total_chunks = len(chunk_files)
        print(f"  [CHUNKER] Generated {total_chunks} absolute-time GOP chunks.")

        print(f"  [WORKERS] Extracting JSON telemetry with {NUM_WORKERS} workers...")
        t_w0 = time.time()
        packets = [None] * total_chunks

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            future_map = {
                pool.submit(worker_extract_chunk, c_path, seq_id, video_id): seq_id
                for seq_id, c_path in enumerate(chunk_files)
            }
            done = 0
            for future in as_completed(future_map):
                seq_id = future_map[future]
                try:
                    packets[seq_id] = future.result()
                except Exception as exc:
                    sys.stdout.write("\r" + " " * 80 + "\r")
                    print(f"    Worker {seq_id} failed: {exc}")
                done += 1
                progress_bar(done, total_chunks, extra=f"chunk {done}/{total_chunks}")
        
        t_worker = time.time() - t_w0
        packets = [p for p in packets if p is not None]
        packets.sort(key=lambda p: p["sequence_id"])
        print(f"  Workers done: {t_worker:.1f}s ({len(packets)}/{total_chunks} chunks OK)")

        out_dir = os.path.join(SCRIPT_DIR, WORKER_DATA_DIR)
        os.makedirs(out_dir, exist_ok=True)
        worker_json_path = os.path.join(out_dir, f"{video_id}_worker_data.json")
        
        with open(worker_json_path, "w") as f:
            json.dump({"video": video_name, "architecture": "gop_chunker", "chunks": packets}, f, indent=2)

if __name__ == "__main__":
    gt_path = os.path.join(SCRIPT_DIR, GROUND_TRUTH_JSON)
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth not found: {gt_path}")
        sys.exit(1)

    # Specify the exact video indexes you want to run here
    SELECTED_VIDEOS = [i for i in list(range(1,42))]
    VIDEOS_TO_MONITOR = [f"attacked_{i}.mp4" for i in SELECTED_VIDEOS]
    print("\n======================================================================")
    print("  STAGE 1: CHUNKER & WORKER INGEST PIPELINE")
    print("======================================================================")

    for vid_idx, video_name in enumerate(VIDEOS_TO_MONITOR):
        video_path = os.path.join(SCRIPT_DIR, ATTACKED_DIR, video_name)
        if not os.path.exists(video_path): continue
        process_video(video_path)
    
    print("\n[+] Worker ingestion complete. JSONs saved to worker_data_temp/")
    print("[+] You can now run leader_temp.py")
