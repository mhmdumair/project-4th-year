import subprocess
import json
import numpy as np
import os
import sys
import time
import datetime
import hashlib
import tempfile
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ATTACKED_DIR        = "dataset"
GROUND_TRUTH_JSON   = "attack.json"
REPORT_FILE         = "report.txt"
WORKER_DATA_DIR     = "worker_data"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE  = "ffmpeg"
FFPROBE_EXE = "ffprobe"

# Distributed
CHUNK_DURATION_SEC  = 5.0
AUDIO_OVERLAP_SEC   = 0.05      # 50ms overlap for spectral flux continuity
NUM_WORKERS         = 10

# --- Fusion / Scoring ---
BOUNDARY_TOLERANCE      = 2.5
TOLERANCE_SEC           = 2.5
MIN_SIGNALS_TO_CONFIRM  = 1
REQUIRE_MULTIMODAL      = False
LOCKOUT_PERIOD          = 5.0
MERGE_WINDOW            = 10.0

# --- Visual ---
VISUAL_MAX_FRAMES       = 30
VISUAL_MIN_WARMUP       = 10
K_VISUAL_THRESHOLD      = 2.5
VISUAL_COOLDOWN         = 3.0
VISUAL_PERSIST_NEEDED   = 2
VISUAL_PERSIST_WINDOW   = 4
SCENE_CHANGE_RATIO      = 8.0
SCENE_CHANGE_TOLERANCE  = 0.5
IFRAME_SANDWICH_ENABLED = True
IFRAME_CONTEXT_WINDOW   = 5
IFRAME_STABILITY_RATIO  = 3.0
IFRAME_MIN_CONTEXT      = 3

# --- Motion ---
MV_WINDOW               = 60
MV_MIN_WARMUP           = 30
K_MV_THRESHOLD          = 4.5
MV_COOLDOWN             = 5.0
MV_PERSIST_NEEDED       = 3
MV_PERSIST_WINDOW       = 5
GOP_SHORT_RATIO         = 0.25
MOTION_JERK_ENABLED     = True
JERK_WINDOW             = 5
JERK_ACCEL_THRESHOLD    = 2.5

# --- Audio ---
AUDIO_SAMPLE_RATE       = 44100
AUDIO_BUFFER_SEC        = 15.0
AUDIO_MICRO_WINDOW_SEC  = 0.5
K_AUDIO_RMS_THRESHOLD   = 3.0
AUDIO_NOISE_FLOOR       = 800.0
AUDIO_COOLDOWN          = 3.0
AUDIO_PERSIST_NEEDED    = 2

# --- Silence ---
SILENCE_THRESHOLD_RATIO = 0.05
SILENCE_MIN_DURATION    = 2.0
SILENCE_COOLDOWN        = 30.0

# --- Spectral Flux ---
FLUX_SAMPLE_RATE        = 22050
FLUX_WINDOW_SEC         = 0.05
FLUX_BUFFER_SEC         = 10.0
K_FLUX_THRESHOLD        = 3.0
FLUX_COOLDOWN           = 3.0
FLUX_PERSIST_NEEDED     = 3

# --- Trust hierarchy ---
VISUAL_MOTION_SIGNALS   = {"Visual", "Motion"}
AUDIO_SIGNALS           = {"AudioRMS", "Silence", "SpectralFlux"}
AUDIO_UPGRADE_WINDOW    = 2.5
MIN_EVENT_GAP           = 0.5

# --- Selection layer ---
SELECTION_WINDOW_SEC    = 30.0
SELECTION_OVERLAP       = 0.1
SELECTION_MIN_GAP       = 0.5

W_CONFIDENCE    = 0.50
W_MAGNITUDE     = 0.05
W_ACCELERATION  = 0.15
W_EDGE_CHANGE   = 0.10
W_UNIFORMITY    = 0.05
W_PERSISTENCE   = 0.15
MAGNITUDE_EXTREME_Z     = 8.0
MAGNITUDE_SWEET_Z_LO    = 2.5
MAGNITUDE_SWEET_Z_HI    = 6.0

DURATION_BUCKETS = [
    (8,   12,  "10s    "),
    (28,  32,  "30s    "),
    (32,  999, "60s+   "),
]

# ==============================================================================
# SHARED HELPERS
# ==============================================================================

def confidence_label(weight):
    if weight >= 3:   return "WIDE"
    elif weight == 2: return "HIGH"
    else:             return "LOW"

def confidence_score(weight):
    if weight >= 3:   return 1.0
    elif weight == 2: return 0.65
    else:             return 0.30

def hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def progress_bar(current, total, width=40, extra=""):
    if total <= 0:
        return
    frac = min(current / total, 1.0)
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    pct = frac * 100
    sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%  {extra}")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

def banner(title):
    w = 70
    print("\n" + "═"*w)
    pad = max(0, (w - len(title) - 2) // 2)
    print(" " * pad + f" {title} ")
    print("═"*w)

def section(title):
    print(f"\n{'─'*70}\n  {title}\n{'─'*70}")

def adaptive_zscore(value, buffer, k):
    arr = np.array(buffer, dtype=np.float64)
    if len(arr) < 2: return 0.0, False
    mu, sigma = np.mean(arr), np.std(arr)
    if sigma < 1e-9: return 0.0, False
    z = (value - mu) / sigma
    return z, abs(z) > k

def compute_trigger_score(weight, features):
    score = (
        W_CONFIDENCE   * confidence_score(weight)           +
        W_MAGNITUDE    * features.get("magnitude",    0.5)  +
        W_ACCELERATION * features.get("acceleration", 0.5)  +
        W_EDGE_CHANGE  * features.get("edge_change",  0.5)  +
        W_UNIFORMITY   * features.get("uniformity",   0.5)  +
        W_PERSISTENCE  * features.get("persistence",  0.5)
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)

def get_video_duration(video_path):
    cmd = [FFPROBE_EXE, "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0

# ==============================================================================
# WORKER: Extract raw data with audio overlap for spectral flux continuity
# ==============================================================================

def _cut_chunk(video_path: str, start_time: float) -> str:
    """Cut 5-second segment to a temp mp4."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    cmd = [
        FFMPEG_EXE, "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(CHUNK_DURATION_SEC),
        "-c", "copy",
        tmp.name
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmp.name

def _probe_fps(chunk_path: str) -> float:
    cmd = [
        FFPROBE_EXE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        chunk_path
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        num, den = r.stdout.strip().split("/")
        fps = float(num) / float(den)
        return fps if 1.0 < fps < 240.0 else 30.0
    except Exception:
        return 30.0

def worker_extract_chunk(video_path: str, start_time: float,
                         seq_id: int, video_id: str) -> dict:
    """
    WORKER — pure extraction, zero detection logic.
    
    KEY IMPROVEMENT: Audio extraction includes 50ms OVERLAP from previous chunk
    so that spectral flux calculation has prev_spectrum context.
    
    Video frames: EXACTLY start_time to start_time + CHUNK_DURATION_SEC
    Audio: start_time - AUDIO_OVERLAP_SEC to start_time + CHUNK_DURATION_SEC
    """
    chunk_path = _cut_chunk(video_path, start_time)

    try:
        fps = _probe_fps(chunk_path)
        audio_start = start_time - AUDIO_OVERLAP_SEC
        audio_duration = CHUNK_DURATION_SEC + AUDIO_OVERLAP_SEC

        packet = {
            "video_id":              video_id,
            "sequence_id":           seq_id,
            "chunk_start_time":      round(start_time, 6),
            "chunk_end_time":        round(start_time + CHUNK_DURATION_SEC, 6),
            "audio_overlap_sec":     AUDIO_OVERLAP_SEC,
            "fps":                   fps,
            "audio_window_sec":      AUDIO_MICRO_WINDOW_SEC,
            "flux_window_sec":       FLUX_WINDOW_SEC,
            "perceptual_chunk_hash": hashlib.md5(
                f"{video_id}|{start_time:.3f}|{seq_id}".encode()
            ).hexdigest(),
            "frames":        [],
            "audio_rms":     [],      # Timestamps include overlap period
            "spectral_flux": [],      # Timestamps include overlap period
        }

        # ------------------------------------------------------------------
        # 1. FRAMES (EXACT chunk, no overlap)
        # ------------------------------------------------------------------
        frame_cmd = [
            FFPROBE_EXE, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_size,pict_type,key_frame",
            "-of", "compact=p=0",
            chunk_path
        ]
        proc = subprocess.Popen(frame_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            parts = {}
            for kv in line.split("|"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    parts[k] = v
            try:
                size = float(parts.get("pkt_size", 0))
                if size == 0:
                    continue
                ptype = parts.get("pict_type", "U")
                if parts.get("key_frame", "0") == "1":
                    ptype = "I"
                packet["frames"].append({"type": ptype, "size": int(size)})
            except Exception:
                continue
        proc.stdout.close()
        proc.wait()

        # ------------------------------------------------------------------
        # 2. AUDIO RMS (with overlap - includes 50ms BEFORE chunk start)
        # ------------------------------------------------------------------
        rms_cmd = [
            FFMPEG_EXE, "-v", "error",
            "-ss", str(audio_start), "-i", video_path,
            "-t", str(audio_duration),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1"
        ]
        rms_proc = subprocess.Popen(rms_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        win_bytes = int(AUDIO_MICRO_WINDOW_SEC * AUDIO_SAMPLE_RATE) * 2
        leftover = b""
        audio_offset = audio_start

        while True:
            raw = rms_proc.stdout.read(win_bytes)
            if not raw:
                break
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

        rms_proc.stdout.close()
        rms_proc.wait()

        # ------------------------------------------------------------------
        # 3. SPECTRAL FLUX (with overlap - has prev_spectrum context!)
        # ------------------------------------------------------------------
        flux_cmd = [
            FFMPEG_EXE, "-v", "error",
            "-ss", str(audio_start), "-i", video_path,
            "-t", str(audio_duration),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(FLUX_SAMPLE_RATE), "-ac", "1",
            "-f", "s16le", "pipe:1"
        ]
        flux_proc = subprocess.Popen(flux_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        hop_samples = max(1, int(FLUX_WINDOW_SEC * FLUX_SAMPLE_RATE))
        n_fft = hop_samples * 2
        hann_win = np.hanning(n_fft)
        hop_bytes = n_fft * 2
        leftover2 = b""
        prev_spec = None
        flux_offset = audio_start

        while True:
            raw = flux_proc.stdout.read(hop_bytes)
            if not raw:
                break
            leftover2 += raw
            while len(leftover2) >= hop_bytes:
                window_raw = leftover2[:hop_bytes]
                leftover2 = leftover2[hop_bytes:]
                samples = np.frombuffer(window_raw, dtype=np.int16).astype(np.float64)
                spectrum = np.abs(np.fft.rfft(samples * hann_win))
                norm = np.linalg.norm(spectrum)
                if norm > 1e-9:
                    spectrum /= norm
                if prev_spec is not None:
                    fv = float(np.sum(np.maximum(spectrum - prev_spec, 0.0)))
                    packet["spectral_flux"].append({
                        "time": round(flux_offset + FLUX_WINDOW_SEC / 2, 6),
                        "flux": round(fv, 6)
                    })
                prev_spec = spectrum
                flux_offset += FLUX_WINDOW_SEC

        flux_proc.stdout.close()
        flux_proc.wait()

        return packet

    finally:
        try:
            os.unlink(chunk_path)
        except Exception:
            pass

# ==============================================================================
# LEADER: Reconstruct timeline (filters out overlap) and run detection
# ==============================================================================

class LeaderDetector:
    @staticmethod
    def _iframe_sandwich(size_list, cur_idx, cur_size):
        if not IFRAME_SANDWICH_ENABLED:
            return True
        if cur_idx < IFRAME_MIN_CONTEXT or cur_idx >= len(size_list) - 1:
            return True
        before = [s for _, s in size_list[max(0, cur_idx - IFRAME_CONTEXT_WINDOW):cur_idx]]
        after = [s for _, s in size_list[cur_idx + 1: min(len(size_list), cur_idx + IFRAME_CONTEXT_WINDOW + 1)]]
        if len(before) < 2 or len(after) < 2:
            return True
        ctx_mean = np.mean(before + after)
        if ctx_mean < 1e-6:
            return True
        ctx_std = max(np.std(before), np.std(after))
        if ctx_std < ctx_mean * 0.1:
            if cur_size / (ctx_mean + 1e-9) > IFRAME_STABILITY_RATIO * 10:
                return False
        return True

    @staticmethod
    def _detect_jerk(motion_history):
        if not MOTION_JERK_ENABLED or len(motion_history) < JERK_WINDOW:
            return True
        ml = list(motion_history)[-JERK_WINDOW:]
        velocities = [ml[i+1] - ml[i] for i in range(len(ml)-1)]
        accelerations = [abs(velocities[i+1] - velocities[i]) for i in range(len(velocities)-1)]
        if not accelerations:
            return True
        mean_a = np.mean(accelerations)
        std_a = np.std(accelerations)
        if std_a < 1e-9:
            return True
        jerk = (max(accelerations) - mean_a) / (std_a + 1e-9)
        return jerk > JERK_ACCEL_THRESHOLD

    def _build_timeline(self, packets: list) -> tuple:
        """Build continuous timeline, filtering out audio/flux overlap"""
        frames = []
        audio = []
        flux = []
        
        for pkt in packets:
            start = pkt["chunk_start_time"]
            overlap = pkt["audio_overlap_sec"]
            fps = pkt["fps"]
            
            # Frames: already exact timestamps
            for idx, frm in enumerate(pkt["frames"]):
                t = start + idx / fps
                frames.append((t, frm["type"], int(frm["size"])))
            
            # Audio RMS: keep only timestamps >= start (filter out overlap)
            for rms_item in pkt["audio_rms"]:
                if rms_item["time"] >= start - 0.001:  # Small tolerance
                    audio.append((rms_item["time"], rms_item["rms"]))
            
            # Spectral flux: keep only timestamps >= start (filter out overlap)
            for flux_item in pkt["spectral_flux"]:
                if flux_item["time"] >= start - 0.001:
                    flux.append((flux_item["time"], flux_item["flux"]))
        
        frames.sort(key=lambda x: x[0])
        audio.sort(key=lambda x: x[0])
        flux.sort(key=lambda x: x[0])
        return frames, audio, flux

    def run_detection(self, packets: list) -> dict:
        frames, audio_samples, flux_samples = self._build_timeline(packets)

        # Visual detection
        v_size_buf = deque(maxlen=VISUAL_MAX_FRAMES)
        v_spike_flags = deque(maxlen=VISUAL_PERSIST_WINDOW)
        v_last_trig = -999.0
        v_triggers = []
        v_sizes_sandwich = []
        scene_trans_end = 0.0

        # Motion detection
        mv_pb_buf = deque(maxlen=MV_WINDOW)
        mv_spike_flags = deque(maxlen=MV_PERSIST_WINDOW)
        mv_last_trig = -999.0
        mv_last_i_t = -999.0
        mv_gop = deque(maxlen=30)
        mv_triggers = []
        jerk_history = deque(maxlen=JERK_WINDOW)

        for (t, ptype, size) in frames:
            is_i = (ptype == "I")

            if is_i:
                v_sizes_sandwich.append((t, size))
                cur_idx = len(v_sizes_sandwich) - 1

                if len(v_size_buf) >= 2:
                    prev = list(v_size_buf)[-1]
                    if prev > 0 and size / prev > SCENE_CHANGE_RATIO:
                        scene_trans_end = t + SCENE_CHANGE_TOLERANCE

                if t < scene_trans_end:
                    spike = False
                elif t > LOCKOUT_PERIOD and len(v_size_buf) >= VISUAL_MIN_WARMUP:
                    _, spike = adaptive_zscore(size, v_size_buf, K_VISUAL_THRESHOLD)
                else:
                    spike = False

                v_spike_flags.append(1 if spike else 0)

                if (t > LOCKOUT_PERIOD and len(v_spike_flags) == VISUAL_PERSIST_WINDOW
                        and sum(v_spike_flags) >= VISUAL_PERSIST_NEEDED
                        and (t - v_last_trig) > VISUAL_COOLDOWN
                        and (t - v_last_trig) > MIN_EVENT_GAP):
                    if self._iframe_sandwich(v_sizes_sandwich, cur_idx, size):
                        v_triggers.append(t)
                        v_last_trig = t
                    v_spike_flags.clear()

                v_size_buf.append(size)

                if mv_last_i_t > 0:
                    interval = t - mv_last_i_t
                    if len(mv_gop) >= 10:
                        mean_gop = np.mean(mv_gop)
                        if (t > LOCKOUT_PERIOD and mean_gop > 0
                                and interval < mean_gop * GOP_SHORT_RATIO
                                and (t - mv_last_trig) > MV_COOLDOWN):
                            mv_triggers.append(t)
                            mv_last_trig = t
                    mv_gop.append(interval)
                mv_last_i_t = t

            if ptype in ("P", "B"):
                jerk_history.append(size)
                mv_pb_buf.append(size)

                if t > LOCKOUT_PERIOD and len(mv_pb_buf) >= MV_MIN_WARMUP:
                    _, spike = adaptive_zscore(size, mv_pb_buf, K_MV_THRESHOLD)
                else:
                    spike = False

                mv_spike_flags.append(1 if spike else 0)

                if (t > LOCKOUT_PERIOD and len(mv_spike_flags) == MV_PERSIST_WINDOW
                        and sum(mv_spike_flags) >= MV_PERSIST_NEEDED
                        and (t - mv_last_trig) > MV_COOLDOWN
                        and (t - mv_last_trig) > MIN_EVENT_GAP):
                    if self._detect_jerk(jerk_history):
                        mv_triggers.append(t)
                        mv_last_trig = t
                    mv_spike_flags.clear()

        # Dedup motion
        mv_triggers.sort()
        deduped_mv, last_t = [], -999.0
        for t in mv_triggers:
            if t - last_t > MV_COOLDOWN:
                deduped_mv.append(t)
                last_t = t

        # Audio RMS + Silence detection (now continuous across chunks!)
        rms_buf = deque(maxlen=int(AUDIO_BUFFER_SEC / AUDIO_MICRO_WINDOW_SEC))
        rms_consec = 0
        rms_last_trig = -999.0
        rms_triggers = []
        sil_in = False
        sil_start = -999.0
        sil_last_trig = -999.0
        sil_triggers = []

        for (micro_t, micro_rms) in audio_samples:
            if micro_t > LOCKOUT_PERIOD and len(rms_buf) == rms_buf.maxlen:
                mu = float(np.mean(rms_buf))
                sigma = float(np.std(rms_buf))
                is_spike = (micro_rms > AUDIO_NOISE_FLOOR and sigma > 0
                            and micro_rms > mu + K_AUDIO_RMS_THRESHOLD * sigma)
                rms_consec = rms_consec + 1 if is_spike else 0

                if (rms_consec >= AUDIO_PERSIST_NEEDED
                        and (micro_t - rms_last_trig) > AUDIO_COOLDOWN
                        and (micro_t - rms_last_trig) > MIN_EVENT_GAP):
                    rms_triggers.append(micro_t)
                    rms_last_trig = micro_t
                    rms_consec = 0

                if mu > AUDIO_NOISE_FLOOR:
                    is_silent = micro_rms < mu * SILENCE_THRESHOLD_RATIO
                    if is_silent and not sil_in:
                        sil_in, sil_start = True, micro_t
                    elif not is_silent and sil_in:
                        sil_in = False
                        dur = micro_t - sil_start
                        if (dur >= SILENCE_MIN_DURATION
                                and (sil_start - sil_last_trig) > SILENCE_COOLDOWN):
                            sil_triggers.append(sil_start)
                            sil_last_trig = sil_start

            rms_buf.append(micro_rms)

        # Spectral flux detection (now has proper prev_spectrum due to overlap!)
        flux_buf = deque(maxlen=max(10, int(FLUX_BUFFER_SEC / FLUX_WINDOW_SEC)))
        flux_consec = 0
        flux_last_trig = -999.0
        flux_triggers = []

        for (flux_t, fv) in flux_samples:
            if flux_t > LOCKOUT_PERIOD and len(flux_buf) >= int(flux_buf.maxlen * 0.3):
                _, spike = adaptive_zscore(fv, flux_buf, K_FLUX_THRESHOLD)
                flux_consec = flux_consec + 1 if spike else 0

                if (flux_consec >= FLUX_PERSIST_NEEDED
                        and (flux_t - flux_last_trig) > FLUX_COOLDOWN
                        and (flux_t - flux_last_trig) > MIN_EVENT_GAP):
                    flux_triggers.append(flux_t)
                    flux_last_trig = flux_t
                    flux_consec = 0

            flux_buf.append(fv)

        return {
            "Visual": v_triggers,
            "Motion": deduped_mv,
            "AudioRMS": rms_triggers,
            "Silence": sil_triggers,
            "SpectralFlux": flux_triggers,
        }

# ==============================================================================
# FUSION AND SELECTION (unchanged)
# ==============================================================================

def fuse_all_signals(signal_dict: dict) -> list:
    all_events = sorted(
        [{"time": t, "signal": lbl} for lbl, ts in signal_dict.items() for t in ts],
        key=lambda x: x["time"]
    )
    used, clusters = [False] * len(all_events), []
    for i, ev in enumerate(all_events):
        if used[i]: continue
        cluster = [ev]; used[i] = True
        for j in range(i + 1, len(all_events)):
            if used[j]: continue
            if all_events[j]["time"] - cluster[0]["time"] > TOLERANCE_SEC: break
            cluster.append(all_events[j]); used[j] = True
        clusters.append(cluster)

    merged = []
    for cluster in clusters:
        sigs = list({e["signal"] for e in cluster})
        vm_sigs = [s for s in sigs if s in VISUAL_MOTION_SIGNALS]
        if not vm_sigs:
            continue
        avg_t = round(float(np.mean([e["time"] for e in cluster])), 2)
        base_weight = len(set(vm_sigs))
        audio_present = {e["signal"] for e in cluster
                         if e["signal"] in AUDIO_SIGNALS
                         and abs(e["time"] - avg_t) <= AUDIO_UPGRADE_WINDOW}
        weight = base_weight + len(audio_present)
        all_sigs = sorted(set(vm_sigs) | audio_present)
        default_feats = {k: 0.5 for k in ["magnitude", "acceleration", "edge_change", "uniformity", "persistence"]}
        score = compute_trigger_score(weight, default_feats)
        merged.append((avg_t, "+".join(all_sigs), len(all_sigs), min(weight, 5), default_feats, score))

    merged.sort(key=lambda x: x[0])
    return merged

def apply_window_merge(fused_events, window=MERGE_WINDOW):
    if not fused_events: return []
    result, i = [], 0
    while i < len(fused_events):
        group, j = [], i
        while j < len(fused_events) and fused_events[j][0] <= fused_events[i][0] + window:
            group.append(fused_events[j]); j += 1
        if len(group) >= 2:
            all_lbl = set(s for e in group for s in e[1].split("+"))
            avg_feats = {k: 0.5 for k in ["magnitude", "acceleration", "edge_change", "uniformity", "persistence"]}
            score = compute_trigger_score(3, avg_feats)
            result.append((round(float(np.mean([e[0] for e in group])), 2),
                           "+".join(sorted(all_lbl)), len(all_lbl), 3, avg_feats, score))
        else:
            result.append(group[0])
        i = j
    return result

def events_to_dicts(fused_events):
    return [{"time": e[0], "label": e[1], "n_sig": e[2], "weight": e[3],
             "features": e[4] if len(e) > 4 else {},
             "score": e[5] if len(e) > 5 else compute_trigger_score(e[3], {})}
            for e in fused_events]

def dicts_to_tuples(event_dicts):
    return [(e["time"], e["label"], e["n_sig"], e["weight"], e["features"], e["score"])
            for e in event_dicts]

def select_strongest_per_window(event_dicts, window_sec=SELECTION_WINDOW_SEC,
                                 overlap=SELECTION_OVERLAP, min_gap=SELECTION_MIN_GAP):
    if not event_dicts:
        return [], []
    hop_sec = window_sec * (1.0 - overlap)
    total_time = max(e["time"] for e in event_dicts) + 1.0
    n_windows = max(1, int(np.ceil((total_time - window_sec) / hop_sec)) + 1)
    kept_set = set()

    for w in range(n_windows):
        win_start = w * hop_sec
        win_end = win_start + window_sec
        in_window = [i for i, e in enumerate(event_dicts) if win_start <= e["time"] < win_end]
        if not in_window: continue
        kept_set.add(max(in_window, key=lambda i: event_dicts[i]["score"]))

    sorted_kept = sorted(kept_set, key=lambda i: event_dicts[i]["time"])
    final_kept, last_t = [], -999.0
    for i in sorted_kept:
        t = event_dicts[i]["time"]
        if t - last_t >= min_gap:
            final_kept.append(i)
            last_t = t

    fk = set(final_kept)
    return ([event_dicts[i] for i in range(len(event_dicts)) if i in fk],
            [event_dicts[i] for i in range(len(event_dicts)) if i not in fk])

# ==============================================================================
# VALIDATION
# ==============================================================================

def classify_hit_position(ts, seg):
    ns = abs(ts - seg["start"]) <= BOUNDARY_TOLERANCE
    ne = abs(ts - seg["end"]) <= BOUNDARY_TOLERANCE
    inside = seg["start"] < ts < seg["end"]
    if ns and not ne: return "START"
    if ne and not ns: return "END"
    if ns and ne: return "START+END"
    if inside: return "IN-BETWEEN"
    return None

def validate_video(video_name, segments, final_events):
    segs = [dict(s, hit=None) for s in segments]
    fp_events = []
    for ev in final_events:
        ts, label, n_sig, weight = ev[0], ev[1], ev[2], ev[3]
        feats = ev[4] if len(ev) > 4 else {}
        score = ev[5] if len(ev) > 5 else 0.0
        claimed = False
        for seg in segs:
            pos = classify_hit_position(ts, seg)
            if pos is not None and seg["hit"] is None:
                seg["hit"] = {"ts": ts, "label": label, "n_sig": n_sig,
                              "weight": weight, "position": pos,
                              "features": feats, "score": score}
                claimed = True
                break
        if not claimed:
            fp_events.append({"ts": ts, "label": label, "n_sig": n_sig,
                              "weight": weight, "features": feats, "score": score})

    seg_results = []
    for seg in segs:
        src = seg.get("source", seg.get("sources", ["unknown"])[0] if seg.get("sources") else "unknown")
        if isinstance(src, list):
            src = ", ".join(src[:2])
        seg_results.append({
            "index": seg["index"], "source": src,
            "start": seg["start"], "end": seg["end"],
            "duration": seg["duration"], "hit": seg["hit"]
        })
    return seg_results, fp_events

# ==============================================================================
# REPORTING
# ==============================================================================

def print_video_report(video_name, seg_results, fp_events, signal_dict,
                       fused_events, final_events, dropped_events):
    total = len(seg_results)
    hits = sum(1 for s in seg_results if s["hit"])
    rate = hits / total * 100 if total else 0

    section(f"VIDEO: {video_name}   [{hits}/{total}  {rate:.1f}%]")
    print("  Raw triggers per signal:")
    for sig, ts in signal_dict.items():
        print(f"    {sig:<15} {len(ts):>4}  {'▪'*min(len(ts),40)}")
    print(f"  Fused: {len(fused_events)}  Final: {len(final_events)}  Dropped: {len(dropped_events)}")

    if hits > 0:
        print(f"\n  HITS:")
        for s in seg_results:
            if s["hit"]:
                h = s["hit"]
                print(f"    Segment {s['index']}: HIT at {h['ts']:.2f}s ({h['position']}) [{h['label']}]")
    
    if len(seg_results) - hits > 0:
        print(f"\n  MISSED:")
        for s in seg_results:
            if not s["hit"]:
                print(f"    Segment {s['index']}: MISSED")
    
    if fp_events:
        print(f"\n  False Positives ({len(fp_events)}):")
        for fp in fp_events[:10]:
            print(f"    @ {fp['ts']:.2f}s [{confidence_label(fp['weight'])}] {fp['label']}")
        if len(fp_events) > 10:
            print(f"    ... and {len(fp_events) - 10} more")

def print_overall_summary(all_video_stats):
    banner("FINAL SUMMARY - ALL VIDEOS")
    
    total_segs = sum(v["total"] for v in all_video_stats)
    total_hits = sum(v["hits"] for v in all_video_stats)
    total_fp = sum(v["fp"] for v in all_video_stats)
    total_worker = sum(v["worker_time"] for v in all_video_stats)
    total_leader = sum(v["leader_time"] for v in all_video_stats)
    overall = total_hits / total_segs * 100 if total_segs else 0
    
    print(f"\n  {'VIDEO':<35}  {'SEGS':>4}  {'HITS':>5}  {'RATE':>6}  {'FP':>4}  {'WORKER':>8}  {'LEADER':>8}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*8}  {'─'*8}")
    
    for v in all_video_stats:
        rate = v["hits"] / v["total"] * 100 if v["total"] else 0
        print(f"  {v['name']:<35}  {v['total']:>4}  {v['hits']:>5}  {rate:>5.1f}%  {v['fp']:>4}  {v['worker_time']:>7.1f}s  {v['leader_time']:>7.3f}s")
    
    print(f"  {'─'*35}  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*8}  {'─'*8}")
    print(f"  {'TOTAL':<35}  {total_segs:>4}  {total_hits:>5}  {overall:>5.1f}%  {total_fp:>4}  {total_worker:>7.1f}s  {total_leader:>7.3f}s")
    
    print(f"\n  {'='*70}")
    print(f"  DETECTION RATE: {total_hits}/{total_segs} ({overall:.2f}%)")
    print(f"  TOTAL WORKER TIME: {total_worker:.1f}s ({total_worker/60:.1f} minutes)")
    print(f"  TOTAL LEADER TIME: {total_leader:.3f}s")
    print(f"  SPEEDUP (worker vs leader): {total_worker / max(total_leader, 0.001):.0f}x")
    print(f"  {'='*70}")

def write_report(report_path, all_video_stats, run_ts):
    total_segs = sum(v["total"] for v in all_video_stats)
    total_hits = sum(v["hits"] for v in all_video_stats)
    total_fp = sum(v["fp"] for v in all_video_stats)
    total_worker = sum(v["worker_time"] for v in all_video_stats)
    total_leader = sum(v["leader_time"] for v in all_video_stats)
    overall = total_hits / total_segs * 100 if total_segs else 0

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("DISTRIBUTED VIOLENCE DETECTION - FINAL REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Run: {run_ts}\n")
        f.write(f"Architecture: {NUM_WORKERS} workers (extract JSON) + 1 leader (detect)\n")
        f.write(f"Chunk size: {CHUNK_DURATION_SEC}s\n")
        f.write(f"Audio overlap: {AUDIO_OVERLAP_SEC*1000:.0f}ms (for spectral flux continuity)\n")
        f.write(f"Total videos: {len(all_video_stats)}\n\n")
        
        f.write(f"{'VIDEO':<35}  {'SEGS':>4}  {'HITS':>5}  {'RATE':>6}  {'FP':>4}  {'WORKER':>8}  {'LEADER':>8}\n")
        f.write("-"*80 + "\n")
        
        for v in all_video_stats:
            rate = v["hits"] / v["total"] * 100 if v["total"] else 0
            f.write(f"{v['name']:<35}  {v['total']:>4}  {v['hits']:>5}  {rate:>5.1f}%  {v['fp']:>4}  {v['worker_time']:>7.1f}s  {v['leader_time']:>7.3f}s\n")
        
        f.write("-"*80 + "\n")
        f.write(f"{'TOTAL':<35}  {total_segs:>4}  {total_hits:>5}  {overall:>5.1f}%  {total_fp:>4}  {total_worker:>7.1f}s  {total_leader:>7.3f}s\n\n")
        f.write(f"DETECTION RATE: {total_hits}/{total_segs} ({overall:.2f}%)\n")
        f.write(f"WORKER TIME: {total_worker:.1f}s ({total_worker/60:.1f} minutes)\n")
        f.write(f"LEADER TIME: {total_leader:.3f}s\n")
        f.write(f"SPEEDUP: {total_worker / max(total_leader, 0.001):.0f}x\n")

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def process_video(video_path: str, ground_truth_entry: dict) -> dict:
    video_name = os.path.basename(video_path)
    video_id = os.path.splitext(video_name)[0]
    segments = ground_truth_entry["segments"]
    
    for seg in segments:
        if seg["end"] < seg["start"]:
            seg["end"] = seg["start"] + seg["duration"]

    duration = get_video_duration(video_path)
    total_chunks = int(np.ceil(duration / CHUNK_DURATION_SEC))

    print(f"\n  Video: {video_name}")
    print(f"  Duration: {duration:.1f}s → {total_chunks} chunks of {CHUNK_DURATION_SEC}s")
    print(f"  Audio overlap: {AUDIO_OVERLAP_SEC*1000:.0f}ms (for flux continuity)")
    print(f"  Segments to detect: {len(segments)}")

    # PHASE 1: Workers extract data
    print(f"\n  [WORKERS] Extracting {total_chunks} chunks with {NUM_WORKERS} workers...")
    t_w0 = time.time()

    packets = [None] * total_chunks

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        future_map = {
            pool.submit(worker_extract_chunk, video_path, seq_id * CHUNK_DURATION_SEC, seq_id, video_id): seq_id
            for seq_id in range(total_chunks)
        }
        done = 0
        for future in as_completed(future_map):
            seq_id = future_map[future]
            try:
                packets[seq_id] = future.result()
            except Exception as exc:
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                print(f"    Worker {seq_id} failed: {exc}")
            done += 1
            progress_bar(done, total_chunks, extra=f"chunk {done}/{total_chunks}")

    print()
    t_worker = time.time() - t_w0

    packets = [p for p in packets if p is not None]
    packets.sort(key=lambda p: p["sequence_id"])
    print(f"  Workers done: {t_worker:.1f}s ({len(packets)}/{total_chunks} chunks OK)")

    # Save worker data
    os.makedirs(os.path.join(SCRIPT_DIR, WORKER_DATA_DIR), exist_ok=True)
    worker_json_path = os.path.join(SCRIPT_DIR, WORKER_DATA_DIR, f"{video_id}_worker_data.json")
    with open(worker_json_path, "w") as f:
        json.dump({"video": video_name, "audio_overlap_sec": AUDIO_OVERLAP_SEC, "chunks": packets}, f, indent=2)

    # PHASE 2: Leader runs detection
    print(f"\n  [LEADER] Running detection on JSON data...")
    t_l0 = time.time()

    detector = LeaderDetector()
    signal_dict = detector.run_detection(packets)

    fused_events = fuse_all_signals(signal_dict)
    merged_events = apply_window_merge(fused_events)
    event_dicts = events_to_dicts(merged_events)
    selected, dropped = select_strongest_per_window(event_dicts)
    final_events = dicts_to_tuples(selected)

    t_leader = time.time() - t_l0
    print(f"  Leader done: {t_leader:.3f}s")

    # Validate
    seg_results, fp_events = validate_video(video_name, segments, final_events)
    hits = sum(1 for s in seg_results if s["hit"])
    
    print(f"\n  RESULTS: {hits}/{len(segments)} segments detected ({hits/len(segments)*100:.1f}%)")
    print(f"  False Positives: {len(fp_events)}")

    return {
        "name": video_name,
        "total": len(segments),
        "hits": hits,
        "fp": len(fp_events),
        "worker_time": t_worker,
        "leader_time": t_leader,
    }

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    gt_path = os.path.join(SCRIPT_DIR, GROUND_TRUTH_JSON)
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        ground_truth = json.load(f)

    VIDEOS_TO_MONITOR = [f"attacked_{i}.mp4" for i in range(1, 42)]
    gt_map = {e["target_video"]: e for e in ground_truth}
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner("DISTRIBUTED VIOLENCE DETECTION (WITH AUDIO OVERLAP)")
    print(f"  Architecture: {NUM_WORKERS} workers → 1 leader")
    print(f"  Chunk size: {CHUNK_DURATION_SEC}s")
    print(f"  Audio overlap: {AUDIO_OVERLAP_SEC*1000:.0f}ms (enables stateless flux detection)")
    print(f"  Workers extract raw data to JSON (with audio overlap)")
    print(f"  Leader filters overlap and runs detection on JSON only")
    print(f"  Videos: {len(VIDEOS_TO_MONITOR)}")

    all_results = []
    for vid_idx, video_name in enumerate(VIDEOS_TO_MONITOR):
        video_path = os.path.join(SCRIPT_DIR, ATTACKED_DIR, video_name)
        
        if not os.path.exists(video_path):
            print(f"\n[!] Not found: {video_path}")
            continue
        if video_name not in gt_map:
            print(f"\n[!] No ground truth: {video_name}")
            continue

        print(f"\n{'='*70}")
        print(f"  VIDEO {vid_idx+1}/{len(VIDEOS_TO_MONITOR)}")
        print(f"{'='*70}")

        result = process_video(video_path, gt_map[video_name])
        all_results.append(result)

    if all_results:
        print_overall_summary(all_results)
        report_path = os.path.join(SCRIPT_DIR, REPORT_FILE)
        write_report(report_path, all_results, run_ts)
        print(f"\n[+] Report saved: {report_path}")
    else:
        print("\n[!] No videos processed.")