import subprocess
import json
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
import os
import sys
import time
import datetime
from collections import deque

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ATTACKED_DIR            = "attacked"
GROUND_TRUTH_JSON       = "attack.json"
REPORT_FILE             = "report.txt"

FFMPEG_DIR  = "ffmpeg-master-latest-win64-gpl"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE  = os.path.join(SCRIPT_DIR, FFMPEG_DIR, "bin", "ffmpeg.exe")
FFPROBE_EXE = os.path.join(SCRIPT_DIR, FFMPEG_DIR, "bin", "ffprobe.exe")

# --- Fusion / Scoring ---
BOUNDARY_TOLERANCE      = 2.5
LOCKOUT_PERIOD          = 5.0

# --- Confidence Grid / KDE Fusion ---
GRID_RESOLUTION         = 0.5
FUSION_SIGMA_SEC        = 1.0
SIGMOID_STEEPNESS       = 1.5
PEAK_MIN_GAP_SEC        = 3.0
PEAK_THRESHOLD_PCT      = 0.20

# --- Signal weights ---
SIGNAL_WEIGHTS = {
    "visual"       : 1.0,
    "gop"          : 0.8,
    "audio_rms"    : 1.0,
    "silence"      : 1.2,
    "spectral_flux": 0.7,
}
PB_WEIGHT_BASE          = 0.1
PB_WEIGHT_MAX           = 0.5

# --- Visual ---
VISUAL_MAX_FRAMES       = 30
VISUAL_MIN_WARMUP       = 10
K_VISUAL                = 3.0
VISUAL_COOLDOWN         = 3.0
VISUAL_PERSIST_NEEDED   = 2
VISUAL_PERSIST_WINDOW   = 4

# --- Motion: GOP ---
K_GOP                   = 3.0
GOP_MIN_WARMUP          = 10
GOP_COOLDOWN            = 3.0
GOP_SHORT_RATIO         = 0.25

# --- Motion: P/B ---
MV_WINDOW               = 60
MV_MIN_WARMUP           = 30
K_PB                    = 3.0
PB_COOLDOWN             = 8.0
PB_PERSIST_NEEDED       = 3
PB_PERSIST_WINDOW       = 5
SG_WINDOW               = 11
SG_POLY                 = 2

# --- Audio RMS ---
AUDIO_BUFFER_SEC        = 15.0
AUDIO_MICRO_WINDOW_SEC  = 0.5
K_AUDIO_RMS             = 3.0
AUDIO_NOISE_FLOOR       = 800.0
AUDIO_COOLDOWN          = 3.0
AUDIO_PERSIST_NEEDED    = 2

# --- Silence ---
SILENCE_THRESHOLD_RATIO = 0.05
SILENCE_MIN_DURATION    = 2.0
SILENCE_COOLDOWN        = 30.0
K_SILENCE               = 3.0

# --- Spectral Flux ---
FLUX_WINDOW_SEC         = 0.05
FLUX_BUFFER_SEC         = 10.0
K_FLUX                  = 3.0
FLUX_COOLDOWN           = 3.0
FLUX_PERSIST_NEEDED     = 3

# --- Pass 2: Bhattacharyya ---
BHATT_ENABLE            = True
BHATT_OFFSET_SEC        = 0.5
BHATT_REJECT_BELOW      = 0.05
BHATT_HIST_BINS         = 64

# ==============================================================================
# PROGRESS HELPERS
# ==============================================================================

def hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def progress_bar(label, current, total, width=35, extra=""):
    frac   = min(current / total, 1.0) if total > 0 else 0
    filled = int(width * frac)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = frac * 100
    sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%  {extra}  ")
    sys.stdout.flush()

def section(title):
    print(f"\n{'─'*65}\n  {title}\n{'─'*65}")

def banner(title):
    w = 65
    print("\n" + "═"*w)
    pad = (w - len(title) - 2) // 2
    print(" "*pad + f" {title} ")
    print("═"*w)

# ==============================================================================
# PRE-FLIGHT: Video duration
# ==============================================================================

def get_video_duration(video_path):
    cmd = [
        FFPROBE_EXE, '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        data = json.loads(result.stdout)
        dur  = float(data["format"]["duration"])
        if dur <= 0:
            raise ValueError(f"Non-positive duration: {dur}")
        print(f"  [PRE-FLIGHT] Duration: {hms(dur)}  ({dur:.3f}s)")
        return dur
    except Exception as e:
        print(f"  [PRE-FLIGHT ERROR] Cannot determine video duration: {e}")
        return None

# ==============================================================================
# ROBUST STATISTICS
# ==============================================================================

def mad_zscore(value, buffer):
    arr    = np.array(buffer, dtype=np.float64)
    if len(arr) < 4:
        return 0.0
    median = np.median(arr)
    mad    = np.median(np.abs(arr - median))
    if mad < 1e-9:
        return 0.0
    return float((value - median) / (0.6745 * mad))


def predict_next_pb(value, buffer):
    arr = np.array(buffer, dtype=np.float64)
    if len(arr) < 4:
        return 0.0
    win = min(SG_WINDOW, len(arr))
    if win % 2 == 0:
        win -= 1
    win = max(win, SG_POLY + 2)
    try:
        smoothed = savgol_filter(arr, window_length=win, polyorder=SG_POLY)
    except Exception:
        smoothed = arr
    x         = np.arange(len(smoothed))
    coeffs    = np.polyfit(x, smoothed, 1)
    trend     = np.polyval(coeffs, x)
    residuals = smoothed - trend
    predicted = np.polyval(coeffs, len(smoothed))
    residual  = value - predicted
    mad       = np.median(np.abs(residuals - np.median(residuals)))
    if mad < 1e-9:
        return 0.0
    return float(residual / (0.6745 * mad))


def sigmoid_confidence(z, k):
    return float(1.0 / (1.0 + np.exp(-SIGMOID_STEEPNESS * (abs(z) - k))))

# ==============================================================================
# CONFIDENCE GRID
# ==============================================================================

def to_confidence_grid(events, n_bins):
    grid = np.zeros(n_bins, dtype=np.float64)
    for t, z, k in events:
        idx = int(t / GRID_RESOLUTION)
        if 0 <= idx < n_bins:
            conf      = sigmoid_confidence(z, k)
            grid[idx] = max(grid[idx], conf)
    return grid

# ==============================================================================
# SIGNAL 1: Visual — I-frame MAD Z-score
# ==============================================================================

def get_visual_events(video_path):
    cmd = [
        FFPROBE_EXE, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time,pts_time,pkt_size,size,pict_type,key_frame',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        frames = json.loads(result.stdout).get('frames', [])
    except Exception:
        return []

    i_frames = [f for f in frames
                if f.get('pict_type') == 'I' or f.get('key_frame') == 1]
    total    = len(i_frames)

    events            = []
    size_buffer       = deque(maxlen=VISUAL_MAX_FRAMES)
    spike_flags       = deque(maxlen=VISUAL_PERSIST_WINDOW)
    last_trigger_time = -999.0

    for idx, frame in enumerate(i_frames):
        size     = float(frame.get('pkt_size') or frame.get('size') or 0)
        time_pts = float(frame.get('pkt_pts_time') or frame.get('pts_time') or -1)
        if time_pts < 0:
            continue

        if idx % max(1, total // 20) == 0:
            progress_bar("Visual", idx, total, extra=f"{hms(time_pts)}")

        z        = 0.0
        is_spike = False
        if time_pts > LOCKOUT_PERIOD and len(size_buffer) >= VISUAL_MIN_WARMUP:
            z        = mad_zscore(size, size_buffer)
            is_spike = abs(z) > K_VISUAL

        spike_flags.append(1 if is_spike else 0)

        if (time_pts > LOCKOUT_PERIOD
                and len(spike_flags) == VISUAL_PERSIST_WINDOW
                and sum(spike_flags) >= VISUAL_PERSIST_NEEDED
                and (time_pts - last_trigger_time) > VISUAL_COOLDOWN):
            events.append((time_pts, z, K_VISUAL))
            last_trigger_time = time_pts
            spike_flags.clear()

        size_buffer.append(size)

    progress_bar("Visual", total, total, extra="done")
    print()
    return events

# ==============================================================================
# SIGNAL 2: Motion — GOP (MAD) + P/B (SG + regression + MAD)
# ==============================================================================

def get_motion_events(video_path):
    cmd = [
        FFPROBE_EXE, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time,pts_time,pkt_size,size,pict_type',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        frames = json.loads(result.stdout).get('frames', [])
    except Exception:
        return [], []

    total         = len(frames)
    gop_events    = []
    pb_events     = []
    pb_buffer     = deque(maxlen=MV_WINDOW)
    spike_flags   = deque(maxlen=PB_PERSIST_WINDOW)
    last_gop_time = -999.0
    last_pb_time  = -999.0
    last_i_time   = -999.0
    gop_intervals = deque(maxlen=30)

    for idx, frame in enumerate(frames):
        ptype    = frame.get('pict_type', 'U')
        size     = float(frame.get('pkt_size') or frame.get('size') or 0)
        time_pts = float(frame.get('pkt_pts_time') or frame.get('pts_time') or -1)
        if time_pts < 0 or size == 0:
            continue

        if idx % max(1, total // 20) == 0:
            progress_bar("Motion", idx, total, extra=f"{hms(time_pts)}")

        if ptype == 'I':
            if last_i_time > 0:
                interval = time_pts - last_i_time
                if len(gop_intervals) >= GOP_MIN_WARMUP:
                    z = mad_zscore(interval, gop_intervals)
                    if (time_pts > LOCKOUT_PERIOD
                            and z < -K_GOP
                            and (time_pts - last_gop_time) > GOP_COOLDOWN):
                        gop_events.append((time_pts, abs(z), K_GOP))
                        last_gop_time = time_pts
                gop_intervals.append(interval)
            last_i_time = time_pts

        if ptype in ('P', 'B'):
            z        = 0.0
            is_spike = False
            if time_pts > LOCKOUT_PERIOD and len(pb_buffer) >= MV_MIN_WARMUP:
                z        = predict_next_pb(size, pb_buffer)
                is_spike = abs(z) > K_PB

            spike_flags.append(1 if is_spike else 0)

            if (time_pts > LOCKOUT_PERIOD
                    and len(spike_flags) == PB_PERSIST_WINDOW
                    and sum(spike_flags) >= PB_PERSIST_NEEDED
                    and (time_pts - last_pb_time) > PB_COOLDOWN):
                pb_events.append((time_pts, z, K_PB))
                last_pb_time = time_pts
                spike_flags.clear()

            pb_buffer.append(size)

    # Deduplicate close GOP + P/B events
    all_mv  = sorted(gop_events + pb_events, key=lambda x: x[0])
    deduped_gop, deduped_pb = [], []
    last = -999.0
    for ev in all_mv:
        if ev[0] - last > min(GOP_COOLDOWN, PB_COOLDOWN):
            if ev in gop_events:
                deduped_gop.append(ev)
            else:
                deduped_pb.append(ev)
            last = ev[0]

    progress_bar("Motion", total, total, extra="done")
    print()
    return deduped_gop, deduped_pb

# ==============================================================================
# AUDIO EXTRACTION
# ==============================================================================

def extract_audio(video_path, sample_rate=44100, tag="tmp"):
    temp_wav = os.path.join(SCRIPT_DIR, f"_audio_{tag}.wav")
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    for exe in [FFMPEG_EXE, "ffmpeg"]:
        try:
            subprocess.run(
                [exe, '-y', '-i', video_path, '-vn',
                 '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', '1', temp_wav],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                break
        except FileNotFoundError:
            continue
    if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
        return None, None
    try:
        sr, data = wavfile.read(temp_wav)
        os.remove(temp_wav)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return sr, data.astype(np.float64)
    except Exception:
        return None, None

# ==============================================================================
# SIGNAL 3 + 4: Audio RMS (MAD) + Silence
# ==============================================================================

def get_audio_rms_events(video_path):
    sample_rate, data = extract_audio(video_path, sample_rate=44100, tag="rms")
    if data is None:
        print("    [!] Audio extraction failed")
        return [], []

    micro_samples    = int(AUDIO_MICRO_WINDOW_SEC * sample_rate)
    bg_chunks_needed = int(AUDIO_BUFFER_SEC / AUDIO_MICRO_WINDOW_SEC)
    total_chunks     = len(data) // micro_samples

    spike_events   = []
    silence_events = []
    rms_buffer     = deque(maxlen=bg_chunks_needed)
    consec_spikes  = 0
    in_silence     = False
    silence_start  = -999.0
    last_spike_t   = -999.0
    last_sil_t     = -999.0

    for i in range(total_chunks):
        chunk     = data[i * micro_samples : (i + 1) * micro_samples]
        micro_rms = float(np.sqrt(np.mean(chunk ** 2))) if np.any(chunk) else 0.0
        time_pts  = (i * micro_samples) / sample_rate

        if i % max(1, total_chunks // 20) == 0:
            progress_bar("AudioRMS", i, total_chunks, extra=f"{hms(time_pts)}")

        if time_pts > LOCKOUT_PERIOD and len(rms_buffer) == bg_chunks_needed:
            mu        = float(np.mean(rms_buffer))
            z         = mad_zscore(micro_rms, rms_buffer)
            is_spike  = (micro_rms > AUDIO_NOISE_FLOOR and z > K_AUDIO_RMS)
            consec_spikes = consec_spikes + 1 if is_spike else 0
            if (consec_spikes >= AUDIO_PERSIST_NEEDED
                    and (time_pts - last_spike_t) > AUDIO_COOLDOWN):
                spike_events.append((time_pts, z, K_AUDIO_RMS))
                last_spike_t  = time_pts
                consec_spikes = 0

            if mu > AUDIO_NOISE_FLOOR:
                silence_z = mad_zscore(micro_rms, rms_buffer)
                is_silent = micro_rms < mu * SILENCE_THRESHOLD_RATIO
                if is_silent and not in_silence:
                    in_silence    = True
                    silence_start = time_pts
                elif not is_silent and in_silence:
                    in_silence = False
                    duration   = time_pts - silence_start
                    if (duration >= SILENCE_MIN_DURATION
                            and (silence_start - last_sil_t) > SILENCE_COOLDOWN):
                        silence_events.append((silence_start, abs(silence_z), K_SILENCE))
                        last_sil_t = silence_start

        rms_buffer.append(micro_rms)

    progress_bar("AudioRMS", total_chunks, total_chunks, extra="done")
    print()
    return spike_events, silence_events

# ==============================================================================
# SIGNAL 5: Spectral Flux (MAD)
# ==============================================================================

def get_spectral_flux_events(video_path):
    sample_rate, data = extract_audio(video_path, sample_rate=22050, tag="flux")
    if data is None:
        print("    [!] Audio extraction failed")
        return []

    hop_samples      = max(1, int(FLUX_WINDOW_SEC * sample_rate))
    n_fft            = hop_samples * 2
    bg_chunks_needed = max(10, int(FLUX_BUFFER_SEC / FLUX_WINDOW_SEC))
    hann_win         = np.hanning(n_fft)
    total_chunks     = len(data) // hop_samples

    events            = []
    flux_buffer       = deque(maxlen=bg_chunks_needed)
    consec_flux       = 0
    last_trigger_time = -999.0
    prev_spectrum     = None

    for i in range(total_chunks):
        start    = i * hop_samples
        chunk    = data[start : start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        time_pts = start / sample_rate

        if i % max(1, total_chunks // 20) == 0:
            progress_bar("SpectFlux", i, total_chunks, extra=f"{hms(time_pts)}")

        spectrum = np.abs(np.fft.rfft(chunk * hann_win))
        norm     = np.linalg.norm(spectrum)
        if norm > 1e-9:
            spectrum = spectrum / norm

        if prev_spectrum is not None:
            flux = float(np.sum(np.maximum(spectrum - prev_spectrum, 0.0)))
            if (time_pts > LOCKOUT_PERIOD
                    and len(flux_buffer) >= int(bg_chunks_needed * 0.3)):
                z           = mad_zscore(flux, flux_buffer)
                is_spike    = z > K_FLUX
                consec_flux = consec_flux + 1 if is_spike else 0
                if (consec_flux >= FLUX_PERSIST_NEEDED
                        and (time_pts - last_trigger_time) > FLUX_COOLDOWN):
                    events.append((time_pts, z, K_FLUX))
                    last_trigger_time = time_pts
                    consec_flux       = 0
            flux_buffer.append(flux)

        prev_spectrum = spectrum

    progress_bar("SpectFlux", total_chunks, total_chunks, extra="done")
    print()
    return events

# ==============================================================================
# PASS 2: Bhattacharyya Histogram Verification
# ==============================================================================

def decode_frame_hue_histogram(video_path, timestamp, bins=BHATT_HIST_BINS):
    cmd = [
        FFMPEG_EXE, '-ss', str(timestamp), '-i', video_path,
        '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=10)
        raw = result.stdout
        if not raw:
            return None
        total_pixels = len(raw) // 3
        if total_pixels == 0:
            return None
        side   = int(np.sqrt(total_pixels))
        pixels = np.frombuffer(raw[:side * side * 3],
                               dtype=np.uint8).reshape(-1, 3)
        r = pixels[:, 0].astype(np.float32)
        g = pixels[:, 1].astype(np.float32)
        b = pixels[:, 2].astype(np.float32)

        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        diff = maxc - minc + 1e-9

        hue      = np.zeros(len(r), dtype=np.float32)
        mask_r   = (maxc == r)
        mask_g   = (maxc == g)
        mask_b   = (maxc == b)
        hue[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r])) % 360
        hue[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        hue[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360

        hist, _ = np.histogram(hue, bins=bins, range=(0, 360))
        hist    = hist.astype(np.float64)
        total   = hist.sum()
        if total > 0:
            hist /= total
        return hist
    except Exception:
        return None


def bhattacharyya_distance(h1, h2):
    bc = np.sum(np.sqrt(h1 * h2 + 1e-12))
    return float(-np.log(bc + 1e-12))


def verify_candidates_bhattacharyya(video_path, candidates):
    if not BHATT_ENABLE:
        return [(t, l, n, c, -1.0) for t, l, n, c in candidates], []

    verified = []
    rejected = []
    print(f"\n  [Pass 2] Bhattacharyya verification on {len(candidates)} candidates...")

    for ts, label, n_sig, confidence in candidates:
        t_before = max(0.0, ts - BHATT_OFFSET_SEC)
        t_after  = ts + BHATT_OFFSET_SEC
        h1 = decode_frame_hue_histogram(video_path, t_before)
        h2 = decode_frame_hue_histogram(video_path, t_after)

        if h1 is None or h2 is None:
            verified.append((ts, label, n_sig, confidence, -1.0))
            continue

        dist = bhattacharyya_distance(h1, h2)
        if dist < BHATT_REJECT_BELOW:
            rejected.append((ts, label, n_sig, confidence, dist))
        else:
            verified.append((ts, label, n_sig, confidence, dist))

    print(f"  [Pass 2] Kept: {len(verified)}  Rejected: {len(rejected)}")
    return verified, rejected

# ==============================================================================
# GATED GAUSSIAN FUSION (KDE)
# ==============================================================================

def fuse_all_signals_kde(event_dict, video_duration):
    n_bins     = int(video_duration / GRID_RESOLUTION) + 1
    sigma_bins = FUSION_SIGMA_SEC / GRID_RESOLUTION

    grids    = {name: to_confidence_grid(events, n_bins)
                for name, events in event_dict.items()}
    smoothed = {name: gaussian_filter1d(g, sigma=sigma_bins)
                for name, g in grids.items()}

    # T-conorm audio gate
    rms_s      = smoothed.get("audio_rms",     np.zeros(n_bins))
    flux_s     = smoothed.get("spectral_flux", np.zeros(n_bins))
    audio_gate = rms_s + flux_s - (rms_s * flux_s)
    audio_gate = np.clip(audio_gate, 0.0, 1.0)

    fused           = np.zeros(n_bins, dtype=np.float64)
    active_capacity = 0.0

    for name, s_grid in smoothed.items():
        if name == "motion_pb":
            dynamic_weight   = PB_WEIGHT_BASE + (PB_WEIGHT_MAX - PB_WEIGHT_BASE) * audio_gate
            fused           += s_grid * dynamic_weight
            active_capacity += PB_WEIGHT_MAX if np.any(s_grid > 0) else 0.0
        else:
            w                = SIGNAL_WEIGHTS.get(name, 1.0)
            fused           += s_grid * w
            active_capacity += w if np.any(s_grid > 0) else 0.0

    tau = max(0.05, PEAK_THRESHOLD_PCT * active_capacity)

    min_gap_bins = int(PEAK_MIN_GAP_SEC / GRID_RESOLUTION)
    peak_indices, _ = find_peaks(fused, height=tau, distance=min_gap_bins)

    candidates = []
    for idx in peak_indices:
        ts           = round(idx * GRID_RESOLUTION, 2)
        confidence   = float(fused[idx])
        contributing = [name for name, s_grid in smoothed.items()
                        if s_grid[idx] > 0.01]
        label        = "+".join(sorted(contributing)) if contributing else "unknown"
        n_signals    = len(contributing)
        candidates.append((ts, label, n_signals, confidence))

    return candidates, fused

# ==============================================================================
# VALIDATE ONE VIDEO
# ==============================================================================

def classify_hit_position(ts, seg):
    near_start = abs(ts - seg["start"]) <= BOUNDARY_TOLERANCE
    near_end   = abs(ts - seg["end"])   <= BOUNDARY_TOLERANCE
    inside     = seg["start"] < ts < seg["end"]

    if near_start and not near_end:
        return "START"
    elif near_end and not near_start:
        return "END"
    elif near_start and near_end:
        return "START+END"
    elif inside:
        return "IN-BETWEEN"
    return None


def validate_video(video_name, segments, verified_candidates, rejected_candidates):
    seg_results = []
    fp_events   = []
    segs        = [dict(s, hit=None) for s in segments]

    for ts, label, n_sig, confidence, bhatt_dist in verified_candidates:
        claimed = False
        for seg in segs:
            pos = classify_hit_position(ts, seg)
            if pos is not None and seg["hit"] is None:
                seg["hit"] = {
                    "ts": ts, "label": label, "n_sig": n_sig,
                    "position": pos, "confidence": confidence,
                    "bhatt_dist": bhatt_dist
                }
                claimed = True
                break
        if not claimed:
            fp_events.append({
                "ts": ts, "label": label, "n_sig": n_sig,
                "confidence": confidence, "bhatt_dist": bhatt_dist
            })

    for seg in segs:
        seg_results.append({
            "index"                : seg["index"],
            "source"               : seg["source"],
            "start"                : seg["start"],
            "end"                  : seg["end"],
            "duration"             : seg["duration"],
            "original_insert_point": seg.get("original_insert_point", seg["start"]),
            "start_frame"          : seg.get("start_frame", None),
            "end_frame"            : seg.get("end_frame",   None),
            "hit"                  : seg["hit"],
        })

    return seg_results, fp_events, rejected_candidates

# ==============================================================================
# PRINT VIDEO REPORT
# ==============================================================================

def confidence_str(n, conf=None):
    if conf is not None:
        return f"{conf:.3f}"
    if n >= 3: return "HIGH"
    if n == 2: return "MED "
    return "LOW "


def print_video_report(video_name, seg_results, fp_events,
                       signal_dict, verified, rejected):
    hits  = sum(1 for s in seg_results if s["hit"] is not None)
    total = len(seg_results)

    section(f"VIDEO: {video_name}   [{hits}/{total} segments detected]")

    print(f"  Raw events per signal:")
    for sig, evs in signal_dict.items():
        bar = "▪" * min(len(evs), 40)
        print(f"    {sig:<15} {len(evs):>4} events  {bar}")

    print(f"  Fused candidates : {len(verified) + len(rejected)}")
    print(f"  Pass 2 verified  : {len(verified)}  rejected: {len(rejected)}")

    print(f"\n  {'#':>3}  {'SOURCE':<16}  {'ORIG_PT':>8}  {'START':>8}  "
          f"{'END':>8}  {'DUR':>5}  {'HIT @':>8}  {'POS':<12}  "
          f"{'CONF':>6}  {'BHATT':>6}  SIGNALS")
    print(f"  {'─'*3}  {'─'*16}  {'─'*8}  {'─'*8}  "
          f"{'─'*8}  {'─'*5}  {'─'*8}  {'─'*12}  "
          f"{'─'*6}  {'─'*6}  {'─'*30}")

    for s in seg_results:
        src  = s["source"][:15]
        orig = s.get("original_insert_point", s["start"])
        if s["hit"]:
            h  = s["hit"]
            bd = f"{h['bhatt_dist']:.3f}" if h['bhatt_dist'] >= 0 else "N/A"
            print(f"  {s['index']:>3}  {src:<16}  {orig:>8.3f}  {s['start']:>8.3f}  "
                  f"{s['end']:>8.3f}  {s['duration']:>5.2f}  {h['ts']:>8.2f}  "
                  f"✓ {h['position']:<10}  {h['confidence']:>6.3f}  "
                  f"{bd:>6}  {h['label'][:30]}")
        else:
            print(f"  {s['index']:>3}  {src:<16}  {orig:>8.3f}  {s['start']:>8.3f}  "
                  f"{s['end']:>8.3f}  {s['duration']:>5.2f}  {'—':>8}  "
                  f"✗ {'MISSED':<10}  {'':>6}  {'':>6}")

    if rejected:
        print(f"\n  Pass 2 Rejected ({len(rejected)}):")
        for ts, label, n, conf, bd in rejected:
            print(f"    @ {ts:>9.2f}s  dist={bd:.4f}  [{label}]  conf={conf:.3f}")

    if fp_events:
        print(f"\n  Remaining False Positives ({len(fp_events)}):")
        for fp in fp_events:
            bd = f"{fp['bhatt_dist']:.3f}" if fp['bhatt_dist'] >= 0 else "N/A"
            print(f"    @ {fp['ts']:>9.2f}s  [{fp['label']}]  "
                  f"conf={fp['confidence']:.3f}  bhatt={bd}")
    else:
        print(f"\n  False Positives: 0  ✓")

    rate = hits / total * 100 if total else 0
    print(f"\n  Detection rate: {hits}/{total}  ({rate:.1f}%)   |   "
          f"FP: {len(fp_events)}  |  Pass2 rejected: {len(rejected)}")

# ==============================================================================
# OVERALL SUMMARY
# ==============================================================================

def print_overall_summary(all_video_stats):
    banner("COMPLETE EVALUATION SUMMARY")

    total_segs = sum(v["total"]    for v in all_video_stats)
    total_hits = sum(v["hits"]     for v in all_video_stats)
    total_fp   = sum(v["fp"]       for v in all_video_stats)
    total_rej  = sum(v["rejected"] for v in all_video_stats)
    total_vids = len(all_video_stats)
    overall    = total_hits / total_segs * 100 if total_segs else 0

    print(f"\n  {'VIDEO':<35}  {'N':>2}  {'SEGS':>4}  {'HITS':>4}  "
          f"{'RATE':>6}  {'FP':>4}  {'REJ':>4}  STATUS")
    print(f"  {'─'*35}  {'─'*2}  {'─'*4}  {'─'*4}  "
          f"{'─'*6}  {'─'*4}  {'─'*4}  {'─'*8}")
    for v in all_video_stats:
        rate   = v["hits"] / v["total"] * 100 if v["total"] else 0
        status = "✓ PASS" if rate >= 50 else "✗ FAIL"
        print(f"  {v['name']:<35}  {v['n_inserts']:>2}  {v['total']:>4}  "
              f"{v['hits']:>4}  {rate:>5.1f}%  {v['fp']:>4}  "
              f"{v['rejected']:>4}  {status}")
    print(f"  {'─'*35}  {'─'*2}  {'─'*4}  {'─'*4}  "
          f"{'─'*6}  {'─'*4}  {'─'*4}  {'─'*8}")
    print(f"  {'TOTAL':<35}  {'':>2}  {total_segs:>4}  {total_hits:>4}  "
          f"{overall:>5.1f}%  {total_fp:>4}  {total_rej:>4}")

    print(f"\n  Pass 2 rejected {total_rej} candidates across all videos")

    print(f"\n  Hit Position Breakdown:")
    pos_counts = {}
    for v in all_video_stats:
        for p in v["positions"]:
            pos_counts[p] = pos_counts.get(p, 0) + 1
    for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"    {pos:<14}: {cnt:>4}  {'█' * cnt}")

    print(f"\n  Per-Signal Contribution:")
    sig_hits, sig_fp = {}, {}
    for v in all_video_stats:
        for sig, cnt in v["sig_hits"].items():
            sig_hits[sig] = sig_hits.get(sig, 0) + cnt
        for sig, cnt in v["sig_fp"].items():
            sig_fp[sig] = sig_fp.get(sig, 0) + cnt

    all_sigs = sorted(set(list(sig_hits.keys()) + list(sig_fp.keys())))
    print(f"  {'Signal':<15}  {'Hits':>5}  {'FP':>6}  Precision")
    print(f"  {'─'*15}  {'─'*5}  {'─'*6}  {'─'*9}")
    for sig in all_sigs:
        h    = sig_hits.get(sig, 0)
        fp   = sig_fp.get(sig, 0)
        tot  = h + fp
        prec = h / tot * 100 if tot > 0 else 0.0
        print(f"  {sig:<15}  {h:>5}  {fp:>6}  {prec:>8.1f}%")

    print(f"\n{'═'*65}")
    print(f"  OVERALL DETECTION RATE   : {total_hits:>4} / {total_segs}  ({overall:.2f}%)")
    print(f"  TOTAL FALSE POSITIVES    : {total_fp}")
    print(f"  PASS 2 REJECTIONS        : {total_rej}")
    fpr = total_fp / (total_fp + total_hits) * 100 if (total_fp + total_hits) > 0 else 0
    print(f"  FALSE POSITIVE RATE      : {fpr:.2f}%")
    print(f"  VIDEOS PROCESSED         : {total_vids}")
    print(f"{'═'*65}\n")

# ==============================================================================
# REPORT WRITER
# ==============================================================================

W = 80

def rl(f, text=""):
    f.write(text + "\n")

def rheader(f, title, char="="):
    f.write("\n" + char * W + "\n")
    pad = max(0, (W - len(title)) // 2)
    f.write(" " * pad + title + "\n")
    f.write(char * W + "\n")

def rsection(f, title, char="-"):
    f.write("\n" + char * W + "\n")
    f.write("  " + title + "\n")
    f.write(char * W + "\n")

def rbar(f, label, value, max_val, width=40, unit=""):
    filled = int(width * min(value / max_val, 1.0)) if max_val > 0 else 0
    bar    = "#" * filled + "." * (width - filled)
    f.write(f"  {label:<22} [{bar}] {value}{unit}\n")


def write_full_report(report_path, all_video_data, run_ts):
    with open(report_path, "a", encoding="utf-8") as f:

        rheader(f, "MULTI-MODAL BOUNDARY DETECTOR  v3.0  -  EVALUATION REPORT")
        rl(f)
        rl(f, f"  Run timestamp      : {run_ts}")
        rl(f, f"  Ground truth file  : {GROUND_TRUTH_JSON}")
        rl(f, f"  Attacked folder    : {ATTACKED_DIR}/")
        rl(f, f"  Videos processed   : {len(all_video_data)}")
        rl(f, f"  Report file        : {report_path}")
        rl(f)
        rl(f, f"  Signals : Visual(MAD) | Motion(GOP+PB) | "
               f"AudioRMS(MAD) | Silence | SpectralFlux(MAD)")
        rl(f, f"  Fusion  : KDE (Gaussian) + T-conorm audio gate + Hadamard P/B gate")
        rl(f, f"  Pass 2  : Bhattacharyya HSV histogram verification")
        rl(f)
        rl(f, f"  Scoring : HIT = any fused+verified event within "
               f"[start-{BOUNDARY_TOLERANCE}s .. end+{BOUNDARY_TOLERANCE}s]")
        rl(f, f"  Positions: START | END | IN-BETWEEN | START+END")

        rheader(f, "CONFIGURATION", char="-")
        params = [
            ("BOUNDARY_TOLERANCE",      BOUNDARY_TOLERANCE),
            ("LOCKOUT_PERIOD",          LOCKOUT_PERIOD),
            ("GRID_RESOLUTION",         GRID_RESOLUTION),
            ("FUSION_SIGMA_SEC",        FUSION_SIGMA_SEC),
            ("SIGMOID_STEEPNESS",       SIGMOID_STEEPNESS),
            ("PEAK_MIN_GAP_SEC",        PEAK_MIN_GAP_SEC),
            ("PEAK_THRESHOLD_PCT",      PEAK_THRESHOLD_PCT),
            ("PB_WEIGHT_BASE",          PB_WEIGHT_BASE),
            ("PB_WEIGHT_MAX",           PB_WEIGHT_MAX),
            ("--- Visual ---",          ""),
            ("K_VISUAL",                K_VISUAL),
            ("VISUAL_COOLDOWN",         VISUAL_COOLDOWN),
            ("VISUAL_PERSIST_NEEDED",   VISUAL_PERSIST_NEEDED),
            ("--- Motion ---",          ""),
            ("K_GOP",                   K_GOP),
            ("GOP_SHORT_RATIO",         GOP_SHORT_RATIO),
            ("K_PB",                    K_PB),
            ("PB_COOLDOWN",             PB_COOLDOWN),
            ("SG_WINDOW",               SG_WINDOW),
            ("SG_POLY",                 SG_POLY),
            ("--- Audio RMS ---",       ""),
            ("K_AUDIO_RMS",             K_AUDIO_RMS),
            ("AUDIO_NOISE_FLOOR",       AUDIO_NOISE_FLOOR),
            ("AUDIO_COOLDOWN",          AUDIO_COOLDOWN),
            ("--- Silence ---",         ""),
            ("SILENCE_THRESHOLD_RATIO", SILENCE_THRESHOLD_RATIO),
            ("SILENCE_MIN_DURATION",    SILENCE_MIN_DURATION),
            ("SILENCE_COOLDOWN",        SILENCE_COOLDOWN),
            ("--- Spectral Flux ---",   ""),
            ("K_FLUX",                  K_FLUX),
            ("FLUX_COOLDOWN",           FLUX_COOLDOWN),
            ("FLUX_PERSIST_NEEDED",     FLUX_PERSIST_NEEDED),
            ("--- Pass 2 ---",          ""),
            ("BHATT_ENABLE",            BHATT_ENABLE),
            ("BHATT_OFFSET_SEC",        BHATT_OFFSET_SEC),
            ("BHATT_REJECT_BELOW",      BHATT_REJECT_BELOW),
            ("BHATT_HIST_BINS",         BHATT_HIST_BINS),
        ]
        rl(f, f"  {'Parameter':<35}  Value")
        rl(f, f"  {'─'*35}  {'─'*20}")
        for k, v in params:
            if v == "":
                rl(f, f"  {k}")
            else:
                rl(f, f"  {k:<35}  {v}")

        rheader(f, "PER-VIDEO DETAILED RESULTS")

        for vd in all_video_data:
            vname     = vd["name"]
            seg_res   = vd["seg_results"]
            fp_events = vd["fp_events"]
            sig_dict  = vd["signal_dict"]
            merged    = vd["merged_events"]
            elapsed   = vd["elapsed"]

            hits  = sum(1 for s in seg_res if s["hit"])
            total = len(seg_res)
            rate  = hits / total * 100 if total else 0
            fp_n  = len(fp_events)
            fpr_v = fp_n / (fp_n + hits) * 100 if (fp_n + hits) > 0 else 0.0

            rsection(f, f"VIDEO: {vname}")
            rl(f)
            rl(f, f"  Carrier source     : {vd.get('carrier_source', 'unknown')}")
            rl(f, f"  Carrier duration   : {vd.get('carrier_duration', '?')}s")
            rl(f, f"  Attacked duration  : {vd.get('attacked_duration', '?')}s")
            rl(f, f"  Resolution         : {vd.get('resolution', 'unknown')}")
            rl(f, f"  FPS                : {vd.get('fps', '?')}")
            rl(f, f"  N inserts          : {vd.get('n_inserts', '?')}")
            rl(f)
            rl(f, f"  Result             : {hits}/{total} detected  ({rate:.1f}%)")
            rl(f, f"  False positives    : {fp_n}  (video FPR: {fpr_v:.1f}%)")
            rl(f, f"  Processing time    : {elapsed:.1f}s")
            rl(f)

            # Insert drift table
            rl(f, "  INSERT DRIFT ANALYSIS:")
            rl(f, f"  {'#':>3}  {'SOURCE':<16}  {'ORIG_PT':>10}  "
                   f"{'ACT_START':>10}  {'DRIFT':>8}  {'DUR':>6}  "
                   f"{'SF':>8}  {'EF':>8}")
            rl(f, f"  {'─'*3}  {'─'*16}  {'─'*10}  "
                   f"{'─'*10}  {'─'*8}  {'─'*6}  "
                   f"{'─'*8}  {'─'*8}")
            for s in seg_res:
                orig  = s.get("original_insert_point", s["start"])
                drift = s["start"] - orig
                sf    = s.get("start_frame", "?")
                ef    = s.get("end_frame",   "?")
                rl(f, f"  {s['index']:>3}  {s['source']:<16}  {orig:>10.3f}  "
                       f"{s['start']:>10.3f}  {drift:>+8.3f}  "
                       f"{s['duration']:>6.3f}  {str(sf):>8}  {str(ef):>8}")
            rl(f)

            # Raw trigger counts
            rl(f, "  RAW TRIGGER COUNTS PER SIGNAL:")
            max_raw = max((len(v) for v in sig_dict.values()), default=1)
            for sig, ts_list in sig_dict.items():
                rbar(f, sig, len(ts_list), max(max_raw, 1),
                     width=35, unit=" triggers")
            rl(f, f"  {'':22}   Fused events: {len(merged)}")
            rl(f)

            # All raw timestamps per signal
            rl(f, "  ALL RAW TRIGGER TIMESTAMPS PER SIGNAL:")
            for sig, ts_list in sig_dict.items():
                times_str = ("  ".join(f"{t:.3f}s" for t in sorted(ts_list))
                             if ts_list else "(none)")
                rl(f, f"  {sig:<15} ({len(ts_list):>4}): {times_str}")
            rl(f)

            # All fused events
            rl(f, "  ALL FUSED EVENTS (chronological):")
            rl(f, f"  {'TIME':>10}  {'SIGNALS':<35}  {'N':>2}  OUTCOME")
            rl(f, f"  {'─'*10}  {'─'*35}  {'─'*2}  {'─'*25}")
            hit_lookup = {}
            for s in seg_res:
                if s["hit"]:
                    hit_lookup[round(s["hit"]["ts"], 2)] = (
                        s["index"], s["hit"]["position"]
                    )
            for ts, label, n_sig in merged:
                key = round(ts, 2)
                if key in hit_lookup:
                    idx, pos = hit_lookup[key]
                    outcome  = f"HIT  seg {idx:>2}  pos={pos}"
                else:
                    outcome = "FALSE POSITIVE"
                rl(f, f"  {ts:>10.3f}s  {label:<35}  {n_sig:>2}  {outcome}")
            rl(f)

            # Segment detection table
            rl(f, "  SEGMENT DETECTION TABLE:")
            rl(f, f"  {'#':>3}  {'SOURCE':<16}  {'ORIG_PT':>9}  "
                   f"{'START':>9}  {'END':>9}  {'DUR':>7}  "
                   f"{'HIT @':>9}  {'OFFSET':>9}  {'POS':<12}  "
                   f"{'CONF':>6}  {'BHATT':>6}  SIGNALS")
            rl(f, f"  {'─'*3}  {'─'*16}  {'─'*9}  "
                   f"{'─'*9}  {'─'*9}  {'─'*7}  "
                   f"{'─'*9}  {'─'*9}  {'─'*12}  "
                   f"{'─'*6}  {'─'*6}  {'─'*25}")
            for s in seg_res:
                src  = s["source"][:15]
                orig = s.get("original_insert_point", s["start"])
                if s["hit"]:
                    h      = s["hit"]
                    offset = h["ts"] - s["start"]
                    bd     = f"{h['bhatt_dist']:.3f}" if h['bhatt_dist'] >= 0 else "N/A"
                    conf   = f"{h['confidence']:.3f}"
                    rl(f, f"  {s['index']:>3}  {src:<16}  {orig:>9.3f}  "
                           f"{s['start']:>9.3f}  {s['end']:>9.3f}  "
                           f"{s['duration']:>7.3f}  {h['ts']:>9.3f}  "
                           f"{offset:>+9.3f}  {h['position']:<12}  "
                           f"{conf:>6}  {bd:>6}  {h['label'][:25]}")
                else:
                    rl(f, f"  {s['index']:>3}  {src:<16}  {orig:>9.3f}  "
                           f"{s['start']:>9.3f}  {s['end']:>9.3f}  "
                           f"{s['duration']:>7.3f}  {'---':>9}  "
                           f"{'---':>9}  {'MISSED':<12}  "
                           f"{'':>6}  {'':>6}")
            rl(f)

            # Missed segment analysis
            missed = [s for s in seg_res if not s["hit"]]
            if missed:
                rl(f, f"  MISSED SEGMENTS ({len(missed)}):")
                for s in missed:
                    mid = (s["start"] + s["end"]) / 2.0
                    if merged:
                        closest    = min(merged, key=lambda e: abs(e[0] - mid))
                        dist_start = abs(closest[0] - s["start"])
                        dist_mid   = abs(closest[0] - mid)
                        dist_end   = abs(closest[0] - s["end"])
                        rl(f, f"  Seg {s['index']:>2}  src={s['source']}  "
                               f"{s['start']:.3f}s-{s['end']:.3f}s  "
                               f"dur={s['duration']:.3f}s")
                        rl(f, f"    Closest fused : {closest[0]:.3f}s  "
                               f"signals={closest[1]}")
                        rl(f, f"    Dist start    : {dist_start:.3f}s  "
                               f"mid: {dist_mid:.3f}s  end: {dist_end:.3f}s")
                        rl(f, f"    Tolerance     : {BOUNDARY_TOLERANCE}s  -> "
                               f"{'JUST MISSED' if dist_start < BOUNDARY_TOLERANCE*2 else 'FAR MISS'}")
                    else:
                        rl(f, f"  Seg {s['index']:>2}  (no fused events in video)")
                rl(f)
            else:
                rl(f, "  MISSED SEGMENTS: none")
                rl(f)

            # False positive detail
            if fp_events:
                rl(f, f"  FALSE POSITIVE EVENTS ({fp_n}):")
                rl(f, f"  {'#':>4}  {'TIME':>10}  {'CONF':>6}  "
                       f"{'BHATT':>6}  SIGNALS")
                rl(f, f"  {'─'*4}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*30}")
                for i, fp in enumerate(fp_events, 1):
                    bd = f"{fp['bhatt_dist']:.3f}" if fp['bhatt_dist'] >= 0 else "N/A"
                    rl(f, f"  {i:>4}  {fp['ts']:>10.3f}s  "
                           f"{fp['confidence']:>6.3f}  {bd:>6}  {fp['label']}")
                if len(fp_events) > 1:
                    gaps = [fp_events[i+1]["ts"] - fp_events[i]["ts"]
                            for i in range(len(fp_events) - 1)]
                    rl(f)
                    rl(f, f"  FP spacing: min={min(gaps):.3f}s  "
                           f"max={max(gaps):.3f}s  "
                           f"mean={np.mean(gaps):.3f}s  "
                           f"median={np.median(gaps):.3f}s")
                    close = sum(1 for g in gaps if g < 10.0)
                    rl(f, f"  Gaps < 10s: {close}  (burst FP region indicator)")
            else:
                rl(f, "  FALSE POSITIVE EVENTS: none")
            rl(f)

            # Per-signal contribution
            rl(f, "  PER-SIGNAL CONTRIBUTION:")
            rl(f, f"  {'Signal':<15}  {'Raw':>5}  {'Hits':>5}  "
                   f"{'FP':>6}  {'Precision':>10}")
            rl(f, f"  {'─'*15}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*10}")
            for sig, ts_list in sig_dict.items():
                raw  = len(ts_list)
                h    = sum(1 for s in seg_res
                           if s["hit"] and sig in s["hit"]["label"].split("+"))
                fp   = sum(1 for e in fp_events
                           if sig in e["label"].split("+"))
                prec = h / (h + fp) * 100 if (h + fp) > 0 else 0.0
                rl(f, f"  {sig:<15}  {raw:>5}  {h:>5}  {fp:>6}  {prec:>9.1f}%")
            rl(f)

        # Overall summary
        rheader(f, "OVERALL SUMMARY")

        total_segs = sum(len(vd["seg_results"]) for vd in all_video_data)
        total_hits = sum(sum(1 for s in vd["seg_results"] if s["hit"])
                         for vd in all_video_data)
        total_fp   = sum(len(vd["fp_events"]) for vd in all_video_data)
        overall    = total_hits / total_segs * 100 if total_segs else 0
        fpr        = total_fp / (total_fp + total_hits) * 100 \
                     if (total_fp + total_hits) > 0 else 0
        total_time = sum(vd["elapsed"] for vd in all_video_data)

        rl(f)
        rl(f, f"  {'VIDEO':<38}  {'N':>2}  {'SEGS':>4}  {'HITS':>4}  "
               f"{'RATE':>6}  {'FP':>5}  {'TIME':>7}  STATUS")
        rl(f, f"  {'─'*38}  {'─'*2}  {'─'*4}  {'─'*4}  "
               f"{'─'*6}  {'─'*5}  {'─'*7}  {'─'*6}")
        for vd in all_video_data:
            sr  = vd["seg_results"]
            fpe = vd["fp_events"]
            h   = sum(1 for s in sr if s["hit"])
            t   = len(sr)
            r   = h / t * 100 if t else 0
            st  = "PASS" if r >= 50 else "FAIL"
            rl(f, f"  {vd['name']:<38}  {vd.get('n_inserts',0):>2}  "
                   f"{t:>4}  {h:>4}  {r:>5.1f}%  "
                   f"{len(fpe):>5}  {vd['elapsed']:>6.1f}s  {st}")
        rl(f, f"  {'─'*38}  {'─'*2}  {'─'*4}  {'─'*4}  "
               f"{'─'*6}  {'─'*5}  {'─'*7}  {'─'*6}")
        rl(f, f"  {'TOTAL':<38}  {'':>2}  {total_segs:>4}  "
               f"{total_hits:>4}  {overall:>5.1f}%  "
               f"{total_fp:>5}  {total_time:>6.1f}s")
        rl(f)

        # Hit position breakdown
        rl(f, "  HIT POSITION BREAKDOWN:")
        pos_counts = {}
        for vd in all_video_data:
            for s in vd["seg_results"]:
                if s["hit"]:
                    p = s["hit"]["position"]
                    pos_counts[p] = pos_counts.get(p, 0) + 1
        max_pos = max(pos_counts.values(), default=1)
        for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
            pct = cnt / total_hits * 100 if total_hits else 0
            rbar(f, pos, cnt, max_pos, width=30,
                 unit=f"  ({pct:.1f}% of hits)")
        rl(f)

        # Trigger offset stats
        offsets = [s["hit"]["ts"] - s["start"]
                   for vd in all_video_data
                   for s in vd["seg_results"] if s["hit"]]
        if offsets:
            rl(f, "  TRIGGER OFFSET FROM SEGMENT START:")
            rl(f, f"  Count  : {len(offsets)}")
            rl(f, f"  Min    : {min(offsets):+.3f}s")
            rl(f, f"  Max    : {max(offsets):+.3f}s")
            rl(f, f"  Mean   : {np.mean(offsets):+.3f}s")
            rl(f, f"  Median : {np.median(offsets):+.3f}s")
            rl(f, f"  Std    : {np.std(offsets):.3f}s")
            neg = sum(1 for o in offsets if o < 0)
            pos = sum(1 for o in offsets if o > 0)
            rl(f, f"  Early (before start): {neg}  ({neg/len(offsets)*100:.1f}%)")
            rl(f, f"  Late  (after  start): {pos}  ({pos/len(offsets)*100:.1f}%)")
            rl(f)

        # Duration bucket analysis
        rl(f, "  DETECTION RATE BY SEGMENT DURATION BUCKET:")
        buckets = [("<5s",   0,  5), ("5-10s",  5, 10),
                   ("10-20s",10, 20), ("20-30s", 20, 30), (">30s", 30, 9999)]
        rl(f, f"  {'Bucket':<8}  {'Hits':>5}  {'Total':>6}  {'Rate':>7}")
        rl(f, f"  {'─'*8}  {'─'*5}  {'─'*6}  {'─'*7}")
        for bname, blo, bhi in buckets:
            bh = sum(1 for vd in all_video_data
                     for s in vd["seg_results"]
                     if blo <= s["duration"] < bhi and s["hit"])
            bt = sum(1 for vd in all_video_data
                     for s in vd["seg_results"]
                     if blo <= s["duration"] < bhi)
            br = bh / bt * 100 if bt > 0 else 0.0
            rl(f, f"  {bname:<8}  {bh:>5}  {bt:>6}  {br:>6.1f}%")
        rl(f)

        # Per-signal accuracy
        rl(f, "  PER-SIGNAL ACCURACY (all videos):")
        sig_hits_all, sig_fp_all, sig_raw_all = {}, {}, {}
        for vd in all_video_data:
            for sig, ts_list in vd["signal_dict"].items():
                sig_raw_all[sig] = sig_raw_all.get(sig, 0) + len(ts_list)
            for s in vd["seg_results"]:
                if s["hit"]:
                    for sig in s["hit"]["label"].split("+"):
                        sig_hits_all[sig] = sig_hits_all.get(sig, 0) + 1
            for e in vd["fp_events"]:
                for sig in e["label"].split("+"):
                    sig_fp_all[sig] = sig_fp_all.get(sig, 0) + 1

        all_sigs = sorted(set(
            list(sig_hits_all.keys()) +
            list(sig_fp_all.keys()) +
            list(sig_raw_all.keys())
        ))
        rl(f, f"  {'Signal':<15}  {'Raw':>6}  {'Hits':>5}  "
               f"{'FP':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
        rl(f, f"  {'─'*15}  {'─'*6}  {'─'*5}  "
               f"{'─'*6}  {'─'*10}  {'─'*8}  {'─'*6}")
        for sig in all_sigs:
            raw  = sig_raw_all.get(sig, 0)
            h    = sig_hits_all.get(sig, 0)
            fp   = sig_fp_all.get(sig, 0)
            prec = h / (h + fp) * 100 if (h + fp) > 0 else 0.0
            rec  = h / total_segs * 100 if total_segs > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else 0.0)
            rl(f, f"  {sig:<15}  {raw:>6}  {h:>5}  "
                   f"{fp:>6}  {prec:>9.1f}%  {rec:>7.1f}%  {f1:>5.1f}%")
        rl(f)

        # Signal co-occurrence on true hits
        rl(f, "  SIGNAL CO-OCCURRENCE ON TRUE HITS:")
        cooc = {a: {b: 0 for b in all_sigs} for a in all_sigs}
        for vd in all_video_data:
            for s in vd["seg_results"]:
                if s["hit"]:
                    sigs = s["hit"]["label"].split("+")
                    for a in sigs:
                        for b in sigs:
                            if a in cooc and b in cooc[a]:
                                cooc[a][b] += 1
        hdr = " " * 17 + "  ".join(f"{s[:10]:>10}" for s in all_sigs)
        rl(f, f"  {hdr}")
        for a in all_sigs:
            row = "  ".join(f"{cooc[a].get(b,0):>10}" for b in all_sigs)
            rl(f, f"  {a:<15}  {row}")
        rl(f)

        # Signal co-occurrence on FPs
        rl(f, "  SIGNAL CO-OCCURRENCE ON FALSE POSITIVES:")
        fp_cooc = {a: {b: 0 for b in all_sigs} for a in all_sigs}
        for vd in all_video_data:
            for e in vd["fp_events"]:
                sigs = e["label"].split("+")
                for a in sigs:
                    for b in sigs:
                        if a in fp_cooc and b in fp_cooc[a]:
                            fp_cooc[a][b] += 1
        rl(f, f"  {hdr}")
        for a in all_sigs:
            row = "  ".join(f"{fp_cooc[a].get(b,0):>10}" for b in all_sigs)
            rl(f, f"  {a:<15}  {row}")
        rl(f)

        rl(f, "=" * W)
        rl(f, f"  OVERALL DETECTION RATE  : "
               f"{total_hits:>4} / {total_segs}  ({overall:.2f}%)")
        rl(f, f"  TOTAL FALSE POSITIVES   : {total_fp}")
        rl(f, f"  FALSE POSITIVE RATE     : {fpr:.2f}%")
        rl(f, f"  VIDEOS PROCESSED        : {len(all_video_data)}")
        rl(f, f"  TOTAL PROCESSING TIME   : {total_time:.1f}s")
        passing = sum(
            1 for vd in all_video_data
            if sum(1 for s in vd["seg_results"] if s["hit"]) /
               len(vd["seg_results"]) * 100 >= 50
        )
        rl(f, f"  VIDEOS PASSING (>=50%)  : {passing}/{len(all_video_data)}")
        rl(f, "=" * W)
        rl(f)

# ==============================================================================
# PROCESS ONE VIDEO
# ==============================================================================

def process_video(video_path, ground_truth_entry):
    video_name = os.path.basename(video_path)
    segments   = ground_truth_entry["segments"]

    # Extract new metadata fields
    carrier_source    = ground_truth_entry.get("carrier_source",    "unknown")
    carrier_duration  = ground_truth_entry.get("carrier_duration",  None)
    attacked_duration = ground_truth_entry.get("attacked_duration", None)
    resolution        = ground_truth_entry.get("resolution",        "unknown")
    fps               = ground_truth_entry.get("fps",               30)
    n_inserts         = ground_truth_entry.get("n_inserts",         len(segments))

    if n_inserts != len(segments):
        print(f"  [WARN] n_inserts={n_inserts} but "
              f"found {len(segments)} segments in JSON")

    # Fix corrupted entries
    for seg in segments:
        if seg["end"] < seg["start"]:
            seg["end"] = seg["start"] + seg["duration"]

    # Print metadata
    print(f"\n  Carrier   : {carrier_source}  "
          f"({carrier_duration:.3f}s)" if carrier_duration else
          f"\n  Carrier   : {carrier_source}")
    print(f"  Attacked  : {video_name}  "
          f"({attacked_duration:.3f}s)" if attacked_duration else
          f"  Attacked  : {video_name}")
    print(f"  Resolution: {resolution}  FPS: {fps}  Inserts: {n_inserts}")

    # Insert drift table
    print(f"\n  Insert drift analysis:")
    print(f"  {'#':>3}  {'SOURCE':<16}  {'ORIG_PT':>10}  "
          f"{'ACT_START':>10}  {'DRIFT':>8}  {'DUR':>8}  FRAMES")
    print(f"  {'─'*3}  {'─'*16}  {'─'*10}  "
          f"{'─'*10}  {'─'*8}  {'─'*8}  {'─'*15}")
    for seg in segments:
        orig  = seg.get("original_insert_point", seg["start"])
        drift = seg["start"] - orig
        sf    = seg.get("start_frame", "?")
        ef    = seg.get("end_frame",   "?")
        print(f"  {seg['index']:>3}  {seg['source']:<16}  {orig:>10.3f}  "
              f"{seg['start']:>10.3f}  {drift:>+8.3f}  "
              f"{seg['duration']:>8.3f}  {sf}-{ef}")

    # PRE-FLIGHT
    video_duration = get_video_duration(video_path)
    if video_duration is None:
        if attacked_duration is not None:
            print(f"  [PRE-FLIGHT] ffprobe failed — using JSON "
                  f"attacked_duration: {attacked_duration:.3f}s")
            video_duration = attacked_duration
        else:
            print(f"  [FATAL] Cannot determine duration for {video_name}")
            return None

    if attacked_duration is not None:
        delta = abs(video_duration - attacked_duration)
        if delta > 1.0:
            print(f"  [WARN] Duration mismatch: ffprobe={video_duration:.3f}s  "
                  f"JSON={attacked_duration:.3f}s  delta={delta:.3f}s")
        else:
            print(f"  [PRE-FLIGHT] Duration OK  "
                  f"(ffprobe={video_duration:.3f}s  "
                  f"JSON={attacked_duration:.3f}s  delta={delta:.3f}s)")

    print(f"\n  Running {n_inserts} insert(s) to detect, "
          f"duration {hms(video_duration)}...")

    t0 = time.time()

    print(f"  [1/5] Visual I-frame spikes (MAD)...")
    v_events = get_visual_events(video_path)
    print(f"        -> {len(v_events)} events")

    print(f"  [2/5] Motion (GOP MAD + P/B regression)...")
    gop_events, pb_events = get_motion_events(video_path)
    print(f"        -> {len(gop_events)} GOP,  {len(pb_events)} P/B events")

    print(f"  [3/5] Audio RMS (MAD) + Silence...")
    rms_events, sil_events = get_audio_rms_events(video_path)
    print(f"        -> {len(rms_events)} RMS,  {len(sil_events)} silence events")

    print(f"  [4/5] Spectral Flux (MAD)...")
    flux_events = get_spectral_flux_events(video_path)
    print(f"        -> {len(flux_events)} events")

    elapsed_extract = time.time() - t0
    print(f"\n  Extraction complete in {elapsed_extract:.1f}s")

    event_dict = {
        "visual"       : v_events,
        "gop"          : gop_events,
        "motion_pb"    : pb_events,
        "audio_rms"    : rms_events,
        "silence"      : sil_events,
        "spectral_flux": flux_events,
    }

    print(f"\n  [Pass 1] KDE fusion...")
    candidates, fused_curve = fuse_all_signals_kde(event_dict, video_duration)
    print(f"  [Pass 1] {len(candidates)} candidate peaks")

    verified, rejected = verify_candidates_bhattacharyya(video_path, candidates)

    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.1f}s")

    seg_results, fp_events, rejected_out = validate_video(
        video_name, segments, verified, rejected
    )

    legacy_signal_dict = {
        "Visual"      : [e[0] for e in v_events],
        "Motion(GOP)" : [e[0] for e in gop_events],
        "Motion(PB)"  : [e[0] for e in pb_events],
        "AudioRMS"    : [e[0] for e in rms_events],
        "Silence"     : [e[0] for e in sil_events],
        "SpectralFlux": [e[0] for e in flux_events],
    }

    print_video_report(video_name, seg_results, fp_events,
                       legacy_signal_dict, verified, rejected_out)

    sig_hits, sig_fp = {}, {}
    for seg in seg_results:
        if seg["hit"]:
            for sig in seg["hit"]["label"].split("+"):
                sig_hits[sig] = sig_hits.get(sig, 0) + 1
    for fp in fp_events:
        for sig in fp["label"].split("+"):
            sig_fp[sig] = sig_fp.get(sig, 0) + 1

    return {
        "name"            : video_name,
        "total"           : len(seg_results),
        "hits"            : sum(1 for s in seg_results if s["hit"] is not None),
        "fp"              : len(fp_events),
        "rejected"        : len(rejected_out),
        "positions"       : [s["hit"]["position"] for s in seg_results if s["hit"]],
        "sig_hits"        : sig_hits,
        "sig_fp"          : sig_fp,
        "raw_triggers"    : {sig: len(ts) for sig, ts in legacy_signal_dict.items()},
        "seg_results"     : seg_results,
        "fp_events"       : fp_events,
        "signal_dict"     : legacy_signal_dict,
        "merged_events"   : [(t, l, n) for t, l, n, c, b in verified],
        "elapsed"         : elapsed,
        "fused_curve"     : fused_curve,
        "carrier_source"  : carrier_source,
        "carrier_duration": carrier_duration,
        "attacked_duration": attacked_duration,
        "resolution"      : resolution,
        "fps"             : fps,
        "n_inserts"       : n_inserts,
    }

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    gt_path = os.path.join(SCRIPT_DIR, GROUND_TRUTH_JSON)
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth not found: {gt_path}")
        sys.exit(1)

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    gt_map = {entry["target_video"]: entry for entry in ground_truth}
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner("MULTI-MODAL BOUNDARY DETECTOR  v3.0  —  FULL EVALUATION")
    print(f"  Ground truth : {GROUND_TRUTH_JSON}  ({len(ground_truth)} videos)")
    print(f"  Attacked dir : {ATTACKED_DIR}/")
    total_inserts = sum(e.get("n_inserts", 0) for e in ground_truth)
    print(f"  Total inserts across all videos: {total_inserts}")
    print(f"  Signals : Visual(MAD) | Motion(GOP+PB) | "
          f"AudioRMS(MAD) | Silence | SpectralFlux(MAD)")
    print(f"  Fusion  : KDE + T-conorm audio gate + Hadamard P/B gate")
    print(f"  Pass 2  : Bhattacharyya HSV histogram verification")
    print(f"  Report  : {REPORT_FILE}")

    attacked_videos = sorted([
        os.path.join(SCRIPT_DIR, ATTACKED_DIR, e["target_video"])
        for e in ground_truth
    ])

    all_video_stats = []

    for vid_idx, video_path in enumerate(attacked_videos):
        video_name = os.path.basename(video_path)
        banner(f"VIDEO {vid_idx+1}/{len(attacked_videos)}: {video_name}")

        if not os.path.exists(video_path):
            print(f"  [!] File not found: {video_path}")
            continue
        if video_name not in gt_map:
            print(f"  [!] No ground truth for {video_name}")
            continue

        stats = process_video(video_path, gt_map[video_name])
        if stats:
            all_video_stats.append(stats)

    if all_video_stats:
        print_overall_summary(all_video_stats)
        report_path = os.path.join(SCRIPT_DIR, REPORT_FILE)
        write_full_report(report_path, all_video_stats, run_ts)
        print(f"[+] Report written to: {report_path}")