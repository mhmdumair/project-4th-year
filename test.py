import subprocess
import json
import numpy as np
from scipy.io import wavfile
import os
import re
from collections import deque

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

LOCKOUT_PERIOD          = 5.0

# --- Visual: I-frame size ---
VISUAL_MAX_FRAMES       = 30
VISUAL_MIN_WARMUP       = 10
K_VISUAL_THRESHOLD      = 2.5
VISUAL_COOLDOWN         = 3.0
VISUAL_PERSIST_NEEDED   = 2
VISUAL_PERSIST_WINDOW   = 4

# --- Motion Vector proxy ---
MV_WINDOW               = 60
MV_MIN_WARMUP           = 30
K_MV_THRESHOLD          = 4.5
MV_COOLDOWN             = 8.0
MV_PERSIST_NEEDED       = 3
MV_PERSIST_WINDOW       = 5
GOP_SHORT_RATIO         = 0.25

# --- Audio: RMS Energy spikes ---
AUDIO_BUFFER_SEC        = 15.0
AUDIO_MICRO_WINDOW_SEC  = 0.5
K_AUDIO_RMS_THRESHOLD   = 3.0
AUDIO_NOISE_FLOOR       = 800.0
AUDIO_COOLDOWN          = 3.0
AUDIO_PERSIST_NEEDED    = 2

# --- Audio: Silence Detection ---
SILENCE_THRESHOLD_RATIO = 0.15
SILENCE_MIN_DURATION    = 0.4
SILENCE_COOLDOWN        = 3.0

# --- Audio: Spectral Flux ---
FLUX_WINDOW_SEC         = 0.05
FLUX_BUFFER_SEC         = 10.0
K_FLUX_THRESHOLD        = 3.0
FLUX_COOLDOWN           = 3.0
FLUX_PERSIST_NEEDED     = 3

# --- Fusion ---
TOLERANCE_SEC           = 2.5
REQUIRE_MULTIMODAL      = False
MIN_SIGNALS_TO_CONFIRM  = 1

# Tolerance for exact boundary hit
BOUNDARY_TOLERANCE      = 2.5

TRUTH_FILE = "merge.txt"

# ==============================================================================
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DEBUG_FILE_PATH = os.path.join(SCRIPT_DIR, "visual_debug.txt")

FFMPEG_DIR  = "ffmpeg-master-latest-win64-gpl"
FFPROBE_EXE = os.path.join(SCRIPT_DIR, FFMPEG_DIR, "bin", "ffprobe.exe")
FFMPEG_EXE  = os.path.join(SCRIPT_DIR, FFMPEG_DIR, "bin", "ffmpeg.exe")


# ==============================================================================
# HELPER: Adaptive z-score
# ==============================================================================

def adaptive_zscore(value, buffer, k):
    arr = np.array(buffer, dtype=np.float64)
    if len(arr) < 2:
        return 0.0, False
    mu    = np.mean(arr)
    sigma = np.std(arr)
    if sigma < 1e-9:
        return 0.0, False
    z = (value - mu) / sigma
    return z, abs(z) > k


# ==============================================================================
# SIGNAL 1: Visual — I-frame size spikes
# ==============================================================================

def get_visual_triggers(video_path):
    print("[*] Signal 1/5 - Visual I-frame Size...")
    cmd = [
        FFPROBE_EXE, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time,pts_time,pkt_size,size,pict_type,key_frame',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        frames = json.loads(result.stdout).get('frames', [])
    except Exception:
        return []

    i_frames = [f for f in frames if f.get('pict_type') == 'I' or f.get('key_frame') == 1]

    with open(DEBUG_FILE_PATH, "w", encoding="utf-8") as f:
        f.write("Index, Size, ZScore\n")

    triggers          = []
    size_buffer       = deque(maxlen=VISUAL_MAX_FRAMES)
    spike_flags       = deque(maxlen=VISUAL_PERSIST_WINDOW)
    last_trigger_time = -999.0

    for idx, frame in enumerate(i_frames):
        size     = float(frame.get('pkt_size') or frame.get('size') or 0)
        time_pts = float(frame.get('pkt_pts_time') or frame.get('pts_time') or -1)
        if time_pts < 0:
            continue

        is_spike = False
        z_score  = 0.0
        if time_pts > LOCKOUT_PERIOD and len(size_buffer) >= VISUAL_MIN_WARMUP:
            z_score, is_spike = adaptive_zscore(size, size_buffer, K_VISUAL_THRESHOLD)

        spike_flags.append(1 if is_spike else 0)

        with open(DEBUG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{idx}, {int(size)}, {z_score:.2f}\n")

        if (time_pts > LOCKOUT_PERIOD
                and len(spike_flags) == VISUAL_PERSIST_WINDOW
                and sum(spike_flags) >= VISUAL_PERSIST_NEEDED
                and (time_pts - last_trigger_time) > VISUAL_COOLDOWN):
            triggers.append(time_pts)
            last_trigger_time = time_pts
            spike_flags.clear()

        size_buffer.append(size)

    print(f"    -> {len(triggers)} triggers")
    return triggers


# ==============================================================================
# SIGNAL 2: Motion Vector proxy
# ==============================================================================

def get_motion_vector_triggers(video_path):
    print("[*] Signal 2/5 - Motion Vector Proxy (GOP + P/B frames)...")
    cmd = [
        FFPROBE_EXE, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pkt_pts_time,pts_time,pkt_size,size,pict_type',
        '-of', 'json', video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        frames = json.loads(result.stdout).get('frames', [])
    except Exception:
        return []

    triggers          = []
    pb_buffer         = deque(maxlen=MV_WINDOW)
    spike_flags       = deque(maxlen=MV_PERSIST_WINDOW)
    last_trigger_time = -999.0
    last_i_time       = -999.0
    gop_intervals     = deque(maxlen=30)

    for frame in frames:
        ptype    = frame.get('pict_type', 'U')
        size     = float(frame.get('pkt_size') or frame.get('size') or 0)
        time_pts = float(frame.get('pkt_pts_time') or frame.get('pts_time') or -1)
        if time_pts < 0 or size == 0:
            continue

        if ptype == 'I':
            if last_i_time > 0:
                interval = time_pts - last_i_time
                if len(gop_intervals) >= 10:
                    mean_gop = np.mean(gop_intervals)
                    if (time_pts > LOCKOUT_PERIOD
                            and mean_gop > 0
                            and interval < mean_gop * GOP_SHORT_RATIO
                            and (time_pts - last_trigger_time) > MV_COOLDOWN):
                        triggers.append(time_pts)
                        last_trigger_time = time_pts
                gop_intervals.append(interval)
            last_i_time = time_pts

        if ptype in ('P', 'B'):
            is_spike = False
            if time_pts > LOCKOUT_PERIOD and len(pb_buffer) >= MV_MIN_WARMUP:
                _, is_spike = adaptive_zscore(size, pb_buffer, K_MV_THRESHOLD)
            spike_flags.append(1 if is_spike else 0)
            if (time_pts > LOCKOUT_PERIOD
                    and len(spike_flags) == MV_PERSIST_WINDOW
                    and sum(spike_flags) >= MV_PERSIST_NEEDED
                    and (time_pts - last_trigger_time) > MV_COOLDOWN):
                triggers.append(time_pts)
                last_trigger_time = time_pts
                spike_flags.clear()
            pb_buffer.append(size)

    triggers.sort()
    deduped, last = [], -999.0
    for t in triggers:
        if t - last > MV_COOLDOWN:
            deduped.append(t)
            last = t

    print(f"    -> {len(deduped)} triggers")
    return deduped


# ==============================================================================
# AUDIO EXTRACTION HELPER
# ==============================================================================

def extract_audio(video_path, sample_rate=44100, tag="tmp"):
    temp_wav = os.path.join(SCRIPT_DIR, f"_audio_{tag}.wav")
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    for exe in [FFMPEG_EXE, "ffmpeg"]:
        try:
            subprocess.run([exe, '-y', '-i', video_path, '-vn',
                            '-acodec', 'pcm_s16le', '-ar', str(sample_rate),
                            '-ac', '1', temp_wav],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    except Exception as e:
        print(f"    -> WAV read error: {e}")
        return None, None


# ==============================================================================
# SIGNAL 3: Audio RMS
# ==============================================================================

def get_audio_rms_triggers(video_path):
    print("[*] Signal 3/5 - Audio RMS (spikes + silence drops)...")
    sample_rate, data = extract_audio(video_path, sample_rate=44100, tag="rms")
    if data is None:
        print("    -> Audio extraction failed")
        return [], []

    micro_samples    = int(AUDIO_MICRO_WINDOW_SEC * sample_rate)
    bg_chunks_needed = int(AUDIO_BUFFER_SEC / AUDIO_MICRO_WINDOW_SEC)
    spike_triggers   = []
    silence_triggers = []
    rms_buffer       = deque(maxlen=bg_chunks_needed)
    consec_spikes    = 0
    in_silence       = False
    silence_start    = -999.0
    last_spike_time  = -999.0
    last_sil_time    = -999.0

    for i in range(len(data) // micro_samples):
        chunk     = data[i * micro_samples : (i + 1) * micro_samples]
        micro_rms = float(np.sqrt(np.mean(chunk ** 2))) if np.any(chunk) else 0.0
        time_pts  = (i * micro_samples) / sample_rate

        if time_pts > LOCKOUT_PERIOD and len(rms_buffer) == bg_chunks_needed:
            mu    = float(np.mean(rms_buffer))
            sigma = float(np.std(rms_buffer))
            is_spike = (micro_rms > AUDIO_NOISE_FLOOR and sigma > 0
                        and micro_rms > mu + K_AUDIO_RMS_THRESHOLD * sigma)
            consec_spikes = consec_spikes + 1 if is_spike else 0
            if (consec_spikes >= AUDIO_PERSIST_NEEDED
                    and (time_pts - last_spike_time) > AUDIO_COOLDOWN):
                spike_triggers.append(time_pts)
                last_spike_time = time_pts
                consec_spikes   = 0
            if mu > AUDIO_NOISE_FLOOR:
                is_silent = micro_rms < mu * SILENCE_THRESHOLD_RATIO
                if is_silent and not in_silence:
                    in_silence    = True
                    silence_start = time_pts
                elif not is_silent and in_silence:
                    in_silence = False
                    duration   = time_pts - silence_start
                    if (duration >= SILENCE_MIN_DURATION
                            and (silence_start - last_sil_time) > SILENCE_COOLDOWN):
                        silence_triggers.append(silence_start)
                        last_sil_time = silence_start
        rms_buffer.append(micro_rms)

    print(f"    -> {len(spike_triggers)} RMS spike triggers, {len(silence_triggers)} silence triggers")
    return spike_triggers, silence_triggers


# ==============================================================================
# SIGNAL 4+5: Spectral Flux
# ==============================================================================

def get_spectral_flux_triggers(video_path):
    print("[*] Signal 5/5 - Spectral Flux...")
    sample_rate, data = extract_audio(video_path, sample_rate=22050, tag="flux")
    if data is None:
        print("    -> Audio extraction failed")
        return []

    hop_samples      = max(1, int(FLUX_WINDOW_SEC * sample_rate))
    n_fft            = hop_samples * 2
    bg_chunks_needed = max(10, int(FLUX_BUFFER_SEC / FLUX_WINDOW_SEC))
    hann_win         = np.hanning(n_fft)
    triggers          = []
    flux_buffer       = deque(maxlen=bg_chunks_needed)
    consec_flux       = 0
    last_trigger_time = -999.0
    prev_spectrum     = None

    for i in range(len(data) // hop_samples):
        start    = i * hop_samples
        chunk    = data[start : start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        time_pts = start / sample_rate
        spectrum = np.abs(np.fft.rfft(chunk * hann_win))
        norm     = np.linalg.norm(spectrum)
        if norm > 1e-9:
            spectrum = spectrum / norm
        if prev_spectrum is not None:
            flux = float(np.sum(np.maximum(spectrum - prev_spectrum, 0.0)))
            if (time_pts > LOCKOUT_PERIOD
                    and len(flux_buffer) >= int(bg_chunks_needed * 0.3)):
                _, is_spike = adaptive_zscore(flux, flux_buffer, K_FLUX_THRESHOLD)
                consec_flux = consec_flux + 1 if is_spike else 0
                if (consec_flux >= FLUX_PERSIST_NEEDED
                        and (time_pts - last_trigger_time) > FLUX_COOLDOWN):
                    triggers.append(time_pts)
                    last_trigger_time = time_pts
                    consec_flux       = 0
            flux_buffer.append(flux)
        prev_spectrum = spectrum

    print(f"    -> {len(triggers)} triggers")
    return triggers


# ==============================================================================
# AUDIO DIAGNOSTIC
# ==============================================================================

def diagnose_audio(video_path):
    print("[*] Audio diagnostic...")
    for exe in [FFMPEG_EXE, "ffmpeg"]:
        try:
            r = subprocess.run([exe, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode == 0:
                print(f"    ffmpeg OK: {exe}")
                break
        except FileNotFoundError:
            print(f"    ffmpeg NOT found: {exe}")
    cmd = [FFPROBE_EXE, '-v', 'error', '-select_streams', 'a',
           '-show_entries', 'stream=codec_name,sample_rate,channels',
           '-of', 'json', video_path]
    try:
        r       = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        streams = json.loads(r.stdout).get('streams', [])
        if streams:
            for s in streams:
                print(f"    Audio stream: codec={s.get('codec_name')} "
                      f"sr={s.get('sample_rate')} ch={s.get('channels')}")
        else:
            print("    WARNING: No audio streams found in video!")
    except Exception as e:
        print(f"    Diagnostic error: {e}")


# ==============================================================================
# FUSION
# ==============================================================================

def fuse_all_signals(signal_dict):
    all_events = []
    for label, timestamps in signal_dict.items():
        for t in timestamps:
            all_events.append({"time": t, "signal": label})
    all_events.sort(key=lambda x: x["time"])
    used     = [False] * len(all_events)
    clusters = []
    for i, ev in enumerate(all_events):
        if used[i]:
            continue
        cluster = [ev]
        used[i] = True
        for j in range(i + 1, len(all_events)):
            if used[j]:
                continue
            if all_events[j]["time"] - cluster[0]["time"] > TOLERANCE_SEC:
                break
            cluster.append(all_events[j])
            used[j] = True
        clusters.append(cluster)

    merged = []
    for cluster in clusters:
        signals_fired = list({e["signal"] for e in cluster})
        n_signals     = len(signals_fired)
        if n_signals < MIN_SIGNALS_TO_CONFIRM:
            continue
        if REQUIRE_MULTIMODAL and n_signals < 2:
            continue
        avg_time = round(float(np.mean([e["time"] for e in cluster])), 2)
        label    = "+".join(sorted(signals_fired))
        merged.append((avg_time, label, n_signals))
    merged.sort(key=lambda x: x[0])
    return merged


# ==============================================================================
# PRINT TABLE
# ==============================================================================

def print_results(merged):
    print("\n" + "=" * 72)
    print(f"{'SIGNALS':<38} | {'TIME (s)':<10} | CONFIDENCE")
    print("-" * 72)
    for ts, label, n in merged:
        conf = "HIGH" if n >= 3 else ("MED" if n == 2 else "LOW")
        print(f"{label:<38} | {ts:<10} | {conf} ({n} signal{'s' if n > 1 else ''})")
    print("=" * 72)
    print(f"Total events detected: {len(merged)}")


# ==============================================================================
# GROUND TRUTH
# ==============================================================================

def load_ground_truth(filepath):
    points = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r") as f:
        for line in f:
            m = re.search(r"(Segment_\d+_(Start|End)):\s+([\d.]+)", line)
            if m:
                points.append({
                    "label": m.group(1),
                    "type":  m.group(2),
                    "time":  float(m.group(3)),
                })
    return points


def build_segments(truth_points):
    """Pair Start/End points into segment windows."""
    segs = {}
    for p in truth_points:
        m = re.match(r"Segment_(\d+)_(Start|End)", p["label"])
        if not m:
            continue
        sid   = int(m.group(1))
        btype = m.group(2)
        if sid not in segs:
            segs[sid] = {"seg_id": sid, "start": None, "end": None, "hit": None}
        if btype == "Start":
            segs[sid]["start"] = p["time"]
        else:
            segs[sid]["end"] = p["time"]
    return sorted(
        [s for s in segs.values() if s["start"] is not None and s["end"] is not None],
        key=lambda x: x["seg_id"]
    )


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_boundaries(detected_events, truth_points):
    segments   = build_segments(truth_points)
    total_segs = len(segments)
    hits        = 0
    fp          = 0

    # For each detected event, try to claim a segment
    for ts, label, n_sig in detected_events:
        claimed = False
        for seg in segments:
            # Hit if within ±BOUNDARY_TOLERANCE of Start, within ±BOUNDARY_TOLERANCE of End,
            # OR strictly between Start and End
            within_start = abs(ts - seg["start"]) <= BOUNDARY_TOLERANCE
            within_end   = abs(ts - seg["end"])   <= BOUNDARY_TOLERANCE
            inside       = seg["start"] < ts < seg["end"]

            if (within_start or within_end or inside) and seg["hit"] is None:
                seg["hit"]  = (ts, label)
                claimed     = True
                break

        if not claimed:
            fp += 1

    # Count hits and print segment timeline
    print("\n[*] Segment Detection Report")
    print("=" * 65)
    for seg in segments:
        start_str = f"{seg['start']}s"
        end_str   = f"{seg['end']}s"
        if seg["hit"] is not None:
            hit_ts, hit_label = seg["hit"]
            hits += 1
            print(f"  Segment_{seg['seg_id']}  "
                  f"[START {start_str}]  -->  "
                  f"[HIT {hit_ts}s | {hit_label}]  -->  "
                  f"[END {end_str}]  ✓")
        else:
            print(f"  Segment_{seg['seg_id']}  "
                  f"[START {start_str}]  -->  "
                  f"[NO DETECTION]  -->  "
                  f"[END {end_str}]  ✗")
    print("=" * 65)

    print("\n" + "-" * 55)
    print("FINAL PERFORMANCE METRICS")
    print("-" * 55)
    print(f"Total Segments:            {total_segs}")
    print(f"Segments Hit:              {hits} / {total_segs}")
    print(f"Total False Positives:     {fp}")
    print("-" * 55)
    rate = (hits / total_segs * 100) if total_segs else 0
    print(f"OVERALL SUCCESS RATE:      {rate:.2f}%")
    print("-" * 55)

    print("\n[*] Per-signal contribution:")
    signal_hits = {}
    for seg in segments:
        if seg["hit"]:
            for sig in seg["hit"][1].split("+"):
                signal_hits[sig] = signal_hits.get(sig, 0) + 1
    if signal_hits:
        for sig, cnt in sorted(signal_hits.items(), key=lambda x: -x[1]):
            print(f"    {sig:<20}: {cnt} hit(s)")
    else:
        print("    (none)")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    VIDEO_FILE = "hidden_output.mp4"
    VIDEO_PATH = os.path.join(SCRIPT_DIR, VIDEO_FILE)

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: {VIDEO_FILE} not found.")
        exit(1)

    print("=" * 60)
    print("  MULTI-MODAL BOUNDARY DETECTOR  v2.3")
    print("  Signals: Visual | Motion | AudioRMS | Silence | SpectralFlux")
    print("  Scoring: 1 point per segment — hit anywhere Start→End")
    print("=" * 60 + "\n")

    diagnose_audio(VIDEO_PATH)
    print()

    v_ts                     = get_visual_triggers(VIDEO_PATH)
    mv_ts                    = get_motion_vector_triggers(VIDEO_PATH)
    rms_spike_ts, silence_ts = get_audio_rms_triggers(VIDEO_PATH)
    flux_ts                  = get_spectral_flux_triggers(VIDEO_PATH)

    signal_dict = {
        "Visual"      : v_ts,
        "Motion"      : mv_ts,
        "AudioRMS"    : rms_spike_ts,
        "Silence"     : silence_ts,
        "SpectralFlux": flux_ts,
    }

    print("\n[*] Signal summary:")
    for name, ts in signal_dict.items():
        print(f"    {name:<15}: {len(ts)} triggers")

    merged = fuse_all_signals(signal_dict)
    print_results(merged)

    truth_points = load_ground_truth(os.path.join(SCRIPT_DIR, TRUTH_FILE))
    if truth_points:
        validate_boundaries(merged, truth_points)
    else:
        print("[!] No ground truth file found -- skipping validation.")
