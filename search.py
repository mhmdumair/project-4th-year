import subprocess
import json
import numpy as np
from scipy.io import wavfile
import os
import sys
import time
from collections import deque

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ATTACKED_DIR            = "attacked"
GROUND_TRUTH_JSON       = "attack.json"

FFMPEG_DIR  = "ffmpeg-master-latest-win64-gpl"
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE  = os.path.join(SCRIPT_DIR, FFMPEG_DIR, "bin", "ffmpeg.exe")
FFPROBE_EXE = os.path.join(SCRIPT_DIR, FFMPEG_DIR, "bin", "ffprobe.exe")

# --- Fusion / Scoring ---
BOUNDARY_TOLERANCE      = 2.5   # seconds either side of start/end to still count as boundary hit
TOLERANCE_SEC           = 2.5   # cluster window for fusing multi-signal events
MIN_SIGNALS_TO_CONFIRM  = 1
REQUIRE_MULTIMODAL      = False
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

# --- Audio RMS ---
AUDIO_BUFFER_SEC        = 15.0
AUDIO_MICRO_WINDOW_SEC  = 0.5
K_AUDIO_RMS_THRESHOLD   = 3.0
AUDIO_NOISE_FLOOR       = 800.0
AUDIO_COOLDOWN          = 3.0
AUDIO_PERSIST_NEEDED    = 2

# --- Silence ---
SILENCE_THRESHOLD_RATIO = 0.15
SILENCE_MIN_DURATION    = 0.4
SILENCE_COOLDOWN        = 3.0

# --- Spectral Flux ---
FLUX_WINDOW_SEC         = 0.05
FLUX_BUFFER_SEC         = 10.0
K_FLUX_THRESHOLD        = 3.0
FLUX_COOLDOWN           = 3.0
FLUX_PERSIST_NEEDED     = 3

# ==============================================================================
# PROGRESS HELPERS
# ==============================================================================

def hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def progress_bar(label, current, total, width=35, extra=""):
    frac  = min(current / total, 1.0) if total > 0 else 0
    filled = int(width * frac)
    bar   = "█" * filled + "░" * (width - filled)
    pct   = frac * 100
    sys.stdout.write(f"\r    [{bar}] {pct:5.1f}%  {extra}  ")
    sys.stdout.flush()

def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")

def banner(title):
    w = 65
    print("\n" + "═"*w)
    pad = (w - len(title) - 2) // 2
    print(" "*pad + f" {title} ")
    print("═"*w)

# ==============================================================================
# ADAPTIVE Z-SCORE
# ==============================================================================

def adaptive_zscore(value, buffer, k):
    arr   = np.array(buffer, dtype=np.float64)
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
    total    = len(i_frames)

    triggers          = []
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

        is_spike = False
        if time_pts > LOCKOUT_PERIOD and len(size_buffer) >= VISUAL_MIN_WARMUP:
            _, is_spike = adaptive_zscore(size, size_buffer, K_VISUAL_THRESHOLD)

        spike_flags.append(1 if is_spike else 0)

        if (time_pts > LOCKOUT_PERIOD
                and len(spike_flags) == VISUAL_PERSIST_WINDOW
                and sum(spike_flags) >= VISUAL_PERSIST_NEEDED
                and (time_pts - last_trigger_time) > VISUAL_COOLDOWN):
            triggers.append(time_pts)
            last_trigger_time = time_pts
            spike_flags.clear()

        size_buffer.append(size)

    progress_bar("Visual", total, total, extra="done")
    print()
    return triggers

# ==============================================================================
# SIGNAL 2: Motion Vector proxy
# ==============================================================================

def get_motion_vector_triggers(video_path):
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

    total             = len(frames)
    triggers          = []
    pb_buffer         = deque(maxlen=MV_WINDOW)
    spike_flags       = deque(maxlen=MV_PERSIST_WINDOW)
    last_trigger_time = -999.0
    last_i_time       = -999.0
    gop_intervals     = deque(maxlen=30)

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
                if len(gop_intervals) >= 10:
                    mean_gop = np.mean(gop_intervals)
                    if (time_pts > LOCKOUT_PERIOD and mean_gop > 0
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

    progress_bar("Motion", total, total, extra="done")
    print()
    return deduped

# ==============================================================================
# AUDIO EXTRACTION
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
    except Exception:
        return None, None

# ==============================================================================
# SIGNAL 3 + 4: Audio RMS + Silence
# ==============================================================================

def get_audio_rms_triggers(video_path):
    sample_rate, data = extract_audio(video_path, sample_rate=44100, tag="rms")
    if data is None:
        print("    [!] Audio extraction failed")
        return [], []

    micro_samples    = int(AUDIO_MICRO_WINDOW_SEC * sample_rate)
    bg_chunks_needed = int(AUDIO_BUFFER_SEC / AUDIO_MICRO_WINDOW_SEC)
    total_chunks     = len(data) // micro_samples
    spike_triggers   = []
    silence_triggers = []
    rms_buffer       = deque(maxlen=bg_chunks_needed)
    consec_spikes    = 0
    in_silence       = False
    silence_start    = -999.0
    last_spike_time  = -999.0
    last_sil_time    = -999.0

    for i in range(total_chunks):
        chunk     = data[i * micro_samples : (i + 1) * micro_samples]
        micro_rms = float(np.sqrt(np.mean(chunk ** 2))) if np.any(chunk) else 0.0
        time_pts  = (i * micro_samples) / sample_rate

        if i % max(1, total_chunks // 20) == 0:
            progress_bar("AudioRMS", i, total_chunks, extra=f"{hms(time_pts)}")

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

    progress_bar("AudioRMS", total_chunks, total_chunks, extra="done")
    print()
    return spike_triggers, silence_triggers

# ==============================================================================
# SIGNAL 5: Spectral Flux
# ==============================================================================

def get_spectral_flux_triggers(video_path):
    sample_rate, data = extract_audio(video_path, sample_rate=22050, tag="flux")
    if data is None:
        print("    [!] Audio extraction failed")
        return []

    hop_samples      = max(1, int(FLUX_WINDOW_SEC * sample_rate))
    n_fft            = hop_samples * 2
    bg_chunks_needed = max(10, int(FLUX_BUFFER_SEC / FLUX_WINDOW_SEC))
    hann_win         = np.hanning(n_fft)
    total_chunks     = len(data) // hop_samples
    triggers         = []
    flux_buffer      = deque(maxlen=bg_chunks_needed)
    consec_flux      = 0
    last_trigger_time = -999.0
    prev_spectrum    = None

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
                _, is_spike = adaptive_zscore(flux, flux_buffer, K_FLUX_THRESHOLD)
                consec_flux = consec_flux + 1 if is_spike else 0
                if (consec_flux >= FLUX_PERSIST_NEEDED
                        and (time_pts - last_trigger_time) > FLUX_COOLDOWN):
                    triggers.append(time_pts)
                    last_trigger_time = time_pts
                    consec_flux       = 0
            flux_buffer.append(flux)
        prev_spectrum = spectrum

    progress_bar("SpectFlux", total_chunks, total_chunks, extra="done")
    print()
    return triggers

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
# VALIDATE ONE VIDEO
# ==============================================================================

def classify_hit_position(ts, seg):
    """Classify where the trigger landed relative to the hidden segment."""
    near_start = abs(ts - seg["start"]) <= BOUNDARY_TOLERANCE
    near_end   = abs(ts - seg["end"])   <= BOUNDARY_TOLERANCE
    inside     = seg["start"] < ts < seg["end"]

    if near_start and not near_end:
        return "START"
    elif near_end and not near_start:
        return "END"
    elif near_start and near_end:
        return "START+END"   # very short clip edge case
    elif inside:
        return "IN-BETWEEN"
    return None


def validate_video(video_name, segments, merged_events):
    """
    Match detected events against ground-truth segments.
    Returns per-segment results + false positives.
    """
    seg_results = []
    fp_events   = []

    # Deep-copy segments with hit tracking
    segs = [dict(s, hit=None) for s in segments]

    for ts, label, n_sig in merged_events:
        claimed = False
        for seg in segs:
            pos = classify_hit_position(ts, seg)
            if pos is not None and seg["hit"] is None:
                seg["hit"] = {"ts": ts, "label": label, "n_sig": n_sig, "position": pos}
                claimed    = True
                break
        if not claimed:
            fp_events.append({"ts": ts, "label": label, "n_sig": n_sig})

    for seg in segs:
        seg_results.append({
            "index":    seg["index"],
            "source":   seg["source"],
            "start":    seg["start"],
            "end":      seg["end"],
            "duration": seg["duration"],
            "hit":      seg["hit"],
        })

    return seg_results, fp_events


# ==============================================================================
# PRINT VIDEO REPORT
# ==============================================================================

SIGNAL_COLORS = {
    "Visual":       "V",
    "Motion":       "M",
    "AudioRMS":     "A",
    "Silence":      "S",
    "SpectralFlux": "F",
}

def confidence_str(n):
    if n >= 3: return "HIGH"
    if n == 2: return "MED "
    return "LOW "


def print_video_report(video_name, seg_results, fp_events, signal_dict, merged_events):
    hits  = sum(1 for s in seg_results if s["hit"] is not None)
    total = len(seg_results)

    section(f"VIDEO: {video_name}   [{hits}/{total} segments detected]")

    # Signal raw counts
    print(f"  Raw triggers per signal:")
    for sig, ts_list in signal_dict.items():
        bar = "▪" * min(len(ts_list), 40)
        print(f"    {sig:<15} {len(ts_list):>4} triggers  {bar}")
    print(f"  Fused events (after clustering): {len(merged_events)}")

    # Per-segment table
    print(f"\n  {'#':>3}  {'SOURCE':<16}  {'START':>8}  {'END':>8}  {'DUR':>5}  {'HIT @':>8}  {'POS':<12}  {'SIGNALS':<35}  {'CONF'}")
    print(f"  {'─'*3}  {'─'*16}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*35}  {'─'*4}")

    for s in seg_results:
        src  = s["source"][:15]
        if s["hit"]:
            h       = s["hit"]
            pos     = h["position"]
            hit_s   = f"{h['ts']:>8.2f}"
            sig_s   = h["label"][:35]
            conf    = confidence_str(h["n_sig"])
            tick    = "✓"
        else:
            pos     = "MISSED"
            hit_s   = "   —   "
            sig_s   = ""
            conf    = "    "
            tick    = "✗"
        print(f"  {s['index']:>3}  {src:<16}  {s['start']:>8.3f}  {s['end']:>8.3f}  {s['duration']:>5.2f}  {hit_s}  {tick} {pos:<10}  {sig_s:<35}  {conf}")

    # False positives
    if fp_events:
        print(f"\n  False Positives ({len(fp_events)}):")
        for fp in fp_events:
            print(f"    @ {fp['ts']:>9.2f}s  [{fp['label']}]  {confidence_str(fp['n_sig'])}")
    else:
        print(f"\n  False Positives: 0  ✓")

    rate = hits / total * 100 if total else 0
    print(f"\n  Detection rate: {hits}/{total}  ({rate:.1f}%)   |   FP: {len(fp_events)}")


# ==============================================================================
# OVERALL SUMMARY
# ==============================================================================

def print_overall_summary(all_video_stats):
    banner("COMPLETE EVALUATION SUMMARY")

    total_segs  = sum(v["total"] for v in all_video_stats)
    total_hits  = sum(v["hits"]  for v in all_video_stats)
    total_fp    = sum(v["fp"]    for v in all_video_stats)
    total_vids  = len(all_video_stats)

    overall_rate = total_hits / total_segs * 100 if total_segs else 0

    # Per-video table
    print(f"\n  {'VIDEO':<35}  {'SEGS':>4}  {'HITS':>4}  {'RATE':>6}  {'FP':>4}  {'STATUS'}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*10}")
    for v in all_video_stats:
        rate   = v["hits"] / v["total"] * 100 if v["total"] else 0
        status = "✓ PASS" if rate >= 50 else "✗ FAIL"
        print(f"  {v['name']:<35}  {v['total']:>4}  {v['hits']:>4}  {rate:>5.1f}%  {v['fp']:>4}  {status}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*10}")
    print(f"  {'TOTAL / AVERAGE':<35}  {total_segs:>4}  {total_hits:>4}  {overall_rate:>5.1f}%  {total_fp:>4}")

    # Hit position breakdown
    print(f"\n  Hit Position Breakdown:")
    pos_counts = {}
    for v in all_video_stats:
        for p in v["positions"]:
            pos_counts[p] = pos_counts.get(p, 0) + 1
    for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {pos:<14}: {cnt:>4}  {bar}")

    # Per-signal accuracy
    print(f"\n  Per-Signal Contribution (how many hits each signal helped claim):")
    sig_hits = {}
    sig_fp   = {}
    for v in all_video_stats:
        for sig, cnt in v["sig_hits"].items():
            sig_hits[sig] = sig_hits.get(sig, 0) + cnt
        for sig, cnt in v["sig_fp"].items():
            sig_fp[sig] = sig_fp.get(sig, 0) + cnt

    all_sigs = sorted(set(list(sig_hits.keys()) + list(sig_fp.keys())))
    print(f"  {'Signal':<15}  {'Hits':>5}  {'FP contrib':>10}  {'Precision'}")
    print(f"  {'─'*15}  {'─'*5}  {'─'*10}  {'─'*9}")
    for sig in all_sigs:
        h  = sig_hits.get(sig, 0)
        fp = sig_fp.get(sig, 0)
        total_fire = h + fp
        prec = h / total_fire * 100 if total_fire > 0 else 0.0
        print(f"  {sig:<15}  {h:>5}  {fp:>10}  {prec:>8.1f}%")

    # Raw trigger totals per signal
    print(f"\n  Raw Trigger Counts (all videos combined):")
    all_raw = {}
    for v in all_video_stats:
        for sig, cnt in v["raw_triggers"].items():
            all_raw[sig] = all_raw.get(sig, 0) + cnt
    for sig, cnt in sorted(all_raw.items(), key=lambda x: -x[1]):
        bar = "▪" * min(cnt // 5 + 1, 40)
        print(f"    {sig:<15}: {cnt:>5} total  {bar}")

    # Final scorecard
    print(f"\n{'═'*65}")
    print(f"  OVERALL DETECTION RATE   : {total_hits:>4} / {total_segs}  ({overall_rate:.2f}%)")
    print(f"  TOTAL FALSE POSITIVES    : {total_fp}")
    print(f"  VIDEOS PROCESSED         : {total_vids}")
    fpr = total_fp / (total_fp + total_hits) * 100 if (total_fp + total_hits) > 0 else 0
    print(f"  FALSE POSITIVE RATE      : {fpr:.2f}%  (FP / all claimed events)")
    print(f"{'═'*65}\n")


# ==============================================================================
# PROCESS ONE VIDEO
# ==============================================================================

def process_video(video_path, ground_truth_entry):
    video_name = os.path.basename(video_path)
    segments   = ground_truth_entry["segments"]

    # Fix obviously corrupted entries (end < start means end was mis-logged)
    for seg in segments:
        if seg["end"] < seg["start"]:
            seg["end"] = seg["start"] + seg["duration"]

    print(f"\n  Running 5 detection signals...")

    t0 = time.time()

    print(f"  [1/5] Visual I-frame spikes...")
    v_ts  = get_visual_triggers(video_path)
    print(f"        → {len(v_ts)} triggers")

    print(f"  [2/5] Motion vector proxy...")
    mv_ts = get_motion_vector_triggers(video_path)
    print(f"        → {len(mv_ts)} triggers")

    print(f"  [3/5] Audio RMS + Silence detection...")
    rms_ts, sil_ts = get_audio_rms_triggers(video_path)
    print(f"        → {len(rms_ts)} RMS spike triggers,  {len(sil_ts)} silence triggers")

    print(f"  [4/5] Spectral Flux...")
    flux_ts = get_spectral_flux_triggers(video_path)
    print(f"        → {len(flux_ts)} triggers")

    elapsed = time.time() - t0
    print(f"\n  Detection complete in {elapsed:.1f}s")

    signal_dict = {
        "Visual":       v_ts,
        "Motion":       mv_ts,
        "AudioRMS":     rms_ts,
        "Silence":      sil_ts,
        "SpectralFlux": flux_ts,
    }

    merged = fuse_all_signals(signal_dict)
    seg_results, fp_events = validate_video(video_name, segments, merged)

    print_video_report(video_name, seg_results, fp_events, signal_dict, merged)

    # Build per-signal stats for overall summary
    sig_hits = {}
    sig_fp   = {}

    for seg in seg_results:
        if seg["hit"]:
            for sig in seg["hit"]["label"].split("+"):
                sig_hits[sig] = sig_hits.get(sig, 0) + 1

    for fp in fp_events:
        for sig in fp["label"].split("+"):
            sig_fp[sig] = sig_fp.get(sig, 0) + 1

    positions = [s["hit"]["position"] for s in seg_results if s["hit"]]

    return {
        "name":         video_name,
        "total":        len(seg_results),
        "hits":         sum(1 for s in seg_results if s["hit"] is not None),
        "fp":           len(fp_events),
        "positions":    positions,
        "sig_hits":     sig_hits,
        "sig_fp":       sig_fp,
        "raw_triggers": {sig: len(ts) for sig, ts in signal_dict.items()},
    }


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    gt_path = os.path.join(SCRIPT_DIR, GROUND_TRUTH_JSON)
    if not os.path.exists(gt_path):
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        sys.exit(1)

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    gt_map = {entry["target_video"]: entry for entry in ground_truth}

    banner("MULTI-MODAL VIOLENT CONTENT DETECTOR  —  FULL EVALUATION")
    print(f"  Ground truth: {GROUND_TRUTH_JSON}  ({len(ground_truth)} videos)")
    print(f"  Attacked folder: {ATTACKED_DIR}/")
    print(f"  Signals: Visual | Motion | AudioRMS | Silence | SpectralFlux")
    print(f"  Scoring: hit = any trigger within [start-{BOUNDARY_TOLERANCE}s ... end+{BOUNDARY_TOLERANCE}s]")
    print(f"  Position labels: START | END | IN-BETWEEN | START+END")

    attacked_videos = sorted([
        os.path.join(SCRIPT_DIR, ATTACKED_DIR, e["target_video"])
        for e in ground_truth
    ])

    all_video_stats = []
    total_vids = len(attacked_videos)

    for vid_idx, video_path in enumerate(attacked_videos):
        video_name = os.path.basename(video_path)

        banner(f"VIDEO {vid_idx+1}/{total_vids}: {video_name}")

        if not os.path.exists(video_path):
            print(f"  [!] File not found, skipping: {video_path}")
            continue

        if video_name not in gt_map:
            print(f"  [!] No ground truth entry for {video_name}, skipping.")
            continue

        stats = process_video(video_path, gt_map[video_name])
        all_video_stats.append(stats)

    print_overall_summary(all_video_stats)