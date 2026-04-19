import subprocess
import json
import numpy as np
from scipy.io import wavfile
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
SILENCE_THRESHOLD_RATIO = 0.05   # was 0.15 — tighter: audio must drop to 5% of bg mean (near-total silence only)
SILENCE_MIN_DURATION    = 2.0    # was 0.4  — silence must hold for 2s (filters breath pauses, word gaps)
SILENCE_COOLDOWN        = 30.0   # was 3.0  — one trigger per 30s max (violent clips are sparse)

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


def validate_video(video_name, segments, merged_events):
    seg_results = []
    fp_events   = []
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
# PRINT VIDEO REPORT  (console — unchanged)
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

    print(f"  Raw triggers per signal:")
    for sig, ts_list in signal_dict.items():
        bar = "▪" * min(len(ts_list), 40)
        print(f"    {sig:<15} {len(ts_list):>4} triggers  {bar}")
    print(f"  Fused events (after clustering): {len(merged_events)}")

    print(f"\n  {'#':>3}  {'SOURCE':<16}  {'START':>8}  {'END':>8}  {'DUR':>5}  {'HIT @':>8}  {'POS':<12}  {'SIGNALS':<35}  {'CONF'}")
    print(f"  {'─'*3}  {'─'*16}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*35}  {'─'*4}")

    for s in seg_results:
        src  = s["source"][:15]
        if s["hit"]:
            h    = s["hit"]
            print(f"  {s['index']:>3}  {src:<16}  {s['start']:>8.3f}  {s['end']:>8.3f}  {s['duration']:>5.2f}  {h['ts']:>8.2f}  ✓ {h['position']:<10}  {h['label'][:35]:<35}  {confidence_str(h['n_sig'])}")
        else:
            print(f"  {s['index']:>3}  {src:<16}  {s['start']:>8.3f}  {s['end']:>8.3f}  {s['duration']:>5.2f}  {'—':>8}  ✗ {'MISSED':<10}  {'':35}  {'':4}")

    if fp_events:
        print(f"\n  False Positives ({len(fp_events)}):")
        for fp in fp_events:
            print(f"    @ {fp['ts']:>9.2f}s  [{fp['label']}]  {confidence_str(fp['n_sig'])}")
    else:
        print(f"\n  False Positives: 0  ✓")

    rate = hits / total * 100 if total else 0
    print(f"\n  Detection rate: {hits}/{total}  ({rate:.1f}%)   |   FP: {len(fp_events)}")


# ==============================================================================
# OVERALL SUMMARY  (console — unchanged)
# ==============================================================================

def print_overall_summary(all_video_stats):
    banner("COMPLETE EVALUATION SUMMARY")

    total_segs   = sum(v["total"] for v in all_video_stats)
    total_hits   = sum(v["hits"]  for v in all_video_stats)
    total_fp     = sum(v["fp"]    for v in all_video_stats)
    total_vids   = len(all_video_stats)
    overall_rate = total_hits / total_segs * 100 if total_segs else 0

    print(f"\n  {'VIDEO':<35}  {'SEGS':>4}  {'HITS':>4}  {'RATE':>6}  {'FP':>4}  {'STATUS'}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*10}")
    for v in all_video_stats:
        rate   = v["hits"] / v["total"] * 100 if v["total"] else 0
        status = "✓ PASS" if rate >= 50 else "✗ FAIL"
        print(f"  {v['name']:<35}  {v['total']:>4}  {v['hits']:>4}  {rate:>5.1f}%  {v['fp']:>4}  {status}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*4}  {'─'*10}")
    print(f"  {'TOTAL / AVERAGE':<35}  {total_segs:>4}  {total_hits:>4}  {overall_rate:>5.1f}%  {total_fp:>4}")

    print(f"\n  Hit Position Breakdown:")
    pos_counts = {}
    for v in all_video_stats:
        for p in v["positions"]:
            pos_counts[p] = pos_counts.get(p, 0) + 1
    for pos, cnt in sorted(pos_counts.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {pos:<14}: {cnt:>4}  {bar}")

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

    print(f"\n  Raw Trigger Counts (all videos combined):")
    all_raw = {}
    for v in all_video_stats:
        for sig, cnt in v["raw_triggers"].items():
            all_raw[sig] = all_raw.get(sig, 0) + cnt
    for sig, cnt in sorted(all_raw.items(), key=lambda x: -x[1]):
        bar = "▪" * min(cnt // 5 + 1, 40)
        print(f"    {sig:<15}: {cnt:>5} total  {bar}")

    print(f"\n{'═'*65}")
    print(f"  OVERALL DETECTION RATE   : {total_hits:>4} / {total_segs}  ({overall_rate:.2f}%)")
    print(f"  TOTAL FALSE POSITIVES    : {total_fp}")
    print(f"  VIDEOS PROCESSED         : {total_vids}")
    fpr = total_fp / (total_fp + total_hits) * 100 if (total_fp + total_hits) > 0 else 0
    print(f"  FALSE POSITIVE RATE      : {fpr:.2f}%  (FP / all claimed events)")
    print(f"{'═'*65}\n")


# ==============================================================================
# REPORT WRITER — writes everything to report.txt
# ==============================================================================

W = 80  # report page width

def rl(f, text=""):
    """Write one line to the report file."""
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
    """
    Appends a full evaluation report to report_path.
    all_video_data: list of rich dicts collected during process_video().
    """
    with open(report_path, "a", encoding="utf-8") as f:

        # ── RUN HEADER ────────────────────────────────────────────────────────
        rheader(f, f"MULTI-MODAL VIOLENT CONTENT DETECTOR  -  EVALUATION REPORT")
        rl(f)
        rl(f, f"  Run timestamp      : {run_ts}")
        rl(f, f"  Ground truth file  : {GROUND_TRUTH_JSON}")
        rl(f, f"  Attacked folder    : {ATTACKED_DIR}/")
        rl(f, f"  Videos processed   : {len(all_video_data)}")
        rl(f, f"  Report file        : {report_path}")
        rl(f)
        rl(f, f"  Signals            : Visual | Motion | AudioRMS | Silence | SpectralFlux")
        rl(f, f"  Fusion window      : {TOLERANCE_SEC}s  (events within this gap merged into one)")
        rl(f, f"  Min signals needed : {MIN_SIGNALS_TO_CONFIRM}")
        rl(f, f"  Require multimodal : {REQUIRE_MULTIMODAL}")
        rl(f, f"  Lockout period     : {LOCKOUT_PERIOD}s  (no triggers in first {LOCKOUT_PERIOD}s)")
        rl(f)
        rl(f, f"  Scoring rule       : A segment counts as HIT if any fused event lands within")
        rl(f, f"                       [start - {BOUNDARY_TOLERANCE}s]  to  [end + {BOUNDARY_TOLERANCE}s]")
        rl(f, f"  Hit positions      : START      = event within {BOUNDARY_TOLERANCE}s of segment start")
        rl(f, f"                       END        = event within {BOUNDARY_TOLERANCE}s of segment end")
        rl(f, f"                       IN-BETWEEN = event inside the segment body")
        rl(f, f"                       START+END  = segment so short both boundaries overlap")

        # ── CONFIGURATION ─────────────────────────────────────────────────────
        rheader(f, "CONFIGURATION PARAMETERS", char="-")
        rl(f, f"  {'Parameter':<35}  Value")
        rl(f, f"  {'-'*35}  {'-'*20}")
        params = [
            ("BOUNDARY_TOLERANCE",      BOUNDARY_TOLERANCE),
            ("TOLERANCE_SEC",           TOLERANCE_SEC),
            ("LOCKOUT_PERIOD",          LOCKOUT_PERIOD),
            ("--- Visual ---",          ""),
            ("VISUAL_MAX_FRAMES",       VISUAL_MAX_FRAMES),
            ("VISUAL_MIN_WARMUP",       VISUAL_MIN_WARMUP),
            ("K_VISUAL_THRESHOLD",      K_VISUAL_THRESHOLD),
            ("VISUAL_COOLDOWN",         VISUAL_COOLDOWN),
            ("VISUAL_PERSIST_NEEDED",   VISUAL_PERSIST_NEEDED),
            ("VISUAL_PERSIST_WINDOW",   VISUAL_PERSIST_WINDOW),
            ("--- Motion ---",          ""),
            ("MV_WINDOW",               MV_WINDOW),
            ("MV_MIN_WARMUP",           MV_MIN_WARMUP),
            ("K_MV_THRESHOLD",          K_MV_THRESHOLD),
            ("MV_COOLDOWN",             MV_COOLDOWN),
            ("MV_PERSIST_NEEDED",       MV_PERSIST_NEEDED),
            ("MV_PERSIST_WINDOW",       MV_PERSIST_WINDOW),
            ("GOP_SHORT_RATIO",         GOP_SHORT_RATIO),
            ("--- Audio RMS ---",       ""),
            ("AUDIO_BUFFER_SEC",        AUDIO_BUFFER_SEC),
            ("AUDIO_MICRO_WINDOW_SEC",  AUDIO_MICRO_WINDOW_SEC),
            ("K_AUDIO_RMS_THRESHOLD",   K_AUDIO_RMS_THRESHOLD),
            ("AUDIO_NOISE_FLOOR",       AUDIO_NOISE_FLOOR),
            ("AUDIO_COOLDOWN",          AUDIO_COOLDOWN),
            ("AUDIO_PERSIST_NEEDED",    AUDIO_PERSIST_NEEDED),
            ("--- Silence ---",         ""),
            ("SILENCE_THRESHOLD_RATIO", SILENCE_THRESHOLD_RATIO),
            ("SILENCE_MIN_DURATION",    SILENCE_MIN_DURATION),
            ("SILENCE_COOLDOWN",        SILENCE_COOLDOWN),
            ("--- Spectral Flux ---",   ""),
            ("FLUX_WINDOW_SEC",         FLUX_WINDOW_SEC),
            ("FLUX_BUFFER_SEC",         FLUX_BUFFER_SEC),
            ("K_FLUX_THRESHOLD",        K_FLUX_THRESHOLD),
            ("FLUX_COOLDOWN",           FLUX_COOLDOWN),
            ("FLUX_PERSIST_NEEDED",     FLUX_PERSIST_NEEDED),
        ]
        for k, v in params:
            if v == "":
                rl(f, f"  {k}")
            else:
                rl(f, f"  {k:<35}  {v}")

        # ── PER-VIDEO DETAILED SECTIONS ────────────────────────────────────────
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
            rl(f, f"  Result          : {hits}/{total} segments detected  ({rate:.1f}%)")
            rl(f, f"  False positives : {fp_n}  (video FPR: {fpr_v:.1f}%)")
            rl(f, f"  Processing time : {elapsed:.1f}s")
            rl(f)

            # -- Raw trigger counts per signal with bar chart
            rl(f, "  RAW TRIGGER COUNTS PER SIGNAL:")
            max_raw = max((len(v) for v in sig_dict.values()), default=1)
            for sig, ts_list in sig_dict.items():
                rbar(f, sig, len(ts_list), max(max_raw, 1), width=35, unit=" triggers")
            rl(f, f"  {'':22}   Total fused events after clustering: {len(merged)}")
            rl(f)

            # -- All raw trigger timestamps per signal
            rl(f, "  ALL RAW TRIGGER TIMESTAMPS PER SIGNAL:")
            for sig, ts_list in sig_dict.items():
                times_str = "  ".join(f"{t:.3f}s" for t in sorted(ts_list)) if ts_list else "(none)"
                rl(f, f"  {sig:<15} ({len(ts_list):>4} total): {times_str}")
            rl(f)

            # -- All fused events with outcome
            rl(f, "  ALL FUSED EVENTS (chronological):")
            rl(f, f"  {'TIME':>10}  {'SIGNALS':<35}  {'N':>2}  {'CONF':<5}  OUTCOME")
            rl(f, f"  {'─'*10}  {'─'*35}  {'─'*2}  {'─'*5}  {'─'*25}")
            # Build hit lookup: trigger time -> which segment it claimed
            hit_lookup = {}
            for s in seg_res:
                if s["hit"]:
                    hit_lookup[round(s["hit"]["ts"], 2)] = (s["index"], s["hit"]["position"])
            for ts, label, n_sig in merged:
                conf = "HIGH" if n_sig >= 3 else ("MED" if n_sig == 2 else "LOW")
                key  = round(ts, 2)
                if key in hit_lookup:
                    idx, pos = hit_lookup[key]
                    outcome  = f"HIT  seg {idx:>2}  pos={pos}"
                else:
                    outcome = "FALSE POSITIVE"
                rl(f, f"  {ts:>10.3f}s  {label:<35}  {n_sig:>2}  {conf:<5}  {outcome}")
            rl(f)

            # -- Segment detection table (full detail)
            rl(f, "  SEGMENT DETECTION TABLE:")
            rl(f,
               f"  {'#':>3}  {'SOURCE':<16}  {'START':>9}  {'END':>9}  {'DUR':>7}  "
               f"{'HIT @':>9}  {'OFFSET':>9}  {'POS':<12}  {'SIGNALS':<30}  CONF")
            rl(f,
               f"  {'─'*3}  {'─'*16}  {'─'*9}  {'─'*9}  {'─'*7}  "
               f"{'─'*9}  {'─'*9}  {'─'*12}  {'─'*30}  {'─'*4}")
            for s in seg_res:
                src = s["source"][:15]
                if s["hit"]:
                    h      = s["hit"]
                    offset = h["ts"] - s["start"]
                    conf   = "HIGH" if h["n_sig"] >= 3 else ("MED" if h["n_sig"] == 2 else "LOW")
                    rl(f,
                       f"  {s['index']:>3}  {src:<16}  {s['start']:>9.3f}  {s['end']:>9.3f}  "
                       f"{s['duration']:>7.3f}  {h['ts']:>9.3f}  {offset:>+9.3f}  "
                       f"  {h['position']:<10}  {h['label'][:30]:<30}  {conf}")
                else:
                    rl(f,
                       f"  {s['index']:>3}  {src:<16}  {s['start']:>9.3f}  {s['end']:>9.3f}  "
                       f"{s['duration']:>7.3f}  {'---':>9}  {'---':>9}  "
                       f"  {'MISSED':<10}  {'':30}  {'':4}")
            rl(f)

            # -- Missed segment analysis
            missed = [s for s in seg_res if not s["hit"]]
            if missed:
                rl(f, f"  MISSED SEGMENTS ({len(missed)}):")
                for s in missed:
                    mid     = (s["start"] + s["end"]) / 2.0
                    if merged:
                        closest    = min(merged, key=lambda e: abs(e[0] - mid))
                        dist_start = abs(closest[0] - s["start"])
                        dist_mid   = abs(closest[0] - mid)
                        dist_end   = abs(closest[0] - s["end"])
                        rl(f, f"  Seg {s['index']:>2}  src={s['source']}  "
                               f"{s['start']:.3f}s-{s['end']:.3f}s  dur={s['duration']:.3f}s")
                        rl(f, f"         Closest fused event : {closest[0]:.3f}s  "
                               f"signals={closest[1]}  conf={'HIGH' if closest[2]>=3 else 'MED' if closest[2]==2 else 'LOW'}")
                        rl(f, f"         Distance from start : {dist_start:.3f}s")
                        rl(f, f"         Distance from mid   : {dist_mid:.3f}s")
                        rl(f, f"         Distance from end   : {dist_end:.3f}s")
                        rl(f, f"         (Boundary tolerance : {BOUNDARY_TOLERANCE}s  -> "
                               f"{'JUST MISSED' if dist_start < BOUNDARY_TOLERANCE * 2 else 'FAR MISS'})")
                    else:
                        rl(f, f"  Seg {s['index']:>2}  src={s['source']}  "
                               f"{s['start']:.3f}s-{s['end']:.3f}s  dur={s['duration']:.3f}s  "
                               f"(no fused events in video at all)")
                rl(f)
            else:
                rl(f, "  MISSED SEGMENTS: none -- all segments detected")
                rl(f)

            # -- False positive detail
            if fp_events:
                rl(f, f"  FALSE POSITIVE EVENTS ({fp_n}):")
                rl(f, f"  {'#':>4}  {'TIME':>10}  {'SIGNALS':<35}  {'N':>2}  CONF")
                rl(f, f"  {'─'*4}  {'─'*10}  {'─'*35}  {'─'*2}  {'─'*4}")
                for i, fp in enumerate(fp_events, 1):
                    conf = "HIGH" if fp["n_sig"] >= 3 else ("MED" if fp["n_sig"] == 2 else "LOW")
                    rl(f, f"  {i:>4}  {fp['ts']:>10.3f}s  {fp['label']:<35}  {fp['n_sig']:>2}  {conf}")
                # FP spacing statistics
                if len(fp_events) > 1:
                    gaps = [fp_events[i+1]["ts"] - fp_events[i]["ts"]
                            for i in range(len(fp_events) - 1)]
                    rl(f)
                    rl(f, f"  FP spacing stats:")
                    rl(f, f"    min gap  : {min(gaps):.3f}s")
                    rl(f, f"    max gap  : {max(gaps):.3f}s")
                    rl(f, f"    mean gap : {np.mean(gaps):.3f}s")
                    rl(f, f"    median   : {np.median(gaps):.3f}s")
                    rl(f, f"    std      : {np.std(gaps):.3f}s")
                    # Count FPs that are very close together (< 10s) vs spread out
                    close = sum(1 for g in gaps if g < 10.0)
                    rl(f, f"    gaps < 10s: {close}  (possible burst FP region)")
            else:
                rl(f, "  FALSE POSITIVE EVENTS: none")
            rl(f)

            # -- Per-signal contribution breakdown for this video
            rl(f, "  PER-SIGNAL CONTRIBUTION (this video):")
            rl(f, f"  {'Signal':<15}  {'Raw':>5}  {'Hits':>5}  {'FP':>6}  {'Precision':>10}  All raw timestamps")
            rl(f, f"  {'─'*15}  {'─'*5}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*30}")
            for sig, ts_list in sig_dict.items():
                raw  = len(ts_list)
                h    = sum(1 for s in seg_res
                           if s["hit"] and sig in s["hit"]["label"].split("+"))
                fp   = sum(1 for e  in fp_events if sig in e["label"].split("+"))
                prec = h / (h + fp) * 100 if (h + fp) > 0 else 0.0
                times_str = "  ".join(f"{t:.2f}s" for t in sorted(ts_list)[:20])
                if len(ts_list) > 20:
                    times_str += f"  ... (+{len(ts_list)-20} more)"
                rl(f, f"  {sig:<15}  {raw:>5}  {h:>5}  {fp:>6}  {prec:>9.1f}%  {times_str}")
            rl(f)

        # ── OVERALL SUMMARY ────────────────────────────────────────────────────
        rheader(f, "OVERALL SUMMARY")

        total_segs = sum(len(vd["seg_results"]) for vd in all_video_data)
        total_hits = sum(sum(1 for s in vd["seg_results"] if s["hit"]) for vd in all_video_data)
        total_fp   = sum(len(vd["fp_events"]) for vd in all_video_data)
        overall    = total_hits / total_segs * 100 if total_segs else 0
        fpr        = total_fp / (total_fp + total_hits) * 100 if (total_fp + total_hits) > 0 else 0
        total_time = sum(vd["elapsed"] for vd in all_video_data)

        # Per-video summary table
        rl(f)
        rl(f, f"  {'VIDEO':<38}  {'SEGS':>4}  {'HITS':>4}  {'RATE':>6}  {'FP':>5}  {'VID FPR':>8}  {'TIME':>7}  STATUS")
        rl(f, f"  {'─'*38}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*8}")
        for vd in all_video_data:
            sr   = vd["seg_results"]
            fpe  = vd["fp_events"]
            h    = sum(1 for s in sr if s["hit"])
            t    = len(sr)
            r    = h / t * 100 if t else 0
            fp   = len(fpe)
            vfpr = fp / (fp + h) * 100 if (fp + h) > 0 else 0.0
            st   = "PASS" if r >= 50 else "FAIL"
            rl(f,
               f"  {vd['name']:<38}  {t:>4}  {h:>4}  {r:>5.1f}%  {fp:>5}  {vfpr:>7.1f}%  "
               f"{vd['elapsed']:>6.1f}s  {st}")
        rl(f, f"  {'─'*38}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*8}")
        rl(f,
           f"  {'TOTAL':<38}  {total_segs:>4}  {total_hits:>4}  {overall:>5.1f}%  "
           f"{total_fp:>5}  {fpr:>7.1f}%  {total_time:>6.1f}s")
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
            rbar(f, pos, cnt, max_pos, width=30, unit=f"  ({pct:.1f}% of hits)")
        rl(f)

        # Trigger offset statistics
        offsets = [s["hit"]["ts"] - s["start"]
                   for vd in all_video_data
                   for s in vd["seg_results"] if s["hit"]]
        if offsets:
            rl(f, "  TRIGGER OFFSET FROM SEGMENT START (all confirmed hits):")
            rl(f, f"  Count  : {len(offsets)}")
            rl(f, f"  Min    : {min(offsets):+.3f}s")
            rl(f, f"  Max    : {max(offsets):+.3f}s")
            rl(f, f"  Mean   : {np.mean(offsets):+.3f}s")
            rl(f, f"  Median : {np.median(offsets):+.3f}s")
            rl(f, f"  Std    : {np.std(offsets):.3f}s")
            rl(f, f"  (negative = trigger fired BEFORE segment start; positive = AFTER)")
            neg = sum(1 for o in offsets if o < 0)
            pos = sum(1 for o in offsets if o > 0)
            rl(f, f"  Early triggers (before start) : {neg}  ({neg/len(offsets)*100:.1f}%)")
            rl(f, f"  Late  triggers (after  start) : {pos}  ({pos/len(offsets)*100:.1f}%)")
            rl(f)

        # Segment duration vs detection rate
        rl(f, "  DETECTION RATE BY SEGMENT DURATION BUCKET:")
        buckets = [("<3s", 0, 3), ("3-5s", 3, 5), ("5-7s", 5, 7), ("7-10s", 7, 10), (">10s", 10, 9999)]
        rl(f, f"  {'Bucket':<8}  {'Hits':>5}  {'Total':>6}  {'Rate':>7}  Notes")
        rl(f, f"  {'─'*8}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*25}")
        for bname, blo, bhi in buckets:
            bh = sum(1 for vd in all_video_data
                     for s in vd["seg_results"]
                     if blo <= s["duration"] < bhi and s["hit"])
            bt = sum(1 for vd in all_video_data
                     for s in vd["seg_results"]
                     if blo <= s["duration"] < bhi)
            br = bh / bt * 100 if bt > 0 else 0.0
            note = "hardest (clip ends before most signals build)" if bname == "<3s" else ""
            rl(f, f"  {bname:<8}  {bh:>5}  {bt:>6}  {br:>6.1f}%  {note}")
        rl(f)

        # Per-signal accuracy across ALL videos
        rl(f, "  PER-SIGNAL ACCURACY (all videos combined):")
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
            list(sig_hits_all.keys()) + list(sig_fp_all.keys()) + list(sig_raw_all.keys())
        ))
        rl(f, f"  {'Signal':<15}  {'Raw':>6}  {'Hits':>5}  {'FP':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
        rl(f, f"  {'─'*15}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*6}")
        for sig in all_sigs:
            raw  = sig_raw_all.get(sig, 0)
            h    = sig_hits_all.get(sig, 0)
            fp   = sig_fp_all.get(sig, 0)
            prec = h / (h + fp) * 100 if (h + fp) > 0 else 0.0
            rec  = h / total_segs * 100 if total_segs > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            rl(f, f"  {sig:<15}  {raw:>6}  {h:>5}  {fp:>6}  {prec:>9.1f}%  {rec:>7.1f}%  {f1:>5.1f}%")
        rl(f)

        # Signal co-occurrence on true hits
        rl(f, "  SIGNAL CO-OCCURRENCE ON TRUE HITS:")
        rl(f, "  (how many confirmed hits had BOTH signals firing within fusion window)")
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
            row = "  ".join(f"{cooc[a].get(b, 0):>10}" for b in all_sigs)
            rl(f, f"  {a:<15}  {row}")
        rl(f)

        # Which signal pairs most often confirm together on FPs
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
            row = "  ".join(f"{fp_cooc[a].get(b, 0):>10}" for b in all_sigs)
            rl(f, f"  {a:<15}  {row}")
        rl(f)

        # Per-video signal raw trigger counts table
        rl(f, "  PER-VIDEO RAW TRIGGER COUNTS PER SIGNAL:")
        sig_names = list(all_video_data[0]["signal_dict"].keys()) if all_video_data else []
        rl(f, f"  {'VIDEO':<38}  " + "  ".join(f"{s[:10]:>10}" for s in sig_names) + "  FUSED")
        rl(f, f"  {'─'*38}  " + "  ".join("─"*10 for _ in sig_names) + "  ─────")
        for vd in all_video_data:
            counts = [len(vd["signal_dict"].get(s, [])) for s in sig_names]
            rl(f,
               f"  {vd['name']:<38}  " +
               "  ".join(f"{c:>10}" for c in counts) +
               f"  {len(vd['merged_events']):>5}")
        rl(f, f"  {'─'*38}  " + "  ".join("─"*10 for _ in sig_names) + "  ─────")
        totals = [sum(len(vd["signal_dict"].get(s, [])) for vd in all_video_data) for s in sig_names]
        rl(f,
           f"  {'TOTAL':<38}  " +
           "  ".join(f"{c:>10}" for c in totals) +
           f"  {sum(len(vd['merged_events']) for vd in all_video_data):>5}")
        rl(f)

        # ── FINAL SCORECARD ───────────────────────────────────────────────────
        rl(f, "=" * W)
        rl(f, f"  OVERALL DETECTION RATE  : {total_hits:>4} / {total_segs}  ({overall:.2f}%)")
        rl(f, f"  TOTAL FALSE POSITIVES   : {total_fp}")
        rl(f, f"  FALSE POSITIVE RATE     : {fpr:.2f}%  (FP / all claimed events)")
        rl(f, f"  VIDEOS PROCESSED        : {len(all_video_data)}")
        rl(f, f"  TOTAL PROCESSING TIME   : {total_time:.1f}s")
        rl(f, f"  VIDEOS PASSING (>=50%)  : {sum(1 for vd in all_video_data if sum(1 for s in vd['seg_results'] if s['hit'])/len(vd['seg_results'])*100 >= 50)}/{len(all_video_data)}")
        rl(f, "=" * W)
        rl(f)


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
    print(f"        -> {len(v_ts)} triggers")

    print(f"  [2/5] Motion vector proxy...")
    mv_ts = get_motion_vector_triggers(video_path)
    print(f"        -> {len(mv_ts)} triggers")

    print(f"  [3/5] Audio RMS + Silence detection...")
    rms_ts, sil_ts = get_audio_rms_triggers(video_path)
    print(f"        -> {len(rms_ts)} RMS spike triggers,  {len(sil_ts)} silence triggers")

    print(f"  [4/5] Spectral Flux...")
    flux_ts = get_spectral_flux_triggers(video_path)
    print(f"        -> {len(flux_ts)} triggers")

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

    # Build per-signal stats for overall summary (console)
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

    # Return everything for the report writer
    return {
        # for console summary (original fields)
        "name":         video_name,
        "total":        len(seg_results),
        "hits":         sum(1 for s in seg_results if s["hit"] is not None),
        "fp":           len(fp_events),
        "positions":    positions,
        "sig_hits":     sig_hits,
        "sig_fp":       sig_fp,
        "raw_triggers": {sig: len(ts) for sig, ts in signal_dict.items()},
        # for report writer (rich fields)
        "seg_results":   seg_results,
        "fp_events":     fp_events,
        "signal_dict":   signal_dict,
        "merged_events": merged,
        "elapsed":       elapsed,
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

    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    banner("MULTI-MODAL VIOLENT CONTENT DETECTOR  -  FULL EVALUATION")
    print(f"  Ground truth: {GROUND_TRUTH_JSON}  ({len(ground_truth)} videos)")
    print(f"  Attacked folder: {ATTACKED_DIR}/")
    print(f"  Signals: Visual | Motion | AudioRMS | Silence | SpectralFlux")
    print(f"  Scoring: hit = any trigger within [start-{BOUNDARY_TOLERANCE}s ... end+{BOUNDARY_TOLERANCE}s]")
    print(f"  Position labels: START | END | IN-BETWEEN | START+END")
    print(f"  Report will be written to: {REPORT_FILE}")

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

    # Console summary (original — unchanged)
    print_overall_summary(all_video_stats)

    # Write detailed report.txt
    report_path = os.path.join(SCRIPT_DIR, REPORT_FILE)
    write_full_report(report_path, all_video_stats, run_ts)
    print(f"[+] Detailed report written to: {report_path}")