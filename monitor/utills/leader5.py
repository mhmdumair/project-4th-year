import subprocess
import json
import numpy as np
import os
import sys
import time
import datetime
from collections import deque

# ==============================================================================
# OPTIMIZATION GRID HYPERPARAMETERS (Directly Synced with try.py Schema)
# ==============================================================================

WORKER_DATA_DIR     = "worker_data_temp"
GROUND_TRUTH_JSON   = "attack.json"
REPORT_FILE         = "report.txt"

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FFMPEG_EXE  = "ffmpeg"
FFPROBE_EXE = "ffprobe"

# Distributed Ingest Constants
CHUNK_DURATION_SEC  = 5.0
AUDIO_OVERLAP_SEC   = 0.05
NUM_WORKERS         = 10

# --- Fusion / Scoring Scoring Matrix ---
BOUNDARY_TOLERANCE      = 2.5
TOLERANCE_SEC           = 2.5
MIN_SIGNALS_TO_CONFIRM  = 1
REQUIRE_MULTIMODAL      = False
LOCKOUT_PERIOD          = 5.0
MERGE_WINDOW            = 10.0

# --- Visual Signal Parameters ---
VISUAL_MAX_FRAMES       = 30
VISUAL_MIN_WARMUP       = 10
K_VISUAL_THRESHOLD      = 2.5  # <--- Modify this constant to optimize recall bounds!
VISUAL_COOLDOWN         = 3.0
VISUAL_PERSIST_NEEDED   = 2
VISUAL_PERSIST_WINDOW   = 4    
SCENE_CHANGE_RATIO      = 8.0
SCENE_CHANGE_TOLERANCE  = 0.5
IFRAME_SANDWICH_ENABLED = True
IFRAME_CONTEXT_WINDOW   = 5
IFRAME_STABILITY_RATIO  = 3.0
IFRAME_MIN_CONTEXT      = 3

# --- Motion Signal Parameters ---
MV_WINDOW               = 60
MV_MIN_WARMUP           = 30
K_MV_THRESHOLD          = 4.5  # <--- Modify this constant to optimize recall bounds!
MV_COOLDOWN             = 5.0
MV_PERSIST_NEEDED       = 3
MV_PERSIST_WINDOW       = 5
GOP_SHORT_RATIO         = 0.25
MOTION_JERK_ENABLED     = True
JERK_WINDOW             = 5
JERK_ACCEL_THRESHOLD    = 2.5

# --- Acoustic Energy Parameters ---
AUDIO_SAMPLE_RATE       = 44100
AUDIO_BUFFER_SEC        = 15.0
AUDIO_MICRO_WINDOW_SEC  = 0.5
K_AUDIO_RMS_THRESHOLD   = 3.0  # <--- Modify this constant to optimize recall bounds!
AUDIO_NOISE_FLOOR       = 800.0
AUDIO_COOLDOWN          = 3.0
AUDIO_PERSIST_NEEDED    = 2

# --- Silence Parameters ---
SILENCE_THRESHOLD_RATIO = 0.05
SILENCE_MIN_DURATION    = 2.0
SILENCE_COOLDOWN        = 30.0

# --- Spectral Flux Parameters ---
FLUX_SAMPLE_RATE        = 22050
FLUX_WINDOW_SEC         = 0.05
FLUX_BUFFER_SEC         = 10.0
K_FLUX_THRESHOLD        = 3.0  # <--- Modify this constant to optimize recall bounds!
FLUX_COOLDOWN           = 3.0
FLUX_PERSIST_NEEDED     = 3

# --- Trust Matrix Constraints ---
VISUAL_MOTION_SIGNALS   = {"Visual", "Motion"}
AUDIO_SIGNALS           = {"AudioRMS", "Silence", "SpectralFlux"}
AUDIO_UPGRADE_WINDOW    = 2.5
MIN_EVENT_GAP           = 0.5

# --- Selection Layer Matrix ---
SELECTION_WINDOW_SEC    = 30.0
SELECTION_OVERLAP       = 0.1
SELECTION_MIN_GAP       = 0.5

W_CONFIDENCE    = 0.50
W_MAGNITUDE     = 0.05
W_ACCELERATION  = 0.15
W_EDGE_CHANGE   = 0.10
W_UNIFORMITY    = 0.05
W_PERSISTENCE   = 0.15

ALL_SIGNALS = ["Visual", "Motion", "AudioRMS", "Silence", "SpectralFlux"]

# ==============================================================================
# SHARED EVALUATION UTILITIES
# ==============================================================================

def confidence_label(weight):
    if weight >= 3:   return "WIDE"
    elif weight == 2: return "HIGH"
    else:             return "LOW"

def confidence_score(weight):
    if weight >= 3:   return 1.0
    elif weight == 2: return 0.65
    else:             return 0.30

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

# ==============================================================================
# STATEFUL LEADER CORE PROCESSING DETECTOR (Direct Port from try.py)
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
        frames = []
        audio = []
        flux = []
        
        for pkt in packets:
            start = pkt["chunk_start_time"]
            fps = pkt["fps"]
            
            for idx, frm in enumerate(pkt["frames"]):
                t = start + idx / fps
                frames.append((t, frm["type"], int(frm["size"])))
            
            for rms_item in pkt["audio_rms"]:
                if rms_item["time"] >= start - 0.001:
                    audio.append((rms_item["time"], rms_item["rms"]))
            
            for flux_item in pkt["spectral_flux"]:
                if flux_item["time"] >= start - 0.001:
                    flux.append((flux_item["time"], flux_item["flux"]))
        
        frames.sort(key=lambda x: x[0])
        audio.sort(key=lambda x: x[0])
        flux.sort(key=lambda x: x[0])
        return frames, audio, flux

    def run_detection(self, packets: list) -> dict:
        frames, audio_samples, flux_samples = self._build_timeline(packets)

        v_size_buf = deque(maxlen=VISUAL_MAX_FRAMES)
        v_spike_flags = deque(maxlen=VISUAL_PERSIST_WINDOW)
        v_last_trig = -999.0
        v_triggers = []
        v_sizes_sandwich = []
        scene_trans_end = 0.0

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

                if (t > LOCKOUT_PERIOD and len(mv_spike_flags) == VISUAL_PERSIST_WINDOW
                        and sum(mv_spike_flags) >= VISUAL_PERSIST_NEEDED
                        and (t - v_last_trig) > VISUAL_COOLDOWN
                        and (t - v_last_trig) > MIN_EVENT_GAP):
                    if self._detect_jerk(jerk_history):
                        mv_triggers.append(t)
                        mv_last_trig = t
                    mv_spike_flags.clear()

        mv_triggers.sort()
        deduped_mv, last_t = [], -999.0
        for t in mv_triggers:
            if t - last_t > MV_COOLDOWN:
                deduped_mv.append(t)
                last_t = t

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
            "Visual": v_triggers, "Motion": deduped_mv,
            "AudioRMS": rms_triggers, "Silence": sil_triggers, "SpectralFlux": flux_triggers,
        }

# ==============================================================================
# PIPELINE INTEGRATION DEEP SIGNAL PROCESSING FUNCTIONS
# ==============================================================================

def fuse_all_signals(signal_dict: dict) -> list:
    all_events = sorted([{"time": t, "signal": lbl} for lbl, ts in signal_dict.items() for t in ts], key=lambda x: x["time"])
    if not all_events: return []
    used = [False] * len(all_events)
    clusters = []
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
        if not vm_sigs: continue
        avg_t = round(float(np.mean([e["time"] for e in cluster])), 2)
        base_weight = len(set(vm_sigs))
        audio_present = {e["signal"] for e in cluster if e["signal"] in AUDIO_SIGNALS and abs(e["time"] - avg_t) <= AUDIO_UPGRADE_WINDOW}
        weight = base_weight + len(audio_present)
        all_sigs = sorted(set(vm_sigs) | audio_present)
        default_feats = {k: 0.5 for k in ["magnitude", "acceleration", "edge_change", "uniformity", "persistence"]}
        score = compute_trigger_score(weight, default_feats)
        merged.append((avg_t, "+".join(all_sigs), len(all_sigs), min(weight, 5), default_feats, score))
    return sorted(merged, key=lambda x: x[0])

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
            result.append((round(float(np.mean([e[0] for e in group])), 2), "+".join(sorted(all_lbl)), len(all_lbl), 3, avg_feats, score))
        else: result.append(group[0])
        i = j
    return result

def select_strongest_per_window(event_dicts, window_sec=SELECTION_WINDOW_SEC, overlap=SELECTION_OVERLAP, min_gap=SELECTION_MIN_GAP):
    if not event_dicts: return [], []
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
        if t - last_t >= min_gap: final_kept.append(i); last_t = t
    return [event_dicts[i] for i in range(len(event_dicts)) if i in kept_set], [event_dicts[i] for i in range(len(event_dicts)) if i not in kept_set]

def classify_hit_position(ts, seg):
    ns = abs(ts - seg["start"]) <= BOUNDARY_TOLERANCE
    ne = abs(ts - seg["end"]) <= BOUNDARY_TOLERANCE
    if ns and not ne: return "START"
    if ne and not ns: return "END"
    if ns and ne: return "START+END"
    if seg["start"] < ts < seg["end"]: return "IN-BETWEEN"
    return None

def validate_video(video_name, segments, final_events):
    segs = [dict(s, hit=None) for s in segments]
    fp_events = []
    for ev in final_events:
        ts, label, n_sig, weight, feats, score = ev
        claimed = False
        for seg in segs:
            pos = classify_hit_position(ts, seg)
            if pos is not None and seg["hit"] is None:
                seg["hit"] = {"ts": ts, "label": label, "n_sig": n_sig, "weight": weight, "position": pos, "features": feats, "score": score}
                claimed = True
                break
        if not claimed: 
            # Appends structured dictionary matching try.py format
            fp_events.append({"ts": ts, "label": label, "n_sig": n_sig, "weight": weight, "features": feats, "score": score})
    return segs, fp_events

def _parse_signals_from_label(label: str) -> set: return set(label.split("+"))
def _signal_contribution_table(events: list) -> dict:
    counts = {s: 0 for s in ALL_SIGNALS}
    for ev in events:
        for s in _parse_signals_from_label(ev.get("label", "")):
            if s in counts: counts[s] += 1
    return counts

# ==============================================================================
# PERFORMANCE VISUALIZATION REPORTING SUMMARY
# ==============================================================================

def print_overall_summary(all_video_stats):
    banner("LEADER ENGINE PRE-FILTER GRID OPTIMIZATION SUMMARY")
    total_segs = sum(v["total"] for v in all_video_stats)
    total_hits = sum(v["hits"] for v in all_video_stats)
    total_fp = sum(v["fp"] for v in all_video_stats)
    total_leader_time = sum(v["leader_time"] for v in all_video_stats)
    overall = total_hits / total_segs * 100 if total_segs else 0
    
    print(f"\n  {'VIDEO':<35}  {'SEGS':>4}  {'HITS':>5}  {'RATE':>6}  {'FP':>4}  {'LEADER TIME':>12}")
    print(f"  {'─'*35}  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*12}")
    for v in all_video_stats:
        rate = v["hits"] / v["total"] * 100 if v["total"] else 0
        print(f"  {v['name']:<35}  {v['total']:>4}  {v['hits']:>5}  {rate:>5.1f}%  {v['fp']:>4}  {v['leader_time']:>11.4f}s")
    print(f"  {'─'*35}  {'─'*4}  {'─'*5}  {'─'*6}  {'─'*4}  {'─'*12}")
    print(f"  {'TOTAL SYSTEM EVAL':<35}  {total_segs:>4}  {total_hits:>5}  {overall:>5.1f}%  {total_fp:>4}  {total_leader_time:>11.4f}s")
    
    print(f"\n  {'='*70}")
    print(f"  OPTIMIZATION GRID RECALL TARGET: {total_hits}/{total_segs} ({overall:.2f}%)")
    print(f"  TOTAL EVALUATION RUN CLOCK: {total_leader_time:.4f} seconds")
    print(f"  {'='*70}")

    section("GLOBAL MULTI-MODAL SIGNAL ANALYSIS MATRIX")
    all_hit_evs = [ev for v in all_video_stats for ev in v.get("hit_events", [])]
    all_fp_evs  = [ev for v in all_video_stats for ev in v.get("fp_events", [])]
    hit_sig, fp_sig = _signal_contribution_table(all_hit_evs), _signal_contribution_table(all_fp_evs)
    th, tf = max(len(all_hit_evs), 1), max(len(all_fp_evs), 1)

    print(f"  {'SIGNAL DIMENSION':<18}  {'HIT CONTRIB':>12}  {'HIT RATIO %':>13}  {'FP CONTRIB':>11}  {'FP RATIO %':>12}")
    print(f"  {'─'*18}  {'─'*12}  {'─'*13}  {'─'*11}  {'─'*12}")
    for s in ALL_SIGNALS:
        print(f"  {s:<18}  {hit_sig[s]:>12}  {hit_sig[s]/th*100:>12.1f}%  {fp_sig[s]:>11}  {fp_sig[s]/tf*100:>11.1f}%")

# ==============================================================================
# TARGET RUN COORDINATOR TUNING ENGINE
# ==============================================================================

def execute_leader_tuning(video_id: str, ground_truth: dict) -> dict:
    json_path = os.path.join(SCRIPT_DIR, WORKER_DATA_DIR, f"{video_id}_worker_data.json")
    if not os.path.exists(json_path): return None

    t0 = time.time()
    with open(json_path, "r") as f: data = json.load(f)
    packets = data["chunks"]

    detector = LeaderDetector()
    signal_dict = detector.run_detection(packets)
    fused_events = fuse_all_signals(signal_dict)
    merged_events = apply_window_merge(fused_events)
    
    event_dicts = [{"time": e[0], "label": e[1], "n_sig": e[2], "weight": e[3], "features": e[4], "score": e[5]} for e in merged_events]
    selected, _ = select_strongest_per_window(event_dicts)
    final_events = [(e["time"], e["label"], e["n_sig"], e["weight"], e["features"], e["score"]) for e in selected]

    leader_time = time.time() - t0
    seg_results, fp_events = validate_video(f"{video_id}.mp4", ground_truth["segments"], final_events)
    hits = sum(1 for s in seg_results if s["hit"])

    hit_events = [{"ts": s["hit"]["ts"], "label": s["hit"]["label"], "weight": s["hit"]["weight"]} for s in seg_results if s["hit"]]
    
    # Fixes KeyError by accessing dictionary keys instead of positional tuple indexes
    formatted_fps = [{"ts": e["ts"], "label": e["label"], "weight": e["weight"]} for e in fp_events]

    return {
        "name": f"{video_id}.mp4", "total": len(ground_truth["segments"]), "hits": hits, "fp": len(fp_events),
        "leader_time": leader_time, "hit_events": hit_events, "fp_events": formatted_fps
    }

if __name__ == "__main__":
    gt_path = os.path.join(SCRIPT_DIR, GROUND_TRUTH_JSON)
    if not os.path.exists(gt_path): sys.exit(1)
    with open(gt_path) as f: ground_truth = json.load(f)
    gt_map = {e["target_video"]: e for e in ground_truth}

    VIDEOS_TO_MONITOR = [f"attacked_{i}" for i in range(1, 42)]
    all_results = []

    banner("STATEFUL LEADER OFFLINE TUNING ENGINE INITIALIZED")
    print(f"  Ingesting JSON tracking matrices from: {WORKER_DATA_DIR}/")
    print(f"  Simulating non-parametric threshold metrics across {len(VIDEOS_TO_MONITOR)} streams...")

    for target_id in VIDEOS_TO_MONITOR:
        vid_filename = f"{target_id}.mp4"
        if vid_filename in gt_map:
            res = execute_leader_tuning(target_id, gt_map[vid_filename])
            if res: all_results.append(res)

    if all_results:
        print_overall_summary(all_results)