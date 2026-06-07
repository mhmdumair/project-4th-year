#!/usr/bin/env python3
"""
analysis.py - Analyze Trigger Features from report.json
========================================================
This script:
1. Loads report.json with all triggers
2. Separates HIT and FP triggers based on ground truth
3. Calculates statistics (mean, median, std, min, max, quartiles) for each feature
4. Exports CSV for further ML analysis
5. Identifies trigger clusters per segment for fusion optimization

Usage: python analysis.py
"""

import json
import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

REPORT_JSON = "report.json"
OUTPUT_CSV = "triggers_analysis.csv"
OUTPUT_STATS = "feature_statistics.txt"
OUTPUT_CLUSTER_REPORT = "trigger_clusters.txt"

BOUNDARY_TOLERANCE = 2.5


# ==============================================================================
# LOAD DATA
# ==============================================================================

def load_report():
    """Load report.json file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, REPORT_JSON)
    
    if not os.path.exists(report_path):
        print(f"[ERROR] {REPORT_JSON} not found in {script_dir}")
        print("Please run the main detection script first.")
        sys.exit(1)
    
    with open(report_path, 'r') as f:
        return json.load(f)


def build_ground_truth_lookup():
    """Build ground truth lookup from attack.json"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(script_dir, "attack.json")
    
    if not os.path.exists(gt_path):
        print(f"[WARN] Ground truth not found at {gt_path}")
        print("Using segment info from report.json instead")
        return None
    
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    gt_lookup = {}
    for entry in ground_truth:
        video_name = entry["target_video"]
        segments = []
        for seg in entry["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "index": seg["index"],
                "source": seg["source"]
            })
        gt_lookup[video_name] = segments
    
    return gt_lookup


# ==============================================================================
# CLASSIFY TRIGGERS AS HIT OR FP
# ==============================================================================

def classify_triggers(report_data, gt_lookup=None):
    """
    Classify each trigger as HIT or FP
    Returns: list of triggers with 'hit_status' added
    """
    all_triggers = []
    
    for video_result in report_data["results"]:
        video_name = video_result["video_name"]
        
        # Get ground truth segments for this video
        segments = []
        if gt_lookup and video_name in gt_lookup:
            segments = gt_lookup[video_name]
        else:
            # Fallback: use segments from report
            segments = video_result.get("segments", [])
        
        for trigger in video_result.get("triggers", []):
            ts = trigger["timestamp"]
            
            # Check if this trigger hits any segment
            is_hit = False
            hit_segment_idx = None
            hit_segment_start = None
            hit_segment_end = None
            
            for seg in segments:
                if (seg["start"] - BOUNDARY_TOLERANCE) <= ts <= (seg["end"] + BOUNDARY_TOLERANCE):
                    is_hit = True
                    hit_segment_idx = seg.get("index", seg.get("source", "unknown"))
                    hit_segment_start = seg["start"]
                    hit_segment_end = seg["end"]
                    break
                elif seg["start"] < ts < seg["end"]:
                    is_hit = True
                    hit_segment_idx = seg.get("index", seg.get("source", "unknown"))
                    hit_segment_start = seg["start"]
                    hit_segment_end = seg["end"]
                    break
            
            trigger_copy = trigger.copy()
            trigger_copy["video_name"] = video_name
            trigger_copy["is_hit"] = is_hit
            trigger_copy["hit_status"] = "HIT" if is_hit else "FP"
            trigger_copy["hit_segment_idx"] = hit_segment_idx
            trigger_copy["hit_segment_start"] = hit_segment_start
            trigger_copy["hit_segment_end"] = hit_segment_end
            
            all_triggers.append(trigger_copy)
    
    return all_triggers


# ==============================================================================
# FEATURE STATISTICS
# ==============================================================================

def get_numeric_features(trigger):
    """Extract all numeric features from a trigger"""
    features = {}
    
    # Basic info
    features["timestamp"] = trigger.get("timestamp", 0)
    features["magnitude"] = trigger.get("magnitude", 0)
    
    # Motion features (Gates 1,4,5)
    features["magnitude_ratio"] = trigger.get("magnitude_ratio", 0)
    features["persistence_count"] = trigger.get("persistence_count", 0)
    features["acceleration"] = trigger.get("acceleration", 0)
    
    # Frame difference features (Gate 2)
    features["frame_diff_prev"] = trigger.get("frame_diff_prev", 0)
    features["frame_diff_next"] = trigger.get("frame_diff_next", 0)
    
    # Histogram differences
    features["hist_diff_prev"] = trigger.get("hist_diff_prev", 0)
    features["hist_diff_next"] = trigger.get("hist_diff_next", 0)
    
    # Uniformity (Gate 3)
    features["uniformity"] = trigger.get("uniformity", 0.5)
    
    # Edge changes (Gate 6)
    features["edge_change_prev"] = trigger.get("edge_change_prev", 0)
    features["edge_change_next"] = trigger.get("edge_change_next", 0)
    
    # Texture features
    features["texture_mean"] = trigger.get("texture_mean", 0)
    features["texture_std"] = trigger.get("texture_std", 0)
    features["texture_entropy"] = trigger.get("texture_entropy", 0)
    
    # Audio features
    features["audio_rms"] = trigger.get("audio_rms", 0)
    features["audio_peak"] = trigger.get("audio_peak", 0)
    features["audio_spectral_centroid"] = trigger.get("audio_spectral_centroid", 0)
    
    return features


def calculate_statistics(hit_features, fp_features, feature_names):
    """Calculate statistics for each feature"""
    stats = {}
    
    for feat in feature_names:
        hit_vals = [f[feat] for f in hit_features if f.get(feat) is not None]
        fp_vals = [f[feat] for f in fp_features if f.get(feat) is not None]
        
        if not hit_vals and not fp_vals:
            continue
            
        stats[feat] = {
            "hit": {
                "count": len(hit_vals),
                "mean": np.mean(hit_vals) if hit_vals else 0,
                "median": np.median(hit_vals) if hit_vals else 0,
                "std": np.std(hit_vals) if hit_vals else 0,
                "min": np.min(hit_vals) if hit_vals else 0,
                "max": np.max(hit_vals) if hit_vals else 0,
                "q1": np.percentile(hit_vals, 25) if hit_vals else 0,
                "q3": np.percentile(hit_vals, 75) if hit_vals else 0,
                "range": (np.max(hit_vals) - np.min(hit_vals)) if hit_vals else 0
            },
            "fp": {
                "count": len(fp_vals),
                "mean": np.mean(fp_vals) if fp_vals else 0,
                "median": np.median(fp_vals) if fp_vals else 0,
                "std": np.std(fp_vals) if fp_vals else 0,
                "min": np.min(fp_vals) if fp_vals else 0,
                "max": np.max(fp_vals) if fp_vals else 0,
                "q1": np.percentile(fp_vals, 25) if fp_vals else 0,
                "q3": np.percentile(fp_vals, 75) if fp_vals else 0,
                "range": (np.max(fp_vals) - np.min(fp_vals)) if fp_vals else 0
            },
            "separation": {
                "mean_diff": (np.mean(hit_vals) - np.mean(fp_vals)) if (hit_vals and fp_vals) else 0,
                "median_diff": (np.median(hit_vals) - np.median(fp_vals)) if (hit_vals and fp_vals) else 0,
                "overlap": calculate_overlap(hit_vals, fp_vals) if (hit_vals and fp_vals) else 1.0
            }
        }
    
    return stats


def calculate_overlap(hit_vals, fp_vals):
    """Calculate overlap coefficient between two distributions"""
    if not hit_vals or not fp_vals:
        return 1.0
    
    hit_min, hit_max = np.min(hit_vals), np.max(hit_vals)
    fp_min, fp_max = np.min(fp_vals), np.max(fp_vals)
    
    overlap_min = max(hit_min, fp_min)
    overlap_max = min(hit_max, fp_max)
    
    if overlap_min > overlap_max:
        return 0.0
    
    overlap_range = overlap_max - overlap_min
    total_range = max(hit_max, fp_max) - min(hit_min, fp_min)
    
    return overlap_range / total_range if total_range > 0 else 1.0


# ==============================================================================
# TRIGGER CLUSTER ANALYSIS
# ==============================================================================

def analyze_trigger_clusters(report_data, gt_lookup=None):
    """
    Analyze how many triggers hit each segment and their timing
    Helps determine fusion window optimization
    """
    cluster_report = []
    cluster_report.append("="*80)
    cluster_report.append("  TRIGGER CLUSTER ANALYSIS PER SEGMENT")
    cluster_report.append("="*80)
    cluster_report.append("")
    
    all_cluster_sizes = []
    all_time_spreads = []
    
    for video_result in report_data["results"]:
        video_name = video_result["video_name"]
        
        # Get segments
        segments = video_result.get("segments", [])
        if not segments:
            continue
        
        # Get all triggers for this video
        triggers = video_result.get("triggers", [])
        
        cluster_report.append(f"\n{'─'*60}")
        cluster_report.append(f"VIDEO: {video_name}")
        cluster_report.append(f"{'─'*60}")
        
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_idx = seg.get("index", "?")
            
            # Find triggers that hit this segment
            hitting_triggers = []
            for trigger in triggers:
                ts = trigger["timestamp"]
                if (seg_start - BOUNDARY_TOLERANCE) <= ts <= (seg_end + BOUNDARY_TOLERANCE):
                    hitting_triggers.append(trigger)
                elif seg_start < ts < seg_end:
                    hitting_triggers.append(trigger)
            
            if hitting_triggers:
                timestamps = [t["timestamp"] for t in hitting_triggers]
                timestamps.sort()
                
                # Calculate time spread
                time_spread = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
                avg_gap = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
                
                cluster_report.append(f"\n  Segment {seg_idx}: {seg_start:.3f}s - {seg_end:.3f}s")
                cluster_report.append(f"    Triggers hitting: {len(hitting_triggers)}")
                cluster_report.append(f"    Time spread: {time_spread:.3f}s")
                cluster_report.append(f"    Avg gap between triggers: {avg_gap:.3f}s")
                cluster_report.append(f"    Timestamps: {[f'{t:.3f}' for t in timestamps]}")
                
                # List trigger types
                type_counts = defaultdict(int)
                for t in hitting_triggers:
                    type_counts[t.get("type", "Unknown")] += 1
                cluster_report.append(f"    Trigger types: {dict(type_counts)}")
                
                all_cluster_sizes.append(len(hitting_triggers))
                all_time_spreads.append(time_spread)
            else:
                cluster_report.append(f"\n  Segment {seg_idx}: {seg_start:.3f}s - {seg_end:.3f}s")
                cluster_report.append(f"    MISSED - No triggers hit this segment")
    
    # Summary statistics
    if all_cluster_sizes:
        cluster_report.append("\n" + "="*80)
        cluster_report.append("  CLUSTER STATISTICS SUMMARY")
        cluster_report.append("="*80)
        cluster_report.append(f"\n  Total segments with hits: {len(all_cluster_sizes)}")
        cluster_report.append(f"  Average triggers per segment: {np.mean(all_cluster_sizes):.2f}")
        cluster_report.append(f"  Median triggers per segment: {np.median(all_cluster_sizes):.0f}")
        cluster_report.append(f"  Min triggers per segment: {min(all_cluster_sizes)}")
        cluster_report.append(f"  Max triggers per segment: {max(all_cluster_sizes)}")
        
        if all_time_spreads:
            cluster_report.append(f"\n  Average time spread: {np.mean(all_time_spreads):.3f}s")
            cluster_report.append(f"  Median time spread: {np.median(all_time_spreads):.3f}s")
            cluster_report.append(f"  95th percentile time spread: {np.percentile(all_time_spreads, 95):.3f}s")
            cluster_report.append(f"  Recommended fusion window: {np.percentile(all_time_spreads, 95):.1f}s")
    
    return "\n".join(cluster_report), all_cluster_sizes, all_time_spreads


# ==============================================================================
# CREATE CSV FOR ML ANALYSIS
# ==============================================================================

def create_analysis_csv(all_triggers, output_path):
    """Create CSV with all features and hit/FP label"""
    
    rows = []
    
    for trigger in all_triggers:
        features = get_numeric_features(trigger)
        features["hit_status"] = trigger["hit_status"]
        features["video_name"] = trigger["video_name"]
        features["trigger_type"] = trigger.get("type", "Unknown")
        rows.append(features)
    
    if not rows:
        print("  No triggers found!")
        return None
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    hit_count = sum(1 for r in rows if r['hit_status'] == 'HIT')
    fp_count = len(rows) - hit_count
    
    print(f"\n  CSV saved to: {output_path}")
    print(f"  Total rows: {len(rows)}")
    print(f"  HIT count: {hit_count}")
    print(f"  FP count: {fp_count}")
    
    return df


# ==============================================================================
# PRINT STATISTICS REPORT
# ==============================================================================

def print_statistics_report(stats, output_path):
    """Write detailed statistics report"""
    
    lines = []
    lines.append("="*80)
    lines.append("  TRIGGER FEATURE STATISTICS - HIT vs FP")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("="*80)
    lines.append("")
    
    if not stats:
        lines.append("No statistics available.")
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
        return
    
    # Feature importance ranking based on median separation
    lines.append("1. FEATURE SEPARATION ANALYSIS (Most important first)")
    lines.append("-"*60)
    
    separations = []
    for feat, stat in stats.items():
        mean_diff = abs(stat["separation"]["mean_diff"])
        median_diff = abs(stat["separation"]["median_diff"])
        overlap = stat["separation"]["overlap"]
        
        # Score: higher difference, lower overlap = better separation
        score = (mean_diff + median_diff) / 2 * (1 - overlap)
        separations.append((feat, score, mean_diff, median_diff, overlap))
    
    separations.sort(key=lambda x: x[1], reverse=True)
    
    for feat, score, mean_diff, median_diff, overlap in separations[:15]:
        lines.append(f"  {feat:<25}: score={score:.4f} | mean_diff={mean_diff:.4f} | overlap={overlap:.2f}")
    
    lines.append("")
    lines.append("2. DETAILED STATISTICS FOR EACH FEATURE")
    lines.append("-"*60)
    
    for feat, stat in stats.items():
        h = stat["hit"]
        f = stat["fp"]
        sep = stat["separation"]
        
        lines.append(f"\n{feat.upper()}:")
        lines.append(f"  {'─'*50}")
        lines.append(f"  {'Metric':<20} {'HIT (True)':<20} {'FP (False)':<20}")
        lines.append(f"  {'─'*50}")
        lines.append(f"  {'Count':<20} {h['count']:<20} {f['count']:<20}")
        lines.append(f"  {'Mean':<20} {h['mean']:<20.4f} {f['mean']:<20.4f}")
        lines.append(f"  {'Median':<20} {h['median']:<20.4f} {f['median']:<20.4f}")
        lines.append(f"  {'Std Dev':<20} {h['std']:<20.4f} {f['std']:<20.4f}")
        lines.append(f"  {'Min':<20} {h['min']:<20.4f} {f['min']:<20.4f}")
        lines.append(f"  {'Max':<20} {h['max']:<20.4f} {f['max']:<20.4f}")
        lines.append(f"  {'Range':<20} {h['range']:<20.4f} {f['range']:<20.4f}")
        lines.append(f"  {'25th % (Q1)':<20} {h['q1']:<20.4f} {f['q1']:<20.4f}")
        lines.append(f"  {'75th % (Q3)':<20} {h['q3']:<20.4f} {f['q3']:<20.4f}")
        lines.append(f"  {'IQR (Q3-Q1)':<20} {h['q3']-h['q1']:<20.4f} {f['q3']-f['q1']:<20.4f}")
        lines.append(f"  {'─'*50}")
        lines.append(f"  Mean Difference: {sep['mean_diff']:+.4f}")
        lines.append(f"  Median Difference: {sep['median_diff']:+.4f}")
        lines.append(f"  Overlap Coefficient: {sep['overlap']:.2f}")
        
        # Interpretation
        if sep['overlap'] < 0.3:
            lines.append(f"  → EXCELLENT separation - use this gate")
        elif sep['overlap'] < 0.6:
            lines.append(f"  → GOOD separation - useful gate")
        elif sep['overlap'] < 0.8:
            lines.append(f"  → POOR separation - limited value")
        else:
            lines.append(f"  → NO separation - discard this gate")
    
    lines.append("")
    lines.append("3. RECOMMENDED THRESHOLDS")
    lines.append("-"*60)
    lines.append("")
    lines.append("  Based on median values and overlap analysis:")
    lines.append("")
    
    for feat, stat in stats.items():
        h_median = stat["hit"]["median"]
        f_median = stat["fp"]["median"]
        overlap = stat["separation"]["overlap"]
        
        if overlap < 0.6:  # Good separation
            if "magnitude_ratio" in feat or "acceleration" in feat:
                # Higher is better for violence
                recommended = (h_median + f_median) / 2
                lines.append(f"  {feat}: > {recommended:.4f}")
            elif "uniformity" in feat:
                # Lower is better for violence
                recommended = (h_median + f_median) / 2
                lines.append(f"  {feat}: < {recommended:.4f}")
            elif "frame_diff" in feat or "edge_change" in feat:
                recommended = min(h_median, f_median)
                lines.append(f"  {feat}: > {recommended:.4f}")
    
    lines.append("")
    lines.append("="*80)
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\n  Statistics report saved to: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("  TRIGGER ANALYSIS - HIT vs FP CLASSIFICATION")
    print("="*80)
    
    # Load report
    print("\n[1/5] Loading report.json...")
    try:
        report_data = load_report()
        print(f"      Loaded {len(report_data['results'])} videos")
    except Exception as e:
        print(f"      Error loading report: {e}")
        sys.exit(1)
    
    # Load ground truth
    print("\n[2/5] Building ground truth lookup...")
    gt_lookup = build_ground_truth_lookup()
    if gt_lookup:
        print(f"      Loaded ground truth for {len(gt_lookup)} videos")
    else:
        print("      Using segment info from report.json")
    
    # Classify triggers
    print("\n[3/5] Classifying triggers as HIT or FP...")
    try:
        all_triggers = classify_triggers(report_data, gt_lookup)
        
        total_triggers = len(all_triggers)
        hit_count = sum(1 for t in all_triggers if t.get("is_hit", False))
        fp_count = total_triggers - hit_count
        
        print(f"      Total triggers: {total_triggers}")
        print(f"      HIT triggers: {hit_count} ({hit_count/total_triggers*100:.1f}%)" if total_triggers > 0 else "      HIT triggers: 0")
        print(f"      FP triggers: {fp_count} ({fp_count/total_triggers*100:.1f}%)" if total_triggers > 0 else "      FP triggers: 0")
    except Exception as e:
        print(f"      Error classifying triggers: {e}")
        sys.exit(1)
    
    # Separate HIT and FP features
    hit_triggers = [t for t in all_triggers if t.get("is_hit", False)]
    fp_triggers = [t for t in all_triggers if not t.get("is_hit", False)]
    
    hit_features = [get_numeric_features(t) for t in hit_triggers]
    fp_features = [get_numeric_features(t) for t in fp_triggers]
    
    if hit_features:
        feature_names = list(hit_features[0].keys())
    elif fp_features:
        feature_names = list(fp_features[0].keys())
    else:
        feature_names = []
    
    # Calculate statistics
    print("\n[4/5] Calculating statistics...")
    stats = calculate_statistics(hit_features, fp_features, feature_names)
    
    # Create CSV
    print("\n[5/5] Creating CSV and reports...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if all_triggers:
        csv_path = os.path.join(script_dir, OUTPUT_CSV)
        create_analysis_csv(all_triggers, csv_path)
    else:
        print("  No triggers to export")
    
    stats_path = os.path.join(script_dir, OUTPUT_STATS)
    print_statistics_report(stats, stats_path)
    
    # Trigger cluster analysis
    try:
        cluster_path = os.path.join(script_dir, OUTPUT_CLUSTER_REPORT)
        cluster_report, cluster_sizes, time_spreads = analyze_trigger_clusters(report_data, gt_lookup)
        
        with open(cluster_path, 'w') as f:
            f.write(cluster_report)
        print(f"\n  Cluster analysis saved to: {cluster_path}")
    except Exception as e:
        print(f"\n  Error in cluster analysis: {e}")
    
    # Quick summary
    print("\n" + "="*80)
    print("  ANALYSIS COMPLETE")
    print("="*80)
    
    if stats:
        print(f"\n  Key Findings:")
        print(f"  ─────────────────────────────────────────────────────────────")
        
        if "magnitude_ratio" in stats:
            print(f"  HIT triggers have HIGHER magnitude_ratio (median: {stats['magnitude_ratio']['hit']['median']:.2f} vs {stats['magnitude_ratio']['fp']['median']:.2f})")
        if "uniformity" in stats:
            print(f"  HIT triggers have LOWER uniformity (median: {stats['uniformity']['hit']['median']:.3f} vs {stats['uniformity']['fp']['median']:.3f})")
        if "acceleration" in stats:
            print(f"  HIT triggers have HIGHER acceleration (median: {stats['acceleration']['hit']['median']:.0f} vs {stats['acceleration']['fp']['median']:.0f})")
        if "frame_diff_prev" in stats:
            print(f"  HIT triggers have HIGHER frame_diff (median: {stats['frame_diff_prev']['hit']['median']:.4f} vs {stats['frame_diff_prev']['fp']['median']:.4f})")
    
    print(f"\n  Output files:")
    print(f"    - {OUTPUT_CSV} (for ML analysis)")
    print(f"    - {OUTPUT_STATS} (feature statistics)")
    print(f"    - {OUTPUT_CLUSTER_REPORT} (trigger cluster analysis)")
    print("="*80)


if __name__ == "__main__":
    main()