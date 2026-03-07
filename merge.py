import subprocess
import os
import random
import glob
import sys
import re
import time

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LONG_VIDEOS_DIR = "long-videos"       # Folder with long carrier videos
VIOLENT_VIDEOS_DIR = "violent-videos" # Folder with violent videos to hide
OUTPUT_DIR = "attacked"               # Output folder for attacked videos
ATTACK_LOG = "attack.txt"             # Log file with all timestamps

FFMPEG_DIR = "ffmpeg-master-latest-win64-gpl"
FFMPEG_EXE = os.path.join(FFMPEG_DIR, "bin", "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_DIR, "bin", "ffprobe.exe")

TARGET_ATTACKED_VIDEOS = 10   # Minimum number of attacked videos to produce
INSERTS_PER_10_MIN = 2        # At least this many violent clips per every 10 minutes
INSERTS_EXTRA_RANDOM = 1      # Extra random inserts added on top (for variety)
# ==============================================================================


def format_hms(seconds):
    """Convert seconds to HH:MM:SS.mmm string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def get_duration(filename):
    """Gets the precise duration of a media file using ffprobe."""
    cmd = [
        FFPROBE_EXE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filename
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        print(f"  [!] WARNING: Could not get duration for {filename}. Skipping.")
        return 0.0


def run_ffmpeg(args, silent=True):
    """Run FFmpeg silently (used for fast copy operations like slicing)."""
    subprocess.run([FFMPEG_EXE, "-y"] + args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_ffmpeg_with_progress(args, label, total_duration_sec):
    """
    Run FFmpeg and print a live progress bar based on encoded time vs total duration.
    Uses ffmpeg's built-in progress output (-progress pipe:1).
    """
    cmd = [FFMPEG_EXE, "-y"] + args + ["-progress", "pipe:1", "-nostats"]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    bar_width = 35
    start_time = time.time()
    current_out_time = 0.0

    def print_bar(out_time_sec, done=False):
        elapsed = time.time() - start_time
        frac = min(out_time_sec / total_duration_sec, 1.0) if total_duration_sec > 0 else 0
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = frac * 100

        if done:
            eta_str = f"elapsed {elapsed:.1f}s"
        else:
            if frac > 0.01:
                eta = (elapsed / frac) - elapsed
                eta_str = f"ETA {eta:.0f}s"
            else:
                eta_str = "ETA --s"

        line = f"\r  {label}: [{bar}] {pct:5.1f}%  {format_hms(out_time_sec)} / {format_hms(total_duration_sec)}  {eta_str}   "
        sys.stdout.write(line)
        sys.stdout.flush()

    for line in process.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                current_out_time = ms / 1_000_000.0
                print_bar(current_out_time)
            except ValueError:
                pass

    process.wait()
    print_bar(total_duration_sec, done=True)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if process.returncode != 0:
        print(f"  [!] WARNING: FFmpeg exited with code {process.returncode} for: {label}")


def reencode_to_common_format(input_file, output_file, label=None):
    """
    Re-encode a video to a common format (H.264 video + AAC audio, 30fps, stereo)
    to ensure seamless concatenation regardless of source format.
    Shows a live progress bar.
    """
    dur = get_duration(input_file)
    lbl = label or os.path.basename(input_file)
    run_ffmpeg_with_progress([
        "-i", input_file,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-r", "30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
        output_file
    ], label=lbl, total_duration_sec=dur)


def get_video_files(directory):
    """Return all mp4/mkv/avi files in a directory."""
    extensions = ["*.mp4", "*.mkv", "*.avi", "*.mov"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def cleanup_temp_files(file_list):
    """Remove temp files safely."""
    for f in file_list:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass


def create_attacked_video(long_video_path, violent_video_paths, output_path, attack_id):
    """
    Hide multiple violent video clips (with audio) inside a single long video.
    Returns a list of truth records: (attacked_video_name, segment_index, start_sec, end_sec, source_violent_video).
    """
    temp_files = []
    truth_records = []

    long_video_name = os.path.basename(long_video_path)
    output_name = os.path.basename(output_path)

    print(f"\n  [*] Processing carrier: {long_video_name}")
    print(f"  [*] Will hide {len(violent_video_paths)} violent clip(s)")

    dur_long = get_duration(long_video_path)
    if dur_long < 120:
        print(f"  [!] Carrier video too short ({dur_long:.1f}s). Skipping.")
        return []

    # -------------------------------------------------------------------------
    # Step 1: Re-encode carrier and violent videos to a common format
    # -------------------------------------------------------------------------
    total_steps = 1 + len(violent_video_paths)
    print(f"  [Step 1/{total_steps}] Re-encoding carrier: {long_video_name}")
    encoded_long = f"temp_enc_long_{attack_id}.mp4"
    reencode_to_common_format(long_video_path, encoded_long, label=f"Carrier: {long_video_name}")
    temp_files.append(encoded_long)
    dur_long_enc = get_duration(encoded_long)

    encoded_violent = []
    for vi, vpath in enumerate(violent_video_paths):
        vname = os.path.basename(vpath)
        print(f"  [Step {vi+2}/{total_steps}] Re-encoding violent clip {vi+1}/{len(violent_video_paths)}: {vname}")
        enc_viol = f"temp_enc_viol_{attack_id}_{vi}.mp4"
        reencode_to_common_format(vpath, enc_viol, label=f"Violent {vi+1}: {vname}")
        temp_files.append(enc_viol)
        encoded_violent.append(enc_viol)

    # -------------------------------------------------------------------------
    # Step 2: Choose random insertion points in the carrier (avoid first/last 60s)
    # -------------------------------------------------------------------------
    buffer = min(60.0, dur_long_enc * 0.05)
    num_inserts = len(violent_video_paths)

    # Space insertion points evenly across the available timeline to avoid overlap
    available_range = dur_long_enc - 2 * buffer
    segment_size = available_range / num_inserts
    insert_points = sorted([
        random.uniform(buffer + i * segment_size, buffer + (i + 1) * segment_size)
        for i in range(num_inserts)
    ])

    print(f"  [*] Insertion points (in carrier): {[round(p, 2) for p in insert_points]}")

    # -------------------------------------------------------------------------
    # Step 3: Slice carrier around insertion points
    # -------------------------------------------------------------------------
    print(f"\n  [Slicing] Cutting carrier into {num_inserts + 1} chunks around insertion points...")
    carrier_chunks = []
    last_p = 0.0
    for i, p in enumerate(insert_points):
        chunk = f"temp_carrier_{attack_id}_part{i}.mp4"
        print(f"    Chunk {i+1}/{num_inserts+1}: {format_hms(last_p)} → {format_hms(p)}")
        run_ffmpeg(["-ss", str(last_p), "-to", str(p), "-i", encoded_long, "-c", "copy", chunk])
        carrier_chunks.append(chunk)
        temp_files.append(chunk)
        last_p = p

    # Final tail of carrier
    final_carrier = f"temp_carrier_{attack_id}_final.mp4"
    print(f"    Chunk {num_inserts+1}/{num_inserts+1}: {format_hms(last_p)} → end")
    run_ffmpeg(["-ss", str(last_p), "-i", encoded_long, "-c", "copy", final_carrier])
    carrier_chunks.append(final_carrier)
    temp_files.append(final_carrier)

    # -------------------------------------------------------------------------
    # Step 4: Build concat list and calculate truth timestamps
    # -------------------------------------------------------------------------
    concat_list_path = f"temp_concat_{attack_id}.txt"
    temp_files.append(concat_list_path)

    cumulative_time = 0.0

    with open(concat_list_path, "w") as f:
        for i in range(num_inserts):
            # Write carrier chunk
            carrier_dur = get_duration(carrier_chunks[i])
            f.write(f"file '{os.path.abspath(carrier_chunks[i])}'\n")
            cumulative_time += carrier_dur

            # Record START of violent segment
            seg_start = round(cumulative_time, 3)

            # Write violent chunk
            violent_dur = get_duration(encoded_violent[i])
            f.write(f"file '{os.path.abspath(encoded_violent[i])}'\n")
            cumulative_time += violent_dur

            # Record END of violent segment
            seg_end = round(cumulative_time, 3)

            truth_records.append({
                "attacked_video": output_name,
                "segment_index": i + 1,
                "violent_source": os.path.basename(violent_video_paths[i]),
                "start_sec": seg_start,
                "end_sec": seg_end,
            })

            print(f"  [+] Segment {i+1}: {seg_start}s → {seg_end}s  (source: {os.path.basename(violent_video_paths[i])})")

        # Final carrier tail
        f.write(f"file '{os.path.abspath(carrier_chunks[-1])}'\n")

    # -------------------------------------------------------------------------
    # Step 5: Final concat merge
    # -------------------------------------------------------------------------
    total_out_dur = cumulative_time + get_duration(carrier_chunks[-1])
    print(f"\n  [Merging] Assembling final output ({format_hms(total_out_dur)} total)...")
    run_ffmpeg_with_progress([
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        output_path
    ], label=f"Merge → {output_name}", total_duration_sec=total_out_dur)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    cleanup_temp_files(temp_files)
    print(f"  [+] Done → {output_path}")
    return truth_records


def main():
    # Validate directories
    for d in [LONG_VIDEOS_DIR, VIOLENT_VIDEOS_DIR]:
        if not os.path.isdir(d):
            print(f"[ERROR] Directory not found: {d}")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    long_videos = get_video_files(LONG_VIDEOS_DIR)
    violent_videos = get_video_files(VIOLENT_VIDEOS_DIR)

    if not long_videos:
        print(f"[ERROR] No video files found in '{LONG_VIDEOS_DIR}'")
        return
    if not violent_videos:
        print(f"[ERROR] No video files found in '{VIOLENT_VIDEOS_DIR}'")
        return

    print(f"[*] Found {len(long_videos)} long video(s): {[os.path.basename(v) for v in long_videos]}")
    print(f"[*] Found {len(violent_videos)} violent video(s): {[os.path.basename(v) for v in violent_videos]}")
    print(f"[*] Target: {TARGET_ATTACKED_VIDEOS} attacked videos\n")

    all_truth_records = []
    attack_count = 0

    # Keep cycling through long videos until we reach TARGET_ATTACKED_VIDEOS
    long_video_cycle = long_videos.copy()
    random.shuffle(long_video_cycle)

    attack_id = 0
    while attack_count < TARGET_ATTACKED_VIDEOS:
        # Pick a carrier (cycle through, repeat if needed)
        carrier = long_video_cycle[attack_count % len(long_video_cycle)]

        # Calculate inserts based on carrier duration: at least 2 per 10 minutes
        carrier_dur_preview = get_duration(carrier)
        carrier_minutes = carrier_dur_preview / 60.0
        num_violent = max(
            INSERTS_PER_10_MIN,                              # always at least base count
            int(carrier_minutes / 10) * INSERTS_PER_10_MIN  # 2 per every 10-min window
        ) + random.randint(0, INSERTS_EXTRA_RANDOM)         # sprinkle random extras

        print(f"    Carrier duration: {carrier_minutes:.1f} min  →  inserting {num_violent} violent clip(s)")

        # Sample violent clips randomly (with replacement if fewer clips than inserts)
        if len(violent_videos) >= num_violent:
            chosen_violent = random.sample(violent_videos, num_violent)
        else:
            chosen_violent = random.choices(violent_videos, k=num_violent)

        carrier_base = os.path.splitext(os.path.basename(carrier))[0]
        output_filename = f"attacked_{attack_count + 1:02d}_{carrier_base}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n{'='*60}")
        print(f"[*] Creating attacked video {attack_count + 1}/{TARGET_ATTACKED_VIDEOS}: {output_filename}")
        print(f"    Carrier: {os.path.basename(carrier)}")
        print(f"    Hiding {num_violent} violent clip(s): {[os.path.basename(v) for v in chosen_violent]}")
        print(f"{'='*60}")

        records = create_attacked_video(
            long_video_path=carrier,
            violent_video_paths=chosen_violent,
            output_path=output_path,
            attack_id=attack_id
        )

        if records:
            all_truth_records.extend(records)
            attack_count += 1
        else:
            print(f"  [!] Skipped (failed or too short). Trying next carrier.")

        attack_id += 1
        # Safety: break if we've tried way too many times
        if attack_id > TARGET_ATTACKED_VIDEOS * 5:
            print("[!] Too many failures. Stopping early.")
            break

    # -------------------------------------------------------------------------
    # Write attack.txt truth log
    # -------------------------------------------------------------------------
    with open(ATTACK_LOG, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ATTACK TRUTH LOG — Violent Segment Timestamps\n")
        f.write("=" * 70 + "\n\n")

        current_video = None
        for rec in all_truth_records:
            if rec["attacked_video"] != current_video:
                current_video = rec["attacked_video"]
                f.write(f"\n[Attacked Video] {current_video}\n")
                f.write("-" * 50 + "\n")

            start_hms = format_hms(rec["start_sec"])
            end_hms = format_hms(rec["end_sec"])
            duration = round(rec["end_sec"] - rec["start_sec"], 3)

            f.write(
                f"  Segment {rec['segment_index']:02d} | "
                f"Source: {rec['violent_source']:<30} | "
                f"Start: {rec['start_sec']:>10.3f}s ({start_hms}) | "
                f"End: {rec['end_sec']:>10.3f}s ({end_hms}) | "
                f"Duration: {duration:.3f}s\n"
            )

    print(f"\n{'='*60}")
    print(f"[+] ALL DONE!")
    print(f"[+] Created {attack_count} attacked video(s) in '{OUTPUT_DIR}/'")
    print(f"[+] Truth log saved to: {ATTACK_LOG}")
    print(f"{'='*60}")

    # Print summary to console
    print(f"\nTruth Log Summary:")
    print("-" * 60)
    with open(ATTACK_LOG) as f:
        print(f.read())



if __name__ == "__main__":
    main()