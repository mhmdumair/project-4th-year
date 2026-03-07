import subprocess
import os
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================
VIDEO_1 = "video1.mp4"      # 1-hour+ Broadcast
VIDEO_2 = "video2.mp4"      # 10-minute+ MMA Fight
OUTPUT_FILE = "hidden_output.mp4"
NUM_SEGMENTS = 5            # Number of MMA chunks to hide
TRUTH_FILE = "merge.txt"    # Truth file with START and END points

FFMPEG_DIR = "ffmpeg-master-latest-win64-gpl"
FFMPEG_EXE = os.path.join(FFMPEG_DIR, "bin", "ffmpeg.exe")
FFPROBE_EXE = os.path.join(FFMPEG_DIR, "bin", "ffprobe.exe")
# ==============================================================================

def get_duration(filename):
    """Gets the precise duration using ffprobe."""
    cmd = [
        FFPROBE_EXE, "-v", "error", "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", filename
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def run_ffmpeg(args):
    """Silent FFmpeg execution."""
    # We use stderr=subprocess.PIPE to capture errors but ignore them if successful
    subprocess.run([FFMPEG_EXE, "-y"] + args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def hide_video():
    if not os.path.exists(VIDEO_1) or not os.path.exists(VIDEO_2):
        print("Error: Source videos not found. Check file names.")
        return

    dur1 = get_duration(VIDEO_1)
    dur2 = get_duration(VIDEO_2)
    seg_dur2 = dur2 / NUM_SEGMENTS

    print(f"[*] Broadcast Duration: {round(dur1/60, 2)} minutes")
    print(f"[*] Secret MMA Duration: {round(dur2/60, 2)} minutes")

    # 1. Generate insertion points in the 1-hour broadcast
    # Buffer: Avoid first 5 mins and last 5 mins of the hour-long video
    buffer_zone = 300.0 if dur1 > 600 else 20.0 
    insert_points = sorted([random.uniform(buffer_zone, dur1 - buffer_zone) for _ in range(NUM_SEGMENTS)])
    
    # 2. Slice Video 2 (MMA) into chunks
    v2_chunks = []
    v2_durs = []
    print(f"[*] Slicing secret video into {NUM_SEGMENTS} segments...")
    for i in range(NUM_SEGMENTS):
        chunk_name = f"temp_mma_part_{i}.mp4"
        start = i * seg_dur2
        run_ffmpeg(["-ss", str(start), "-t", str(seg_dur2), "-i", VIDEO_2, "-c", "copy", chunk_name])
        v2_chunks.append(chunk_name)
        v2_durs.append(get_duration(chunk_name))

    # 3. Slice Video 1 (Broadcast) into chunks
    v1_chunks = []
    v1_durs = []
    last_p = 0
    print(f"[*] Slicing carrier video around insertion points...")
    for i, p in enumerate(insert_points):
        chunk_name = f"temp_broadcast_part_{i}.mp4"
        run_ffmpeg(["-ss", str(last_p), "-to", str(p), "-i", VIDEO_1, "-c", "copy", chunk_name])
        v1_chunks.append(chunk_name)
        v1_durs.append(get_duration(chunk_name))
        last_p = p
    
    # Final piece of Video 1
    final_v1 = "temp_broadcast_part_final.mp4"
    run_ffmpeg(["-ss", str(last_p), "-i", VIDEO_1, "-c", "copy", final_v1])
    v1_chunks.append(final_v1)
    v1_durs.append(get_duration(final_v1))

    # 4. Concatenate and Calculate Truth Timestamps
    current_cumulative_time = 0.0
    truth_report = []

    with open("join_list.txt", "w") as f:
        for i in range(NUM_SEGMENTS):
            # Carrier Segment
            f.write(f"file '{v1_chunks[i]}'\n")
            current_cumulative_time += v1_durs[i]
            
            # THE START POINT
            start_time = round(current_cumulative_time, 2)
            
            # MMA Segment
            f.write(f"file '{v2_chunks[i]}'\n")
            current_cumulative_time += v2_durs[i]
            
            # THE END POINT
            end_time = round(current_cumulative_time, 2)
            
            truth_report.append(f"Segment_{i+1}_Start: {start_time}")
            truth_report.append(f"Segment_{i+1}_End:   {end_time}")

        # Final Carrier Segment
        f.write(f"file '{v1_chunks[-1]}'\n")

    # Save to merge.txt
    with open(TRUTH_FILE, "w") as f:
        f.write("\n".join(truth_report))

    # 5. Final Merge
    print("[*] Performing final merge into 'hidden_output.mp4'...")
    # Using concat demuxer is fast for large files as it doesn't re-encode
    run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "join_list.txt", "-c", "copy", OUTPUT_FILE])

    # Cleanup temp files
    print("[*] Cleaning up temporary files...")
    for f in v1_chunks + v2_chunks + ["join_list.txt"]:
        if os.path.exists(f): os.remove(f)

    print(f"\n[+] SUCCESS!")
    print(f"[+] Precise truth log saved to: {TRUTH_FILE}")
    print("-" * 40)
    for line in truth_report:
        print(line)

if __name__ == "__main__":
    hide_video()