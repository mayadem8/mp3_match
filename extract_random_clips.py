import os
import random
import subprocess

INPUT_FILE = "chunk_20250127_135809.mp3"
OUTPUT_DIR = "samples"
NUM_CLIPS = 50
CLIP_LENGTH = 10
TOTAL_DURATION = 59 * 60

def ensure_output_folder():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output folder: {OUTPUT_DIR}")

def generate_random_timestamp():
    start_sec = random.randint(0, TOTAL_DURATION - CLIP_LENGTH)
    hh = start_sec // 3600
    mm = (start_sec % 3600) // 60
    ss = start_sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def extract_clip(index):
    start_time = generate_random_timestamp()
    output_file = os.path.join(OUTPUT_DIR, f"sample_{index:02d}.mp3")

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_time,
        "-i", INPUT_FILE,
        "-t", str(CLIP_LENGTH),
        "-acodec", "copy",
        output_file
    ]

    print(f"[{index:02d}] Extracting from {start_time} -> {output_file}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    ensure_output_folder()

    print("\n=== Generating Random Samples ===\n")
    for i in range(1, NUM_CLIPS + 1):
        extract_clip(i)

    print("\n✅ ALL DONE! 50 samples created in /samples folder.\n")

if __name__ == "__main__":
    main()
