import os
import time
import numpy as np
import subprocess

SAMPLES_DIR = "samples"
OUTPUT_FILE = "samples_base.npy"
SR = 44100  


def decode_audio_ffmpeg(path, sr=44100):
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{err.decode()}")

    data = np.frombuffer(raw, dtype=np.float32).copy()
    data /= np.max(np.abs(data)) + 1e-12
    return data, sr


def main():
    print("\n=== Building Sample Database (samples_base.npy) ===\n")
    t0 = time.time()

    samples_data = []  

    sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".mp3")])
    print(f"Found {len(sample_files)} sample files in '{SAMPLES_DIR}'")

    for i, filename in enumerate(sample_files, start=1):
        path = os.path.join(SAMPLES_DIR, filename)
        print(f"[{i:02d}] Decoding {filename} ...", end="", flush=True)
        t_load = time.time()
        audio, _ = decode_audio_ffmpeg(path, SR)
        norm = np.linalg.norm(audio)
        samples_data.append((filename, audio, norm))
        print(f" done ({len(audio)/SR:.2f}s, load {time.time()-t_load:.2f}s)")

  
    np.save(OUTPUT_FILE, np.array(samples_data, dtype=object), allow_pickle=True)

    print(f"\n✅ Saved all samples to {OUTPUT_FILE}")
    print(f"Total build time: {time.time() - t0:.2f} seconds\n")


if __name__ == "__main__":
    main()
