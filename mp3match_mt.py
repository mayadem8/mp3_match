import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.signal import correlate


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


def calculate_energy(x, w):
    x2 = x**2
    cumsum = np.concatenate(([0], np.cumsum(x2)))
    energy = cumsum[w:] - cumsum[:-w]
    return np.sqrt(energy)


def process_chunk(args):
    long_chunk, short_audio, short_norm, start_idx, sr = args
    corr = correlate(long_chunk, short_audio, mode="valid", method="fft")
    energy_long = calculate_energy(long_chunk, len(short_audio))
    corr_norm = corr / (energy_long * short_norm + 1e-12)
    idx = np.argmax(corr_norm)
    score = corr_norm[idx]
    abs_time = (start_idx + idx) / sr
    return score, abs_time


def find_audio_segment_multithreaded(
    long_path,
    short_path,
    sr=44100,
    chunk_dur=60.0,
    overlap=5.0,
    max_workers=4
):
    print(f"Loading long audio: {long_path}")
    t0 = time.time()
    long_audio, _ = decode_audio_ffmpeg(long_path, sr)
    total_dur = len(long_audio) / sr
    print(f"Loaded long audio ({total_dur/60:.1f} min) in {time.time() - t0:.2f}s\n")

    print(f"Loading sample: {short_path}")
    t0 = time.time()
    short_audio, _ = decode_audio_ffmpeg(short_path, sr)
    short_len = len(short_audio)
    short_norm = np.linalg.norm(short_audio)
    print(f"Loaded sample ({short_len/sr:.2f}s) in {time.time() - t0:.2f}s\n")

    step = chunk_dur - overlap
    chunk_size = int(chunk_dur * sr)
    chunks = []
    for chunk_start_s in np.arange(0, total_dur - chunk_dur, step):
        start_idx = int(chunk_start_s * sr)
        end_idx = start_idx + chunk_size
        long_chunk = long_audio[start_idx:end_idx]
        if len(long_chunk) < short_len:
            break
        chunks.append((long_chunk, short_audio, short_norm, start_idx, sr))

    print(f"Searching in {len(chunks)} chunks using {max_workers} threads...\n")
    t_search = time.time()

    best_score, best_offset = -1.0, 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, c) for c in chunks]
        for future in as_completed(futures):
            score, abs_time = future.result()
            if score > best_score:
                best_score = score
                best_offset = abs_time

    best_score = min(best_score * 100, 100.0)
    print("\nDone.")
    print(f"   Best match at: {best_offset:.2f}s (confidence={best_score:.2f}%)")
    print(f"   Total search time: {time.time() - t_search:.2f}s\n")

    return best_offset, best_score


if __name__ == "__main__":
    find_audio_segment_multithreaded(
        "chunk_20250127_135809.mp3",
        "sample.mp3",
        sr=44100,
        chunk_dur=120.0,
        overlap=5.0,
        max_workers=14
    )