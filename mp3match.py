import subprocess
import numpy as np
from scipy.signal import correlate
import time

def decode_audio_ffmpeg(path, sr=44100):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
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

def find_audio_segment(long_path, short_path, sr=44100,
                       chunk_dur=30.0, overlap=5.0,
                       stop_conf=0.99):

    print(f"Loading audio from: {long_path}\n")
    start_load = time.time()
    long_audio, _ = decode_audio_ffmpeg(long_path, sr)
    long_len = len(long_audio)
    total_dur = long_len / sr
    print(f"Loaded audio: {total_dur/60:.1f} min")
    print(f"   (load time: {time.time() - start_load:.2f} s)\n")

    print(f"Loading sample from: {short_path}\n")
    start_load = time.time()
    short_audio, _ = decode_audio_ffmpeg(short_path, sr)
    short_len = len(short_audio)
    print(f"Loaded sample: {short_len/sr:.2f} s")
    print(f"   (load time: {time.time() - start_load:.2f} s)\n")

    print(f"Searching for the sample in the audio file...\n")
    
    start_search = time.time()
    step = chunk_dur - overlap
    chunk_size = int(chunk_dur * sr)
    short_norm = np.linalg.norm(short_audio)
    best_score, best_offset = -1.0, 0.0

    for chunk_start_s in np.arange(0, total_dur - chunk_dur, step):
        start_idx = int(chunk_start_s * sr)
        end_idx = start_idx + chunk_size
        long_chunk = long_audio[start_idx:end_idx]
        if len(long_chunk) < short_len:
            break

        corr = correlate(long_chunk, short_audio, mode="valid", method="fft")
        energy_long = calculate_energy(long_chunk, short_len)
        corr_norm = corr / (energy_long * short_norm + 1e-12)

        idx = np.argmax(corr_norm)
        score = corr_norm[idx]
        abs_time = (start_idx + idx) / sr

        if score > best_score:
            best_score = score
            best_offset = abs_time

        if best_score >= stop_conf:
            best_score = min(best_score*100, 100.0)
            print(f"\nFound sample with confidence {best_score:.2f}%, exiting now...")
            break
    
    print("\nDone.")
    print(f"   Match found at time point: {best_offset:.2f} s (confidence={best_score:.2f}%)\n")
    print(f"   Search time: {time.time() - start_search:.2f} s")
    
    return best_offset, best_score


if __name__ == "__main__":
    find_audio_segment(
        "chunk_20250127_135809.mp3",
        "sample.mp3",
        sr=44100,
        chunk_dur=120.0,
        overlap=5.0,
        stop_conf=0.99
    )