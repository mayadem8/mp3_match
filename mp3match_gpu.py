import time
import subprocess
import numpy as np      
import cupy as cp       


LONG_PATH   = "chunk_20250127_135809.mp3"
SAMPLES_DB  = "samples_base.npy"
SR          = 44100
CHUNK_DUR   = 120.0
OVERLAP     = 5.0


# ==========================
# AUDIO LOADING (CPU) -> move to GPU after
# ==========================
def decode_audio_ffmpeg(path, sr=44100):
    t0 = time.time()
    print(f"Loading audio from: {path}")
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
    data /= np.max(np.abs(data)) + 1e-12  # Normalize
    print(f"\nLoaded audio: {len(data)/sr/60:.1f} min")
    print(f"   (load time: {time.time() - t0:.2f} s)\n")
    return data, sr



def gpu_normalized_correlation(long_chunk_gpu, short_gpu, short_norm_gpu, start_idx):
    """
    Normalized cross-correlation using FFT (GPU).
    """
    # Full convolution via FFT
    corr_fft = cp.fft.ifft(cp.fft.fft(long_chunk_gpu) * cp.conj(cp.fft.fft(short_gpu, long_chunk_gpu.size)))
    corr_fft = cp.real(corr_fft[:long_chunk_gpu.size - short_gpu.size + 1])

    # Sliding energy for normalization
    x2 = long_chunk_gpu * long_chunk_gpu
    cumsum = cp.concatenate((cp.array([0.0], dtype=cp.float32), cp.cumsum(x2)))
    win = short_gpu.size
    energy = cp.sqrt(cumsum[win:] - cumsum[:-win] + 1e-12)

    corr_norm = corr_fft / (energy * short_norm_gpu + 1e-12)

    idx = int(cp.argmax(corr_norm).item())
    score = float(corr_norm[idx].item())
    abs_time = (start_idx + idx) / SR
    return score, abs_time


def find_audio_segment_gpu(long_gpu, short_gpu):
    short_len = short_gpu.size
    short_norm_gpu = cp.linalg.norm(short_gpu)

    total_dur = long_gpu.size / SR
    step = CHUNK_DUR - OVERLAP
    chunk_size = int(CHUNK_DUR * SR)

    print(f"Searching on GPU...\n")
    t_search = time.time()

    best_score, best_offset = -1.0, 0.0

    for chunk_start_s in np.arange(0, total_dur - CHUNK_DUR, step):
        start_idx = int(chunk_start_s * SR)
        end_idx = start_idx + chunk_size
        long_chunk_gpu = long_gpu[start_idx:end_idx]

        if long_chunk_gpu.size < short_len:
            break

        score, abs_time = gpu_normalized_correlation(
            long_chunk_gpu, short_gpu, short_norm_gpu, start_idx
        )

        if score > best_score:
            best_score = score
            best_offset = abs_time

    best_pct = min(best_score * 100.0, 100.0)
    print(f"\n   Best match at: {best_offset:.2f}s (confidence={best_pct:.2f}%)")
    print(f"   Total search time: {time.time() - t_search:.2f}s\n")
    return best_offset, best_pct



if __name__ == "__main__":

    long_cpu, _ = decode_audio_ffmpeg(LONG_PATH, SR)
    print("Transferring chunk to GPU...")
    long_gpu = cp.asarray(long_cpu, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    print("   ✅ Ready on GPU\n")


    print("Loading sample database: samples_base.npy")
    samples = np.load(SAMPLES_DB, allow_pickle=True)
    print(f"   ✅ {len(samples)} samples loaded (CPU)\n")

    print("=== STARTING GPU BATCH MATCHING ===")
    results = []
    t0 = time.time()

    for filename, short_cpu, _ in samples:
        print(f"\n==============================")
        print(f"MATCHING SAMPLE: {filename}")
        print(f"==============================")

        short_gpu = cp.asarray(short_cpu, dtype=cp.float32)
        offset, score = find_audio_segment_gpu(long_gpu, short_gpu)
        results.append((filename, offset, score))

    cp.cuda.Stream.null.synchronize()
    print(f"\n✅ GPU BATCH COMPLETE in {time.time() - t0:.2f}s\n")

    print("=== MATCH SUMMARY ===")
    for name, off, sc in sorted(results):
        print(f"{name:<20} → {off:8.2f}s  ({sc:6.2f}%)")
