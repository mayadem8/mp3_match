# MP3-in-MP3 (GPU)

## Installation

Clone the repository
```bash
https://github.com/mayadem8/mp3_match.git
cd mp3_match
```

Download ffmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z

Put ffmpeg.exe in mp3_match folder

Download CUDA Toolkit Installer: https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_windows.exe

Update Nvidia driver, so it supports CUDA 13, can be found here: https://www.nvidia.com/en-us/drivers/


Install dependencies
```bash
pip install -r requirements.txt
```


Run code 
```bash
python .\mp3match_gpu.py
```

Example Output: 
```bash
✅ GPU BATCH COMPLETE in 8.32s

=== MATCH SUMMARY ===
sample_01.mp3        →  2342.99s  ( 99.90%)
sample_02.mp3        →   833.99s  ( 99.77%)
sample_03.mp3        →   326.99s  ( 99.94%)
sample_04.mp3        →    65.99s  ( 99.67%)
sample_05.mp3        →  2276.99s  ( 99.98%)
sample_06.mp3        →   680.00s  ( 99.05%)
sample_07.mp3        →  2456.99s  ( 99.71%)
sample_08.mp3        →   809.99s  ( 99.86%)
sample_09.mp3        →   367.00s  ( 99.00%)
sample_10.mp3        →  1577.00s  ( 99.46%)
sample_11.mp3        →  1420.00s  ( 99.44%)
sample_12.mp3        →  1958.99s  ( 99.88%)
sample_13.mp3        →  3302.99s  ( 99.88%)
sample_14.mp3        →  2897.99s  ( 99.99%)
sample_15.mp3        →   497.00s  ( 98.78%)
sample_16.mp3        →  2198.00s  ( 99.30%)
sample_17.mp3        →   470.00s  ( 99.67%)
sample_18.mp3        →   676.00s  ( 99.40%)
sample_19.mp3        →   791.00s  ( 99.36%)
sample_20.mp3        →   490.00s  ( 99.31%)
sample_21.mp3        →   716.00s  ( 99.69%)
sample_22.mp3        →  2633.99s  ( 99.88%)
sample_23.mp3        →  1234.00s  ( 99.51%)
sample_24.mp3        →  1754.00s  ( 99.49%)
sample_25.mp3        →  2915.99s  ( 99.68%)
sample_26.mp3        →  2602.00s  ( 99.60%)
sample_27.mp3        →  1945.00s  ( 99.27%)
sample_28.mp3        →  3290.99s  ( 99.19%)
sample_29.mp3        →   943.00s  ( 99.79%)
sample_30.mp3        →    98.99s  (  6.75%)
sample_31.mp3        →  1532.99s  ( 99.88%)
sample_32.mp3        →  2939.99s  ( 99.69%)
sample_33.mp3        →  3497.00s  ( 99.35%)
sample_34.mp3        →  2839.00s  ( 99.94%)
sample_35.mp3        →  2234.00s  ( 99.16%)
sample_36.mp3        →  2990.99s  ( 99.45%)
sample_37.mp3        →   260.00s  ( 99.66%)
sample_38.mp3        →  1516.00s  ( 99.30%)
sample_39.mp3        →  1781.99s  ( 99.79%)
sample_40.mp3        →  3341.99s  ( 99.75%)
sample_41.mp3        →   877.00s  ( 99.63%)
sample_42.mp3        →  3125.00s  ( 99.83%)
sample_43.mp3        →  1334.99s  ( 99.83%)
sample_44.mp3        →   829.00s  ( 99.21%)
sample_45.mp3        →  2545.00s  ( 99.44%)
sample_46.mp3        →   638.00s  ( 99.30%)
sample_47.mp3        →  3508.00s  ( 99.30%)
sample_48.mp3        →  3450.00s  ( 18.77%)
sample_49.mp3        →  2458.00s  ( 98.99%)
sample_50.mp3        →  3002.00s  ( 99.58%)
```
