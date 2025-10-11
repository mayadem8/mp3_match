# MP3-in-MP3

## Installation

Clone the repository
```bash
https://github.com/mayadem8/mp3_match.git
cd mp3_match
```

Install dependencies
```bash
pip install -r requirements.txt
```

Also download ffmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z

Paste ffmpeg.exe in mp3_match folder

Run code 
```bash
python .\mp3match_mt.py
```

Example Output: 
```bash
Loading long audio: chunk_20250127_135809.mp3
Loaded long audio (60.0 min) in 12.01s

Loading sample: sample.mp3
Loaded sample (9.80s) in 0.06s

Searching in 65 chunks using 14 threads...


Done.
   Best match at: 7.00s (confidence=100.00%)
   ✅ Total search time: 3.33s
```
