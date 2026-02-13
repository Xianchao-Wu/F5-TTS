import os
from pathlib import Path

LIBRITTS_ROOT = Path("/workspace/asr/CosyVoice/data/tts/openslr/libritts/LibriTTS")
OUTPUT = LIBRITTS_ROOT / "metadata.csv"

lines = []

for wav_path in LIBRITTS_ROOT.rglob("*.wav"):

    # 对应文本文件
    txt_path = wav_path.with_suffix(".normalized.txt")

    if not txt_path.exists():
        txt_path = wav_path.with_suffix(".txt")

    if not txt_path.exists():
        print("Missing text:", wav_path)
        continue

    with open(txt_path, "r", encoding="utf8") as f:
        text = f.read().strip()

    abs_wav = wav_path.resolve()

    lines.append(f"{abs_wav}|{text}")

# 写入 metadata.csv
with open(OUTPUT, "w", encoding="utf8") as f:
    for line in lines:
        f.write(line + "\n")

print("Done. total:", len(lines))
print("Saved to:", OUTPUT)

