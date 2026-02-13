import os
from pathlib import Path

# NOTE 这个代码，只针对custom dataset，即我自己的私有数据的时候，才需要
# TODO 如果是libritts，不需要走这一步的
ROOT = Path("/workspace/asr/CosyVoice/data/tts/openslr/libritts/LibriTTS")

subsets = [p for p in ROOT.iterdir() if p.is_dir()]

for subset in subsets:

    print("Processing a subset:", subset.name)

    lines = []

    for wav_path in subset.rglob("*.wav"):

        txt_path = wav_path.with_suffix(".normalized.txt")

        if not txt_path.exists():
            txt_path = wav_path.with_suffix(".txt")

        if not txt_path.exists():
            continue

        with open(txt_path, "r", encoding="utf8") as f:
            text = f.read().strip()

        lines.append(f"{wav_path.resolve()}|{text}")

    output_file = subset / "metadata.csv"

    with open(output_file, "w", encoding="utf8") as f:
        for l in lines:
            f.write(l + "\n")

    print("saved:", output_file, "lines:", len(lines))

print("All done.")

