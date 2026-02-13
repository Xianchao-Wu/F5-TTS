#########################################################################
# File Name: 1.prepare.train.data.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Feb 13 05:56:59 2026
#########################################################################
#!/bin/bash

# Prepare the Emilia dataset
#python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
#python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/f5_tts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
#python src/f5_tts/train/datasets/prepare_ljspeech.py
