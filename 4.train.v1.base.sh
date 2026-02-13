#########################################################################
# File Name: 4.train.v1.base.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Feb 13 06:58:03 2026
#########################################################################
#!/bin/bash

#accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml
#accelerate launch 

config="src/f5_tts/configs/F5TTS_v1_Base.yaml"

CUDA_VISIBLE_DEVICES=7 python -m ipdb \
    src/f5_tts/train/train.py --config-name $config 
