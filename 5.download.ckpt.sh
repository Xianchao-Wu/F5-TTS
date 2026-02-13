#########################################################################
# File Name: 5.download.ckpt.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Feb 13 07:21:03 2026
#########################################################################
#!/bin/bash

hf download SWivid/F5-TTS \
  --repo-type model \
  --include "F5TTS_v1_Base/*" \
  --local-dir ./ckpts

