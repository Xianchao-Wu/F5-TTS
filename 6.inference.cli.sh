#########################################################################
# File Name: 6.inference.cli.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Feb 13 07:42:25 2026
#########################################################################
#!/bin/bash

modelname="F5TTS_v1_Base"
ckpt="/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors"
vocab="/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/vocab.txt"

mcfg="/workspace/asr/F5-TTS/src/f5_tts/configs/F5TTS_v1_Base.yaml"

gen_text="那是当然的啦，我们都找到了自己的真正的幸福。"

prompt_wav_ch='/workspace/asr/CosyVoice/asset/zero_shot_prompt.wav'
prompt_txt_ch="希望你以后能够做的比我还好呦。"

CUDA_VISIBLE_DEVICES=7 python -m ipdb src/f5_tts/infer/infer_cli.py \
	--model $modelname \
	--ckpt_file $ckpt \
	--vocab_file $vocab \
	--model_cfg $mcfg \
	--ref_audio $prompt_wav_ch \
	--ref_text $prompt_txt_ch \
	--gen_text $gen_text 
