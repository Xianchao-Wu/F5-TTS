import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from unidecode import unidecode

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--no_legacy_text",
    action="store_false",
    help="Not to use lossy ASCII transliterations of unicode text in saved file names.",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)

import ipdb; ipdb.set_trace()
args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/basic/basic_ref_en.wav")
ref_text = (
    args.ref_text
    if args.ref_text is not None
    else config.get("ref_text", "Some call me nature, others call me mother nature.")
)
gen_text = args.gen_text or config.get("gen_text", "Here we generate something just for test.")
gen_file = args.gen_file or config.get("gen_file", "")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

save_chunk = args.save_chunk or config.get("save_chunk", False)
use_legacy_text = args.no_legacy_text or config.get("no_legacy_text", False)  # no_legacy_text is a store_false arg
if save_chunk and use_legacy_text:
    print(
        "\nWarning to --save_chunk: lossy ASCII transliterations of unicode text for legacy (.wav) file names, --no_legacy_text to disable.\n"
    )

remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms) # NOTE rms = root mean square for loudness
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step) # number of feature evaluation for number of forward calls in inference algorithm such as Euler algorithm
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)


# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("f5_tts").joinpath(f"{ref_audio}"))
if "infer/examples/" in gen_file:
    gen_file = str(files("f5_tts").joinpath(f"{gen_file}"))
if "voices" in config:
    for voice in config["voices"]:
        voice_ref_audio = config["voices"][voice]["ref_audio"]
        if "infer/examples/" in voice_ref_audio:
            config["voices"][voice]["ref_audio"] = str(files("f5_tts").joinpath(f"{voice_ref_audio}"))


# ignore gen_text if gen_file provided

if gen_file:
    gen_text = codecs.open(gen_file, "r", "utf-8").read()


# output path

wave_path = Path(output_dir) / output_file # PosixPath('tests/infer_cli_basic.wav') 保存生成的音频文件 NOTE, 这个是从config文件里面读取的，目前命令行里没有输入
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    #vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    vocoder_local_path = "./ckpts/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load TTS model
import ipdb; ipdb.set_trace()

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
) # args.model_cfg='/workspace/asr/F5-TTS/src/f5_tts/configs/F5TTS_v1_Base.yaml'
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}") # 'DiT' - <class 'f5_tts.model.backbones.dit.DiT'>
model_arc = model_cfg.model.arch # architecture parameters: {'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'text_mask_padding': True, 'qk_norm': None, 'conv_layers': 4, 'pe_attn_head': None, 'attn_backend': 'torch', 'attn_mask_enabled': False, 'checkpoint_activations': False}

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))
elif ckpt_file.startswith("hf://"):
    ckpt_file = str(cached_path(ckpt_file))

if vocab_file.startswith("hf://"):
    vocab_file = str(cached_path(vocab_file))
import ipdb; ipdb.set_trace()
print(f"Using {model}...") # NOTE load the model here: 初始化model对象：
ema_model = load_model(
    model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
)
# 1 model_cls=<class 'f5_tts.model.backbones.dit.DiT'>; 
# 2 model_arc={'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'text_mask_padding': True, 'qk_norm': None, 'conv_layers': 4, 'pe_attn_head': None, 'attn_backend': 'torch', 'attn_mask_enabled': False, 'checkpoint_activations': False}
# 3 ckpt_file='/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors'
# 4 mel_spec_type='vocos'
# 5 vocab_file='/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/vocab.txt'
# 6 device='cuda'

# inference process


def main():
    import ipdb; ipdb.set_trace() # 看着逻辑是，先初始化一些参数，导入模型，然后再到这里，继续搞事情
    main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        print("Voice:", voice)
        print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_text"]
        )
        print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print("No voice tag found, using main.")
            voice = "main"
        if voice not in voices:
            print(f"Voice {voice} not found, using main.")
            voice = "main"
        text = re.sub(reg2, "", text)
        ref_audio_ = voices[voice]["ref_audio"]
        ref_text_ = voices[voice]["ref_text"]
        local_speed = voices[voice].get("speed", speed)
        gen_text_ = text.strip()
        print(f"Voice: {voice}")
        import ipdb; ipdb.set_trace() # NOTE very important here:
        audio_segment, final_sample_rate, spectrogram = infer_process(
            ref_audio_, # '/tmp/tmp726iz92m.wav', the ref wav filename with path
            ref_text_, # '希望你以后能够做的比我还好呦。'
            gen_text_, # '那是当然的啦，我们都找到了自己的真正的幸福。'
            ema_model, # <class 'f5_tts.model.cfm.CFM'>, model size=337,096,804=337M parameters
            vocoder, # <class 'vocos.pretrained.Vocos'>, model size=13,531,650=13M parameters
            mel_spec_type=vocoder_name, # 'vocos'
            target_rms=target_rms, # 0.1
            cross_fade_duration=cross_fade_duration, # 0.15, NOTE TODO what for? 交叉淡化时长, 在两段语音拼接时，用多长时间（或多少帧）对前一段音频逐渐减弱，同时对后一段音频逐渐增强，从而避免“咔嗒声”或突变。
            nfe_step=nfe_step, # 32, number of feature evaluation, forward call times
            cfg_strength=cfg_strength, # 2.0
            sway_sampling_coef=sway_sampling_coef, # -1.0
            speed=local_speed, # 1.0
            fix_duration=fix_duration, # None
            device=device, # 'cuda'
        )
        generated_audio_segments.append(audio_segment)

        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + " ... "
            if use_legacy_text:
                gen_text_ = unidecode(gen_text_)
            sf.write(
                os.path.join(output_chunk_dir, f"{len(generated_audio_segments) - 1}_{gen_text_}.wav"),
                audio_segment,
                final_sample_rate,
            )

    import ipdb; ipdb.set_trace()
    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments) # final_wave.shape=(99840,)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(wave_path, "wb") as f: # wave_path=PosixPath('tests/infer_cli_basic.wav')
            sf.write(f.name, final_wave, final_sample_rate) # file.name, (99840,), and sr=24k
            # Remove silence
            if remove_silence: # False
                remove_silence_for_generated_wav(f.name)
            print(f.name)


if __name__ == "__main__":
    main()

'''
ipdb> ema_model
CFM(
  (mel_spec): MelSpec()
  (transformer): DiT(
    (time_embed): TimestepEmbedding(
      (time_embed): SinusPositionEmbedding()
      (time_mlp): Sequential(
        (0): Linear(in_features=256, out_features=1024, bias=True)
        (1): SiLU()
        (2): Linear(in_features=1024, out_features=1024, bias=True)
      )
    )
    (text_embed): TextEmbedding(
      (text_embed): Embedding(2546, 512)
      (text_blocks): Sequential(
        (0): ConvNeXtV2Block(
          (dwconv): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), groups=512)
          (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (pwconv1): Linear(in_features=512, out_features=1024, bias=True)
          (act): GELU(approximate='none')
          (grn): GRN()
          (pwconv2): Linear(in_features=1024, out_features=512, bias=True)
        )
        (1): ConvNeXtV2Block(
          (dwconv): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), groups=512)
          (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (pwconv1): Linear(in_features=512, out_features=1024, bias=True)
          (act): GELU(approximate='none')
          (grn): GRN()
          (pwconv2): Linear(in_features=1024, out_features=512, bias=True)
        )
        (2): ConvNeXtV2Block(
          (dwconv): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), groups=512)
          (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (pwconv1): Linear(in_features=512, out_features=1024, bias=True)
          (act): GELU(approximate='none')
          (grn): GRN()
          (pwconv2): Linear(in_features=1024, out_features=512, bias=True)
        )
        (3): ConvNeXtV2Block(
          (dwconv): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), groups=512)
          (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (pwconv1): Linear(in_features=512, out_features=1024, bias=True)
          (act): GELU(approximate='none')
          (grn): GRN()
          (pwconv2): Linear(in_features=1024, out_features=512, bias=True)
        )
      )
    )
    (input_embed): InputEmbedding(
      (proj): Linear(in_features=712, out_features=1024, bias=True)
      (conv_pos_embed): ConvPositionEmbedding(
        (conv1d): Sequential(
          (0): Conv1d(1024, 1024, kernel_size=(31,), stride=(1,), padding=(15,), groups=16)
          (1): Mish()
          (2): Conv1d(1024, 1024, kernel_size=(31,), stride=(1,), padding=(15,), groups=16)
          (3): Mish()
        )
      )
    )
    (rotary_embed): RotaryEmbedding()
    (transformer_blocks): ModuleList(
      (0-21): 22 x DiTBlock(
        (attn_norm): AdaLayerNorm(
          (silu): SiLU()
          (linear): Linear(in_features=1024, out_features=6144, bias=True)
          (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=False)
        )
        (attn): Attention(
          (to_q): Linear(in_features=1024, out_features=1024, bias=True)
          (to_k): Linear(in_features=1024, out_features=1024, bias=True)
          (to_v): Linear(in_features=1024, out_features=1024, bias=True)
          (to_out): ModuleList(
            (0): Linear(in_features=1024, out_features=1024, bias=True)
            (1): Dropout(p=0.1, inplace=False)
          )
        )
        (ff_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=False)
        (ff): FeedForward(
          (ff): Sequential(
            (0): Sequential(
              (0): Linear(in_features=1024, out_features=2048, bias=True)
              (1): GELU(approximate='tanh')
            )
            (1): Dropout(p=0.1, inplace=False)
            (2): Linear(in_features=2048, out_features=1024, bias=True)
          )
        )
      )
    )
    (norm_out): AdaLayerNorm_Final(
      (silu): SiLU()
      (linear): Linear(in_features=1024, out_features=2048, bias=True)
      (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=False)
    )
    (proj_out): Linear(in_features=1024, out_features=100, bias=True)
  )
)

'''

'''
ipdb> vocoder
Vocos(
  (feature_extractor): MelSpectrogramFeatures(
    (mel_spec): MelSpectrogram(
      (spectrogram): Spectrogram()
      (mel_scale): MelScale()
    )
  )
  (backbone): VocosBackbone(
    (embed): Conv1d(100, 512, kernel_size=(7,), stride=(1,), padding=(3,))
    (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    (convnext): ModuleList(
      (0-7): 8 x ConvNeXtBlock(
        (dwconv): Conv1d(512, 512, kernel_size=(7,), stride=(1,), padding=(3,), groups=512)
        (norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        (pwconv1): Linear(in_features=512, out_features=1536, bias=True)
        (act): GELU(approximate='none')
        (pwconv2): Linear(in_features=1536, out_features=512, bias=True)
      )
    )
    (final_layer_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
  )
  (head): ISTFTHead(
    (out): Linear(in_features=512, out_features=1026, bias=True)
    (istft): ISTFT()
  )
)
'''
