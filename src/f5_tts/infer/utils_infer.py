# A unified script for inference process
# Make adjustments inside functions, and consider both gradio and cli scripts if need to change func output format
import os
import sys
from concurrent.futures import ThreadPoolExecutor


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS device compatibility
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../third_party/BigVGAN/")

import hashlib
import re
import tempfile
from importlib.resources import files

import matplotlib


matplotlib.use("Agg")

import matplotlib.pylab as plt
import numpy as np
import torch
import torchaudio
import tqdm
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos

from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer


_ref_audio_cache = {}
_ref_text_cache = {}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

tempfile_kwargs = {"delete_on_close": False} if sys.version_info >= (3, 12) else {"delete": False}

# -----------------------------------------

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256 # frame shift，帧移，stride，相邻两个分析帧之间，在时间轴上间隔的采样点数(samples)
win_length = 1024 # window length，一个frame里面包括了1024个points。1st frame: [0, 1023], 2nd frame: [256, 1279], 3rd frame: [512, 1535]
n_fft = 1024
mel_spec_type = "vocos"
target_rms = 0.1
cross_fade_duration = 0.15
ode_method = "euler"
nfe_step = 32  # 16, 32
cfg_strength = 2.0
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

# -----------------------------------------


# chunk text into smaller pieces


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# load vocoder
def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device=device, hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        try:
            from third_party.BigVGAN import bigvgan
        except ImportError:
            print("You need to follow the README to init submodule and change the BigVGAN source code.")
        if is_local:
            # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, cache_dir=hf_cache_dir
            )

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    return vocoder


# load asr pipeline

asr_pipe = None


def initialize_asr_pipeline(device: str = device, dtype=None):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


# transcribe


def transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline(device=device)
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={"task": "transcribe", "language": language} if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


# load model checkpoint for inference


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = (
            torch.float16
            if "cuda" in device
            and torch.cuda.get_device_properties(device).major >= 7
            and not torch.cuda.get_device_name().endswith("[ZLUDA]")
            else torch.float32
        )
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }

        # patch for backward compatibility, 305e3ea
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()

    return model.to(device)


# load model for inference


def load_model(
    model_cls, # <class 'f5_tts.model.backbones.dit.DiT'>
    model_cfg, # {'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'text_mask_padding': True, 'qk_norm': None, 'conv_layers': 4, 'pe_attn_head': None, 'attn_backend': 'torch', 'attn_mask_enabled': False, 'checkpoint_activations': False}
    ckpt_path, # /workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.safetensors
    mel_spec_type=mel_spec_type, # 'vocos'
    vocab_file="", # /workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/vocab.txt
    ode_method=ode_method, # 'euler'
    use_ema=True,
    device=device, # 'cuda'
):
    import ipdb; ipdb.set_trace()
    if vocab_file == "":
        vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
    tokenizer = "custom"

    print("\nvocab : ", vocab_file)
    print("token : ", tokenizer)
    print("model : ", ckpt_path, "\n")

    vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer) # vocab_size=2545; 其中的2545个元素为：' ':0 ... '𠮶':2544
    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels), # n_mel_channels=100
        mel_spec_kwargs=dict(
            n_fft=n_fft, # 1024
            hop_length=hop_length, # 256
            win_length=win_length, # 1024
            n_mel_channels=n_mel_channels, # 100
            target_sample_rate=target_sample_rate, # 24000
            mel_spec_type=mel_spec_type, # 'vocos'
        ),
        odeint_kwargs=dict(
            method=ode_method, # 'euler'
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
    import ipdb; ipdb.set_trace()
    return model
    # in model.transformer: 
    # 1. time_embed : TimestepEmbedding
    # 2. text_embed : TextEmbedding
    # 3. input_embed : InputEmbedding
    # 4. rotary_embed : RotaryEmbedding
    # 5. transformer_blocks : ModuleList
    # 6. norm_out : AdaLayerNorm_Final
    # 7. proj_out : Linear

def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


# preprocess reference audio and text


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    global _ref_audio_cache

    if audio_hash in _ref_audio_cache:
        show_info("Using cached preprocessed reference audio...")
        ref_audio = _ref_audio_cache[audio_hash]

    else:  # first pass, do preprocess
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as f:
            temp_path = f.name

        aseg = AudioSegment.from_file(ref_audio_orig)

        # 1. try to find long silence for clipping
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (1)")
                break
            non_silent_wave += non_silent_seg

        # 2. try to find short silence for clipping if 1. failed
        if len(non_silent_wave) > 12000:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    show_info("Audio is over 12s, clipping short. (2)")
                    break
                non_silent_wave += non_silent_seg

        aseg = non_silent_wave

        # 3. if no proper silence found for clipping
        if len(aseg) > 12000:
            aseg = aseg[:12000]
            show_info("Audio is over 12s, clipping short. (3)")

        aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        aseg.export(temp_path, format="wav")
        ref_audio = temp_path

        # Cache the processed reference audio
        _ref_audio_cache[audio_hash] = ref_audio

    if not ref_text.strip():
        global _ref_text_cache
        if audio_hash in _ref_text_cache:
            # Use cached asr transcription
            show_info("Using cached reference text...")
            ref_text = _ref_text_cache[audio_hash]
        else:
            show_info("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(ref_audio)
            # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
            _ref_text_cache[audio_hash] = ref_text
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text


# infer process: chunk text -> infer batches [i.e. infer_batch_process()]

def infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model_obj,
    vocoder,
    mel_spec_type=mel_spec_type,
    show_info=print,
    progress=tqdm,
    target_rms=target_rms,
    cross_fade_duration=cross_fade_duration,
    nfe_step=nfe_step,
    cfg_strength=cfg_strength,
    sway_sampling_coef=sway_sampling_coef,
    speed=speed,
    fix_duration=fix_duration,
    device=device,
):
    import ipdb; ipdb.set_trace() # NOTE TODO
    # Split the input text into batches
    audio, sr = torchaudio.load(ref_audio) # torch.Size([1, 45663]), sr=16000
    max_chars = int(len(ref_text.encode("utf-8")) / (audio.shape[-1] / sr) * (22 - audio.shape[-1] / sr) * speed) # 301
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars) # ['那是当然的啦，我们都找到了自己的真正的幸福。']
    for i, gen_text in enumerate(gen_text_batches):
        print(f"gen_text {i}", gen_text) # gen_text 0 那是当然的啦，我们都找到了自己的真正的幸福。
    print("\n")

    show_info(f"Generating audio in {len(gen_text_batches)} batches...")
    return next(
        infer_batch_process(
            (audio, sr), # audio.shape=[1, 45663], sr=16000
            ref_text, # '希望你以后能够做的比我还好呦。'
            gen_text_batches, # ['那是当然的啦，我们都找到了自己的真正的幸福。']
            model_obj, # <class 'f5_tts.model.cfm.CFM'>
            vocoder, # <class 'vocos.pretrained.Vocos'>
            mel_spec_type=mel_spec_type, # 'vocos'
            progress=progress, # <module 'tqdm' from '/usr/local/lib/python3.10/dist-packages/tqdm/__init__.py'>
            target_rms=target_rms, # 0.1
            cross_fade_duration=cross_fade_duration, # 0.15
            nfe_step=nfe_step, # 32
            cfg_strength=cfg_strength, # 2.0
            sway_sampling_coef=sway_sampling_coef, # -1.0
            speed=speed, # 1.0
            fix_duration=fix_duration, # None
            device=device, # 'cuda'
        )
    )


# infer batches

def infer_batch_process(
    ref_audio,
    ref_text,
    gen_text_batches,
    model_obj,
    vocoder,
    mel_spec_type="vocos",
    progress=tqdm,
    target_rms=0.1,
    cross_fade_duration=0.15,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1,
    speed=1,
    fix_duration=None,
    device=None,
    streaming=False,
    chunk_size=2048,
):
    import ipdb; ipdb.set_trace() # NOTE TODO
    audio, sr = ref_audio # (tensor([[ 0.0047, -0.0043,  0.0012,  ...,  0.0000,  0.0000,  0.0000]]), 16000); ref_audio[0].shape=torch.Size([1, 45663]) and 16k are the two parameters of the reference audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio))) # tensor(0.0742) NOTE root mean square 这个的计算真的很不错啊
    if rms < target_rms: # 0.0742 < 0.1, in
        audio = audio * target_rms / rms # * 0.1/0.0742 = 1.3480 --> 1.3480 * audio
    if sr != target_sample_rate: # 16000 != 24000
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate) # sr=16000, 24000=target_sample_rate
        audio = resampler(audio) # 16k to 24k : torch.Size([1, 45663]) -> torch.Size([1, 68495])
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    def process_batch(gen_text):
        import ipdb; ipdb.set_trace()
        local_speed = speed # 1.0
        if len(gen_text.encode("utf-8")) < 10: # 66 > 10
            local_speed = 0.3

        # Prepare the text
        text_list = [ref_text + gen_text] # NOTE ['希望你以后能够做的比我还好呦。那是当然的啦，我们都找到了自己的真正的幸福。'] 这是直接对ref_text和gen_text进行了拼接操作了，两个字符串，合并成为了一个字符串了. so what about after?
        final_text_list = convert_char_to_pinyin(text_list) # [[' ', 'xi1', ' ', 'wang4', ' ', 'ni3', ' ', 'yi3', ' ', 'hou4', ' ', 'neng2', ' ', 'gou4', ' ', 'zuo4', ' ', 'de', ' ', 'bi3', ' ', 'wo3', ' ', 'hai2', ' ', 'hao3', ' ', 'you1', '。', ' ', 'na4', ' ', 'shi4', ' ', 'dang1', ' ', 'ran2', ' ', 'de', ' ', 'la', '，', ' ', 'wo3', ' ', 'men', ' ', 'dou1', ' ', 'zhao3', ' ', 'dao4', ' ', 'le', ' ', 'zi4', ' ', 'ji3', ' ', 'de', ' ', 'zhen1', ' ', 'zheng4', ' ', 'de', ' ', 'xing4', ' ', 'fu2', '。']] # TODO 需要注意的是这里，这里是用了full pinyin，而且每个full pinyin内部是带有声调的，而且外部是带有空格的。相当于说，一个完整的发音，例如you1，就是一个独立的'character', pay attention to this special character unit!

        ref_audio_len = audio.shape[-1] // hop_length # NOTE, 267 = 68495 // 256, 相当于说ref audio里面有267个mel frames
        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Calculate duration
            ref_text_len = len(ref_text.encode("utf-8")) # 45 for '希望你以后能够做的比我还好呦。'
            gen_text_len = len(gen_text.encode("utf-8")) # 66 for '那是当然的啦，我们都找到了自己的真正的幸福。'
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed) # 658=267+391, NOTE very important ，首先是ref语音的长度，267；然后，是 267 / 45 * 66 / local_speed=1.0，就是根据ref的语音和文本的比例，来近似推算一下当前的待生成的文本的对应的语音的长度，大概是多少的节奏，这个还是很不错的。例如，如果是唱歌的时候，那么我们的唱歌的语音/文本的比例，就可以影响在唱歌模式下的，一个待生成的文本对应的唱歌的语音的长度了. 267 + ?391.6 -> 391 = 658

        # inference
        with torch.inference_mode():
            import ipdb; ipdb.set_trace()
            generated, _ = model_obj.sample( # NOTE TODO 
                cond=audio, # torch.tensor with shape = [1, 68495], with sampling rate=24k, the reference audio
                text=final_text_list, # 参考Line 478's value for reference
                duration=duration, # 658, 参考语音的长度267 + 推断出来的待生成的语音的长度391
                steps=nfe_step, # 32
                cfg_strength=cfg_strength, # 2.0, classifier-free guidance
                sway_sampling_coef=sway_sampling_coef, # -1.0
            )
            del _ # _ for the trajectory with 33 tensors of shape=[1, 658, 100]

            generated = generated.to(torch.float32)  # generated mel spectrogram, [1, 658, 100] with the late 390 elements are for the generated audio mel spectrogram frame
            generated = generated[:, ref_audio_len:, :] # ref_audio_len=267 ... Oh! generated.shape=[1, 391, 100]
            generated = generated.permute(0, 2, 1)
            if mel_spec_type == "vocos": # True, here
                generated_wave = vocoder.decode(generated) # torch.Size([1, 99840])
            elif mel_spec_type == "bigvgan":
                generated_wave = vocoder(generated)
            if rms < target_rms:
                generated_wave = generated_wave * rms / target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy() # (99840,)

            if streaming: # False
                for j in range(0, len(generated_wave), chunk_size):
                    # NOTE yield generated_wave[j : j + chunk_size], target_sample_rate
                    return generated_wave[j : j + chunk_size], target_sample_rate
            else:
                generated_cpu = generated[0].cpu().numpy() # generated[0].shape=[100, 391]
                del generated
                # NOTE yield generated_wave, generated_cpu
                return generated_wave, generated_cpu # (99840,) and (100, 391)
    # NOTE for 'streaming':
    if streaming:
        for gen_text in progress.tqdm(gen_text_batches) if progress is not None else gen_text_batches:
            for chunk in process_batch(gen_text):
                yield chunk
    else:
        if False: # NOTE debug only, the parallel inference algorithm is so good for learning in the future
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_batch, gen_text) for gen_text in gen_text_batches]
                for future in progress.tqdm(futures) if progress is not None else futures:
                    result = future.result()
                    if result:
                        generated_wave, generated_mel_spec = next(result)
                        generated_waves.append(generated_wave)
                        spectrograms.append(generated_mel_spec)

        #### for debug only  NOTE TODO ####
        for gen_text in gen_text_batches: # ['那是当然的啦，我们都找到了自己的真正的幸福。']
            generated_wave, generated_mel_spec = process_batch(gen_text) # NOTE TODO important here
            generated_waves.append(generated_wave) # (99840,)是经过了vocoder之后的语音信息了，可以直接保存到.wav文件了
            spectrograms.append(generated_mel_spec) # (100, 391) 是生成的梅尔谱张量
        #### end for debug only ####

        if generated_waves:
            if cross_fade_duration <= 0:
                # Simply concatenate
                final_wave = np.concatenate(generated_waves)
            else:
                # Combine all generated waves with cross-fading
                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                    cross_fade_samples = int(cross_fade_duration * target_sample_rate)
                    cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                    if cross_fade_samples <= 0:
                        # No overlap possible, concatenate
                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    # Overlapping parts
                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    # Fade out and fade in
                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    # Cross-faded overlap
                    cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                    # Combine
                    new_wave = np.concatenate(
                        [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]]
                    )

                    final_wave = new_wave

            # Create a combined spectrogram
            combined_spectrogram = np.concatenate(spectrograms, axis=1)

            yield final_wave, target_sample_rate, combined_spectrogram

        else:
            yield None, target_sample_rate, None


# remove silence from generated wav


def remove_silence_for_generated_wav(filename):
    aseg = AudioSegment.from_file(filename)
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        non_silent_wave += non_silent_seg
    aseg = non_silent_wave
    aseg.export(filename, format="wav")


# save spectrogram


def save_spectrogram(spectrogram, path):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram, origin="lower", aspect="auto")
    plt.colorbar()
    plt.savefig(path)
    plt.close()
