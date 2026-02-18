# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

# '/workspace/asr/F5-TTS/src/f5_tts/../..'
os.chdir(str(files("f5_tts").joinpath("../..")))  # change working directory to root of project (local editable)

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    import ipdb; ipdb.set_trace() 
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}") # <class 'f5_tts.model.backbones.dit.DiT'>
    model_arc = model_cfg.model.arch # {'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'text_mask_padding': True, 'qk_norm': None, 'conv_layers': 4, 'pe_attn_head': None, 'attn_backend': 'torch', 'attn_mask_enabled': False, 'checkpoint_activations': False} model architecture 模型的架构
    tokenizer = model_cfg.model.tokenizer # 'custom'
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type # 'vocos'

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}" # 'F5TTS_v1_Base_vocos_custom_LibriTTS'
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path # '/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/vocab.txt' 词典文件
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer) # vocab_size=2545

    # set model : <class 'f5_tts.model.cfm.CFM'> with 337,096,804=337M parameters
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels), # <class 'f5_tts.model.backbones.dit.DiT'> for transformer
        mel_spec_kwargs=model_cfg.model.mel_spec, # {'target_sample_rate': 24000, 'n_mel_channels': 100, 'hop_length': 256, 'win_length': 1024, 'n_fft': 1024, 'mel_spec_type': 'vocos'}
        vocab_char_map=vocab_char_map, # lexicon str:int with 2545 elements
    )

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs, # 11
        learning_rate=model_cfg.optim.learning_rate, # 7.5e-05
        num_warmup_updates=model_cfg.optim.num_warmup_updates, # 20000
        save_per_updates=model_cfg.ckpts.save_per_updates, # 50000
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints, # -1
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")), # '/workspace/asr/F5-TTS/src/f5_tts/../../ckpts/F5TTS_v1_Base_vocos_custom_LibriTTS' = '/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base_vocos_custom_LibriTTS'
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu, # 2
        batch_size_type=model_cfg.datasets.batch_size_type, # 'frame'
        max_samples=model_cfg.datasets.max_samples, # 2
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps, # 1
        max_grad_norm=model_cfg.optim.max_grad_norm, # 1.0
        logger=model_cfg.ckpts.logger, # 'tensorboard'
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name, # 'F5TTS_v1_Base_vocos_custom_LibriTTS'
        wandb_resume_id=wandb_resume_id, # None
        last_per_updates=model_cfg.ckpts.last_per_updates, # 5000
        log_samples=model_cfg.ckpts.log_samples, # True
        bnb_optimizer=model_cfg.optim.bnb_optimizer, # False
        mel_spec_type=mel_spec_type, # 'vocos'
        is_local_vocoder=model_cfg.model.vocoder.is_local, # False
        local_vocoder_path=model_cfg.model.vocoder.local_path, # None
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True), # {'datasets': {'name': 'LibriTTS', 'batch_size_per_gpu': 2, 'batch_size_type': 'frame', 'max_samples': 2, 'num_workers': 1}, 'optim': {'epochs': 11, 'learning_rate': 7.5e-05, 'num_warmup_updates': 20000, 'grad_accumulation_steps': 1, 'max_grad_norm': 1.0, 'bnb_optimizer': False}, 'model': {'name': 'F5TTS_v1_Base', 'tokenizer': 'custom', 'tokenizer_path': '/workspace/asr/F5-TTS/ckpts/F5TTS_v1_Base/vocab.txt', 'backbone': 'DiT', 'arch': {'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'text_mask_padding': True, 'qk_norm': None, 'conv_layers': 4, 'pe_attn_head': None, 'attn_backend': 'torch', 'attn_mask_enabled': False, 'checkpoint_activations': False}, 'mel_spec': {'target_sample_rate': 24000, 'n_mel_channels': 100, 'hop_length': 256, 'win_length': 1024, 'n_fft': 1024, 'mel_spec_type': 'vocos'}, 'vocoder': {'is_local': False, 'local_path': None}}, 'ckpts': {'logger': 'tensorboard', 'log_samples': True, 'save_per_updates': 50000, 'keep_last_n_checkpoints': -1, 'last_per_updates': 5000, 'save_dir': 'ckpts/F5TTS_v1_Base_vocos_custom_LibriTTS'}}
    )
    import ipdb; ipdb.set_trace()
    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    import ipdb; ipdb.set_trace()
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
