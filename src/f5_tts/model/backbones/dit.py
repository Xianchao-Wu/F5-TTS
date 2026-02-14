"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True, average_upsampling=False, conv_layers=0, conv_mult=2
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 is ~87.38s of 24khz audio; 4096 is ~43.69s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask):
        batch, text_len, text_dim = text.shape

        audio_len = text_len  # cuz text already padded to same length as audio sequence
        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            text_len = text_lens[i].item()

            if text_len == 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [text_len, text_dim]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (1 if j >= text_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:audio_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], seq_len, drop_text=False):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0
        import ipdb; ipdb.set_trace() # NOTE TODO 所以，这里，drop_text是的确把text设置为0了。
        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"], # noisy audio, [1, 658, 100]
        cond: float["b n d"], # ref audio + 0 of to-be-gen audio, [1, 658, 100]
        text_embed: float["b n d"], # only the first 71 is with value, [1, 658, 100]
        drop_audio_cond=False, # False
        audio_mask: bool["b n"] | None = None, # None
    ):
        if drop_audio_cond:  # cfg for cond audio; False
            cond = torch.zeros_like(cond) # [1, 658, 100] all 0

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1)) # NOTE TODO noisy_audio ref_audio+gen_audio text_embed --> [1, 658, 100], [1, 658, 100], [1, 658, 512] --> [1, 658, 712] --> 712 to 1024 --> [1, 658, 1024]
        x = self.conv_pos_embed(x, mask=audio_mask) + x
        return x # [1, 658, 1024]
        # NOTE when drop_audio_cond, 只有中间的ref_audio is all 0 --> no --> Line 94 文本也是被设置为了0了！！！

# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d, noisy audio, [1, 658, 100]
        cond,  # b n d, ref audio + 0 for to-be-gen audio [1, 658, 100]
        text,  # b nt, [1, 71]
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(text, x.shape[1], drop_text=drop_text) # NOTE 重要，这里是4层convnextv2 blocks，用于对文本进行embedding; text: from [1, 71] to [1, 658, 512], 这里是让文本的长度71，更新为了目标audio的mel frame的长度658; 这个长度658里面，只有前面的71个位置上有取值，后面的都是0！
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)  # Calculate the actual sequence length for each sample
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(text_embed_list, batch_first=True, padding_value=0)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond # [1, 658, 512]
        import ipdb; ipdb.set_trace()
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond, audio_mask=audio_mask)
        # x=noised audio [1, 658, 100]; cond=ref audio + to-be-gen audio [1, 658, 100]; text_embed of [1, 658, 100] and only first 71 elements are with value; drop_audio_cond=False; audio_mask=None
        return x # x.shape=[1, 658, 1024], 这是按照最后一个dim对三个张量进行拼接!

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
    def forward(
        self,
        x: float["b n d"],  # noised input audio 带噪声的语音张量, [1, 658, 100]
        cond: float["b n d"],  # masked cond audio, [1, 658, 100], 左边是268个mel frames，ref audio的；右边是390个全0的mel frames
        text: int["b nt"],  # text, shape=[1, 71]
        time: float["b"] | float[""],  # time step, e.g., tensor(0., device='cuda:0', dtype=torch.float16)
        mask: bool["b n"] | None = None, # batch seq len mask, 当前为None
        drop_audio_cond: bool = False,  # cfg for cond audio; False
        drop_text: bool = False,  # cfg for text; False; 只有当采用了cfg=classifier-free guidance的时候，才有drop_text=True NOTE TODO
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward; True
        cache: bool = False, # True
    ):
        import ipdb; ipdb.set_trace()
        batch, seq_len = x.shape[0], x.shape[1] # 1, 658
        if time.ndim == 0:
            time = time.repeat(batch) # tensor(0., device='cuda:0', dtype=torch.float16) --> tensor([0.], device='cuda:0', dtype=torch.float16)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time) # NOTE embed time, from shape=[1] to [1, 1024]
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache, audio_mask=mask
            ) # x=noised audio [1,658,100]; cond=masked cond audio [1,658,100] with 268 ref mel frames and 390 all-0 mel frames; text=[1,71] --> x_cond.shape=[1, 658, 712=100+100+512 for noisy audio, ref_audio+gen_audio, text_embed] 
            x_uncond = self.get_input_embed(
                x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache, audio_mask=mask
            ) # NOTE TODO 上面的drop_text没有起到作用啊... 只有drop_audio_cond的时候，被设置为0了
            x = torch.cat((x_cond, x_uncond), dim=0) # [1, 658, 1024] + [1, 658, 1024] -> [2, 658, 1024]
            t = torch.cat((t, t), dim=0) # [2, 1024]
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None # mask=None
        else:
            x = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache, audio_mask=mask
            )
        import ipdb; ipdb.set_trace()
        rope = self.rotary_embed.forward_from_seq_len(seq_len) # seq_len=658, -> rope[0].shape=[1, 658, 64]

        if self.long_skip_connection is not None:
            residual = x
        import ipdb; ipdb.set_trace()
        # block = <class 'f5_tts.model.modules.DiTBlock'>
        for block in self.transformer_blocks: # NOTE 22 blocks: AdaLayerNorm + self-Attention + LayerNorm + FeedForward
            if self.checkpoint_activations: # NOTE TODO False
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope) # x.shape=[2, 658, 1024], t.shape=[2, 1024], mask=None, rope[0].shape=[1, 658, 64] --> output x.shape=[2, 658, 1024]
        import ipdb; ipdb.set_trace()
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))
        import ipdb; ipdb.set_trace()
        x = self.norm_out(x, t)
        output = self.proj_out(x) # Linear(in_features=1024, out_features=100, bias=True), from 1024 to 100 (mel dim)

        return output # [2, 658, 100]
