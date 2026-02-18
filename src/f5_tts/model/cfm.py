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

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module, # <class 'f5_tts.model.backbones.dit.DiT'>
        sigma=0.0, # 0.0
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask # (0.7, 1.0)

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs)) # 若mel_spec_module!=None,则用它；否则用后面的MelSpec(**mel_spec_kwargs), 目前是用后面的，重新初始化一个梅尔谱对象; mel_spec_kwargs={'n_fft': 1024, 'hop_length': 256, 'win_length': 1024, 'n_mel_channels': 100, 'target_sample_rate': 24000, 'mel_spec_type': 'vocos'} NOTE
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob # 0.3
        self.cond_drop_prob = cond_drop_prob # 0.2

        # transformer
        self.transformer = transformer # <class 'f5_tts.model.backbones.dit.DiT'> NOTE
        dim = transformer.dim # 1024
        self.dim = dim

        # conditional flow related
        self.sigma = sigma # 0.0

        # sampling related
        self.odeint_kwargs = odeint_kwargs # {'method': 'euler'}

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map # str: int 的词典，里面有2545个元素
        '''
        ipdb> for name, module in self.named_children():
        print(name, ":", module.__class__.__name__)

            mel_spec : MelSpec
            transformer : DiT
        '''

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"], # ref audio, [1, 68495] 
        text: int["b nt"] | list[str], # [ref_text + gen_text]
        duration: int | int["b"], # 658, ref audio len + estimated gen audio len
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        import ipdb; ipdb.set_trace()
        self.eval()
        # raw wave

        if cond.ndim == 2: # 的确是二维的, batch.size and points.with.sampling.rate=24k
            cond = self.mel_spec(cond) # NOTE 语音转梅尔谱 [1, 68495] --> torch.Size([1, 100, 268]), 把采样率为24k的ref audio，转成mel spectrogram的张量，维度为100，长度为268; 之前是近似为267，这里把最后的一个frame也保留下来，所以是268
            cond = cond.permute(0, 2, 1) # [1, 268=N=len, 100=F=feat of mel]
            assert cond.shape[-1] == self.num_channels # 100

        cond = cond.to(next(self.parameters()).dtype) # torch.float16, shape=[1, 268, 100]

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long) # tensor([268], device='cuda:0')

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map): # NOTE character to id的词典，2545个条目
                text = list_str_to_idx(text, self.vocab_char_map).to(device) # full pinyin -> id
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch # text.shape=[1, 71], a tensor now

        # duration

        cond_mask = lens_to_mask(lens) # [1, 268] all True; 268是ref audio的mel frame的数量
        if edit_mask is not None: # edit_mask=None
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int): # duration=658
            duration = torch.full((batch,), duration, device=device, dtype=torch.long) # duration=tensor([658], device='cuda:0')

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration) # max_duration=65536
        max_duration = duration.amax() # duration=tensor([658], device='cuda:0')

        # duplicate test corner for inner time step oberservation
        if duplicate_test: # False
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0) # 进来的时候，是cond.shape=[1, 268, 100] ref audio's mel spectrogram, ---> pad之后，是在最右边，增加了658-268=390个0，即得到的是[1, 658, 100]这个张量; 这390个mel frames是为待生成的audio准备的。
        if no_ref_audio: # False, NOTE 这个有意思，这是ref audio为空的时候，自动用0填充一下！这个可以作为消融实验的一个设置 TODO
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False) # [1,268] with 268 True --> [1,658] with 390 False at the end
        cond_mask = cond_mask.unsqueeze(-1) # [1, 658, 1]
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond) # TODO what is torch.where? step_cond == cond, yes
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def fn(t, x): # t=tensor(0., device='cuda:0', dtype=torch.float16); x.shape=torch.Size([1, 658, 100]), 这是268个ref audio mel frames + 390个待生成的目标audio的mel frames了
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            #import ipdb; ipdb.set_trace() # NOTE TODO here
            # predict flow (cond)
            if cfg_strength < 1e-5: # 2.0 > 1e-5
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance NOTE TODO
            pred_cfg = self.transformer( # self.transformer : <class 'f5_tts.model.backbones.dit.DiT'>
                x=x, # x.shape=[1,658,100], pure noise
                cond=step_cond, # [1,658,100], ref audio + to-be-generated audio
                text=text, # [1,71] ref text + gen text
                time=t, # tensor(0., device='cuda:0', dtype=torch.float16); t.shape=torch.Size([])
                mask=mask, # None, length mask for different seq in one batch, 现在因为是batch size=1，所以mask=None
                cfg_infer=True,
                cache=True,
            ) # pred_cfg.shape=[2, 658, 100]
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0) # pred.shape=[1, 658, 100], null_pred.shape=[1, 658, 100]
            return pred + (pred - null_pred) * cfg_strength # NOTE 非常重要, 论文中的Equation (4)

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration: # tensor([658], device='cuda:0')
            if exists(seed): # seed=None
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype)) # NOTE [658, 100] pure noise tensor, act as x0 in the paper of F5-TTS
        y0 = pad_sequence(y0, padding_value=0, batch_first=True) # y0.shape=[1,658,100]

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test: # False
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype) # NOTE t.shape=[33], 一个等差数列
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t) # NOTE 这里非常重要，这是sway sampling的 重要使用，摇摆采样; alike x^2了; 抛物线在直线的下方 NOTE
        import ipdb; ipdb.set_trace()
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs) # NOTE TODO y0=pure audio noise; self.odeint_kwargs={'method': 'euler'}; t=tensor([0.0000e+00, 9.7656e-04, 4.8828e-03, 1.0742e-02, 1.9043e-02, 2.9785e-02,...) with a shape of [33] 这个时间schedule是个非线性的! 这是跑完全部time points的! ---> len(trajectory)=33, trajectory[0].shape=[1, 658, 100] all the 33 elements are with the same shape!
        self.transformer.clear_cache()

        sampled = trajectory[-1] # [1, 658, 100]
        out = sampled # out.shape=[1, 658, 100] this is a weighted sum of conditional and null_conditional preds.
        out = torch.where(cond_mask, cond, out) # cond_mask=[268*True and then 390*False] with shape=[1, 658, 1], 

        if exists(vocoder): # NOTE TODO why vocoder=None here?
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        import ipdb; ipdb.set_trace()
        return out, trajectory # out.shape=[1, 658, 100] with the later 390 elements are meaningful

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave; curerntly mel-spectrogram [2, 528, 100]
        text: int["b nt"] | list[str], # a list with 2 textual sequences
        *,
        lens: int["b"] | None = None, # lengths for the mel-spectrogram: [528, 121]
        noise_scheduler: str | None = None, # None
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels
        import ipdb; ipdb.set_trace()
        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        # 2, 528, torch.float32, cuda:0, 0.0=self.sigma
        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device) # [2, 97] with token.ids, and padding=-1
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):  # if lens not acquired by trainer from collate_fn
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len) # [2, 528], and mask[0] all 528*True and mask[1] first 121*True and other spaces are all False, this mask is for length masking in current batch! NOTE

        # get a random span (RANDOM SPAN) to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask) # TODO what for? (0.7, 1.0) -> tensor([0.9656, 0.7461], device='cuda:0')
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths) # [528, 121] + (0.9656, 0.7461) --> [509, 90] for the positions in the mel-spectrogram to be kept! NOTE TODO 从结果上看，是一个连续的区域被赋值为了False，也就是说被mask掉的梅尔谱，是连续的一个子区域！！！连续的！这个和论文里面的逻辑就对上了。

        if exists(mask):
            rand_span_mask &= mask # rand_span_mask no change after & with mask! NO CHANGE

        # mel is x1
        x1 = inp # real-world data, audio mel spectrogram

        # x0 is gaussian noise
        x0 = torch.randn_like(x1) # NOTE this is the pure noise audio sampled from Gaussian distribution!

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device) # TODO 这个重要，这个是随机出来两个time points，供当前的一个batch中的两个样本使用。这个逻辑和diffusion model中的训练的逻辑是一样的; time=[0.0315, 0.2665] for different period of noise (different levels of noise during current x1-> x0 trajectory)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1) # t.shape=[2, 1, 1]
        φ = (1 - t) * x0 + t * x1 # (1-t)*noise + t*data
        flow = x1 - x0 # derivation on t of Line 287 results x1 - x0! data - noise! target - source! NOTE TODO flow.shape=[2, 528, 100]

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1) # where(cond, x, y) and when cond=True, return x part; otherwise return y part NOTE TODO 从这里可以看到，这里的cond是把大部分的内容都设置为了0了，也就是说，当rand_span_mask=True的时候，是设置为0, mask掉了；如果是rand_span_mask=False的时候，则表示是使用real data x1的内容了。所以，看到cond的结果是的确是大部分都设置为了0了，等待模型的下一步的预测了。 seq0 masked 509 elements of 528 elements in total; and seq1 masked 90 elements of 121 elements in total; 96.4% and 74.4% which are hugh

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper; this time drop_audio_cond=False
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper NOTE very important!
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False
        import ipdb; ipdb.set_trace()
        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred = self.transformer(
            x=φ, cond=cond, text=text, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, mask=mask
        ) # forward() method in <class 'f5_tts.model.backbones.dit.DiT'> 1. x=noisy audio of shape=[2, 528, 100], 2. cond.shape=[2,528,100] 有意思了，这个是把一部分语音mask之后的结果，作为条件！！！在inference的时候，是ref_audio_mel+to-be-gen_audio_mel all 0; NOTE TODO 3. text.shape=[2, 97] are token.ids, before textual embedding! 4. time.shape=[2] with a batch of randomly generated timepoints for current datasets! NOTE this is only one time point for one sequence; not a time schedule! 5. drop_audio_cond=False, 6. drop_text=False, 7. mask.lengths=[528, 121] ---> pred.shape=[2, 528, 100] is the predicted noisy audio
        import ipdb; ipdb.set_trace()
        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none") # NOTE TODO okay I see now, pred=predicated noisy audio; and flow = reference (training target which is made up on-the-fly! interesting); output loss.shape=[2, 528, 100]
        loss = loss[rand_span_mask] # rand_span_mask with 509/90 elements for the two sequences; meaning that we are predicting these 509/90 elements in the noisy audio. Thus, we only compute the losses of the masked places.

        import ipdb; ipdb.set_trace()
        return loss.mean(), cond, pred
        # loss.mean() -> tensor(5.2501, device='cuda:0', grad_fn=<MeanBackward0>)
        # cond.shape=[2, 528, 100], 只留下少数没有被mask掉的mel frames作为参考, alike ref_audio+to-be-gen audio in the inference step! NOTE
        # pred.shape=[2, 528, 100] the prediction output of the self.transformer: DiT module

