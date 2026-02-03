from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torchaudio
import torchvision
import random
import numpy as np
from python_speech_features import logfbank
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class FBanksAndStack(torch.nn.Module):
    def __init__(self, stack_order=4):
        super().__init__()
        self.stack_order = stack_order

    def stacker(self, feats):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % self.stack_order != 0:
            res = self.stack_order - len(feats) % self.stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, self.stack_order, feat_dim)).reshape(-1, self.stack_order*feat_dim)
        return feats

    def forward(self, x):
        # x: T x 1
        # return: T x F*stack_order
        audio_feats = logfbank(x.squeeze().numpy(), samplerate=16000).astype(np.float32) # [T, F]
        audio_feats = self.stacker(audio_feats) # [T/stack_order_audio, F*stack_order_audio]
        
        with torch.no_grad():
            audio_feats = F.layer_norm(torch.from_numpy(audio_feats), audio_feats.shape[1:])
        return audio_feats

def normalize_audio(waveform):
    max_val = torch.abs(waveform).max()
    return waveform / max_val if max_val > 0 else waveform

class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=None,
        snr_target=None,
    ):
        super().__init__()
        self.snr_levels = [snr_target] if snr_target else [-5, 0, 5, 10, 15, 20, 999999]
        if noise_filename is None:
            # self.noise = torch.randn(1, 16000)
            self.noise = None
        else:
            self.noise, sample_rate = torchaudio.load(noise_filename)
            assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise is None:
            return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


@dataclass
class TemporalRandomWalk1D(torch.nn.Module):
    """
    Generic temporally-coherent 1D-parameter augmentation for video clips.

    You provide:
      - frame_transform(frame, param) -> frame

    This class:
      - samples a smooth param trajectory per clip (OU process),
      - clamps and optionally smooths it,
      - applies your frame_transform with per-frame param.

    Supported clip formats:
      - torch.Tensor: (T, C, H, W)
      - list/tuple of frames (e.g., PIL images)

    Notes:
      - 'param' is a float (Python float) passed to frame_transform.
      - Use bounds to prevent unrealistic drift.
    """
    frame_transform: Callable[[torch.Tensor, float], torch.Tensor]

    # Probability to apply to the whole clip (coherent decision)
    p: float = 1.0

    # Parameter bounds
    min_val: float = -10.0
    max_val: float = 10.0

    # OU process params (mean-reverting random walk)
    alpha: float = 0.15  # mean reversion strength
    sigma: float = 1.0   # noise std per step (in same units as param)
    mu: float = 0.0      # long-run mean

    # Optional smoothing over time (moving average window)
    smooth_window: int = 1  # 1 => no smoothing; try 3 or 5

    # Optional: start value (None => sample from N(mu, sigma))
    start: float | None = None

    # Torch random generator for reproducibility
    generator: torch.Generator | None = None

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _sample_trajectory(self, T: int, device=None) -> torch.Tensor:
        a = torch.empty((T,), device=device, dtype=torch.float32)

        if self.start is None:
            a0 = torch.empty((), device=device).normal_(
                mean=self.mu, std=self.sigma, generator=self.generator
            )
            a[0] = a0
        else:
            a[0] = float(self.start)

        for t in range(1, T):
            eps = torch.empty((), device=device).normal_(0.0, 1.0, generator=self.generator)
            a[t] = a[t - 1] + self.alpha * (self.mu - a[t - 1]) + self.sigma * eps

        a = a.clamp(self.min_val, self.max_val)

        # Optional moving-average smoothing
        if self.smooth_window > 1 and T > 1:
            w = int(self.smooth_window)
            pad = w // 2
            ap = torch.nn.functional.pad(a[None, None, :], (pad, pad), mode="reflect")
            kernel = torch.ones((1, 1, w), device=device, dtype=a.dtype) / w
            a = torch.nn.functional.conv1d(ap, kernel).squeeze(0).squeeze(0)
            a = a.clamp(self.min_val, self.max_val)

        return a

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # Coherent apply/skip decision for the whole clip
        if self.p < 1.0:
            r = torch.rand((), generator=self.generator).item()
            if r > self.p:
                return clip

        # Tensor clip: (T,C,H,W)
        if isinstance(clip, torch.Tensor):
            if clip.ndim != 4:
                raise ValueError(f"Expected tensor clip (T,C,H,W), got {tuple(clip.shape)}")
            T = clip.shape[0]
            device = clip.device
            params = self._sample_trajectory(T, device=device)

            out_frames = []
            for t in range(T):
                frame = clip[t]
                out_frames.append(self.frame_transform(frame, float(params[t].item())))
            return torch.stack(out_frames, dim=0)

        # List/tuple clip
        if isinstance(clip, (list, tuple)):
            T = len(clip)
            params = self._sample_trajectory(T, device=None)  # CPU
            out = []
            for t in range(T):
                out.append(self.frame_transform(clip[t], float(params[t].item())))
            return out

        raise TypeError(f"Unsupported clip type: {type(clip)}")

def rotate_frame(frame, angle_deg: float):
    return TF.rotate(
        frame,
        angle=angle_deg,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        center=None,
        fill=0,
    )

def brightness_frame(frame, brightness_factor: float):
    return TF.adjust_brightness(frame, brightness_factor + 1.0)


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),
                # torchvision.transforms.RandomRotation(10),
                # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                torchvision.transforms.GaussianBlur((5,5), (0.01, 0.5)),
                TemporalRandomWalk1D(
                    frame_transform=rotate_frame,
                    p=0.9,
                    min_val=-12.0,
                    max_val=12.0,
                    alpha=0.2,
                    sigma=2,
                    mu=0.0,
                    smooth_window=3,
                ),
                TemporalRandomWalk1D(
                    frame_transform=brightness_frame,
                    p=0.9,
                    min_val=-1,
                    max_val=1.0,
                    alpha=0.2,
                    sigma=0.1,
                    mu=0.0,
                    smooth_window=3,
                ),
                # torchvision.transforms.Grayscale(),
                # AdaptiveTimeMask(50, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.CenterCrop(88),
                # torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        assert len(sample.shape) == 4
        assert sample.shape[1] == 1
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, snr_target=None):
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                AddNoise(),
                FBanksAndStack(),
                # FunctionalModule(
                #     lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                # ),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target)
                if snr_target is not None
                else FunctionalModule(lambda x: x),
                FBanksAndStack(),
                # FunctionalModule(
                #     lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
                # ),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)