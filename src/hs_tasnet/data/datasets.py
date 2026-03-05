from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hs_tasnet.utils.audio import load_audio


@dataclass
class AudioExample:
    mixture: torch.Tensor
    stems: torch.Tensor


class TinySyntheticDataset(Dataset):
    """Synthetic dataset for smoke tests/CI."""

    def __init__(self, length: int, segment_samples: int, num_stems: int, channels: int = 1):
        self.length = length
        self.segment_samples = segment_samples
        self.num_stems = num_stems
        self.channels = channels

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> AudioExample:
        rng = np.random.RandomState(idx)
        stems = rng.randn(self.num_stems, self.channels, self.segment_samples).astype(np.float32)
        mixture = stems.sum(axis=0)
        return AudioExample(
            mixture=torch.from_numpy(mixture),
            stems=torch.from_numpy(stems),
        )


class AudioStemDataset(Dataset):
    """Dataset expecting track directories with mixture + stems wavs."""

    def __init__(
        self,
        root: str | pathlib.Path,
        stems: List[str],
        segment_samples: int,
        sample_rate: int,
    ) -> None:
        self.root = pathlib.Path(root)
        self.stems = stems
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.tracks = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not self.tracks:
            raise FileNotFoundError(f"No track directories found in {self.root}")

    def __len__(self) -> int:
        return len(self.tracks)

    def _load_track(self, track_dir: pathlib.Path) -> Tuple[np.ndarray, List[np.ndarray]]:
        mixture, sr = load_audio(track_dir / "mixture.wav", target_sr=self.sample_rate)
        stems = []
        for stem in self.stems:
            audio, _ = load_audio(track_dir / f"{stem}.wav", target_sr=self.sample_rate)
            stems.append(audio)
        return mixture, stems

    def __getitem__(self, idx: int) -> AudioExample:
        track_dir = self.tracks[idx]
        mixture, stems = self._load_track(track_dir)
        total_samples = mixture.shape[1]
        if total_samples < self.segment_samples:
            pad = self.segment_samples - total_samples
            mixture = np.pad(mixture, ((0, 0), (0, pad)))
            stems = [np.pad(s, ((0, 0), (0, pad))) for s in stems]
            start = 0
        else:
            start = random.randint(0, total_samples - self.segment_samples)
        end = start + self.segment_samples
        mixture = mixture[:, start:end]
        stems = [s[:, start:end] for s in stems]
        stems = np.stack(stems, axis=0)
        return AudioExample(
            mixture=torch.from_numpy(mixture.astype(np.float32)),
            stems=torch.from_numpy(stems.astype(np.float32)),
        )


class MusdbStemDataset(Dataset):
    """MUSDB18 loader using the musdb package (supports .stem.mp4 or wav)."""

    def __init__(
        self,
        root: str | pathlib.Path,
        subset: str,
        split: str | None,
        stems: List[str],
        segment_samples: int,
        sample_rate: int,
        audio_channels: int,
        is_wav: bool = False,
    ) -> None:
        try:
            import musdb  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "musdb is required for data.loader=musdb. Install with the musdb extra."
            ) from exc

        self.root = pathlib.Path(root)
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.stems = stems
        self.audio_channels = audio_channels

        self.db = musdb.DB(
            root=str(self.root),
            subsets=subset,
            split=split,
            is_wav=is_wav,
        )
        self.tracks = list(self.db.tracks)
        if not self.tracks:
            raise FileNotFoundError(f"No tracks found in MUSDB root {self.root}")

    def __len__(self) -> int:
        return len(self.tracks)

    def _mix_channels(self, audio: np.ndarray) -> np.ndarray:
        # audio: [T, C]
        if self.audio_channels == audio.shape[1]:
            return audio
        if self.audio_channels == 1 and audio.shape[1] > 1:
            return audio.mean(axis=1, keepdims=True)
        raise ValueError("Channel mismatch for MUSDB audio")

    def __getitem__(self, idx: int) -> AudioExample:
        track = self.tracks[idx]
        if hasattr(track, "rate") and track.rate != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {track.rate}")

        mixture = track.audio  # [T, C]
        total_samples = mixture.shape[0]
        if total_samples < self.segment_samples:
            pad = self.segment_samples - total_samples
            mixture = np.pad(mixture, ((0, pad), (0, 0)))
            start = 0
        else:
            start = random.randint(0, total_samples - self.segment_samples)
        end = start + self.segment_samples
        mixture = mixture[start:end]

        stem_audio = []
        for stem in self.stems:
            if stem not in track.targets:
                raise KeyError(f"Stem '{stem}' not found in MUSDB targets")
            audio = track.targets[stem].audio[start:end]
            if audio.shape[0] < self.segment_samples:
                pad = self.segment_samples - audio.shape[0]
                audio = np.pad(audio, ((0, pad), (0, 0)))
            stem_audio.append(audio)

        mixture = self._mix_channels(mixture)
        stem_audio = [self._mix_channels(a) for a in stem_audio]
        stems = np.stack(stem_audio, axis=0)  # [S, T, C]

        mixture = mixture.T  # [C, T]
        stems = stems.transpose(0, 2, 1)  # [S, C, T]

        return AudioExample(
            mixture=torch.from_numpy(mixture.astype(np.float32)),
            stems=torch.from_numpy(stems.astype(np.float32)),
        )
