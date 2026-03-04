from hs_tasnet import HSTasNet
from neutone_sdk.utils import save_neutone_model
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn.functional as F
from torch import Tensor, nn

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter,
ContinuousNeutoneParameter


class HSTasNetCore(nn.Module):
    """
    Thin wrapper to make HS‑TasNet traceable.
    Returns only the separated audio (no hiddens).
    """

    def __init__(self, model: HSTasNet):
        super().__init__()
        self.model = model

    def forward(self, audio: Tensor) -> Tensor:
        # Force no internal truncation. We'll pad ourselves.
        recon_audio, _ = self.model(
            audio, auto_curtail_length_to_multiple=False)
        return recon_audio  # [B, num_sources, channels, N]


class HSTasNetWrapper(WaveformToWaveformBase):
    def __init__(self, model: nn.Module, segment_len: int, sample_rate: int, num_sources:
                 int, audio_channels: int):
        super().__init__(model)
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.num_sources = num_sources
        self.audio_channels = audio_channels

    def get_model_name(self) -> str:
        return "HS-TasNet"

    def get_model_authors(self) -> List[str]:
        return ["Sahal Sajeer Kalandan"]

    def get_model_short_description(self) -> str:
        return "Realtime HS-TasNet"

    def get_model_long_description(self) -> str:
        return "Realtime-capable HS-TasNet separation."

    def get_technical_description(self) -> str:
        return "HS-TasNet wrapped for Neutone FX."

    def get_technical_links(self) -> Dict[str, str]:
        return {}

    def get_tags(self) -> List[str]:
        return ["separation"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        # Simple 4‑stem mixer (normalized)
        return [
            ContinuousNeutoneParameter(
                "s1", "Source 1 gain", default_value=1.0),
            ContinuousNeutoneParameter(
                "s2", "Source 2 gain", default_value=1.0),
            ContinuousNeutoneParameter(
                "s3", "Source 3 gain", default_value=1.0),
            ContinuousNeutoneParameter(
                "s4", "Source 4 gain", default_value=1.0),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return self.audio_channels == 1

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return self.audio_channels == 1

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [self.sample_rate]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        # multiples of 1024; add more if you want
        return [self.segment_len, self.segment_len * 2, self.segment_len * 4]

    @tr.jit.export
    def calc_model_delay_samples(self) -> int:
        return 0

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # x: [channels, N]
        n = x.size(1)
        pad = (-n) % self.segment_len
        if pad:
            x = F.pad(x, (0, pad))

        # model expects [B, C, N]
        y = self.model(x.unsqueeze(0))  # [1, S, C, Npad]

        # mix sources with normalized gains
        gains = tr.stack([params["s1"], params["s2"], params["s3"],
                          params["s4"]]).view(1, self.num_sources, 1, 1)
        gains = gains.clamp(min=0)
        gains = gains / (gains.sum() + 1e-8)

        y = (y * gains).sum(dim=1)  # [1, C, Npad]
        y = y[..., :n]
        return y.squeeze(0)


def load_hs_tasnet(trace_len: int = 2048):
    model = HSTasNet()
    model.eval()

    segment_len = model.segment_len
    sample_rate = model.sample_rate
    num_sources = model.num_sources
    audio_channels = model.audio_channels

    core = HSTasNetCore(model)
    example = tr.randn(1, audio_channels, trace_len)
    traced = tr.jit.trace(core, example, strict=False)
    traced.eval()

    return traced, segment_len, sample_rate, num_sources, audio_channels


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_hs_tasnet")
    args = parser.parse_args()

    traced, segment_len, sample_rate, num_sources, audio_channels = load_hs_tasnet()
    wrapper = HSTasNetWrapper(traced, segment_len, sample_rate, num_sources,
                              audio_channels)

    save_neutone_model(wrapper, pathlib.Path(args.output), dump_samples=True,
                       submission=True)
