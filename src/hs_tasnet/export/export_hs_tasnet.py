from __future__ import annotations

import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn.functional as F
from torch import Tensor, nn

from hs_tasnet.models.hs_tasnet import HSTasNet, HSTasNetConfig
from hs_tasnet.train.checkpointing import load_checkpoint
from hs_tasnet.utils.config import load_config


def _lazy_import_neutone():
    try:
        from neutone_sdk import (  # type: ignore
            ContinuousNeutoneParameter,
            NeutoneParameter,
            WaveformToWaveformBase,
        )
        from neutone_sdk.utils import save_neutone_model  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("neutone_sdk is required for export") from exc
    return (
        WaveformToWaveformBase,
        NeutoneParameter,
        ContinuousNeutoneParameter,
        save_neutone_model,
    )


class HSTasNetCore(nn.Module):
    def __init__(self, model: HSTasNet):
        super().__init__()
        self.model = model

    def forward(self, audio: Tensor) -> Tensor:
        recon_audio, _ = self.model(
            audio, auto_curtail_length_to_multiple=False, return_aux=True
        )
        return recon_audio  # [B, S, T]


class HSTasNetWrapper:
    def __init__(
        self,
        model: nn.Module,
        segment_len: int,
        sample_rate: int,
        num_sources: int,
        audio_channels: int,
    ):
        WaveformToWaveformBase, NeutoneParameter, ContinuousNeutoneParameter, _ = (
            _lazy_import_neutone()
        )

        class _Wrapper(WaveformToWaveformBase):
            def __init__(
                self,
                model: nn.Module,
                segment_len: int,
                sample_rate: int,
                num_sources: int,
                audio_channels: int,
            ):
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
                return [
                    ContinuousNeutoneParameter("s1", "Source 1 gain", default_value=1.0),
                    ContinuousNeutoneParameter("s2", "Source 2 gain", default_value=1.0),
                    ContinuousNeutoneParameter("s3", "Source 3 gain", default_value=1.0),
                    ContinuousNeutoneParameter("s4", "Source 4 gain", default_value=1.0),
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
                return [self.segment_len, self.segment_len * 2, self.segment_len * 4]

            @tr.jit.export
            def calc_model_delay_samples(self) -> int:
                return 0

            def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
                n = x.size(1)
                pad = (-n) % self.segment_len
                if pad:
                    x = F.pad(x, (0, pad))

                y, _ = self.model(x.unsqueeze(0))  # [1, S, T]
                gains = tr.stack(
                    [params["s1"], params["s2"], params["s3"], params["s4"]]
                ).view(1, self.num_sources, 1)
                gains = gains.clamp(min=0)
                gains = gains / (gains.sum() + 1e-8)

                y = (y * gains).sum(dim=1)  # [1, T]
                y = y[..., :n]
                return y.squeeze(0)

        self.wrapper = _Wrapper(
            model, segment_len, sample_rate, num_sources, audio_channels
        )

    def get(self):
        return self.wrapper


def load_hs_tasnet(
    trace_len: int = 2048,
    cfg: HSTasNetConfig | None = None,
    checkpoint: str | None = None,
):
    model = HSTasNet(cfg or HSTasNetConfig())
    if checkpoint:
        load_checkpoint(checkpoint, model, map_location="cpu")
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


def export_from_cfg(cfg_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_config(cfg_path)
    if overrides:
        from hs_tasnet.utils.config import apply_overrides

        cfg = apply_overrides(cfg, overrides)
    export_cfg = cfg.get("export", {})
    model_cfg = HSTasNetConfig(**cfg.get("model", {}))
    trace_len = export_cfg.get("trace_len", 2048)
    checkpoint = export_cfg.get("checkpoint")
    output_dir = export_cfg.get("output_dir", "export_hs_tasnet")

    _, _, _, save_neutone_model = _lazy_import_neutone()

    traced, segment_len, sample_rate, num_sources, audio_channels = load_hs_tasnet(
        trace_len=trace_len, cfg=model_cfg, checkpoint=checkpoint
    )
    wrapper = HSTasNetWrapper(
        traced, segment_len, sample_rate, num_sources, audio_channels
    )
    save_neutone_model(wrapper.get(), pathlib.Path(output_dir), dump_samples=True, submission=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--trace-len", type=int, default=None)
    parser.add_argument("--cfg", default=None)
    args = parser.parse_args()

    if args.cfg:
        export_from_cfg(args.cfg)
        return

    _, _, _, save_neutone_model = _lazy_import_neutone()
    trace_len = args.trace_len or 2048
    output_dir = args.output or "export_hs_tasnet"

    traced, segment_len, sample_rate, num_sources, audio_channels = load_hs_tasnet(trace_len)
    wrapper = HSTasNetWrapper(
        traced, segment_len, sample_rate, num_sources, audio_channels
    )
    save_neutone_model(wrapper.get(), pathlib.Path(output_dir), dump_samples=True, submission=True)


if __name__ == "__main__":
    main()
