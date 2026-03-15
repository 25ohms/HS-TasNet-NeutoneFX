import torch
import torch.nn.functional as F


def _align_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 3 and target.dim() == 4:
        if target.size(2) == 1:
            target = target.squeeze(2)
        else:
            target = target.mean(dim=2)
    elif pred.dim() == 4 and target.dim() == 3:
        target = target.unsqueeze(2)
    return target


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _align_target(pred, target)
    return F.l1_loss(pred, target)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = _align_target(pred, target)
    return F.mse_loss(pred, target)


def signal_distortion_ratio(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    target = _align_target(pred, target)
    error = pred - target
    signal_power = torch.sum(target**2, dim=-1)
    noise_power = torch.sum(error**2, dim=-1)
    sdr = 10 * torch.log10((signal_power + eps) / (noise_power + eps))
    return sdr.mean()


def si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: [B, S, T]
    target = _align_target(pred, target)
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    s_target = (
        torch.sum(pred * target, dim=-1, keepdim=True)
        / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    ) * target
    e_noise = pred - s_target
    ratio = (torch.sum(s_target**2, dim=-1) + eps) / (
        torch.sum(e_noise**2, dim=-1) + eps
    )
    return -10 * torch.log10(ratio + eps).mean()
