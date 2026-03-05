import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def si_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # pred/target: [B, S, T]
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
