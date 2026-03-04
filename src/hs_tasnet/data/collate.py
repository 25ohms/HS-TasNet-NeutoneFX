from typing import List, Tuple

import torch

from .datasets import AudioExample


def collate_examples(batch: List[AudioExample]) -> Tuple[torch.Tensor, torch.Tensor]:
    mixtures = torch.stack([ex.mixture for ex in batch], dim=0)
    stems = torch.stack([ex.stems for ex in batch], dim=0)
    return mixtures, stems
