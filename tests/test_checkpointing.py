from pathlib import Path

import pytest

from hs_tasnet.train.checkpointing import prune_checkpoints


def test_prune_checkpoints_keeps_latest_files(tmp_path: Path):
    for epoch in range(1, 6):
        (tmp_path / f"checkpoint_epoch{epoch}.pt").write_bytes(b"checkpoint")

    prune_checkpoints(tmp_path, keep_last=2)

    remaining = sorted(path.name for path in tmp_path.glob("checkpoint_epoch*.pt"))
    assert remaining == ["checkpoint_epoch4.pt", "checkpoint_epoch5.pt"]


def test_prune_checkpoints_rejects_non_positive_keep_last(tmp_path: Path):
    with pytest.raises(ValueError):
        prune_checkpoints(tmp_path, keep_last=0)
