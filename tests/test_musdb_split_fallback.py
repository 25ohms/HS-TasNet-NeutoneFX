from hs_tasnet.data.datasets import _split_train_valid_indices


def test_split_train_valid_indices_uses_80_20_default_math():
    train_idx, valid_idx = _split_train_valid_indices(num_items=10, train_fraction=0.8, seed=42)
    assert len(train_idx) == 8
    assert len(valid_idx) == 2
    assert set(train_idx).isdisjoint(valid_idx)
    assert sorted(train_idx + valid_idx) == list(range(10))


def test_split_train_valid_indices_is_deterministic_for_seed():
    train_a, valid_a = _split_train_valid_indices(num_items=17, train_fraction=0.8, seed=7)
    train_b, valid_b = _split_train_valid_indices(num_items=17, train_fraction=0.8, seed=7)
    assert train_a == train_b
    assert valid_a == valid_b
