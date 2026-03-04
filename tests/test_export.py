import pytest


def test_export_smoke():
    try:
        import neutone_sdk  # noqa: F401
    except Exception:
        pytest.skip("neutone_sdk not available")

    from hs_tasnet.export.export_hs_tasnet import load_hs_tasnet

    traced, segment_len, sample_rate, num_sources, audio_channels = load_hs_tasnet(trace_len=256)
    assert traced is not None
    assert segment_len > 0
    assert sample_rate > 0
    assert num_sources > 0
    assert audio_channels > 0
