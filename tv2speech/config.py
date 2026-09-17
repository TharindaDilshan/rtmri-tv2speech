"""Configuration loading.

Every constant in the pipeline lives in configs/default.yaml so that the signal
geometry cannot drift between stages. The three rates that must agree are the mel
hop (16000/320 = 50 Hz), WavLM's output rate (also 50 Hz), and the vocoder's
upsampling product (8*5*4*2 = 320). load_config asserts the last of these.
"""
from pathlib import Path
from types import SimpleNamespace

import yaml


def _ns(d):
    """Recursively convert nested dicts to attribute-access namespaces."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_ns(v) for v in d]
    return d


def load_config(path="configs/default.yaml"):
    cfg = _ns(yaml.safe_load(Path(path).read_text()))

    prod = 1
    for r in cfg.vocoder.upsample_rates:
        prod *= r
    assert prod == cfg.audio.hop_length, (
        f"vocoder upsample product {prod} != hop_length {cfg.audio.hop_length}; "
        "the vocoder would not reconstruct the frame rate the synthesizer predicts"
    )
    assert cfg.synthesizer.n_input == cfg.inversion.n_tv + 2, (
        "synthesizer input width must be n_tv + 2 source channels"
    )
    return cfg
