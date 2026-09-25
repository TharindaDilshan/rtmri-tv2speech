#!/usr/bin/env python3
"""Synthesize speech from rtMRI tract variables.

Two input modes:

  --features PREFIX   loads PREFIX_x.npy (T,10) and PREFIX_spk.npy (192,).
                      This is the format of the released samples and needs no audio.

  --tv-csv FILE --audio FILE
                      builds the input from a tract-variable csv at the native rtMRI
                      frame rate plus a reference recording. Audio is required: the
                      source features and the speaker embedding are derived from it.

Examples
--------
    python scripts/synthesize.py \\
        --features samples/inputs/sub013_2drt_17_topic1 \\
        --synth-ckpt checkpoints/synth.pt \\
        --voc-ckpt checkpoints/vocoder.pt \\
        --mel-stats checkpoints/mel_stats.pt \\
        --out out.wav
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv2speech.config import load_config
from tv2speech.infer import (load_mel_stats, synthesize_from_features,
                             synthesize_from_raw)
from tv2speech.synthesizer import load_synthesizer
from tv2speech.vocoder import load_vocoder


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--synth-ckpt", required=True, help="acoustic model checkpoint")
    p.add_argument("--voc-ckpt", required=True, help="vocoder checkpoint")
    p.add_argument("--mel-stats", required=True,
                   help="mel_stats.pt with per-bin mean/std; required to de-standardise")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--features", help="prefix for PREFIX_x.npy and PREFIX_spk.npy")
    g.add_argument("--tv-csv", help="tract variables at the native rtMRI frame rate")
    p.add_argument("--audio", help="reference recording (required with --tv-csv)")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = load_synthesizer(a.synth_ckpt, device)
    vocoder = load_vocoder(a.voc_ckpt, device)
    mel_mean, mel_std = load_mel_stats(a.mel_stats)
    print(f"device {device} | synthesizer {sum(q.numel() for q in model.parameters()):,} params")

    if a.features:
        x = np.load(f"{a.features}_x.npy").astype(np.float32)
        spk = torch.from_numpy(np.load(f"{a.features}_spk.npy")).float()
        assert x.shape[1] == cfg.synthesizer.n_input, (
            f"expected {cfg.synthesizer.n_input} input channels, got {x.shape[1]}")
        wav = synthesize_from_features(x, spk, model, vocoder, device,
                                       mel_mean, mel_std, cfg)
    else:
        if not a.audio:
            p.error("--tv-csv requires --audio: the source features and speaker "
                    "embedding are derived from the reference recording")
        wav = synthesize_from_raw(a.tv_csv, a.audio, model, vocoder, device,
                                  mel_mean, mel_std, cfg)

    import soundfile as sf
    sf.write("samples/outputs/pred.wav", wav, cfg.audio.sample_rate)
    print(f"wrote samples/outputs/pred.wav  ({len(wav) / cfg.audio.sample_rate:.2f} s)")


if __name__ == "__main__":
    main()
