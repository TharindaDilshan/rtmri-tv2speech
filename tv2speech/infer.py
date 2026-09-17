"""End-to-end inference: articulatory features -> waveform.

Two entry points:

    synthesize_from_features(x, spk, ...)   cached (T,10) features + (192,) embedding
    synthesize_from_raw(tv_csv, audio, ...) tract-variable csv + a reference recording

SCOPE. This is not a tract-variable-only synthesizer. Input channels 8 and 9 are
log-F0 and voicing probability extracted with pYIN from the reference recording, and
the speaker embedding is extracted from the same recording. Reconstruction therefore
requires audio from the target utterance, which is the resynthesis setting evaluated
in the paper. A silent-speech or voice-restoration deployment would have to predict
these signals from articulation or supply them externally.

WINDOWED DECODING. The acoustic model is trained on crops of at most
cfg.synthesizer.max_seq_frames, so its absolute positional encoding is defined only
over that range. Feeding a longer sequence in a single pass supplies untrained
positions and degrades output. predict_mel decodes long utterances in overlapping
windows with a tapered overlap-add, which keeps every position within the trained
range and is the inference path used for all reported results.
"""
import numpy as np
import torch


@torch.no_grad()
def _mel_window(model, x, spk, device):
    """One forward pass over a sequence no longer than the training crop."""
    xt = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(device)
    pad = torch.zeros(1, xt.shape[1], dtype=torch.bool, device=device)
    _, refined = model(xt, spk.unsqueeze(0).to(device), pad)
    return refined.squeeze(0)


@torch.no_grad()
def predict_mel(model, x, spk, device, mel_mean, mel_std, cfg):
    """(T,10) features -> (T,n_mels) log-mel, de-standardised and ready to vocode."""
    win = cfg.synthesizer.window
    hop = cfg.synthesizer.window_hop
    taper = cfg.synthesizer.window_taper
    n_mels = cfg.audio.n_mels
    T = x.shape[0]

    if T <= win:
        std_mel = _mel_window(model, x, spk, device)
    else:
        starts = list(range(0, T - win + 1, hop))
        if starts[-1] + win < T:
            starts.append(T - win)
        acc = torch.zeros(T, n_mels, device=device)
        wsum = torch.zeros(T, 1, device=device)
        for st in starts:
            seg = x[st:st + win]
            L = seg.shape[0]
            r = _mel_window(model, seg, spk, device)
            w = torch.ones(L, 1, device=device)
            k = min(taper, L // 4)
            if st > 0:
                w[:k, 0] = torch.linspace(0, 1, k, device=device)
            if st + L < T:
                w[-k:, 0] = torch.linspace(1, 0, k, device=device)
            acc[st:st + L] += r * w
            wsum[st:st + L] += w
        std_mel = acc / wsum.clamp(min=1e-6)

    return std_mel * mel_std.to(device) + mel_mean.to(device)


@torch.no_grad()
def synthesize_from_features(x, spk, model, vocoder, device, mel_mean, mel_std, cfg):
    """x: (T,10) float32 array. spk: (192,) tensor. -> (N,) waveform at cfg sample rate."""
    log_mel = predict_mel(model, x, spk, device, mel_mean, mel_std, cfg)
    return vocoder(log_mel.T.unsqueeze(0)).squeeze().cpu().numpy()


@torch.no_grad()
def build_input(tv_csv, audio_path, cfg, ecapa=None, device=None, mel_transform=None):
    """Assemble the (T,10) input and speaker embedding from raw inputs.

    Steps follow the order used to build the training caches:
      1. read the eight tract variables at the native rtMRI frame rate
      2. load the reference recording at 16 kHz
      3. compute log-mel to fix the target frame count T at 50 Hz
      4. extract log-F0 and voicing at the same hop
      5. resample the tract variables from the native rate to T frames
      6. concatenate, and take the speaker embedding from the peak-normalised audio
    """
    import pandas as pd
    from .features import (load_audio_16k, peak_norm, log_mel, extract_f0_voiced,
                           resample_tv_to_len, ecapa_embed, load_ecapa)

    tv = pd.read_csv(tv_csv)[list(cfg.tv.columns)].to_numpy(np.float32)
    wav = load_audio_16k(audio_path, cfg.audio.sample_rate)
    wav_n = peak_norm(wav, cfg.audio.peak_norm_target)

    mel = log_mel(wav_n, cfg, mel_transform)
    src = extract_f0_voiced(wav.numpy(), cfg)
    T = min(mel.shape[1], src.shape[0])

    tv50 = resample_tv_to_len(torch.from_numpy(tv), T).numpy()
    x = np.concatenate([tv50, src[:T]], axis=-1).astype(np.float32)

    if ecapa is None:
        ecapa = load_ecapa(device)
    spk = ecapa_embed(wav_n, ecapa, device).float()
    return x, spk


@torch.no_grad()
def synthesize_from_raw(tv_csv, audio_path, model, vocoder, device,
                        mel_mean, mel_std, cfg, ecapa=None):
    x, spk = build_input(tv_csv, audio_path, cfg, ecapa=ecapa, device=device)
    return synthesize_from_features(x, spk, model, vocoder, device,
                                    mel_mean, mel_std, cfg)


def load_mel_stats(path, device=None):
    """Per-bin mean and standard deviation used to de-standardise predictions.

    These are computed once over the pretraining corpus and held fixed across both
    training stages. Predictions cannot be vocoded without them.
    """
    d = torch.load(path, map_location="cpu")
    mean, std = d["mean"], d["std"]
    if device is not None:
        mean, std = mean.to(device), std.to(device)
    return mean, std
