"""Feature extraction shared by cache building and inference.

The order of operations here matters and mirrors how the training caches were built.
Two conventions in particular are easy to get wrong:

  * Mel spectrograms and speaker embeddings are computed from peak-normalised audio,
    but F0 is computed from the un-normalised waveform. pYIN's pitch estimate is
    amplitude-invariant, so this is harmless, but keeping it consistent with training
    avoids any doubt.
  * Unvoiced frames keep log-F0 exactly 0 rather than being interpolated through. The
    voicing channel stores pYIN's continuous voiced_probability, not a hard flag. A
    model trained under this convention degrades badly if given interpolated
    continuous-F0 at inference, because the zero is itself a voicing cue.
"""
import numpy as np
import torch
import torch.nn.functional as F


def load_audio_16k(path, target_sr=16000):
    """Load any audio file as a mono waveform tensor at target_sr."""
    import torchaudio
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def peak_norm(wav, target=0.95):
    """Scale to a fixed peak amplitude. Equalises dynamic range across corpora; this is
    not loudness normalisation and does not equalise perceived level."""
    peak = wav.abs().max()
    return wav / (peak + 1e-8) * target if peak > 0 else wav


def make_mel_transform(cfg, device=None):
    import torchaudio.transforms as T
    t = T.MelSpectrogram(
        sample_rate=cfg.audio.sample_rate, n_fft=cfg.audio.n_fft,
        win_length=cfg.audio.win_length, hop_length=cfg.audio.hop_length,
        n_mels=cfg.audio.n_mels, f_min=cfg.audio.f_min, f_max=cfg.audio.f_max,
        center=True, power=1.0)
    return t.to(device) if device is not None else t


def log_mel(wav, cfg, mel_transform=None):
    """(N,) waveform -> (n_mels, T) log-mel. power=1.0 is amplitude, not power."""
    tf = mel_transform if mel_transform is not None else make_mel_transform(cfg)
    m = tf(wav.unsqueeze(0)).squeeze(0)
    return torch.log(m.clamp(min=cfg.audio.mel_clamp_min))


def extract_f0_voiced(wav_np, cfg):
    """(N,) waveform at 16 kHz -> (T,2): normalised log-F0 and voicing probability.

    log-F0 is mapped to [0,1] between f0_min and f0_max; unvoiced frames stay at 0.
    """
    import librosa
    f0, _, voiced_prob = librosa.pyin(
        wav_np.astype(np.float64), fmin=cfg.source.f0_min, fmax=cfg.source.f0_max,
        sr=cfg.audio.sample_rate, hop_length=cfg.audio.hop_length,
        frame_length=cfg.audio.win_length, center=True)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0).astype(np.float32)

    lo, hi = np.log(cfg.source.f0_min), np.log(cfg.source.f0_max)
    log_f0 = np.zeros_like(f0)
    m = f0 > 0
    log_f0[m] = np.clip((np.log(f0[m]) - lo) / (hi - lo), 0.0, 1.0)
    return np.stack([log_f0, voiced_prob], axis=-1).astype(np.float32)


def resample_tv_to_len(tv, target_len):
    """(T_in, n_tv) tract variables -> (target_len, n_tv) by linear interpolation.

    Used to bring tract variables from the native rtMRI frame rate to the 50 Hz audio
    frame rate.
    """
    if not torch.is_tensor(tv):
        tv = torch.from_numpy(np.asarray(tv, dtype=np.float32))
    out = F.interpolate(tv.T.unsqueeze(0).float(), size=int(target_len),
                        mode="linear", align_corners=True)
    return out.squeeze(0).T


def load_ecapa(device, savedir=".cache/ecapa"):
    """ECAPA-TDNN speaker encoder (speechbrain/spkrec-ecapa-voxceleb)."""
    try:
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": str(device)}, savedir=savedir)


@torch.no_grad()
def ecapa_embed(wav, ecapa, device=None):
    """(N,) waveform at 16 kHz -> (192,) speaker embedding tensor on CPU.

    Note that in the resynthesis setting evaluated in the paper this embedding is
    taken from the utterance being reconstructed, which gives the model access to
    information from the target recording. See the README for the implications.
    """
    if not torch.is_tensor(wav):
        wav = torch.from_numpy(wav)
    w = wav.float()
    if device is not None:
        w = w.to(device)
    return ecapa.encode_batch(w.unsqueeze(0)).squeeze().cpu()
