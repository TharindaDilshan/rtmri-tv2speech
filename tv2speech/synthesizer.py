"""Acoustic model: articulatory features -> log-mel spectrogram.

A Transformer encoder maps a per-frame sequence of tract variables and source
features to an 80-bin log-mel spectrogram, refined by a convolutional postnet. An
utterance-level speaker embedding is projected and broadcast-added as global
conditioning.

The model predicts *standardized* log-mel: per-bin mean and standard deviation are
computed once over the pretraining corpus and held fixed. Predictions must be
de-standardized with those same statistics before vocoding, so mel_stats.pt is
required at inference and is distributed alongside the weights.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Only positions below the training crop length are ever trained. Sequences longer
    than that must be decoded in windows (see infer.predict_mel), or the encoder
    receives positions it has never seen.
    """

    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Postnet(nn.Module):
    """Five-layer convolutional postnet predicting a residual refinement of the
    coarse mel prediction (Tacotron 2 style)."""

    def __init__(self, n_mels, channels, kernel, n_layers, dropout):
        super().__init__()
        pad = kernel // 2
        self.n_layers = n_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel, padding=pad), nn.BatchNorm1d(channels)))
        for _ in range(n_layers - 2):
            self.convs.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel, padding=pad), nn.BatchNorm1d(channels)))
        self.convs.append(nn.Sequential(
            nn.Conv1d(channels, n_mels, kernel, padding=pad), nn.BatchNorm1d(n_mels)))

    def forward(self, x):
        x = x.transpose(1, 2)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < self.n_layers - 1:
                x = torch.tanh(x)
            x = F.dropout(x, self.dropout, self.training)
        return x.transpose(1, 2)


class SynthModel(nn.Module):
    """Tract variables and source features -> standardized log-mel.

    Args mirror configs/default.yaml under `synthesizer`. Hyperparameters are passed
    explicitly rather than read from module globals so that a checkpoint can carry its
    own configuration and the same class loads models of different sizes.

    forward returns (coarse, refined); refined = coarse + postnet(coarse). Use the
    refined output at inference.
    """

    def __init__(self, n_input=10, spk_emb_dim=192, n_mels=80, d_model=384, n_head=6,
                 n_layers=6, dim_ff=1536, dropout=0.1, max_pe_len=4000,
                 postnet_channels=256, postnet_kernel=5, postnet_layers=5,
                 postnet_dropout=0.5):
        super().__init__()
        self.hparams = dict(
            n_input=n_input, spk_emb_dim=spk_emb_dim, n_mels=n_mels, d_model=d_model,
            n_head=n_head, n_layers=n_layers, dim_ff=dim_ff, dropout=dropout,
            max_pe_len=max_pe_len, postnet_channels=postnet_channels,
            postnet_kernel=postnet_kernel, postnet_layers=postnet_layers,
            postnet_dropout=postnet_dropout)

        self.input_proj = nn.Linear(n_input, d_model)
        self.spk_proj = nn.Linear(spk_emb_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_pe_len)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_ff, dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, n_mels)
        self.post = Postnet(n_mels, postnet_channels, postnet_kernel,
                            postnet_layers, postnet_dropout)

    def forward(self, x, spk, pad_mask):
        """x: (B,T,n_input)  spk: (B,spk_emb_dim)  pad_mask: (B,T) bool, True = padding."""
        h = self.input_proj(x) + self.spk_proj(spk).unsqueeze(1)
        h = self.pos(h)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        coarse = self.out(h)
        refined = coarse + self.post(coarse.masked_fill(pad_mask.unsqueeze(-1), 0.0))
        return coarse, refined


# def load_synthesizer(ckpt_path, device, **overrides):
#     """Instantiate from a checkpoint, using stored hyperparameters when present."""
#     ckpt = torch.load(ckpt_path, map_location=device)
#     hp = dict(ckpt.get("hparams", {}))
#     hp.update(overrides)
#     model = SynthModel(**hp).to(device)
#     model.load_state_dict(ckpt["model"])
#     model.eval()
#     return model

_LEGACY_KEY_MAP = (("post.c.", "post.convs."),)


def _remap_state_dict(sd):
    out, n = {}, 0
    for k, v in sd.items():
        nk = k
        for old, new in _LEGACY_KEY_MAP:
            if nk.startswith(old):
                nk = new + nk[len(old):]
                n += 1
        out[nk] = v
    return out, n


def load_synthesizer(ckpt_path, device, **overrides):
    """Instantiate from a checkpoint, using stored hyperparameters when present."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = dict(ckpt.get("hparams", {}))
    hp.update(overrides)
    model = SynthModel(**hp).to(device)

    sd, n_remapped = _remap_state_dict(ckpt["model"])
    if n_remapped:
        print(f"  note: remapped {n_remapped} legacy postnet keys (post.c -> post.convs)")
    model.load_state_dict(sd)          # strict: any residual mismatch raises here
    model.eval()
    return model
