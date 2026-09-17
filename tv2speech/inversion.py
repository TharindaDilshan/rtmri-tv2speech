"""Acoustic-to-articulatory inversion: speech -> 8 tract variables.

A frozen WavLM-Large encoder supplies per-layer hidden states; a learned softmax over
layers pools them, and a two-layer BiLSTM head regresses the eight tract variables.
Freezing the encoder is deliberate: the head is trained on roughly two hours of paired
articulatory data, which is far too little to adapt a large self-supervised model
without overfitting.

WavLM emits 50 Hz features. Ground-truth tract variables are recorded at the native
rtMRI frame rate and are linearly resampled to the WavLM output length during training,
so the model's native output rate is 50 Hz regardless of the rate of its targets.

Input normalisation matters and must match training exactly: waveforms are
standardised per utterance, (w - mean) / (std + 1e-8), before entering WavLM.
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LayerWeights(nn.Module):
    """Learned softmax-weighted sum over WavLM's per-layer hidden states.

    Different layers of a self-supervised encoder carry different information; rather
    than committing to one, the head learns which to attend to.
    """

    def __init__(self, n_layers):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, feats_list):
        stack = torch.stack(feats_list, dim=0)                 # (L,B,T,D)
        a = torch.softmax(self.w, dim=0).view(-1, 1, 1, 1)
        return (stack * a).sum(dim=0)                          # (B,T,D)


class InversionModel(nn.Module):
    """WavLM hidden states -> (B,T,n_tv) tract variables at 50 Hz."""

    def __init__(self, n_layers_wavlm, feat_dim=1024, proj_dim=256, lstm_hidden=384,
                 lstm_layers=2, dropout=0.1, n_tv=8):
        super().__init__()
        self.hparams = dict(
            n_layers_wavlm=n_layers_wavlm, feat_dim=feat_dim, proj_dim=proj_dim,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout, n_tv=n_tv)
        self.layer_weights = LayerWeights(n_layers_wavlm)
        self.proj = nn.Linear(feat_dim, proj_dim)
        self.lstm = nn.LSTM(proj_dim, lstm_hidden, lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.out = nn.Linear(lstm_hidden * 2, n_tv)

    def forward(self, feats_list, out_lengths):
        x = torch.tanh(self.proj(self.layer_weights(feats_list)))
        packed = pack_padded_sequence(x, out_lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed_out, batch_first=True,
                                   total_length=feats_list[0].size(1))
        return self.out(x)


def load_wavlm(device):
    """Frozen WavLM-Large feature extractor. Expects 16 kHz input."""
    import torchaudio
    bundle = torchaudio.pipelines.WAVLM_LARGE
    model = bundle.get_model().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    assert bundle.sample_rate == 16000
    return model


@torch.no_grad()
def wavlm_features(wavlm, wav16):
    """wav16: (B,N) at 16 kHz, equal lengths (no padding) -> list of (B,T,1024)."""
    feats, _ = wavlm.extract_features(wav16)
    return feats


@torch.no_grad()
def predict_tvs(inv_model, wavlm, wav16, device):
    """Single utterance (N,) numpy or tensor at 16 kHz -> (T,n_tv) numpy at 50 Hz.

    Applies the per-utterance standardisation used during training.
    """
    if not torch.is_tensor(wav16):
        wav16 = torch.from_numpy(wav16)
    w = wav16.float()
    w = (w - w.mean()) / (w.std() + 1e-8)
    feats = wavlm_features(wavlm, w.unsqueeze(0).to(device))
    T = feats[0].shape[1]
    out = inv_model(feats, torch.full((1,), T, device=device))
    return out.squeeze(0).cpu().numpy()


def load_inversion(ckpt_path, device, n_layers_wavlm=None, wavlm=None, **overrides):
    """Instantiate from a checkpoint. The WavLM layer count is probed if not given."""
    if n_layers_wavlm is None:
        assert wavlm is not None, "pass either n_layers_wavlm or a loaded wavlm model"
        with torch.no_grad():
            n_layers_wavlm = len(wavlm_features(wavlm, torch.zeros(1, 16000).to(device)))
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = dict(ckpt.get("hparams", {}))
    hp.setdefault("n_layers_wavlm", n_layers_wavlm)
    hp.update(overrides)
    model = InversionModel(**hp).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
