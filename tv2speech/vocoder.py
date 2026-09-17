"""HiFi-GAN v1 generator: log-mel -> waveform.

Inference path only. The multi-period and multi-scale discriminators used during
training are not included here, since they are not needed to run the model and are
unmodified from Kong et al. (2020).

The upsampling rates multiply to the mel hop length (8*5*4*2 = 320 at 16 kHz), so one
mel frame produces exactly 320 output samples. This must match the frame rate the
acoustic model predicts; config.load_config asserts it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pad(kernel, dilation=1):
    return (kernel * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel, dilations, lrelu=0.1):
        super().__init__()
        wn = nn.utils.weight_norm
        self.lrelu = lrelu
        self.convs1 = nn.ModuleList([
            wn(nn.Conv1d(channels, channels, kernel, dilation=d, padding=_pad(kernel, d)))
            for d in dilations])
        self.convs2 = nn.ModuleList([
            wn(nn.Conv1d(channels, channels, kernel, padding=_pad(kernel, 1)))
            for _ in dilations])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(F.leaky_relu(x, self.lrelu))
            xt = c2(F.leaky_relu(xt, self.lrelu))
            x = x + xt
        return x


class Generator(nn.Module):
    """log-mel (B, n_mels, T) -> waveform (B, 1, T * prod(upsample_rates))."""

    def __init__(self, n_mels=80, upsample_rates=(8, 5, 4, 2),
                 upsample_kernels=(16, 10, 8, 4), resblock_kernels=(3, 7, 11),
                 resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 initial_channels=512, lrelu_slope=0.1):
        super().__init__()
        self.hparams = dict(
            n_mels=n_mels, upsample_rates=list(upsample_rates),
            upsample_kernels=list(upsample_kernels),
            resblock_kernels=list(resblock_kernels),
            resblock_dilations=[list(d) for d in resblock_dilations],
            initial_channels=initial_channels, lrelu_slope=lrelu_slope)
        wn = nn.utils.weight_norm
        h = initial_channels
        self.lrelu = lrelu_slope
        self.n_kernels = len(resblock_kernels)

        self.conv_pre = wn(nn.Conv1d(n_mels, h, 7, padding=3))
        self.ups = nn.ModuleList([
            wn(nn.ConvTranspose1d(h // (2 ** i), h // (2 ** (i + 1)), k,
                                  stride=u, padding=(k - u) // 2))
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernels))])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h // (2 ** (i + 1))
            for k, d in zip(resblock_kernels, resblock_dilations):
                self.resblocks.append(ResBlock(ch, k, d, lrelu_slope))
        self.conv_post = wn(nn.Conv1d(h // (2 ** len(self.ups)), 1, 7, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, self.lrelu))
            xs = None
            for j in range(self.n_kernels):
                r = self.resblocks[i * self.n_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.n_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x, self.lrelu)))


# def load_vocoder(ckpt_path, device, **overrides):
#     """Instantiate from a checkpoint. Training checkpoints store the generator under
#     the key 'gen' alongside discriminator state, which is ignored here."""
#     ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
#     hp = dict(ckpt.get("hparams", {}))
#     hp.update(overrides)
#     gen = Generator(**hp).to(device)
#     gen.load_state_dict(ckpt["gen"] if "gen" in ckpt else ckpt["model"])
#     gen.eval()
#     return gen

_LEGACY_PREFIX_MAP = (("res.", "resblocks."),)
_LEGACY_INNER_MAP = ((".c1.", ".convs1."), (".c2.", ".convs2."))


def _remap_state_dict(sd):
    out, n = {}, 0
    for k, v in sd.items():
        nk, changed = k, False
        for old, new in _LEGACY_PREFIX_MAP:
            if nk.startswith(old):
                nk = new + nk[len(old):]
                changed = True
        for old, new in _LEGACY_INNER_MAP:
            if old in nk:
                nk = nk.replace(old, new)
                changed = True
        out[nk] = v
        n += int(changed)
    return out, n


def load_vocoder(ckpt_path, device, **overrides):
    """Instantiate from a checkpoint. Training checkpoints store the generator under
    the key 'gen' alongside discriminator state, which is ignored here."""
    ckpt = torch.load(ckpt_path, map_location=device)
    hp = dict(ckpt.get("hparams", {}))
    hp.update(overrides)
    gen = Generator(**hp).to(device)

    raw = ckpt["gen"] if "gen" in ckpt else ckpt["model"]
    sd, n_remapped = _remap_state_dict(raw)
    if n_remapped:
        print(f"  note: remapped {n_remapped} legacy generator keys "
              f"(res -> resblocks, c1/c2 -> convs1/convs2)")
    gen.load_state_dict(sd)            # strict: any residual mismatch raises here
    gen.eval()
    return gen