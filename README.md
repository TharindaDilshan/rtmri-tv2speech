# rtMRI Tract-Variable Speech Synthesis

Reference implementation for _[paper title]_ ([venue, year]).

Speech synthesis from vocal tract variables (TVs) measured in real-time MRI. Tract
variables are extracted from tracked midsagittal articulator contours, an
acoustic-to-articulatory inversion model pseudo-labels a large speech corpus, and a
Transformer maps tract variables to mel-spectrograms which a HiFi-GAN vocoder renders
as audio.

This repository contains the **model implementations, the tract-variable extraction
code, and inference**. Training loops, evaluation harnesses, and analysis code are not
included.

## Scope: what this system takes as input

The synthesizer is **not** driven by articulation alone. Its per-frame input has ten
channels: eight tract variables, plus log-F0 and voicing probability extracted with
pYIN **from the target recording**. It is additionally conditioned on an ECAPA-TDNN
speaker embedding, also taken from that recording. Reconstruction therefore requires
audio of the utterance being synthesized, which is the resynthesis setting evaluated
in the paper.

An ablation in the paper shows the source features are strongly load-bearing: removing
them raises character error rate substantially. A silent-speech or voice-restoration
application would not have access to them and would need to predict them from
articulation or supply them externally. We state this prominently because it bounds
what the reported numbers mean.

## Install

```bash
git clone https://github.com/<user>/rtmri-tv2speech.git
cd rtmri-tv2speech
pip install -r requirements.txt
```

## Checkpoints

Weights and the mel standardisation statistics are archived separately; see
`checkpoints/README.md`. Three files are needed for inference:

| File           | Purpose                                    |
| -------------- | ------------------------------------------ |
| `synth.pt`     | acoustic model, tract variables to log-mel |
| `vocoder.pt`   | HiFi-GAN generator, log-mel to waveform    |
| `mel_stats.pt` | per-bin mean and standard deviation        |

`mel_stats.pt` is **required**. The acoustic model predicts standardised log-mel;
without these statistics its output cannot be de-standardised and the vocoder receives
a spectrogram in the wrong space.

## Usage

From the released sample features, which need no audio:

```bash
python scripts/synthesize.py \
    --features samples/inputs/<id> \
    --synth-ckpt checkpoints/synth.pt \
    --voc-ckpt checkpoints/vocoder.pt \
    --mel-stats checkpoints/mel_stats.pt \
    --out out.wav
```

From a tract-variable csv and a reference recording:

```bash
python scripts/synthesize.py \
    --tv-csv path/to/tvs.csv --audio path/to/reference.wav \
    --synth-ckpt checkpoints/synth.pt \
    --voc-ckpt checkpoints/vocoder.pt \
    --mel-stats checkpoints/mel_stats.pt \
    --out out.wav
```

Extracting tract variables from tracked contours:

```bash
python scripts/extract_tvs.py --track-json path/to/contours.json --out tvs.csv
```

## Repository layout

```
tv2speech/
    config.py         configuration loading and geometry
    assertions        input validation and error checking
    tv_extraction.py  tracked contours -> 8 tract variables
    inversion.py      frozen WavLM + BiLSTM head -> tract variables
    synthesizer.py    Transformer, tract variables + source -> log-mel
    vocoder.py        HiFi-GAN v1 generator (inference path only)
    features.py       mel, pYIN source features, TV resampling, speaker embedding
    infer.py          windowed decoding and the end-to-end chain
scripts/            command-line entry points
configs/            all constants in one file
samples/            example inputs and outputs
tv_extraction/      contour tracking and TV extraction logic
```

## Signal geometry

All stages operate at 16 kHz with a hop of 320 samples, giving a 50 Hz frame rate that
matches WavLM's output rate and the vocoder's upsampling product (8x5x4x2 = 320).
`config.load_config` asserts this relationship, since a mismatch would silently produce
audio at the wrong rate. Tract variables are recorded at the native rtMRI frame rate
and linearly resampled to 50 Hz before entering the synthesizer.

## Windowed decoding

The acoustic model is trained on crops of at most 1000 frames, so its absolute
positional encoding is only defined over that range. Utterances longer than 20 seconds
are decoded in overlapping 1000-frame windows with a tapered overlap-add. Decoding a
long utterance in a single pass supplies untrained positions and measurably degrades
output; all reported results use the windowed path implemented in `infer.predict_mel`.

## Data

Experiments use the USC 75-Speaker rtMRI database with Annot-16 contour and phonetic
annotations, and LibriTTS-R for pseudo-labelled pretraining. Neither corpus is
redistributed here; see their respective sources for access.

## Citation

```bibtex
@inproceedings{...}
```
