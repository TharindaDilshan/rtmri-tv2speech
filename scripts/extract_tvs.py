#!/usr/bin/env python3
"""Extract tract variables from tracked midsagittal contours.

Reads a task JSON of tracked articulator contours, computes the eight tract
variables frame by frame, and writes a CSV in the form the synthesizer expects.

    python scripts/extract_tvs.py --track-json samples/contours/<id>.json

By default this reads from samples/contours/ and writes to samples/inputs/.

WHAT THIS DOES BEYOND RAW EXTRACTION
------------------------------------
tv_extraction.process_task emits raw measurements in image units, together with
closure flags and carried-forward constriction-location values in separate columns.
The synthesizer consumes eight normalised channels, so this script additionally:

  1. Fills constriction location during closure. When a constrictor is fully closed
     the location is geometrically undefined and process_task writes NaN, holding the
     last valid value in TTCL_last_valid / TBCL_last_valid. We substitute that carried
     value, consistent with treating place as a gestural target rather than as
     instantaneous geometry.
  2. Fills any remaining gaps by forward then backward filling along time, matching
     the cache-building procedure used for the reported experiments.
  3. Normalises each channel to [0,1].

NORMALISATION IS NOT OPTIONAL AND THE CHOICE MATTERS
-----------------------------------------------------
The models were trained on tract variables normalised per speaker using 2nd and 98th
percentiles computed over all of that speaker's training utterances. To reproduce the
paper's conditions you must supply those statistics with --norm-stats and --speaker.

--norm-from-task computes percentiles from the single task being processed. This is a
convenience for users bringing their own data who have no training statistics, but it
is not equivalent: a short task rarely spans a speaker's full articulatory range, so
the resulting scale differs from training and synthesis quality will suffer. The
script warns when this path is used.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tv2speech.config import load_config
from tv_extraction.tv_extraction import process_task

NORM_P_LOW, NORM_P_HIGH = 2, 98


def fill_closures(df):
    """Replace NaN constriction location during closure with the carried-forward value.

    process_task records closure in TTCD_closed / TBCD_closed and the last valid
    location in TTCL_last_valid / TBCL_last_valid. Returns the frame counts affected so
    the caller can report them: a task where most frames are in closure is worth
    inspecting before trusting the output.
    """
    counts = {}
    for cl, closed, last in [("TTCL", "TTCD_closed", "TTCL_last_valid"),
                             ("TBCL", "TBCD_closed", "TBCL_last_valid")]:
        if cl not in df or last not in df:
            continue
        mask = df[cl].isna()
        if closed in df:
            mask = mask | (df[closed] == 1)
        df.loc[mask, cl] = df.loc[mask, last]
        counts[cl] = int(mask.sum())
    return df, counts


def fill_gaps(df, columns):
    """Forward then backward fill along time, then zero any column that is all NaN.

    Mirrors the gap handling in the cache-building code, so a task processed here is
    treated the same way as the training data.
    """
    before = int(df[columns].isna().sum().sum())
    df[columns] = df[columns].ffill().bfill().fillna(0.0)
    return df, before


def normalise(tv, columns, stats):
    """Map each channel to [0,1] using 2nd/98th percentile bounds, clipping outliers."""
    out = tv.copy()
    for c in columns:
        lo, hi = stats[c]["p2"], stats[c]["p98"]
        if hi - lo < 1e-6:
            raise ValueError(f"{c}: degenerate percentile range [{lo}, {hi}]")
        out[c] = np.clip((tv[c] - lo) / (hi - lo), 0.0, 1.0)
    return out


def stats_from_task(tv, columns):
    return {c: {"p2": float(np.nanpercentile(tv[c], NORM_P_LOW)),
                "p98": float(np.nanpercentile(tv[c], NORM_P_HIGH))}
            for c in columns}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track-json", required=True,
                   help="task JSON of tracked contours; a bare filename is resolved "
                        "relative to --contour-dir")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--contour-dir", default="samples/contours")
    p.add_argument("--out-dir", default="samples/inputs")
    p.add_argument("--out", default=None,
                   help="output CSV path; defaults to <out-dir>/<json stem>_tv.csv")
    n = p.add_mutually_exclusive_group(required=True)
    n.add_argument("--norm-stats",
                   help="JSON of per-speaker percentile statistics used in training")
    n.add_argument("--norm-from-task", action="store_true",
                   help="derive percentiles from this task alone (see the note above)")
    n.add_argument("--raw", action="store_true",
                   help="emit unnormalised values; not usable by synthesize.py")
    p.add_argument("--speaker", help="speaker key within --norm-stats")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="also write the full 14-column extraction output")
    a = p.parse_args()

    cfg = load_config(a.config)
    columns = list(cfg.tv.columns)

    json_path = Path(a.track_json)
    if not json_path.exists():
        json_path = Path(a.contour_dir) / a.track_json
    if not json_path.exists():
        p.error(f"contour JSON not found: {a.track_json} (also looked in {a.contour_dir})")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(a.out) if a.out else out_dir / f"{json_path.stem}_tv.csv"

    print(f"extracting from {json_path}")
    work = out_dir / "_intermediate" if not a.keep_intermediate else out_dir
    df = process_task(json_path, work)
    print(f"  {len(df)} frames, {len(df.columns)} raw columns")

    missing = [c for c in columns if c not in df.columns]
    if missing:
        p.error(f"extraction did not produce expected columns: {missing}")

    df, closure_counts = fill_closures(df)
    for k, v in closure_counts.items():
        pct = 100 * v / max(len(df), 1)
        note = "   <- inspect this task" if pct > 40 else ""
        print(f"  {k}: {v} closure frames filled ({pct:.0f}%){note}")

    df, n_gaps = fill_gaps(df, columns)
    if n_gaps:
        print(f"  filled {n_gaps} remaining missing values by forward/backward fill")

    if a.raw:
        print("  WARNING: writing unnormalised values; synthesize.py expects [0,1]")
        tv = df[columns]
    else:
        if a.norm_stats:
            stats = json.loads(Path(a.norm_stats).read_text())
            if a.speaker:
                if a.speaker not in stats:
                    p.error(f"speaker '{a.speaker}' not in {a.norm_stats}; "
                            f"available: {sorted(stats)[:8]}")
                stats = stats[a.speaker]
            elif set(columns) - set(stats):
                p.error("--norm-stats appears to be keyed by speaker; pass --speaker")
            print(f"  normalising with training statistics"
                  + (f" for {a.speaker}" if a.speaker else ""))
        else:
            stats = stats_from_task(df, columns)
            print("  WARNING: percentiles derived from this task alone. These will not "
                  "match the\n           statistics the model was trained with, and "
                  "synthesis quality will suffer.\n           Use --norm-stats where "
                  "training statistics are available.")
        tv = normalise(df[columns], columns, stats)

    out = pd.DataFrame({"frame_id": df["frame_id"]}).join(tv)
    out.to_csv(out_csv, index=False, float_format="%.6f")

    rng = f"[{tv.to_numpy().min():.3f}, {tv.to_numpy().max():.3f}]"
    print(f"\nwrote {out_csv}")
    print(f"  {len(out)} frames at the native rtMRI rate "
          f"({len(out)/cfg.tv.mri_frame_rate:.2f} s at {cfg.tv.mri_frame_rate} Hz)")
    print(f"  {len(columns)} channels, range {rng}")
    print(f"\nsynthesize with:\n"
          f"  python scripts/synthesize.py --tv-csv {out_csv} --audio <reference.wav> \\\n"
          f"      --synth-ckpt checkpoints/synth.pt --voc-ckpt checkpoints/vocoder.pt \\\n"
          f"      --mel-stats checkpoints/mel_stats.pt --out out.wav")
    print("  (audio is required: source features and the speaker embedding come from it)")


if __name__ == "__main__":
    main()