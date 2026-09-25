"""
TV Visualization

plot_tv_overview(json_path) will randomly selects 5 frames from a task JSON and plots each one with:

- All available articulatory contours
- Dashed lines showing each CD measurement (tongue ↔ reference surface)
- A filled dot on the hard palate marking each CL (TTCL, TBCL)
- A dashed line showing LA (lip gap)
- A dotted line showing LP (horizontal protrusion)
- Frame title with closure flags if any are active

Scalar TV values come from the saved CSV. This guarantees the visualisation shows exactly what was extracted and saved, not a re-computation that could differ due to any future code changes.

Geometric coordinates (where to draw lines and markers) are derived from the JSON contours using the saved TTCL / TBCL values as anchors:
- Given TTCL, find the palate point at t=TTCL → that is the line endpoint and the CL marker location.
- The tongue endpoint is the closest tongue point to that palate point.
- TBCD geometry: find closest tongue point to posterior palate+velum, and find palate point at t=TBCL for the CL marker.
- TRCD, LA: closest-point pairs from their respective contours.
- LP: anterior lip x-coordinates and palate t=0.

The CSV path is derived automatically from the JSON path:
- json:  subXXX/track_json/task.json
- csv:   subXXX/extracted_TVs/task.csv
""" 

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from .tv_extraction import (
    build_dense_spline,
    min_dist_two_curves,
    N_DENSE_PALATE,
    N_DENSE_PHARYNX,
    N_DENSE_VELUM,
    N_DENSE_LIPS,
    TB_PALATE_START,
    TR_TONGUE_END,
)

OUTPUT_SUBDIR     = "extracted_TVs"

# ── Per-contour colours ────────────────────────────────────────────────────
CONTOUR_STYLE = {
    "hard_palate":     {"color": "#4e79a7", "lw": 2.0, "zorder": 3},
    "velum":           {"color": "#9467bd", "lw": 2.0, "zorder": 3},
    "tongue":          {"color": "#e15759", "lw": 2.0, "zorder": 3},
    "pharyngeal_wall": {"color": "#59a14f", "lw": 2.0, "zorder": 3},
    "upper_lip":       {"color": "#f28e2b", "lw": 2.0, "zorder": 3},
    "lower_lip":       {"color": "#c49c94", "lw": 2.0, "zorder": 3},
    "chin":            {"color": "#bab0ac", "lw": 1.5, "zorder": 2},
}
 
TV_COLOR = {
    "TTCD": "#e15759",
    "TBCD": "#4e79a7",
    "TRCD": "#59a14f",
    "LA":   "#f28e2b",
    "LP":   "#9467bd",
    "VEL":  "#17becf",
}
 
 
# ── Internal drawing helpers ───────────────────────────────────────────────
 
def _draw_cd_line(ax, pt_a, pt_b, color, label=None):
    if np.any(np.isnan(pt_a)) or np.any(np.isnan(pt_b)):
        return
    ax.plot(
        [pt_a[0], pt_b[0]], [pt_a[1], pt_b[1]],
        color=color, lw=1.8, ls="--", zorder=6, alpha=0.9,
    )
    if label:
        mid = (pt_a + pt_b) / 2.0
        ax.annotate(
            label, xy=mid, fontsize=6.5, color=color,
            ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=color, alpha=0.8, lw=0.8),
            zorder=7,
        )
 
 
def _draw_cl_marker(ax, pt, color, label=None):
    if np.any(np.isnan(pt)):
        return
    ax.scatter(pt[0], pt[1], color=color, s=65,
               zorder=8, edgecolors="white", linewidths=0.9)
    if label:
        ax.annotate(
            label, xy=pt, xytext=(0, 7), textcoords="offset points",
            fontsize=6, color=color, ha="center", zorder=9,
        )
 
 
# ── Geometry derivation from saved TV values ───────────────────────────────
 
def _geom_from_tvs(frame_data: dict, tv_row: pd.Series) -> dict:
    """
    Derive 2D visualisation coordinates from contour data (JSON) and
    saved scalar TV values (CSV row).
 
    Returns a dict of 2D coordinates used only for drawing.
    """
    geom = {k: np.full(2, np.nan) for k in [
        "_tt_palate_pt", "_tt_tongue_pt",
        "_tb_ref_pt",    "_tb_palate_pt", "_tb_tongue_pt",
        "_tr_pharynx_pt","_tr_tongue_pt",
        "_vel_velum_pt", "_vel_pharynx_pt",
        "_la_ul_pt",     "_la_ll_pt",
    ]}
    geom["_lp_lip_x"] = np.nan
    geom["_lp_lip_y"] = np.nan
 
    def get(name):
        raw = frame_data.get(name)
        return np.asarray(raw, dtype=float) if raw is not None else None
 
    palate  = get("hard_palate")
    velum   = get("velum")
    tongue  = get("tongue")
    pharynx = get("pharyngeal_wall")
    ul      = get("upper_lip")
    ll      = get("lower_lip")
 
    if palate is not None:
        pd_pts, pd_t = build_dense_spline(palate, N_DENSE_PALATE)
    else:
        return geom
 
    # ── TTCL → palate point; closest tongue point → tongue endpoint ────────
    ttcl = tv_row.get("TTCL", np.nan) if hasattr(tv_row, "get") else tv_row["TTCL"]
    if tongue is not None and not np.isnan(ttcl):
        # Find palate point at t = TTCL
        tt_pal_idx = int(np.argmin(np.abs(pd_t - ttcl)))
        tt_pal_pt  = pd_pts[tt_pal_idx]
        # Closest tongue point to that palate point
        dists      = np.hypot(tongue[:, 0] - tt_pal_pt[0],
                              tongue[:, 1] - tt_pal_pt[1])
        tt_tng_pt  = tongue[int(np.argmin(dists))]
        geom["_tt_palate_pt"] = tt_pal_pt
        geom["_tt_tongue_pt"] = tt_tng_pt
 
    # ── TBCL → palate CL marker; TB tongue + ref point → TBCD line ─────────
    tbcl = tv_row.get("TBCL", np.nan) if hasattr(tv_row, "get") else tv_row["TBCL"]
    if tongue is not None and not np.isnan(tbcl):
        # CL marker: palate point at t = TBCL
        tb_pal_idx = int(np.argmin(np.abs(pd_t - tbcl)))
        geom["_tb_palate_pt"] = pd_pts[tb_pal_idx]
 
        # TBCD line: closest tongue point to posterior palate only (velum ignored)
        post_mask   = pd_t >= TB_PALATE_START
        post_palate = pd_pts[post_mask]

        _, hp_idx, t_idx_hp = min_dist_two_curves(post_palate, tongue)

        geom["_tb_tongue_pt"] = tongue[t_idx_hp]
        geom["_tb_ref_pt"]    = post_palate[hp_idx]
 
    # ── TRCD: closest tongue-pharynx pair ──────────────────────────────────
    if tongue is not None and pharynx is not None:
        ph_dense = build_dense_spline(pharynx, N_DENSE_PHARYNX)[0]
        # Restrict tongue to root-side arc-length (same logic as extract_TRCD)
        diffs    = np.diff(tongue, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        arc      = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total    = arc[-1]
        if total > 1e-9:
            root_mask   = (arc / total) <= TR_TONGUE_END
            tongue_root = tongue[root_mask]
        else:
            tongue_root = tongue
        if len(tongue_root) > 0:
            _, ph_idx, t_idx = min_dist_two_curves(ph_dense, tongue_root)
            geom["_tr_pharynx_pt"] = ph_dense[ph_idx]
            geom["_tr_tongue_pt"]  = tongue_root[t_idx]

    # ── VEL: closest velum-pharynx pair ────────────────────────────────────
    if velum is not None and pharynx is not None:
        vel_dense_vel = build_dense_spline(velum,   N_DENSE_VELUM)[0]
        ph_dense_vel  = build_dense_spline(pharynx, N_DENSE_PHARYNX)[0]
        _, v_idx, p_idx = min_dist_two_curves(vel_dense_vel, ph_dense_vel)
        geom["_vel_velum_pt"]   = vel_dense_vel[v_idx]
        geom["_vel_pharynx_pt"] = ph_dense_vel[p_idx]
 
    # ── LA: closest lip pair ───────────────────────────────────────────────
    if ul is not None and ll is not None:
        ul_dense = build_dense_spline(ul, N_DENSE_LIPS)[0]
        ll_dense = build_dense_spline(ll, N_DENSE_LIPS)[0]
        _, ul_idx, ll_idx = min_dist_two_curves(ul_dense, ll_dense)
        geom["_la_ul_pt"] = ul_dense[ul_idx]
        geom["_la_ll_pt"] = ll_dense[ll_idx]
 
    # ── LP: anterior lip midpoint ──────────────────────────────────────────
    if ul is not None and ll is not None:
        ul_ant = ul[int(np.argmin(ul[:, 0]))]
        ll_ant = ll[int(np.argmin(ll[:, 0]))]
        geom["_lp_lip_x"] = float((ul_ant[0] + ll_ant[0]) / 2.0)
        geom["_lp_lip_y"] = float((ul_ant[1] + ll_ant[1]) / 2.0)
 
    return geom
 
 
# ── Single-frame plot ──────────────────────────────────────────────────────
 
def _plot_single_frame(
    ax,
    frame_data: dict,
    tv_row: pd.Series,
    frame_id: str | int,
) -> None:
    """
    Plot one frame.  tv_row is the corresponding CSV row (pd.Series).
    Scalar values come from tv_row; geometry is derived from frame_data.
    """
    # ── Contours ───────────────────────────────────────────────────────────
    for name, coords in frame_data.items():
        pts   = np.asarray(coords, dtype=float)
        style = CONTOUR_STYLE.get(name, {"color": "gray", "lw": 1.5, "zorder": 2})
        ax.plot(pts[:, 0], pts[:, 1], "-o",
                color=style["color"], lw=style["lw"],
                ms=2.5, zorder=style["zorder"], alpha=0.85, label=name)
 
    # ── Derive geometry ────────────────────────────────────────────────────
    geom = _geom_from_tvs(frame_data, tv_row)
 
    # ── TTCD / TTCL ────────────────────────────────────────────────────────
    ttcd = tv_row["TTCD"]
    if not np.isnan(ttcd):
        _draw_cd_line(ax,
                      geom["_tt_palate_pt"], geom["_tt_tongue_pt"],
                      TV_COLOR["TTCD"],
                      label=f"TTCD {ttcd:.1f}px")
        _draw_cl_marker(ax, geom["_tt_palate_pt"], TV_COLOR["TTCD"],
                        label=f"TTCL={tv_row['TTCL']:.2f}")
 
    # ── TBCD / TBCL ────────────────────────────────────────────────────────
    tbcd = tv_row["TBCD"]
    if not np.isnan(tbcd):
        _draw_cd_line(ax,
                      geom["_tb_ref_pt"], geom["_tb_tongue_pt"],
                      TV_COLOR["TBCD"],
                      label=f"TBCD {tbcd:.1f}px")
        _draw_cl_marker(ax, geom["_tb_palate_pt"], TV_COLOR["TBCD"],
                        label=f"TBCL={tv_row['TBCL']:.2f}")
 
    # ── TRCD ───────────────────────────────────────────────────────────────
    trcd = tv_row["TRCD"]
    if not np.isnan(trcd):
        _draw_cd_line(ax,
                      geom["_tr_pharynx_pt"], geom["_tr_tongue_pt"],
                      TV_COLOR["TRCD"],
                      label=f"TRCD {trcd:.1f}px")

    # ── VEL ────────────────────────────────────────────────────────────────
    vel = tv_row["VEL"]
    if not np.isnan(vel):
        _draw_cd_line(ax,
                      geom["_vel_velum_pt"], geom["_vel_pharynx_pt"],
                      TV_COLOR["VEL"],
                      label=f"VEL {vel:.1f}px")
    
    # ── LA ─────────────────────────────────────────────────────────────────
    la = tv_row["LA"]
    if not np.isnan(la):
        _draw_cd_line(ax,
                      geom["_la_ul_pt"], geom["_la_ll_pt"],
                      TV_COLOR["LA"],
                      label=f"LA {la:.1f}px")
 
    # ── LP ─────────────────────────────────────────────────────────────────
    lp   = tv_row["LP"]
    lx   = geom["_lp_lip_x"]
    ly   = geom["_lp_lip_y"]
    if not np.isnan(lp) and not np.isnan(lx):
        palate_t0_x = lx + lp          # LP = palate_t0_x - lip_x  →  palate_t0_x = lx + lp
        ax.plot([lx, palate_t0_x], [ly, ly],
                color=TV_COLOR["LP"], lw=1.8, ls=":", zorder=6, alpha=0.9)
        ax.annotate(
            f"LP {lp:.1f}px",
            xy=((lx + palate_t0_x) / 2.0, ly),
            xytext=(0, 5), textcoords="offset points",
            fontsize=6.5, color=TV_COLOR["LP"],
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=TV_COLOR["LP"], alpha=0.8, lw=0.8),
            zorder=7,
        )
 
    # ── Title ─────────────────────────────────────────────────────────────
    flags = []
    if tv_row["LA_closed"]:   flags.append("LA")
    if tv_row["TTCD_closed"]: flags.append("TT")
    if tv_row["TBCD_closed"]: flags.append("TB")
    closed_str = f"\n[closed: {', '.join(flags)}]" if flags else ""
    ax.set_title(f"Frame {frame_id}{closed_str}", fontsize=9)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
 
 
# ── Public API ─────────────────────────────────────────────────────────────
 
def _csv_path_from_json(json_path: Path) -> Path:
    """
    Derive CSV path from JSON path.
    subXXX/track_json/task.json  →  subXXX/extracted_TVs/task.csv
    """
    return json_path.parent.parent / OUTPUT_SUBDIR / (json_path.stem + ".csv")
 
 
def plot_tv_overview(
    json_path: str | Path,
    csv_path: str | Path | None = None,
    n_frames: int = 5,
    seed: int = 42,
    frame_ids: list | None = None,
) -> None:
    """
    Plot TV overlays for n_frames frames from a task, reading scalar TV
    values from the saved CSV and geometry from the JSON.
 
    Parameters
    ----------
    json_path  : path to task JSON (contours)
    csv_path   : path to extracted TV CSV. If None, derived automatically
                 from json_path (subXXX/extracted_TVs/task.csv).
    n_frames   : number of random frames to show (default 5)
    seed       : random seed
    frame_ids  : if given, plot exactly these frames (overrides n_frames/seed)
 
    Usage
    -----
    plot_tv_overview("/data/75annot16/sub009/track_json/task.json")
    plot_tv_overview("/data/75annot16/sub009/track_json/task.json",
                     frame_ids=[150, 320, 870])
    """
    json_path = Path(json_path)
    csv_path  = Path(csv_path) if csv_path else _csv_path_from_json(json_path)
 
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run process_task() or process_all_speakers() first."
        )
 
    # ── Load data ──────────────────────────────────────────────────────────
    with open(json_path) as f:
        json_data = json.load(f)
 
    csv_df = pd.read_csv(csv_path)
    csv_df = csv_df.set_index("frame_id")   # index by int frame ID
 
    # ── Choose frames ──────────────────────────────────────────────────────
    # Only frames present in BOTH json and csv
    available = sorted(
        [k for k in json_data if int(k) in csv_df.index],
        key=lambda k: int(k),
    )
    if not available:
        raise ValueError("No frames found in both JSON and CSV.")
 
    if frame_ids is not None:
        chosen = [str(f) for f in frame_ids if str(f) in json_data
                  and int(f) in csv_df.index]
        if not chosen:
            raise ValueError("None of the specified frame_ids are in both JSON and CSV.")
    else:
        rng    = np.random.default_rng(seed)
        chosen = sorted(
            rng.choice(available, size=min(n_frames, len(available)), replace=False),
            key=lambda k: int(k),
        )
 
    # ── Plot ───────────────────────────────────────────────────────────────
    n    = len(chosen)
    # fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), squeeze=False)
    fig, axes = plt.subplots(n, 1, figsize=(11, 9 * n), squeeze=False)
    # axes = axes[0]
    axes = axes[:,0]
 
    for ax, fid in zip(axes, chosen):
        tv_row = csv_df.loc[int(fid)]
        _plot_single_frame(ax, json_data[fid], tv_row, fid)
 
    # ── Contour legend (first axis) ────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    seen   = {}
    unique = [(h, l) for h, l in zip(handles, labels)
              if l not in seen and not seen.update({l: True})]
    axes[0].legend([h for h, _ in unique], [l for _, l in unique],
                   fontsize=6.5, loc="lower left", framealpha=0.85)
 
    # ── TV colour legend (last axis) ───────────────────────────────────────
    import matplotlib.patches as mpatches
    tv_patches = [mpatches.Patch(color=c, label=name)
                  for name, c in TV_COLOR.items()]
    axes[-1].legend(handles=tv_patches, fontsize=6.5, loc="lower right",
                    framealpha=0.85, title="TVs", title_fontsize=7)
 
    fig.suptitle(
        f"{json_path.name}   |   CSV: {csv_path.name}   |   {n} frame(s)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.show()
 
 
# =============================================================================
# Usage
# =============================================================================
 
plot_tv_overview("/home/data/75SpeakerAnnot16/sub074/track_json/sub074_2drt_21_topic5_track.json")
 
# Specific frames:
# plot_tv_overview("/data/75annot16/sub009/track_json/task.json",
#                  frame_ids=[150, 500, 1000, 1500, 2000])
 
# Explicit CSV path (if directory layout differs):
# plot_tv_overview("/data/75annot16/sub009/track_json/task.json",
#                  csv_path="/data/75annot16/sub009/extracted_TVs/task.csv")
