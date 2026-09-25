import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.interpolate import interp1d

# Global Configuration

# Hard palate t-parameter upper bound for TTCD reference region
TT_PALATE_END   = 0.45
# Hard palate t-parameter lower bound for TBCD reference region
TB_PALATE_START = 0.60
# Tongue arc-length upper bound for TRCD reference region (root-side portion)
TR_TONGUE_END   = 0.30
 
# Pixels; CD below this flags the frame as a closure
CLOSURE_THRESHOLD = 1.0
 
# Resampling densities (more points = higher accuracy, modest cost)
N_DENSE_PALATE  = 200
N_DENSE_VELUM   = 150
N_DENSE_PHARYNX = 150
N_DENSE_LIPS    = 100
 
# Filesystem layout
TRACK_JSON_SUBDIR = "track_json"
OUTPUT_SUBDIR     = "extracted_TVs"
# Common Utility Functions

def build_dense_spline(points: np.ndarray, n_dense: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a contour to n_dense arc-length-uniform points.
 
    Parameters
    ----------
    points  : (N, 2) array of (x, y) coordinates
    n_dense : number of output points
 
    Returns
    -------
    dense_pts : (n_dense, 2)  resampled coordinates
    t_vals    : (n_dense,)    arc-length parameter in [0, 1]
    """
    points = np.asarray(points, dtype=float)
 
    if len(points) < 2:
        return np.tile(points[0], (n_dense, 1)), np.linspace(0.0, 1.0, n_dense)
 
    diffs    = np.diff(points, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    arc      = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total    = arc[-1]
 
    if total < 1e-9:
        return np.tile(points[0], (n_dense, 1)), np.linspace(0.0, 1.0, n_dense)
 
    arc_norm = arc / total
 
    # Remove duplicate arc-length values (coincident points)
    _, keep  = np.unique(arc_norm, return_index=True)
    arc_norm = arc_norm[keep]
    points   = points[keep]
 
    t_query  = np.linspace(0.0, 1.0, n_dense)
    fx = interp1d(arc_norm, points[:, 0], kind="linear")
    fy = interp1d(arc_norm, points[:, 1], kind="linear")
 
    return np.column_stack([fx(t_query), fy(t_query)]), t_query
 
 
def min_dist_two_curves(curve_a: np.ndarray, curve_b: np.ndarray) -> tuple[float, int, int]:
    """
    Find the minimum Euclidean distance between two dense curves.
    Fully vectorised via numpy broadcasting — O(n_a × n_b).
 
    Parameters
    ----------
    curve_a : (n_a, 2)
    curve_b : (n_b, 2)
 
    Returns
    -------
    min_dist : float   — minimum distance
    idx_a    : int     — index in curve_a achieving the minimum
    idx_b    : int     — index in curve_b achieving the minimum
    """
    if len(curve_a) == 0 or len(curve_b) == 0:
        return np.inf, -1, -1
 
    diff = curve_a[:, np.newaxis, :] - curve_b[np.newaxis, :, :]  # (n_a, n_b, 2)
    dist = np.hypot(diff[:, :, 0], diff[:, :, 1])                  # (n_a, n_b)
    flat_idx = np.argmin(dist)
    idx_a, idx_b = np.unravel_index(flat_idx, dist.shape)
 
    return float(dist[idx_a, idx_b]), int(idx_a), int(idx_b)

# Individual TV Extraction Functions

# Each function returns its TV value(s) plus visualisation info (prefixed _). 
# Visualisation keys are plain numpy coordinates, not indices into dense arrays, 
# so the visualisation cell can draw lines without rebuilding splines.

def extract_LA(
    upper_lip: np.ndarray,
    lower_lip: np.ndarray,
    closure_threshold: float = CLOSURE_THRESHOLD,
    n_dense: int = N_DENSE_LIPS,
) -> dict:
    """
    Lip Aperture: minimum Euclidean distance between upper and lower lip.
 
    Returns
    -------
    dict with keys:
        LA         : float  — distance in pixels
        LA_closed  : int    — 1 if LA < closure_threshold
        _la_ul_pt  : (2,)   — upper lip point for visualisation
        _la_ll_pt  : (2,)   — lower lip point for visualisation
    """
    ul_dense, _ = build_dense_spline(upper_lip, n_dense)
    ll_dense, _ = build_dense_spline(lower_lip, n_dense)
 
    dist, ul_idx, ll_idx = min_dist_two_curves(ul_dense, ll_dense)
 
    return {
        "LA":        float(dist),
        "LA_closed": int(dist < closure_threshold),
        "_la_ul_pt": ul_dense[ul_idx],
        "_la_ll_pt": ll_dense[ll_idx],
    }

def extract_LP(
    upper_lip: np.ndarray,
    lower_lip: np.ndarray,
    palate_t0_x: float,
) -> dict:
    """
    Lip Protrusion: signed horizontal displacement of the lip midpoint
    relative to the hard palate anterior reference (t = 0).
 
    lip_midpoint_x  = mean of (min-x of upper lip, min-x of lower lip)
    LP              = palate_t0_x - lip_midpoint_x
    Positive        = lips anterior to the alveolar reference (protruded).
 
    Returns
    -------
    dict with keys:
        LP          : float  — displacement in pixels
        _lp_lip_x   : float  — x-coordinate of lip midpoint (for visualisation)
        _lp_lip_y   : float  — y-coordinate of lip midpoint (midpoint between
                               the two anterior points, for drawing the LP line)
    """
    ul = np.asarray(upper_lip, dtype=float)
    ll = np.asarray(lower_lip, dtype=float)
 
    ul_ant_idx = int(np.argmin(ul[:, 0]))
    ll_ant_idx = int(np.argmin(ll[:, 0]))
 
    ul_ant = ul[ul_ant_idx]
    ll_ant = ll[ll_ant_idx]
 
    lip_midpoint_x = float((ul_ant[0] + ll_ant[0]) / 2.0)
    lip_midpoint_y = float((ul_ant[1] + ll_ant[1]) / 2.0)
 
    return {
        "LP":       float(palate_t0_x - lip_midpoint_x),
        "_lp_lip_x": lip_midpoint_x,
        "_lp_lip_y": lip_midpoint_y,
    }

def extract_TTCD_TTCL(
    tongue: np.ndarray,
    palate_dense: np.ndarray,
    palate_t_vals: np.ndarray,
    tt_palate_end: float     = TT_PALATE_END,
    closure_threshold: float = CLOSURE_THRESHOLD,
) -> dict:
    """
    Tongue Tip Constriction Degree and Location.
 
    Reference region: anterior hard palate t ∈ [0.00, tt_palate_end].
    Strategy: start from the anterior palate, find the closest tongue point.
 
    TTCD = min distance from the anterior palate to any tongue point.
    TTCL = t-parameter of the palate point achieving TTCD.
 
    Returns
    -------
    dict with keys:
        TTCD           : float
        TTCL           : float  in [0, tt_tb_boundary]
        TTCD_closed    : int
        _tt_palate_pt  : (2,)   palate point (= CL location, = line endpoint)
        _tt_tongue_pt  : (2,)   tongue point (= line endpoint)
    """
    tongue = np.asarray(tongue, dtype=float)
 
    ant_mask   = palate_t_vals < tt_palate_end
    ant_palate = palate_dense[ant_mask]
    ant_t      = palate_t_vals[ant_mask]
 
    if len(ant_palate) == 0:
        return {
            "TTCD": np.nan, "TTCL": np.nan, "TTCD_closed": 0,
            "_tt_palate_pt": np.array([np.nan, np.nan]),
            "_tt_tongue_pt": np.array([np.nan, np.nan]),
        }
 
    dist, p_idx, t_idx = min_dist_two_curves(ant_palate, tongue)
 
    return {
        "TTCD":          float(dist),
        "TTCL":          float(ant_t[p_idx]),
        "TTCD_closed":   int(dist < closure_threshold),
        "_tt_palate_pt": ant_palate[p_idx].copy(),
        "_tt_tongue_pt": tongue[t_idx].copy(),
    }

def extract_TBCD_TBCL(
    tongue: np.ndarray,
    palate_dense: np.ndarray,
    palate_t_vals: np.ndarray,
    velum_dense: np.ndarray | None,
    tb_palate_start: float   = TB_PALATE_START,
    closure_threshold: float = CLOSURE_THRESHOLD,
) -> dict:
    """
    Tongue Body Constriction Degree and Location.
 
    Reference for CD  : posterior hard palate t ∈ [tb_palate_start, 1.0] UNION velum.
                        Velum is included so velar constrictions (/k/, /g/) are
                        correctly captured.
    Reference for CL  : full hard palate only (t-parameter).
                        Even if TBCD is achieved via the velum, TBCL reports
                        the t-parameter of the closest hard palate point to the
                        tongue point that achieved TBCD. This keeps CL on the
                        canonical hard-palate scale (Browman & Goldstein, 1992).
 
    Strategy: start from the reference surface, find closest tongue point.
 
    Returns
    -------
    dict with keys:
        TBCD           : float
        TBCL           : float  in [0, 1]  (hard palate t-parameter)
        TBCD_closed    : int
        _tb_ref_pt     : (2,)   actual closest reference surface point (for TBCD line)
        _tb_palate_pt  : (2,)   hard palate point at TBCL (for CL marker)
        _tb_tongue_pt  : (2,)   tongue point achieving TBCD (shared by line + CL)
    """
    tongue = np.asarray(tongue, dtype=float)
 
    # ── Posterior hard palate ──────────────────────────────────────────────
    post_mask   = palate_t_vals >= tb_palate_start
    post_palate = palate_dense[post_mask]
 
    dist_hp, hp_idx, t_idx_hp = min_dist_two_curves(post_palate, tongue)
 
    # ── Velum (optional) ──────────────────────────────────────────────────
    # if velum_dense is not None and len(velum_dense) > 0:
    #     dist_vel, vel_idx, t_idx_vel = min_dist_two_curves(velum_dense, tongue)
    # else:
    dist_vel, vel_idx, t_idx_vel = np.inf, -1, -1
 
    # ── TBCD = minimum across both surfaces ───────────────────────────────
    if dist_hp <= dist_vel:
        TBCD      = dist_hp
        tongue_pt = tongue[t_idx_hp].copy()
        ref_pt    = post_palate[hp_idx].copy()
    else:
        TBCD      = dist_vel
        tongue_pt = tongue[t_idx_vel].copy()
        ref_pt    = velum_dense[vel_idx].copy()
 
    # ── TBCL: closest full hard palate point to the TB tongue point ────────
    hp_dists   = np.hypot(
        palate_dense[:, 0] - tongue_pt[0],
        palate_dense[:, 1] - tongue_pt[1],
    )
    cl_idx     = int(np.argmin(hp_dists))
    TBCL       = float(palate_t_vals[cl_idx])
    palate_pt  = palate_dense[cl_idx].copy()
 
    return {
        "TBCD":          float(TBCD),
        "TBCL":          TBCL,
        "TBCD_closed":   int(TBCD < closure_threshold),
        "_tb_ref_pt":    ref_pt,
        "_tb_palate_pt": palate_pt,
        "_tb_tongue_pt": tongue_pt,
    }

def extract_TRCD(
    tongue: np.ndarray,
    pharynx_dense: np.ndarray,
    tr_tongue_end: float = TR_TONGUE_END,
) -> dict:
    """
    Tongue Root Constriction Degree.
 
    TRCD = minimum distance from the tongue root portion to the pharyngeal wall.
    Only the root-side of the tongue (arc-length t ∈ [0.00, tr_tongue_end]) is
    considered, preventing tongue body points from being selected.
    No CL for tongue root.
 
    Returns
    -------
    dict with keys:
        TRCD            : float
        _tr_pharynx_pt  : (2,)  pharyngeal wall point (for line endpoint)
        _tr_tongue_pt   : (2,)  tongue point (for line endpoint)
    """
    tongue = np.asarray(tongue, dtype=float)
 
    # ── Restrict tongue to root-side arc-length portion ────────────────────
    if len(tongue) >= 2:
        diffs    = np.diff(tongue, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        arc      = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total    = arc[-1]
        if total > 1e-9:
            arc_norm   = arc / total
            root_mask  = arc_norm <= tr_tongue_end
            tongue_root = tongue[root_mask]
        else:
            tongue_root = tongue
    else:
        tongue_root = tongue
 
    if len(tongue_root) == 0:
        return {
            "TRCD":           np.nan,
            "_tr_pharynx_pt": np.full(2, np.nan),
            "_tr_tongue_pt":  np.full(2, np.nan),
        }
 
    dist, ph_idx, t_idx = min_dist_two_curves(pharynx_dense, tongue_root)
 
    return {
        "TRCD":           float(dist),
        "_tr_pharynx_pt": pharynx_dense[ph_idx].copy(),
        "_tr_tongue_pt":  tongue_root[t_idx].copy(),
    }

def extract_VEL(
    velum_dense: np.ndarray,
    pharynx_dense: np.ndarray,
) -> dict:
    """
    Velum (velopharyngeal) aperture: minimum distance between the velum
    and the pharyngeal wall.
 
    Small VEL → velopharyngeal port closed (oral sounds, velum raised).
    Large VEL → velopharyngeal port open (nasal sounds, velum lowered).
 
    Both contours are passed in as pre-built dense arrays (already computed
    in extract_frame_tvs for other TVs, so no redundant resampling).
 
    Returns
    -------
    dict with keys:
        VEL              : float
        _vel_velum_pt    : (2,)  velum point (for line endpoint)
        _vel_pharynx_pt  : (2,)  pharyngeal wall point (for line endpoint)
    """
    dist, vel_idx, ph_idx = min_dist_two_curves(velum_dense, pharynx_dense)
 
    return {
        "VEL":             float(dist),
        "_vel_velum_pt":   velum_dense[vel_idx].copy(),
        "_vel_pharynx_pt": pharynx_dense[ph_idx].copy(),
    }

# Frame-Level, Task-Level, Speaker-Level, and Full Pipeline

def extract_frame_tvs(
    frame_data: dict,
    tt_palate_end: float     = TT_PALATE_END,
    tb_palate_start: float   = TB_PALATE_START,
    tr_tongue_end: float     = TR_TONGUE_END,
    closure_threshold: float = CLOSURE_THRESHOLD,
    n_dense_palate: int      = N_DENSE_PALATE,
    n_dense_velum: int       = N_DENSE_VELUM,
    n_dense_pharynx: int     = N_DENSE_PHARYNX,
) -> dict:
    """
    Extract all TVs from one frame's articulatory contours.
 
    Missing contours (not in frame_data) result in NaN for affected TVs.
    Visualisation coordinates are stored under keys prefixed with '_'.
    These are NOT written to the CSV.
 
    Parameters
    ----------
    frame_data       : dict {articulator_name: [[x,y], ...]}
    tt_tb_boundary   : hard palate t-parameter boundary for TT/TB split
    closure_threshold: pixel threshold for closure flags
    n_dense_*        : resampling densities
 
    Returns
    -------
    dict containing all TV values and visualisation coordinates
    """
 
    # ── Helpers ────────────────────────────────────────────────────────────
    def get(name):
        raw = frame_data.get(name)
        return np.asarray(raw, dtype=float) if raw is not None else None
 
    def nan_tv_block():
        return {
            "LA": np.nan, "LA_closed": 0,
            "LP": np.nan,
            "TTCD": np.nan, "TTCL": np.nan,  "TTCD_closed": 0,
            "TBCD": np.nan, "TBCL": np.nan,  "TBCD_closed": 0,
            "TRCD": np.nan,
            "VEL":  np.nan,
            "_la_ul_pt":    np.full(2, np.nan),
            "_la_ll_pt":    np.full(2, np.nan),
            "_lp_lip_x":    np.nan,
            "_lp_lip_y":    np.nan,
            "_tt_palate_pt": np.full(2, np.nan),
            "_tt_tongue_pt": np.full(2, np.nan),
            "_tb_ref_pt":    np.full(2, np.nan),
            "_tb_palate_pt": np.full(2, np.nan),
            "_tb_tongue_pt": np.full(2, np.nan),
            "_tr_pharynx_pt": np.full(2, np.nan),
            "_tr_tongue_pt":  np.full(2, np.nan),
            "_vel_velum_pt":   np.full(2, np.nan),
            "_vel_pharynx_pt": np.full(2, np.nan),
        }
 
    result = nan_tv_block()
 
    palate  = get("hard_palate")
    velum   = get("velum")
    tongue  = get("tongue")
    pharynx = get("pharyngeal_wall")
    ul      = get("upper_lip")
    ll      = get("lower_lip")
 
    # ── Build dense splines (only if contour exists) ───────────────────────
    if palate is not None:
        pd_pts, pd_t  = build_dense_spline(palate,  n_dense_palate)
        palate_t0_x   = float(pd_pts[0, 0])
    else:
        pd_pts = pd_t = palate_t0_x = None
 
    vel_dense = build_dense_spline(velum,  n_dense_velum)[0]   if velum   is not None else None
    ph_dense  = build_dense_spline(pharynx, n_dense_pharynx)[0] if pharynx is not None else None
 
    # ── LA ─────────────────────────────────────────────────────────────────
    if ul is not None and ll is not None:
        result.update(extract_LA(ul, ll, closure_threshold=closure_threshold))
 
    # ── LP ─────────────────────────────────────────────────────────────────
    if ul is not None and ll is not None and palate_t0_x is not None:
        result.update(extract_LP(ul, ll, palate_t0_x))
 
    # ── TTCD / TTCL ────────────────────────────────────────────────────────
    if tongue is not None and pd_pts is not None:
        result.update(extract_TTCD_TTCL(
            tongue, pd_pts, pd_t,
            tt_palate_end=tt_palate_end,
            closure_threshold=closure_threshold,
        ))
 
    # ── TBCD / TBCL ────────────────────────────────────────────────────────
    if tongue is not None and pd_pts is not None:
        result.update(extract_TBCD_TBCL(
            tongue, pd_pts, pd_t,
            velum_dense=vel_dense,
            tb_palate_start=tb_palate_start,
            closure_threshold=closure_threshold,
        ))
 
    # ── TRCD ───────────────────────────────────────────────────────────────
    if tongue is not None and ph_dense is not None:
        result.update(extract_TRCD(tongue, ph_dense, tr_tongue_end=tr_tongue_end))

    # ── VEL ────────────────────────────────────────────────────────────────
    if vel_dense is not None and ph_dense is not None:
        result.update(extract_VEL(vel_dense, ph_dense))
 
    return result
 
 
# CSV columns (in output order) — no visualisation keys
_CSV_COLUMNS = [
    "frame_id",
    "LA",  "LA_closed",
    "LP",
    "TTCD", "TTCL", "TTCD_closed", "TTCL_last_valid",
    "TBCD", "TBCL", "TBCD_closed", "TBCL_last_valid",
    "TRCD",
    "VEL",
]
 
 
def process_task(
    json_path: str | Path,
    output_dir: str | Path,
    tt_palate_end: float     = TT_PALATE_END,
    tb_palate_start: float   = TB_PALATE_START,
    tr_tongue_end: float     = TR_TONGUE_END,
    closure_threshold: float = CLOSURE_THRESHOLD,
    n_dense_palate: int      = N_DENSE_PALATE,
    n_dense_velum: int       = N_DENSE_VELUM,
    n_dense_pharynx: int     = N_DENSE_PHARYNX,
) -> pd.DataFrame:
    """
    Extract all TVs for every frame in a task JSON and save as a CSV.
 
    - Frames are processed in ascending numerical order for correct
      temporal tracking of last_valid_TTCL / last_valid_TBCL.
    - Output CSV has the same stem as the input JSON.
 
    Parameters
    ----------
    json_path    : path to task JSON  (e.g. sub009/track_json/task.json)
    output_dir   : directory to write the CSV into
    ...          : TV parameters (defaults from Cell 1)
 
    Returns
    -------
    pd.DataFrame of all TV rows (same as what is saved to CSV)
    """
    json_path  = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv    = output_dir / (json_path.stem + ".csv")
 
    with open(json_path) as f:
        frames = json.load(f)
 
    # Sort by numeric frame ID for temporal ordering
    sorted_ids = sorted(frames.keys(), key=lambda k: int(k))
 
    rows = []
    last_valid_ttcl = np.nan
    last_valid_tbcl = np.nan
 
    for frame_id in sorted_ids:
        tvs = extract_frame_tvs(
            frames[frame_id],
            tt_palate_end=tt_palate_end,
            tb_palate_start=tb_palate_start,
            tr_tongue_end=tr_tongue_end,
            closure_threshold=closure_threshold,
            n_dense_palate=n_dense_palate,
            n_dense_velum=n_dense_velum,
            n_dense_pharynx=n_dense_pharynx,
        )
 
        # Update last-valid CL only when not in closure and value is real
        if tvs["TTCD_closed"] == 0 and not np.isnan(tvs["TTCL"]):
            last_valid_ttcl = tvs["TTCL"]
 
        if tvs["TBCD_closed"] == 0 and not np.isnan(tvs["TBCL"]):
            last_valid_tbcl = tvs["TBCL"]
 
        rows.append({
            "frame_id":        int(frame_id),
            "LA":              tvs["LA"],
            "LA_closed":       tvs["LA_closed"],
            "LP":              tvs["LP"],
            "TTCD":            tvs["TTCD"],
            "TTCL":            tvs["TTCL"],
            "TTCD_closed":     tvs["TTCD_closed"],
            "TTCL_last_valid": last_valid_ttcl,
            "TBCD":            tvs["TBCD"],
            "TBCL":            tvs["TBCL"],
            "TBCD_closed":     tvs["TBCD_closed"],
            "TBCL_last_valid": last_valid_tbcl,
            "TRCD":            tvs["TRCD"],
            "VEL":             tvs["VEL"],
        })
 
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"  [{json_path.parent.parent.name}]  {json_path.stem}  →  "
          f"{len(df)} frames  →  {out_csv.name}")
    return df
 
 
def process_speaker(
    speaker_dir: str | Path,
    track_json_subdir: str   = TRACK_JSON_SUBDIR,
    output_subdir: str       = OUTPUT_SUBDIR,
    tt_palate_end: float     = TT_PALATE_END,
    tb_palate_start: float   = TB_PALATE_START,
    tr_tongue_end: float     = TR_TONGUE_END,
    closure_threshold: float = CLOSURE_THRESHOLD,
) -> None:
    """
    Process all task JSONs under speaker_dir/track_json_subdir/.
    Saves one CSV per task to speaker_dir/output_subdir/.
 
    Expected input  : speaker_dir / track_json_subdir / *.json
    Output          : speaker_dir / output_subdir     / *.csv
    """
    speaker_dir = Path(speaker_dir)
    json_dir    = speaker_dir / track_json_subdir
    output_dir  = speaker_dir / output_subdir
 
    task_jsons  = sorted(json_dir.glob("*.json"))
 
    if not task_jsons:
        print(f"  [WARNING] No JSON files in {json_dir} — skipping.")
        return
 
    print(f"\n{'='*60}")
    print(f"  Speaker : {speaker_dir.name}   ({len(task_jsons)} tasks)")
    print(f"{'='*60}")
 
    for task_json in task_jsons:
        process_task(
            json_path=task_json,
            output_dir=output_dir,
            tt_palate_end=tt_palate_end,
            tb_palate_start=tb_palate_start,
            tr_tongue_end=tr_tongue_end,
            closure_threshold=closure_threshold,
        )
 
 
# def process_all_speakers(
#     data_root: str | Path,
#     track_json_subdir: str   = TRACK_JSON_SUBDIR,
#     output_subdir: str       = OUTPUT_SUBDIR,
#     tt_palate_end: float     = TT_PALATE_END,
#     tb_palate_start: float   = TB_PALATE_START,
#     tr_tongue_end: float     = TR_TONGUE_END,
#     closure_threshold: float = CLOSURE_THRESHOLD,
# ) -> None:
#     """
#     Process every speaker directory matching sub* under data_root.
 
#     Expected structure
#     ------------------
#     data_root/
#       subXXX/
#         track_json/
#           task1.json
#           task2.json
#           ...
#       subYYY/
#         ...
 
#     Output
#     ------
#     data_root/
#       subXXX/
#         extracted_TVs/
#           task1.csv
#           task2.csv
#           ...
#     """
#     data_root = Path(data_root)
#     speakers  = sorted(data_root.glob("sub*"))
#     # speakers  = sorted(data_root.glob("sub009"))
 
#     if not speakers:
#         print(f"[ERROR] No sub* directories found under {data_root}")
#         return
 
#     print(f"Found {len(speakers)} speaker(s) under {data_root}\n")
 
#     for spk in speakers:
#         process_speaker(
#             speaker_dir=spk,
#             track_json_subdir=track_json_subdir,
#             output_subdir=output_subdir,
#             tt_palate_end=tt_palate_end,
#             tb_palate_start=tb_palate_start,
#             tr_tongue_end=tr_tongue_end,
#             closure_threshold=closure_threshold,
#         )
 
#     print("\n✓ All speakers processed.")


# # Batch Run

# DATA_ROOT = "/home/data/75SpeakerAnnot16"  
 
# process_all_speakers(
#     data_root         = DATA_ROOT,
#     tt_palate_end = TT_PALATE_END,
#     tb_palate_start = TB_PALATE_START,
#     tr_tongue_end = TR_TONGUE_END,
#     closure_threshold = CLOSURE_THRESHOLD,
# )