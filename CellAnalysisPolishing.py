#!/usr/bin/env python3

import sys, os, re, shutil, textwrap, warnings
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import chi2_contingency
from scipy.signal import welch
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ValueWarning as sm_ValueWarning
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import tifffile
import json


# -----------------------------------------------------------------------------
#                  CONFIGURATION CHEAT SHEET & GUIDE (v2)
# -----------------------------------------------------------------------------
# This guide explains the key parameters you can tune for your analysis.
#
# --- [ 1. CORE ANALYSIS TOGGLES ] ---
#
# GLM_METHOD:
#   - What it does: Selects the statistical model used for response detection.
#   - Options:
#     - "explicit_baseline_contrast" (Recommended): A robust two-stage model.
#       1. ASW response is tested against a 60s pre-stimulus baseline.
#       2. Chemical responses are tested against the ASW application period,
#          effectively controlling for mechanical droplet artifacts.
#     - "explicit_baseline_contrast_2drugs" 
#       1. ASW response is tested against a 60s pre-stimulus baseline.
#       2. First chemical responses are tested against the ASW application period,
#          effectively controlling for mechanical droplet artifacts. Subsequent 
#          chemical responses are tested against the final 30 seconds of the previous period
#     - "full_trace_intercept": A simpler, single-stage model where all stimuli
#       are tested against the recording's overall baseline (all time points
#       not occupied by a stimulus). Faster to run but less controlled.
#
#
# APPLY_LOWESS_DETREND:
#   - What it does: Removes slow, non-linear drifts from each neuron's dF/F trace.
#   - Recommended: "True" if necessary for correcting photobleaching or z-drift.
#   - "False" if stimuli induce global intensity changes, as "True" will normalize out these responses.
#
# USE_GLOBAL_REGRESSION:
#   - What it does: Subtracts the average activity of ALL neurons from each
#     individual neuron at every time point.
#   - Recommended: False. This is a very aggressive correction.
#   - Use with extreme caution. It can remove motion artifacts that affect the
#     entire field of view, but it will also remove genuine, widespread neural
#     responses (e.g., a global inhibition signal).
#
# --- [ 2. RESPONSE CLASSIFICATION THRESHOLDS ] ---
# These settings add an additional "effect size" gate on top of the GLM's
# statistical significance (p-value). A response must be both statistically
# significant AND cross this dF/F threshold to be counted.
#
# USE_NOISE_BASED_THRESHOLD:
#   - What it does: Toggles between an adaptive or a fixed dF/F threshold.
#   - Recommended: True. An adaptive threshold is more robust as it adjusts
#     to the noise level of each individual neuron.
#   - Set to False to use a single, fixed dF/F value for all neurons.
#
# NOISE_MULTIPLIER:
#   - What it does: Sets the adaptive threshold when USE_NOISE_BASED_THRESHOLD=True.
#     A value of 3 means a response peak must be at least 3 times the standard
#     deviation of that neuron's baseline noise.
#   - Recommended Range: 3 to 5.
#   - Higher value = stricter threshold, fewer false positives, but might miss
#     weaker real responses.
#
# FIXED_DFF_THRESHOLD:
#   - What it does: Sets the fixed dF/F threshold when USE_NOISE_BASED_THRESHOLD=False.
#     A value of 0.2 means a response peak must exceed 0.2 dF/F.
#   - This setting is IGNORED if USE_NOISE_BASED_THRESHOLD is True.
#
# USE_ASW_EFFECT_SIZE_CONTROL:
#   - What it does: The "mechanical artifact filter". If True, a chemical
#     response is only counted if its peak dF/F is stronger than the ASW peak.
#   - Recommended: True only if using GLM_METHOD = "full_trace_intercept". 
#     In this case, this is a critical control for droplet-application
#     experiments to prevent misclassifying mechanosensitive cells.
#   - Note: This is a separate control from the GLM baseline comparison.
#
# ASW_FOLD_THRESHOLD:
#   - What it does: Sets how much stronger the chemical response peak must be
#     when USE_ASW_EFFECT_SIZE_CONTROL=True.
#   - Recommended Range: 1.1 to 2.0.
#   - 1.1 = Chemical peak must be at least 10% larger than ASW peak.
#
# --- [ 3. FIRE-FRACTION ANALYSIS PARAMETERS ] ---
# These settings control the "Fraction of Active Cells" plots.
#
# FIRE_FRACTION_DFF_MULTIPLIER:
#   - What it does: Sets the dF/F threshold for a cell to be considered "active"
#     in the fire-fraction plots. It is an adaptive threshold, similar to
#     NOISE_MULTIPLIER.
#   - Recommended Range: 3.0 to 5.0.
#
# --- [ 4. GLM & TIMING PARAMETERS ] ---
#
# STIM_CONFIG:
#   - What it does: Allows different analysis windows for different stimuli.
#   - 'window_seconds': The duration of the analysis window after stimulus onset.
#   - 'bin_size_seconds': The temporal resolution of the GLM. Smaller bins can
#     capture faster dynamics but may be noisier.
# -----------------------------------------------------------------------------

RUN_PEAK_THRESHOLD_ANALYSIS = False
GLM_METHOD = "explicit_baseline_contrast"  # Options: "full_trace_intercept" (DON'T use unless specifically necessary), "explicit_baseline_contrast" or "explicit_baseline_contrast_2drugs'
# For 'explicit_baseline_contrast_2drugs' method:Length of the local pre-stimulus baseline window (seconds) for the 2nd+ chemical stimuli (and for the 1st chemical if ASW is absent).
CHEM_LOCAL_BASELINE_SECONDS = 60  #Set to number of seconds prior (leading up) to a stimulus that you want to be used as a baseline
MANUAL_FRAMERATE_HZ    = 0.68 #Overrides Suite2p output
APPLY_LOWESS_DETREND   = True
LOWESS_FRACTION = 0.5
USE_GLOBAL_REGRESSION  = False
USE_NOISE_BASED_THRESHOLD = True
NOISE_MULTIPLIER          = 3
FIXED_DFF_THRESHOLD       = 0.2
USE_ASW_EFFECT_SIZE_CONTROL = False
ASW_FOLD_THRESHOLD        = 1.1
FDR_ALPHA = 0.05
# --- DISPLAY / SORTING OPTIONS ---
FIG1A_SORT_MODE = "peak"   # "peak" (current) or "roi"

# --- Neuropil toggle ---
USE_NEUROPIL_SUBTRACTION = True     # False => α = 0 (skip neuropil)
FORCE_NEUROPIL_ALPHA = 0.5         # e.g., 0.5 to override ops; None to use ops or fallback
NEUROPIL_ALPHA_FALLBACK = 0.5       # used if ops has no 'neucoeff'



STIM_CONFIG = {
    "ASW": {
        "delay_seconds": 0,  
        "window_seconds": 60,
        "bin_size_seconds": 10,
    },
    "default": {
        "delay_seconds": 0,  
        "window_seconds": 300,
        "bin_size_seconds": 30,
    }
}

FIRE_FRACTION_DFF_MULTIPLIER   = 3.0

COLOR_MAP = {
    "activated": "#c3195d", "inhibited": "#0077b6", "unresponsive": "#d3d3d3",
    "exc": "#c3195d", "inh": "#0077b6", "unresp": "#d3d3d3", 
}

GEM_DIV_CMAP = LinearSegmentedColormap.from_list(
    "gem_diverging", [COLOR_MAP["inhibited"], "white", COLOR_MAP["activated"]]
)
# --- Symbol remapping for outputs ---
OUTPUT_SYMBOLS = {
    "activated": "E",        # excitation
    "inhibited": "i",        # inhibition
    "unresponsive": "x"      # no response
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
SAFE_COL_PREFIX = "col__"

_UNSAFE_COL_PATTERNS = [
    re.compile(r"^\s*$"),                     # empty
    re.compile(r"^[\+\-\=\@]"),               # formula-like / leading symbols
    re.compile(r"^\d+$"),                     # pure integer
    re.compile(r"^\d+\.\d+$"),                # pure float
    re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$"),  # date-like (e.g., 1/2/2025)
    re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$"),      # ISO-ish date
    re.compile(r"^\d+e[\+\-]?\d+$", re.IGNORECASE),    # scientific notation
]

def _is_unsafe_header(name: str) -> bool:
    s = str(name).strip()
    if s == "": 
        return True
    for pat in _UNSAFE_COL_PATTERNS:
        if pat.match(s):
            return True
    return False

def _make_safe_headers(columns) -> (dict, list):
    """
    Returns (mapping, safe_list). Ensures uniqueness after prefixing.
    """
    mapping = {}
    used = set()
    safe_cols = []
    for col in columns:
        col_str = str(col)
        if _is_unsafe_header(col_str):
            new = f"{SAFE_COL_PREFIX}{col_str}"
        else:
            new = col_str

        # ensure uniqueness
        base = new
        k = 1
        while new in used:
            k += 1
            new = f"{base}__{k}"
        used.add(new)
        mapping[col] = new
        safe_cols.append(new)
    return mapping, safe_cols

def _normalize_category(x: str) -> str:
    if pd.isna(x): 
        return "unresponsive"
    s = str(x).strip().lower()
    mapping = {
        "activated": "activated", "exc": "activated", "e": "activated",
        "inhibited": "inhibited", "inh": "inhibited", "i": "inhibited",
        "unresponsive": "unresponsive", "unresp": "unresponsive", "x": "unresponsive",
        # common capitalized forms
        "activated ": "activated", "inhibited ": "inhibited", "unresponsive ": "unresponsive",
    }
    return mapping.get(s, "unresponsive")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")

def save_df(df, fname, local_dir, central_dir, glm_method_tag):
    """
    Save a DataFrame to both local and central dirs with Excel-safe headers.
    Also writes a sidecar JSON with the original->safe header map.

    Example:
      - CSV:   myfile_GLM-Contrast.csv
      - MAP:   myfile_GLM-Contrast__columns_map.json
    """
    fname_tagged = fname.replace('.csv', f'_{glm_method_tag}.csv')
    local_dir.mkdir(parents=True, exist_ok=True)
    central_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        return

    col_map, safe_cols = _make_safe_headers(df.columns)
    df_to_save = df.copy()
    df_to_save.columns = safe_cols

    #JSON mapping 
    sidecar_name = fname_tagged.replace('.csv', '__columns_map.json')
    sidecar_map = {str(k): v for k, v in col_map.items()}

    # Write both copies
    df_to_save.to_csv(local_dir / fname_tagged, index=False)
    df_to_save.to_csv(central_dir / fname_tagged, index=False)

    with open(local_dir / sidecar_name, 'w') as f:
        json.dump(sidecar_map, f, indent=2)
    with open(central_dir / sidecar_name, 'w') as f:
        json.dump(sidecar_map, f, indent=2)

    log(f"Saved CSV (safe headers): {fname_tagged}")
    log(f"Saved column map: {sidecar_name}")


def save_json(obj, fname, local_dir, central_dir, glm_method_tag):
    fname_tagged = fname.replace('.json', f'_{glm_method_tag}.json')
    local_dir.mkdir(parents=True, exist_ok=True)
    central_dir.mkdir(parents=True, exist_ok=True)
    with open(local_dir / fname_tagged, 'w') as f:
        json.dump(obj, f, indent=2)
    with open(central_dir / fname_tagged, 'w') as f:
        json.dump(obj, f, indent=2)
    log(f"Saved JSON: {fname_tagged}")


def add_figure_title(fig, fignum, title):
    fig.text(0.01, 0.99, f"Figure {fignum}: {title}", ha="left", va="top", fontsize=12, weight="bold")

def add_text_page_to_pdf(pdf_obj, title, text_lines, fignum="", fontsize=10, line_spacing=0.025):
    fig_txt = plt.figure(figsize=(8.5, 11))
    plt.axis("off")
    full_title = f"Figure {fignum}: {title}" if fignum else title
    fig_txt.text(0.05, 0.95, full_title, va="top", family="monospace", fontsize=12, weight="bold")
    y = 0.90
    for line in text_lines:
        wrapped = textwrap.fill(line, width=100)
        for sub in wrapped.split("\n"):
            fig_txt.text(0.05, y, sub, va="top", family="monospace", fontsize=fontsize)
            y -= line_spacing
            if y < 0.05: break
        if y < 0.05: break
        y -= 0.01
    pdf_obj.savefig(fig_txt, bbox_inches="tight")
    plt.close(fig_txt)

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
def load_data(sess_dir, stim_frames):
    log("Loading Suite2p data...")
    F = np.load(sess_dir / "F.npy")
    Fneu = np.load(sess_dir / "Fneu.npy")
    iscell = np.load(sess_dir / "iscell.npy", allow_pickle=True)
    good_rois = np.where(iscell[:, 0] == 1)[0]

    ops = np.load(sess_dir / "ops.npy", allow_pickle=True).item()
    stat_all = np.load(sess_dir / "stat.npy", allow_pickle=True)

    F_good, Fneu_good = F[good_rois], Fneu[good_rois]
    stat_sel = stat_all[good_rois]
    log(f"Found {len(good_rois)} manually curated cells.")

    # --- Neuropil subtraction ---
    alpha, alpha_tag = get_neuropil_alpha_and_tag(ops)  # uses ops['neucoeff'] / fallback / forced / off
    if alpha > 0:
        log(f"Neuropil subtraction ON ({alpha_tag})")
        Fcorr = F_good - alpha * Fneu_good
    else:
        log("Neuropil subtraction OFF (alpha=0.0)")
        Fcorr = F_good

    # record for PDF metadata page
    ops['neuropil_alpha_used'] = float(alpha)
    ops['neuropil_alpha_note'] = str(alpha_tag)


    # dF/F baseline: frames between Stim 1 and Stim 2  →  [start_of_stim1 : start_of_stim2)
    baseline_start_frame = int(stim_frames[0][1])   # Stim 1 onset
    baseline_end_frame   = int(stim_frames[1][1])   # Stim 2 onset

    nF = Fcorr.shape[1]
    lo = max(0, min(baseline_start_frame, nF))
    hi = max(0, min(baseline_end_frame, nF))

    # If the slice is invalid/empty, fall back to [0 : baseline_end_frame)
    if hi <= lo:
        lo, hi = 0, max(0, min(baseline_end_frame, nF))

    ops['dff_baseline_frames'] = f"{lo}-{hi}"


    F0 = np.percentile(Fcorr[:, lo:hi], 20, axis=1, keepdims=True)
    F0[F0 <= 0] = 1e-6
    dff = (Fcorr - F0) / F0

    return dff, good_rois, ops, stat_sel




def extract_stims(sess_dir, summary_csv_path):
    log("Extracting stimulus frames/labels...")
    summary = pd.read_csv(summary_csv_path)
    run_folder = next((p for p in Path(sess_dir).parents if "_zout_" in p.name), None)
    if run_folder is None:
        raise ValueError(f"Could not find a '_zout_' parent folder for {sess_dir}")
    m = re.match(r'^(.*)_zout_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(t\S+)$', run_folder.name)
    if not m:
        raise ValueError(f"Cannot parse run_id from folder name: {run_folder.name}")
    run_id = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"

    row = summary[summary.run_id.str.strip().str.lower() == run_id.lower()]
    if row.empty:
        raise ValueError(f"No entry found in summary CSV for run_id={run_id}")
    row = row.iloc[0]

    stim_cols = sorted(
        [c for c in row.index if re.match(r"stim\d+_frame", c) and pd.notna(row[c])],
        key=lambda s: int(re.search(r"\d+", s).group())
    )
    frames = [int(row[c]) for c in stim_cols]

    neuro_map = {
        'g': 'GABA', 'a': 'ACh', 'd': 'DA', 'o': 'Oct',
        's': '5HT', 'i': 'INK', 'f': 'FOOD', 'l': 'Glut',
        'w': 'ASW', 'v': 'Vibr', 'n': 'NoStim', 'p': 'FLRIamide'
    }

    code_x = re.search(r'_x([A-Za-z]+)x', run_id)
    code_z = re.search(r'_z([A-Za-z]+)(?:x|_|$)', run_id)

    # zN override: first is mechanical control (ASW1), next two are treated as chemicals for negative control
    if code_z and code_z.group(1).lower() == 'n':
        frames = [202, 404, 606]
        raw_labels = ['ASW1', 'CHEM1', 'CHEM2']
    elif code_x:
        raw_labels = ['ASW'] + [neuro_map.get(l, l) for l in code_x.group(1).lower()]
    elif code_z:
        raw_labels = [neuro_map.get(l, l) for l in code_z.group(1).lower()]
    else:
        raw_labels = []

    if len(raw_labels) != len(frames):
        log(f"Warning: Mismatch between labels and frames. Using generic labels.")
        raw_labels = [f"Stim{i+1}" for i in range(len(frames))]

    labels, counts = [], {}
    for L in raw_labels:
        counts[L] = counts.get(L, 0) + 1
        labels.append(f"{L}_{counts[L]}" if counts[L] > 1 else L)

    return run_id, list(zip(labels, frames)), row


def detrend_traces(dff):
    log("Applying LOWESS detrending...")
    detrended_dff = np.zeros_like(dff)
    for i in range(dff.shape[0]):
        trend = lowess(dff[i], np.arange(dff.shape[1]), frac=LOWESS_FRACTION, return_sorted=False)
        detrended_dff[i] = dff[i] - trend
    return detrended_dff

def get_neuropil_alpha_and_tag(ops):
    """
    Returns (alpha, tag_string) based on the three configuration knobs:
    - USE_NEUROPIL_SUBTRACTION
    - FORCE_NEUROPIL_ALPHA
    - NEUROPIL_ALPHA_FALLBACK
    and Suite2p's ops['neucoeff'] if available.
    """
    if not USE_NEUROPIL_SUBTRACTION:
        return 0.0, "off"

    if FORCE_NEUROPIL_ALPHA is not None:
        return float(FORCE_NEUROPIL_ALPHA), f"forced={FORCE_NEUROPIL_ALPHA:.3f}"

    # Suite2p stores the scalars in 'neucoeff' for the run
    alpha = float(ops.get('neucoeff', float('nan')))
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = float(NEUROPIL_ALPHA_FALLBACK)
        return alpha, f"fallback={alpha:.3f}"

    return alpha, f"ops.neucoeff={alpha:.3f}"


def classify_activity_windows(dff, stims, fs, good_rois):
    """
    Classify each ROI in 5-minute windows after the first ASW (or first stim if no ASW)
    and after each subsequent stimulus: 'stable',  'oscillatory', 'sporadic'.
    Returns (df_long, df_wide).
    """
    log("Classifying 5-minute post-stim activity windows...")
    import numpy as _np
    from scipy.signal import welch as _welch
    n_cells, n_frames = dff.shape
    win_frames = int(300 * fs)  # 5 minutes

    # Choose baseline end as second stim if present, else 120s
    baseline_end_frame = stims[1][1] if len(stims) > 1 else min(n_frames, int(120 * fs))
    sigma_baseline = dff[:, :baseline_end_frame].std(axis=1)

    # First window anchor: first ASW if present; else first stim
    asw_idx = next((i for i, (lbl, _) in enumerate(stims) if str(lbl).startswith("ASW")), 0)
    window_defs = []
    first_lbl, first_fr = stims[asw_idx]
    window_defs.append((f"Post-{first_lbl} (first 5 min)", first_fr, min(n_frames, first_fr + win_frames)))
    # Subsequent windows: every stimulus (except the one already added)
    for i, (lbl, fr) in enumerate(stims):
        if i == asw_idx: 
            continue
        window_defs.append((f"Post-{lbl} (5 min)", fr, min(n_frames, fr + win_frames)))

    f_lo, f_hi = 0.003, 0.2  # Hz band for slow oscillations
    rows = []
    for ci in range(n_cells):
        thr = (NOISE_MULTIPLIER * sigma_baseline[ci]) if USE_NOISE_BASED_THRESHOLD else FIXED_DFF_THRESHOLD
        for win_lbl, w0, w1 in window_defs:
            if w1 <= w0 + 2:
                cls = "stable"
            else:
                seg = _np.nan_to_num(dff[ci, w0:w1])
                has_event = (_np.max(_np.abs(seg)) >= thr)
                var_seg = _np.nanvar(seg); var_base = sigma_baseline[ci]**2
                low_var = var_seg <= var_base + 1e-12
                try:
                    frq, pxx = _welch(seg, fs=fs, nperseg=min(len(seg), 256))
                    band = (frq >= f_lo) & (frq <= f_hi)
                    if band.any():
                        band_pxx = pxx[band]; med = _np.nanmedian(band_pxx) if band_pxx.size else 0.0
                        if med > 0:
                            peak_idx = _np.nanargmax(band_pxx)
                            peak_f = frq[band][peak_idx]
                            ratio = band_pxx[peak_idx] / (med + 1e-12)
                            enough_cycles = (peak_f * (w1 - w0) / fs) >= 2.0
                            is_osc = (ratio >= 5.0) and enough_cycles
                        else:
                            is_osc = False
                    else:
                        is_osc = False
                except Exception:
                    is_osc = False

                if is_osc:
                    cls = "oscillatory"
                elif has_event:
                    cls = "sporadic"
                else:
                    cls = "stable"


            rows.append({
                "ROI_ID": int(good_rois[ci]) if hasattr(good_rois, '__len__') else int(ci),
                "WindowLabel": win_lbl,
                "WindowStartFrame": int(w0),
                "WindowEndFrame": int(w1),
                "Class": cls
            })

    df_long = pd.DataFrame(rows)
    pivot = df_long.pivot(index="ROI_ID", columns="WindowLabel", values="Class")
    return df_long, pivot
def global_regression(dff):
    log("Applying global signal regression...")
    return dff - dff.mean(axis=0)

# -----------------------------------------------------------------------------
# ANALYSIS PIPELINE 1: SIMPLE PEAK-THRESHOLD
# -----------------------------------------------------------------------------
def classify_responses_by_peak(dff, stims, fs, good_rois):
    log("Running simple peak-threshold classification (dF/F only)...")
    n_cells, n_frames = dff.shape
    baseline_end_frame = stims[1][1]
    sigma_dff = dff[:, :baseline_end_frame].std(axis=1)
    results = []
    for i in range(n_cells):
        roi_id = good_rois[i]
        for label, frame_start in stims:
            stim_key = label.split('_')[0]
            cfg = STIM_CONFIG.get(stim_key, STIM_CONFIG["default"])
            win_end = min(n_frames, frame_start + int(cfg["window_seconds"] * fs))
            
            trace_dff = dff[i, frame_start:win_end]
            thresh_dff = NOISE_MULTIPLIER * sigma_dff[i] if USE_NOISE_BASED_THRESHOLD else FIXED_DFF_THRESHOLD
            
            dff_resp = "unresponsive"
            peak_dff, trough_dff = np.nanmax(trace_dff), np.nanmin(trace_dff)
            if peak_dff > thresh_dff:
                dff_resp = "activated"
            elif trough_dff < -thresh_dff:
                dff_resp = "inhibited"
            
            if dff_resp == "activated" and USE_ASW_EFFECT_SIZE_CONTROL and "ASW" in [s[0] for s in stims] and not label.startswith("ASW"):
                asw_label = next(s[0] for s in stims if s[0].startswith("ASW"))
                asw_fr = next(s[1] for s in stims if s[0] == asw_label)
                asw_win_end = asw_fr + int(STIM_CONFIG["default"]["window_seconds"] * fs)
                asw_peak = np.nanmax(dff[i, asw_fr : asw_win_end])
                if peak_dff < ASW_FOLD_THRESHOLD * asw_peak:
                    dff_resp = "unresponsive"
            
            results.append({"ROI_ID": roi_id, "Stimulus": label, "Response": dff_resp})
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# ANALYSIS PIPELINE 2: GLM-BASED (TWO-STAGE)
# -----------------------------------------------------------------------------
def make_design_matrix(stims, n_frames, fs):
    log("Building binned FIR design matrix with stimulus-specific windows...")
    design_matrix = {}
    all_potential_cols = []
    

    for label, t0 in stims:
        stim_key = label.split('_')[0]
        cfg = STIM_CONFIG.get(stim_key, STIM_CONFIG["default"])

        # Get the delay and calculate the true start frame for the analysis window
        delay_frames = int(cfg.get('delay_seconds', 0) * fs)
        analysis_start_frame = t0 + delay_frames
        
        n_bins = int(cfg['window_seconds'] / cfg['bin_size_seconds'])
        
        for b in range(n_bins):
            start = int(analysis_start_frame + b * cfg['bin_size_seconds'] * fs)
            end   = int(analysis_start_frame + (b + 1) * cfg['bin_size_seconds'] * fs)
            name = f"{label}_bin{b}"
            all_potential_cols.append(name)
            vec = np.zeros(n_frames)
            if start < n_frames: vec[start:min(n_frames, end)] = 1
            if vec.sum() > 0: design_matrix[name] = vec
                
        design_df = pd.DataFrame(design_matrix)
        for col in all_potential_cols:
            if col not in design_df: design_df[col] = 0.0
            
        return sm.add_constant(design_df[sorted(design_df.columns)], prepend=True)

def run_glm(traces, data_type, design_matrix, rois):
    log(f"Running FIR-GLM on {data_type} data ({traces.shape[0]} cells)...")
    results, stim_groups = [], {}
    for col in design_matrix.columns:
        if 'bin' in col:
            stim_name = col.split('_bin')[0]
            if stim_name not in stim_groups: stim_groups[stim_name] = []
            stim_groups[stim_name].append(col)
    for i, trace in enumerate(traces):
        if np.all(trace == 0):
            for stim, cols in stim_groups.items(): results.append({"ROI_ID": int(rois[i]), "Stimulus": stim, "f_stat": 0, "p_value": 1.0, "betas": np.zeros(len(cols)), "type": data_type})
            continue
        model = sm.OLS(trace, design_matrix).fit()
        for stim, cols in stim_groups.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", sm_ValueWarning)
                    ftest = model.f_test(cols)
                f_val, p_val = float(ftest.fvalue), float(ftest.pvalue)
            except (ValueError, np.linalg.LinAlgError): f_val, p_val = np.nan, np.nan
            results.append({"ROI_ID": int(rois[i]), "Stimulus": stim, "f_stat": f_val, "p_value": p_val, "betas": model.params[cols].values, "type": data_type})
    df = pd.DataFrame(results)
    if df.empty: return df
    log(f"Performing FDR correction for {data_type}...")
    df["p_adj"], df["is_sig"] = 1.0, False
    valid_pvals = df["p_value"].dropna()
    if not valid_pvals.empty:
        reject, p_adj, _, _ = multipletests(valid_pvals, alpha=FDR_ALPHA, method="fdr_bh")
        df.loc[valid_pvals.index, "p_adj"], df.loc[valid_pvals.index, "is_sig"] = p_adj, reject
    def classify(row):
        if not row["is_sig"]: return "unresponsive"
        peak_idx = np.argmax(np.abs(row["betas"]))
        return "activated" if row["betas"][peak_idx] > 0 else "inhibited"
    df['Response'] = df.apply(classify, axis=1)
    df['beta_sum'] = [np.sum(b) if s else 0 for s, b in zip(df['is_sig'], df['betas'])]
    return df

def make_design_matrix_for_chemicals(stims, n_frames, fs):
    log("Building GLM design matrix for chemical stimuli...")
    design_matrix = {}
    
    chem_stims = [s for s in stims if not s[0].startswith("ASW")]
    

    for label, t0 in chem_stims:
        stim_key = label.split('_')[0]
        cfg = STIM_CONFIG.get(stim_key, STIM_CONFIG["default"])
        
        # Get the delay and calculate the true start frame for the analysis window
        delay_frames = int(cfg.get('delay_seconds', 0) * fs)
        analysis_start_frame = t0 + delay_frames
        
        n_bins = int(cfg['window_seconds'] / cfg['bin_size_seconds'])
        for b in range(n_bins):
            start = int(analysis_start_frame + b * cfg['bin_size_seconds'] * fs)
            end   = int(analysis_start_frame + (b + 1) * cfg['bin_size_seconds'] * fs)
            name = f"{label}_bin{b}"
            vec = np.zeros(n_frames)
            if start < n_frames: vec[start:min(n_frames, end)] = 1
            if vec.sum() > 0: design_matrix[name] = vec
            
    asw_label = next(s[0] for s in stims if s[0].startswith("ASW"))
    asw_fr = next(s[1] for s in stims if s[0] == asw_label)
    
    baseline_start_frame = asw_fr
    baseline_end_frame = stims[1][1]
    
    baseline_vec = np.zeros(n_frames)
    baseline_vec[baseline_start_frame:baseline_end_frame] = 1
    design_matrix["baseline"] = baseline_vec
    
    design_df = pd.DataFrame(design_matrix)
    return design_df[sorted(design_df.columns)]

def make_design_matrix_for_single_chemical_vs_baseline(
    stim_label, stim_frame, n_frames, fs, *,
    baseline_start_frame, baseline_end_frame
):
    """
    Build a design matrix for ONE chemical stimulus with a custom baseline interval.
    - Bins for the target chemical only (using its STIM_CONFIG window/binning).
    - A single 'baseline' column covering [baseline_start_frame : baseline_end_frame).
    """
    design_matrix = {}

    stim_key = stim_label.split('_')[0]
    cfg = STIM_CONFIG.get(stim_key, STIM_CONFIG["default"])

    # Respect per-stim delay
    delay_frames = int(cfg.get('delay_seconds', 0) * fs)
    analysis_start_frame = stim_frame + delay_frames

    n_bins = int(cfg['window_seconds'] / cfg['bin_size_seconds'])
    for b in range(n_bins):
        start = int(analysis_start_frame + b * cfg['bin_size_seconds'] * fs)
        end   = int(analysis_start_frame + (b + 1) * cfg['bin_size_seconds'] * fs)
        name = f"{stim_label}_bin{b}"
        vec = np.zeros(n_frames)
        if start < n_frames:
            vec[start:min(n_frames, end)] = 1
        if vec.sum() > 0:
            design_matrix[name] = vec

    # Custom baseline column
    bl_start = max(0, int(baseline_start_frame))
    bl_end   = min(n_frames, int(baseline_end_frame))
    baseline_vec = np.zeros(n_frames)
    if bl_start < bl_end:
        baseline_vec[bl_start:bl_end] = 1
    design_matrix["baseline"] = baseline_vec

    design_df = pd.DataFrame(design_matrix)
    return design_df[sorted(design_df.columns)]


def make_design_matrix_for_asw(stims, n_frames, fs):
    log("Building GLM design matrix for ASW stimulus...")
    design_matrix = {}
    
    asw_stim = next((s for s in stims if s[0].startswith("ASW")), None)
    if asw_stim is None: return pd.DataFrame()
    
    label, t0 = asw_stim
    cfg = STIM_CONFIG.get("ASW", STIM_CONFIG["default"])

    # Get the delay and calculate the true start frame for the analysis window
    delay_frames = int(cfg.get('delay_seconds', 0) * fs)
    analysis_start_frame = t0 + delay_frames
    
    n_bins = int(cfg['window_seconds'] / cfg['bin_size_seconds'])
    for b in range(n_bins):
        start = int(analysis_start_frame + b * cfg['bin_size_seconds'] * fs)
        end   = int(analysis_start_frame + (b + 1) * cfg['bin_size_seconds'] * fs)
        name = f"{label}_bin{b}"
        vec = np.zeros(n_frames)
        if start < n_frames: vec[start:min(n_frames, end)] = 1
        if vec.sum() > 0: design_matrix[name] = vec
    pre_asw_baseline_end = t0
    pre_asw_baseline_start = max(0, t0 - int(60 * fs))
    baseline_vec = np.zeros(n_frames)
    baseline_vec[pre_asw_baseline_start:pre_asw_baseline_end] = 1
    design_matrix["pre_asw_baseline"] = baseline_vec
    
    design_df = pd.DataFrame(design_matrix)
    return design_df[sorted(design_df.columns)]


def run_narrow_glm(traces, data_type, design_matrix, rois, baseline_col='baseline'):
    log(f"Running Explicit Baseline Contrast GLM on {data_type} data...")
    results, stim_groups = [], {}
    for col in design_matrix.columns:
        if 'bin' in col:
            stim_name = col.split('_bin')[0]
            if stim_name not in stim_groups: stim_groups[stim_name] = []
            stim_groups[stim_name].append(col)

    if baseline_col not in design_matrix.columns:
        raise ValueError(f"Specified baseline column '{baseline_col}' not in design matrix.")

    for i, trace in enumerate(traces):
        model = sm.OLS(trace, design_matrix).fit()
        
        for stim, cols in stim_groups.items():
            contrast_matrix = []
            for col_name in cols:
                contrast = np.zeros(len(design_matrix.columns))
                contrast[design_matrix.columns.get_loc(col_name)] = 1
                contrast[design_matrix.columns.get_loc(baseline_col)] = -1
                contrast_matrix.append(contrast)
            
            try:
                if not contrast_matrix: raise ValueError("Empty contrast matrix")
                ftest = model.f_test(np.array(contrast_matrix))
                f_val, p_val = float(ftest.fvalue), float(ftest.pvalue)
            except (ValueError, np.linalg.LinAlgError):
                f_val, p_val = np.nan, np.nan

            stim_betas = model.params[cols].values
            results.append({"ROI_ID": int(rois[i]), "Stimulus": stim, "f_stat": f_val, "p_value": p_val, "betas": stim_betas, "type": data_type})

    df = pd.DataFrame(results)
    if df.empty: return df
    log(f"Performing FDR correction for {data_type}...")
    df["p_adj"], df["is_sig"] = 1.0, False
    valid_pvals = df["p_value"].dropna()
    if not valid_pvals.empty:
        reject, p_adj, _, _ = multipletests(valid_pvals, alpha=FDR_ALPHA, method="fdr_bh")
        df.loc[valid_pvals.index, "p_adj"], df.loc[valid_pvals.index, "is_sig"] = p_adj, reject
    def classify(row):
        if not row["is_sig"]: return "unresponsive"
        peak_idx = np.argmax(np.abs(row["betas"]))
        return "activated" if row["betas"][peak_idx] > 0 else "inhibited"
    df['Response'] = df.apply(classify, axis=1)
    df['beta_sum'] = [np.sum(b) if s else 0 for s, b in zip(df['is_sig'], df['betas'])]
    return df

def gate_glm_responses_by_effect_size(glm_df, dff, stims, fs, good_rois):
    log("Gating GLM responses by effect size and ASW control...")
    
    analysis_windows = {}
    for label, frame_start in stims:
        stim_key = label.split('_')[0]
        cfg = STIM_CONFIG.get(stim_key, STIM_CONFIG["default"])
        analysis_windows[label] = (frame_start, frame_start + int(cfg["window_seconds"] * fs))

    roi_id_to_idx = {roi_id: idx for idx, roi_id in enumerate(good_rois)}
    baseline_end_frame = stims[1][1]
    sigma_dff = dff[:, :baseline_end_frame].std(axis=1)
    
    def check_effect(row):
        if row['Response'] == 'unresponsive': return "unresponsive"
        roi_idx = roi_id_to_idx[row['ROI_ID']]
        win_start, win_end = analysis_windows[row['Stimulus']]
        
        trace_drug = dff[roi_idx, win_start:win_end]
        if USE_ASW_EFFECT_SIZE_CONTROL and "ASW" in [s[0] for s in stims] and not row['Stimulus'].startswith("ASW"):
            asw_label = next(s[0] for s in stims if s[0].startswith("ASW"))
            asw_fr = next(s[1] for s in stims if s[0] == asw_label)
            asw_win_end = asw_fr + int(STIM_CONFIG["default"]["window_seconds"] * fs)
            asw_peak = np.nanmax(dff[roi_idx, asw_fr:asw_win_end])
            if np.max(trace_drug) < ASW_FOLD_THRESHOLD * asw_peak: return "unresponsive"
        
        thresh = NOISE_MULTIPLIER * sigma_dff[roi_idx] if USE_NOISE_BASED_THRESHOLD else FIXED_DFF_THRESHOLD
        if np.max(np.abs(trace_drug)) < thresh: return "unresponsive"
            
        return row['Response']
        
    glm_df['Response'] = glm_df.apply(check_effect, axis=1)
    glm_df['is_sig'] = glm_df['Response'] != 'unresponsive'
    return glm_df

def create_glm_consensus_responses(glm_df):
    log("Finalizing GLM responses...")
    if glm_df.empty: return pd.DataFrame()
    consensus_df = glm_df.copy()
    consensus_df.rename(columns={'Response': 'Category'}, inplace=True)
    return consensus_df[['ROI_ID', 'Stimulus', 'Category']]

# -----------------------------------------------------------------------------
# OTHER ANALYSIS & PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_exploratory_metrics(dff, good_rois):
    log("Calculating exploratory metrics for summary plots...")
    n_cells, n_frames = dff.shape
    noise_floor = np.array([np.nanstd(trace[trace <= np.nanmedian(trace)]) if np.any(trace <= np.nanmedian(trace)) else np.nanstd(trace) for trace in dff])
    dff_snr = np.nanmedian(dff, axis=1) / (noise_floor + 1e-8)
    
    dff_clean = np.nan_to_num(dff)
    if n_cells > 2:
        pca_coords = PCA(n_components=2).fit_transform(dff_clean)
        tsne_coords = TSNE(n_components=2, perplexity=min(30.0, n_cells - 1), random_state=0).fit_transform(dff_clean)
    else:
        pca_coords, tsne_coords = np.full((n_cells, 2), np.nan), np.full((n_cells, 2), np.nan)
        
    sparseness_per_frame = np.zeros(n_frames)
    for t in range(n_frames):
        R = np.maximum(dff[:, t], 0)
        if len(R) < 2 or np.all(R == 0): sparseness_per_frame[t] = np.nan
        else: sparseness_per_frame[t] = (1 - (np.mean(R)**2 / (np.mean(R**2) + 1e-8))) / (1 - 1/len(R))
    
    return pd.DataFrame({"ROI_ID": good_rois, "dff_snr": dff_snr, "pca_coord_1": pca_coords[:, 0], "pca_coord_2": pca_coords[:, 1], "tsne_coord_1": tsne_coords[:, 0], "tsne_coord_2": tsne_coords[:, 1]}), sparseness_per_frame

def calculate_temporal_properties(glm_df, stims):
    log("Calculating temporal properties of response kernels...")
    coms, peaks = [], []
    for _, row in glm_df.iterrows():
        if row['is_sig']:
            stim_key = row['Stimulus'].split('_')[0]
            cfg = STIM_CONFIG.get(stim_key, STIM_CONFIG["default"])
            bin_size_s = cfg['bin_size_seconds']
            
            betas, times = np.array(row['betas']), np.arange(len(row['betas'])) * bin_size_s
            abs_b = np.abs(betas)
            coms.append((times * abs_b).sum() / abs_b.sum() if abs_b.sum() > 1e-6 else np.nan)
            if row['Response'] == 'activated': peaks.append(times[np.argmax(betas)])
            elif row['Response'] == 'inhibited': peaks.append(times[np.argmin(betas)])
            else: peaks.append(np.nan)
        else:
            coms.append(np.nan); peaks.append(np.nan)
    glm_df['center_of_mass_s'], glm_df['time_to_peak_s'] = coms, peaks
    return glm_df

def calculate_fire_fraction_summary(dff, stims):
    log("Calculating fire-fraction summary table...")
    n_cells, n_frames = dff.shape
    epochs = {"Pre-Stim (1st 100 frames)": np.arange(0, min(100, n_frames)), "Full Recording": np.arange(0, n_frames), "Post-Stim (last 100 frames)": np.arange(max(0, n_frames - 100), n_frames)}
    
    baseline_end_frame = stims[1][1]
    sigma_dff = dff[:, :baseline_end_frame].std(axis=1)
    
    dff_abs_thresh = FIRE_FRACTION_DFF_MULTIPLIER * sigma_dff

    out = []
    for name, idx in epochs.items():
        if len(idx) == 0: continue
        active_dff = (dff[:, idx] >= dff_abs_thresh[:, np.newaxis]).any(axis=1).sum()
        pct_dff = 100 * active_dff / n_cells
        out.append([name, f"{active_dff} ({pct_dff:.1f}%)"])
    return out

def _compute_frac_dff(dff, dff_thresh_per_cell, time_idx):
    if dff.size == 0 or time_idx.size == 0: return np.nan
    return (dff[:, time_idx].max(axis=1) >= dff_thresh_per_cell).sum() / dff.shape[0]

def _compute_state_combinations(activity_long: pd.DataFrame, window_order: list, *, use_first_n_windows: int = 3):
    """
    Build 'state codes' per ROI across the first N post-stim windows.
    Mapping: stable->'n', oscillatory->'o', sporadic->'s'.
    Returns (summary_df, ordered_patterns):
      summary_df columns: ['Pattern','Count','Percent']
    """
    if activity_long is None or activity_long.empty:
        return pd.DataFrame(columns=['Pattern','Count','Percent']), []

    # Keep a stable, chronological window order and trim to first N windows
    window_order = [w for w in window_order][:use_first_n_windows]

    # Map classes -> single letters
    map_letter = {'stable': 'n', 'oscillatory': 'o', 'sporadic': 's'}

    # Pivot to wide: rows=ROI_ID, cols=WindowLabel, vals=Class
    wide = (activity_long
            .pivot(index='ROI_ID', columns='WindowLabel', values='Class')
            .reindex(columns=window_order))

    # Build pattern strings per ROI (default to 'n' if a cell is missing a class)
    patterns = []
    for _, row in wide.iterrows():
        code = ''.join(map_letter.get(str(v).lower(), 'n') for v in row.tolist())
        patterns.append(code)

    s = pd.Series(patterns)
    counts = s.value_counts().sort_index()
    total = counts.sum() if counts.sum() else 1
    summary = (pd.DataFrame({'Pattern': counts.index, 'Count': counts.values})
               .assign(Percent=lambda df: 100.0 * df['Count'] / total))

    # Deterministic 3^N lexicographic order for bar labels
    alphabet = ['n','o','s']
    from itertools import product
    ordered_patterns = [''.join(p) for p in product(alphabet, repeat=len(window_order))]

    # ensure all combos present even if zero
    summary = summary.set_index('Pattern').reindex(ordered_patterns, fill_value=0).reset_index()
    summary.rename(columns={'index':'Pattern'}, inplace=True)
    return summary, ordered_patterns


def plot_functional_clusters(pdf_obj, glm_all, stim_order, stat_sel, good_rois, ops, fignum, local_output_dir, central_csv_dir, run_id, timestamp_str, glm_method_tag):
    log("Performing functional clustering for dF/F responses...")
    heat = glm_all[glm_all['type'] == 'dF/F'].pivot(index='ROI_ID', columns='Stimulus', values='Response').reindex(columns=stim_order)
    code = {'activated': 1, 'inhibited': -1, 'unresponsive': 0}
    mat = heat.applymap(lambda x: code.get(x, 0)).fillna(0)
    if mat.shape[0] < 10:
        log("Not enough cells for clustering.")
        return None # Return None if no clusters
        
    Z = linkage(pdist(mat, metric='hamming'), method='ward')
    thr = 0.7 * Z[:, 2].max()
    clusters = fcluster(Z, thr, criterion='distance')
    mat['cluster'] = clusters
    ncl = len(np.unique(clusters))
    title = "Functional Clustering of Neurons (dF/F Responses)"
    if ncl > 1:
        try:
            score = silhouette_score(mat.iloc[:, :-1], clusters, metric='hamming')
            title += f"\nSilhouette Score: {score:.2f}"
        except ValueError:
            title += "\nSilhouette Score: N/A (not enough clusters)"
    sorted_df = mat.sort_values('cluster')
    fig = plt.figure(figsize=(12, 10)); add_figure_title(fig, fignum, title)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0, 0])
    sns.heatmap(sorted_df.drop('cluster', axis=1), cmap=[COLOR_MAP["inhibited"], COLOR_MAP["unresponsive"], COLOR_MAP["activated"]], cbar=False, yticklabels=False, ax=ax0)
    ax0.set_title("Clustered Response Vectors")
    boundaries = np.where(np.diff(sorted_df['cluster']))[0] + 1
    for b in boundaries: ax0.axhline(b, color='white', lw=1.5)
    ax1 = fig.add_subplot(gs[0, 1])
    tune = sorted_df[stim_order + ['cluster']].groupby('cluster').mean()
    sns.heatmap(tune, cmap=GEM_DIV_CMAP, center=0, annot=True, fmt=".2f", ax=ax1)
    ax1.set_title("Average Cluster Response Profile"); ax1.set_ylabel("Cluster ID")
    ax2 = fig.add_subplot(gs[1, :]); ax2.imshow(ops['meanImg'], cmap='gray'); ax2.axis('off'); ax2.set_title("Spatial Location of Clusters")
    roi_to_pos = {r: i for i, r in enumerate(good_rois)}
    colors = plt.cm.get_cmap('tab10', ncl)
    for cid in np.unique(clusters):
        idxs = sorted_df[sorted_df['cluster'] == cid].index
        xs, ys = [], []
        for roi in idxs:
            pos = roi_to_pos.get(roi)
            if pos is None: continue
            cy, cx = stat_sel[pos]['med']
            xs.append(cx); ys.append(cy)
        ax2.scatter(xs, ys, color=colors((cid - 1) / max(1, ncl-1)), s=20, alpha=0.8, label=f"Cluster {cid} (n={len(idxs)})", edgecolors='white', linewidth=0.5)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf_obj.savefig(fig); plt.close(fig)
    
    dfc = pd.DataFrame({'ROI_ID': sorted_df.index, 'cluster_id': sorted_df['cluster']})
    save_df(dfc, f"{run_id}_dff_clusters.csv", local_output_dir, central_csv_dir, glm_method_tag)
    return dfc # Return the dataframe for use in other plots

def export_region_counts_csv(df_region_resp, local_output_dir, central_csv_dir, run_id, glm_method_tag, prefix):
    """
    Summarize, for each anatomical region, how many ROIs are:
      - Activated at least once to any stimulus
      - Inhibited at least once (and never activated)
      - Unresponsive to all stimuli

    Fractions are ROI fractions within each region (0-1).
    Assumes df_region_resp has columns: ROI_ID, region, Simple_Response.
    """
    # Drop 'Outside' so  only report labeled masks/regions
    df = df_region_resp[df_region_resp['region'] != 'Outside'].copy()
    if df.empty:
        return

    # One row per ROI/region (region is constant per ROI)
    roi_region = df[['ROI_ID', 'region']].drop_duplicates()

    # For each ROI, check if it is ever Activated or Inhibited for any stimulus
    flags = (
        df.assign(
            is_act=df['Simple_Response'] == 'Activated',
            is_inh=df['Simple_Response'] == 'Inhibited'
        )
        .groupby('ROI_ID')[['is_act', 'is_inh']]
        .any()
        .reset_index()
    )

    def classify_any(row):
        if row['is_act']:
            return 'Activated'
        if row['is_inh']:
            return 'Inhibited'
        return 'Unresponsive'

    flags['Any_Response'] = flags.apply(classify_any, axis=1)

    merged = roi_region.merge(flags[['ROI_ID', 'Any_Response']], on='ROI_ID', how='left')
    merged['Any_Response'] = merged['Any_Response'].fillna('Unresponsive')

    rows = []
    for region_name, grp in merged.groupby('region'):
        total = len(grp)
        n_act = (grp['Any_Response'] == 'Activated').sum()
        n_inh = (grp['Any_Response'] == 'Inhibited').sum()
        n_unr = (grp['Any_Response'] == 'Unresponsive').sum()

        if total > 0:
            frac_act = n_act / total
            frac_inh = n_inh / total
            frac_unr = n_unr / total
        else:
            frac_act = frac_inh = frac_unr = np.nan

        rows.append({
            "Region": region_name,
            "total_cells_in_mask": total,
            "cells_in_mask_activated": n_act,
            "cells_in_mask_no_response": n_unr,
            "cells_in_mask_inhibited": n_inh,
            "fraction_activated": frac_act,
            "fraction_no_response": frac_unr,
            "fraction_inhibited": frac_inh,
        })

    summary_df = pd.DataFrame(rows)
    save_df(
        summary_df,
        f"{run_id}_{prefix}_anatomical_counts.csv",
        local_output_dir,
        central_csv_dir,
        glm_method_tag
    )
def export_region_counts_by_stim(df_region_resp, local_output_dir, central_csv_dir,
                                 run_id, glm_method_tag, prefix):
    """
    Per-stimulus anatomical table.

    For each Stimulus + Region combo, compute:
      - total_cells_in_mask
      - cells_in_mask_activated
      - cells_in_mask_no_response
      - cells_in_mask_inhibited
      - fraction_* (within that region for that stimulus)

    Assumes df_region_resp has columns:
      ROI_ID, Stimulus, region, Simple_Response
    """
    # Ignore unlabeled ROIs
    df = df_region_resp[df_region_resp['region'] != 'Outside'].copy()
    if df.empty:
        return

    rows = []

    # loop over each stimulus (DA_1, DA_2, CHEM1, CHEM2, etc.)
    for stim in sorted(df['Stimulus'].unique()):
        df_stim = df[df['Stimulus'] == stim]

        for region_name, grp in df_stim.groupby('region'):
            # Each ROI appears once per stim already, but be explicit:
            total = grp['ROI_ID'].nunique()

            n_act = grp.loc[grp['Simple_Response'] == 'Activated', 'ROI_ID'].nunique()
            n_inh = grp.loc[grp['Simple_Response'] == 'Inhibited', 'ROI_ID'].nunique()
            n_unr = grp.loc[grp['Simple_Response'] == 'Unresponsive', 'ROI_ID'].nunique()

            if total > 0:
                frac_act = n_act / total
                frac_inh = n_inh / total
                frac_unr = n_unr / total
            else:
                frac_act = frac_inh = frac_unr = np.nan

            rows.append({
                "Stimulus": stim,
                "Region": region_name,
                "total_cells_in_mask": total,
                "cells_in_mask_activated": n_act,
                "cells_in_mask_no_response": n_unr,
                "cells_in_mask_inhibited": n_inh,
                "fraction_activated": frac_act,
                "fraction_no_response": frac_unr,
                "fraction_inhibited": frac_inh,
            })

    summary_df = pd.DataFrame(rows)

    save_df(
        summary_df,
        f"{run_id}_{prefix}_anatomical_counts_by_stim.csv",
        local_output_dir,
        central_csv_dir,
        glm_method_tag,
    )

def analyze_anatomical_regions(pdf_obj, data_folder, df_consensus, stat_sel, good_rois, ops, pattern_df, pattern_counts, chemical_stim_order, fignum_base, local_output_dir, central_csv_dir, run_id, stim_order, timestamp_str, glm_method_tag):
    log("Analyzing anatomical regions...")
    masks = {}
    for mp in data_folder.glob("*Mask.tif"):
        name = mp.stem.replace("Mask", "")
        try:
            img = tifffile.imread(mp)
            masks[name] = img > 0
        except Exception as e:
            log(f"Warning: Could not read mask file {mp}. Skipping. Error: {e}")
            continue
    if not masks:
        log("No valid region masks found; skipping anatomical analysis.")
        return
    
    df_consensus['region'] = 'Outside'
    pos_map = {r: i for i, r in enumerate(good_rois)}
    for roi_id in df_consensus['ROI_ID'].unique():
        pos = pos_map.get(roi_id)
        if pos is None: continue
        cy, cx = map(int, stat_sel[pos]['med'])
        for nm, m in masks.items():
            if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and m[cy, cx]:
                df_consensus.loc[df_consensus['ROI_ID'] == roi_id, 'region'] = nm
                break
    
    region_df = df_consensus[['ROI_ID', 'region']].drop_duplicates()
    save_df(region_df, f"{run_id}_anatomical_regions.csv", local_output_dir, central_csv_dir, glm_method_tag)
    
    df_consensus['Simple_Response'] = df_consensus['Category'].replace({
        'activated': 'Activated', 'inhibited': 'Inhibited', 'unresponsive': 'Unresponsive'
    })
    export_region_counts_csv(
        df_region_resp=df_consensus,
        local_output_dir=local_output_dir,
        central_csv_dir=central_csv_dir,
        run_id=run_id,
        glm_method_tag=glm_method_tag,
        prefix="mask"
    )

    export_region_counts_by_stim(
        df_region_resp=df_consensus,
        local_output_dir=local_output_dir,
        central_csv_dir=central_csv_dir,
        run_id=run_id,
        glm_method_tag=glm_method_tag,
        prefix="mask"
    )
    

    fig, ax = plt.subplots(figsize=(10, 8)); 
    add_figure_title(fig, f"{fignum_base}A", "Spatial Distribution of Top 10 Chemical Response Patterns")
    ax.imshow(ops['meanImg'], cmap='gray'); ax.axis('off')
    cmap_regions = plt.cm.get_cmap('tab10', len(masks))
    for i, (nm, m) in enumerate(masks.items()):
        ax.contour(m, levels=[0.5], colors=[cmap_regions(i)], linewidths=2, label=nm)
    
    plot_pattern_overlay_logic(ax, pattern_df, pattern_counts, chemical_stim_order, ops, stat_sel, pos_map)
    
    fig.tight_layout(rect=[0, 0, 0.8, 0.95])
    pdf_obj.savefig(fig); plt.close(fig)

    per_page = 6
    for i in range(0, len(stim_order), per_page):
        subs = stim_order[i:i + per_page]
        fig, axes = plt.subplots(3, 2, figsize=(10, 12)); fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
        add_figure_title(fig, f"{fignum_base}B", "Cell Counts per Region for Each Stimulus")
        ax_list = axes.ravel()
        for j, stim in enumerate(subs):
            ax = ax_list[j]
            tbl = pd.crosstab(df_consensus[df_consensus['Stimulus'] == stim]['region'], df_consensus[df_consensus['Stimulus'] == stim]['Simple_Response'])
            for cat in ['Activated', 'Inhibited', 'Unresponsive']:
                if cat not in tbl: tbl[cat] = 0
            tbl = tbl[['Activated', 'Inhibited', 'Unresponsive']]
            pct = tbl.div(tbl.sum(axis=1), axis=0).fillna(0) * 100
            labels = tbl.astype(str) + "\n(" + pct.round(1).astype(str) + "%)"
            sns.heatmap(tbl, annot=labels, fmt='s', cmap='mako', ax=ax)
            ax.set_title(stim); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for k in range(len(subs), len(ax_list)): ax_list[k].axis('off')
        pdf_obj.savefig(fig); plt.close(fig)
    
    raw = []
    for stim in stim_order:
        ct = pd.crosstab(df_consensus[df_consensus['Stimulus'] == stim]['region'], df_consensus[df_consensus['Stimulus'] == stim]['Simple_Response'])
        if ct.shape[0] < 2 or ct.shape[1] < 2: continue
        chi2, p, _, exp = chi2_contingency(ct)
        raw.append((stim, chi2, p, ct, exp))
    if raw:
        pvals = [r[2] for r in raw]
        rej, padj, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method='fdr_bh')
        results, sigs = [], []
        for i, (stim, chi2, p, ct, exp) in enumerate(raw):
            sig = rej[i]
            results.append([stim, f"{chi2:.2f}", f"{padj[i]:.4f}", "Yes" if sig else "No"])
            if sig:
                res = (ct - exp) / np.sqrt(exp + 1e-8)
                sigs.append((stim, res))
        
        fig_chi, ax_chi = plt.subplots(figsize=(8.5, 5))
        add_figure_title(fig_chi, f"{fignum_base}C", "Statistical Test for Anatomical Enrichment (FDR Corrected)")
        ax_chi.axis('off')
        tbl_df = pd.DataFrame(results, columns=["Stimulus", "Chi-Squared", "p-value (adj)", "Significant"])
        table = ax_chi.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
        fig_chi.tight_layout()
        pdf_obj.savefig(fig_chi); plt.close(fig_chi)

        if sigs:
            add_text_page_to_pdf(pdf_obj, "Post-Hoc Analysis of Anatomical Enrichment", ["Standardized residuals >1.96 (warm)=enrichment, <-1.96 (cool)=depletion."], fignum=f"{fignum_base}D")
            for stim, res in sigs:
                fig, ax = plt.subplots(figsize=(8, 6)); add_figure_title(fig, f"{fignum_base}D (cont.)", f"Post-Hoc Residuals for {stim}")
                sns.heatmap(res, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Standardized Residuals")
                pdf_obj.savefig(fig); plt.close(fig)

def analyze_outer_inner_regions(pdf_obj, data_folder, df_consensus, stat_sel, good_rois, ops, pattern_df, pattern_counts, chemical_stim_order, fignum_base, local_output_dir, central_csv_dir, run_id, timestamp_str, stim_order, glm_method_tag):
    log("Checking for PeriphRegion and InnerRegion masks...")
    masks = {}
    for mp in data_folder.glob("*Region.tif"):
        nm = mp.stem.replace("Region", "")
        if nm in ("Periph", "Inner"):
            try:
                masks[nm] = tifffile.imread(mp) > 0
            except Exception: continue
    if set(masks) != {"Periph", "Inner"}:
        log("Periph/Inner masks missing; skipping Periph vs. Inner analysis.")
        return
    
    roi_to_region, pos_map = {}, {r: i for i, r in enumerate(good_rois)}
    for roi in good_rois:
        pos = pos_map.get(roi)
        if pos is None: continue
        cy, cx = map(int, stat_sel[pos]['med'])
        roi_to_region[roi] = "Periph" if masks['Periph'][cy, cx] else "Inner" if masks['Inner'][cy, cx] else "Outside"
    
    sub = df_consensus.copy()
    sub['region'] = sub['ROI_ID'].map(roi_to_region)
    sub = sub[sub['region'].isin(["Periph", "Inner"])]
    sub['Simple_Response'] = sub['Category'].replace({
        'activated': 'Activated', 'inhibited': 'Inhibited', 'unresponsive': 'Unresponsive'
    })

    # Export pooled anatomical counts/fractions for Periph vs Inner regions
    export_region_counts_csv(
        df_region_resp=sub,
        local_output_dir=local_output_dir,
        central_csv_dir=central_csv_dir,
        run_id=run_id,
        glm_method_tag=glm_method_tag,
        prefix="periph_inner"
    )

    fig, ax = plt.subplots(figsize=(8.5, 6))
    add_figure_title(fig, f"{fignum_base}A", "Spatial Distribution of Top 10 Patterns (Periph vs. Inner)")
    ax.imshow(ops['meanImg'], cmap='gray'); ax.axis('off')
    
    cmap_regions = {'Periph': 'cyan', 'Inner': 'magenta'}
    for nm, m in masks.items():
        ax.contour(m, levels=[0.5], colors=[cmap_regions[nm]], linewidths=2)

    plot_pattern_overlay_logic(ax, pattern_df, pattern_counts, chemical_stim_order, ops, stat_sel, pos_map)

    fig.tight_layout(rect=[0, 0, 0.8, 0.95])
    pdf_obj.savefig(fig)
    plt.close(fig)
    
    stimuli = sorted(sub['Stimulus'].unique())
    per_page = 6
    for i in range(0, len(stimuli), per_page):
        subs = stimuli[i:i + per_page]
        fig, axes = plt.subplots(3, 2, figsize=(10, 12)); fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
        add_figure_title(fig, f"{fignum_base}B", "Periph vs Inner Cell Counts")
        ax_list = axes.ravel()
        for j, stim in enumerate(subs):
            ax = ax_list[j]
            ct = pd.crosstab(sub[sub['Stimulus'] == stim]['region'], sub[sub['Stimulus'] == stim]['Simple_Response'])
            for cat in ['Activated', 'Inhibited', 'Unresponsive']:
                if cat not in ct: ct[cat] = 0
            ct = ct[['Activated', 'Inhibited', 'Unresponsive']]
            pct = ct.div(ct.sum(axis=1), axis=0).fillna(0) * 100
            labels = ct.astype(str) + "\n(" + pct.round(1).astype(str) + "%)"
            sns.heatmap(ct, annot=labels, fmt='s', cmap='viridis', ax=ax)
            ax.set_title(stim); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for k in range(len(subs), len(ax_list)): ax_list[k].axis('off')
        pdf_obj.savefig(fig); plt.close(fig)
    
    raw = []
    for stim in stimuli:
        ct = pd.crosstab(sub[sub['Stimulus'] == stim]['region'], sub[sub['Stimulus'] == stim]['Simple_Response'])
        if ct.shape[0] < 2 or ct.shape[1] < 2: continue
        chi2, p, _, exp = chi2_contingency(ct)
        raw.append((stim, chi2, p, ct, exp))
    if not raw:
        log("No valid chi-squared tests for Periph vs Inner"); return
    pvals = [r[2] for r in raw]
    rej, padj, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method='fdr_bh')
    results, sigs = [], []
    for i, (stim, chi2, p, ct, exp) in enumerate(raw):
        sig = rej[i]
        results.append([stim, f"{chi2:.2f}", f"{padj[i]:.4f}", "Yes" if sig else "No"])
        if sig: sigs.append((stim, (ct - exp) / np.sqrt(exp + 1e-8)))
    
    fig_stats, ax_stats = plt.subplots(figsize=(8.5, 5))
    add_figure_title(fig_stats, f"{fignum_base}C", "Periph vs Inner: Chi-Squared Summary")
    ax_stats.axis('off')
    tbl_df = pd.DataFrame(results, columns=["Stimulus", "Chi-Squared", "p-value (adj)", "Significant"])
    table = ax_stats.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    fig_stats.tight_layout()
    pdf_obj.savefig(fig_stats)
    plt.close(fig_stats)
    
    if sigs:
        add_text_page_to_pdf(pdf_obj, "Post-Hoc Analysis of Periph vs. Inner", ["Standardized residuals >1.96 (warm)=enrichment, <-1.96 (cool)=depletion."], fignum=f"{fignum_base}D")
        for stim, res in sigs:
            fig, ax = plt.subplots(figsize=(6, 4)); add_figure_title(fig, f"{fignum_base}D (cont.)", f"Post-Hoc Residuals for {stim}")
            sns.heatmap(res, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax); ax.set_title("Standardized Residuals")
            pdf_obj.savefig(fig); plt.close(fig)

# -----------------------------------------------------------------------------
# CSV SAVING
# -----------------------------------------------------------------------------
def save_all_csv_outputs(exploratory_df, glm_all, glm_consensus, peak_df,
                         stim_order, local_output_dir, central_csv_dir, run_id, timestamp_str, glm_method_tag):
    log("--- Saving all CSV outputs ---")
    if RUN_PEAK_THRESHOLD_ANALYSIS:
        save_df(peak_df, f"{run_id}_peak_threshold_results.csv", local_output_dir, central_csv_dir, glm_method_tag)
        peak_summary = peak_df.groupby('Stimulus')['Response'].value_counts().unstack(fill_value=0)
        save_df(peak_summary.reset_index(), f"{run_id}_peak_threshold_summary.csv", local_output_dir, central_csv_dir, glm_method_tag)

    glm_wide = glm_all.pivot_table(index='ROI_ID', columns=['Stimulus'], values=['Response', 'p_adj', 'f_stat', 'beta_sum', 'center_of_mass_s', 'time_to_peak_s'], aggfunc='first').fillna(0)
    glm_wide.columns = [f"{val}_{stim}" for val, stim in glm_wide.columns]
    
    consensus_wide = glm_consensus.pivot(index='ROI_ID', columns='Stimulus', values='Category').add_suffix('_glm_category')
    
    df_joint = exploratory_df.merge(glm_wide, on='ROI_ID', how='outer').merge(consensus_wide, on='ROI_ID', how='outer')
    save_df(df_joint, f"{run_id}_joint_analysis_results.csv", local_output_dir, central_csv_dir, glm_method_tag)
    
    if stim_order:
        current, name_map = [], {}
        for stim in stim_order:
            current.append(stim); name_map[stim] = "+".join(current)
        df_cum = glm_all.copy(); df_cum['Stimulus'] = df_cum['Stimulus'].map(name_map); df_cum = df_cum.dropna(subset=['Stimulus'])
        save_df(df_cum, f"{run_id}_cumulative_stim_results.csv", local_output_dir, central_csv_dir, glm_method_tag)

# -----------------------------------------------------------------------------
# PDF REPORT GENERATION
# -----------------------------------------------------------------------------
def plot_clustered_trends(pdf_obj, glm_all, stim_order, fignum):
    fig, ax = plt.subplots(1, 1, figsize=(8, 11))
    add_figure_title(fig, fignum, "Clustered Continuous Response Trends (GLM Beta Sum, dF/F)")
    
    mat = glm_all[glm_all['type'] == 'dF/F'].pivot(index='ROI_ID', columns='Stimulus', values='beta_sum').reindex(columns=stim_order).fillna(0)
    if not mat.empty and mat.shape[0] > 1:
        Z = linkage(pdist(mat, metric='euclidean'), method='ward')
        row_order = leaves_list(Z)
        mat_sorted = mat.iloc[row_order]
        vmax = np.percentile(np.abs(mat.values), 98)
        sns.heatmap(mat_sorted, cmap=GEM_DIV_CMAP, center=0, vmax=vmax, vmin=-vmax, yticklabels=False, ax=ax)
    ax.set_title("dF/F")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf_obj.savefig(fig); plt.close(fig)

def plot_consensus_summary_table(ax, consensus_df, stim_order, fignum, title_suffix):
    add_figure_title(ax.figure, fignum, f"Response Summary ({title_suffix})"); ax.axis('off')
    summary = consensus_df.groupby('Stimulus')['Response'].value_counts().unstack(fill_value=0).reindex(stim_order)
    n_rois = len(consensus_df['ROI_ID'].unique())
    
    col_map = {"activated": "Activated", "inhibited": "Inhibited", "unresponsive": "Unresponsive"}
    table_data, col_labels = [], ["Stimulus"]
    
    for cat in col_map.keys():
        if cat not in summary.columns: summary[cat] = 0
    
    for cat_key, cat_name in col_map.items():
        col_labels.append(cat_name)
        
    for stim, row in summary.iterrows():
        row_data = [stim]
        for cat_key in col_map.keys():
            count = row.get(cat_key, 0)
            percent = 100 * count / n_rois
            row_data.append(f"{count} ({percent:.1f}%)")
        table_data.append(row_data)
        
    tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1.2, 1.5)

def plot_peak_threshold_bar_charts(pdf_obj, peak_df, stim_order, fignum):
    fig, ax = plt.subplots(figsize=(8, 6))
    add_figure_title(fig, fignum, "ROI Response Fractions (Peak-Threshold Method, dF/F)")
    
    summary = peak_df.groupby('Stimulus')['Response'].value_counts().unstack(fill_value=0).reindex(stim_order)
    for cat in ['activated', 'inhibited', 'unresponsive']:
        if cat not in summary.columns: summary[cat] = 0

    fracs = summary.div(summary.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(fracs))
    for resp, color in zip(['activated', 'inhibited', 'unresponsive'], [COLOR_MAP['exc'], COLOR_MAP['inh'], COLOR_MAP['unresp']]):
        bars = ax.bar(fracs.index, fracs[resp], bottom=bottom, color=color, label=resp)
        for bar, count, pct in zip(bars, summary[resp], fracs[resp]):
            if count > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2, f"{int(count)}\n({pct:.1f}%)", ha='center', va='center', fontsize=8)
        bottom += fracs[resp].values
    ax.set_title("dF/F Responses"); ax.set_ylabel("Fraction of ROIs (%)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf_obj.savefig(fig); plt.close(fig)

def plot_response_patterns(ax, data_df, stim_order, title_suffix, chemical_only=False):
    stims_to_plot = [s for s in stim_order if not s.startswith("ASW")] if chemical_only else stim_order
    if not stims_to_plot:
        log("Not enough chemical stimuli to generate chemical-only response patterns.")
        ax.text(0.5, 0.5, "Not enough chemical stimuli to plot.", ha='center', va='center')
        ax.axis('off')
        return None, None

    patterns_df = data_df.pivot(index='ROI_ID', columns='Stimulus', values='Response').reindex(columns=stims_to_plot).fillna('unresponsive')
    sign_map = {
        'activated': OUTPUT_SYMBOLS['activated'],
        'inhibited': OUTPUT_SYMBOLS['inhibited'],
        'unresponsive': OUTPUT_SYMBOLS['unresponsive']
    }

    # Build pattern strings using the configured output symbols; default to "no response"
    patterns = patterns_df.apply(
        lambda row: ''.join(row.map(sign_map).fillna(OUTPUT_SYMBOLS['unresponsive'])),
        axis=1
    )
    pattern_counts = patterns.value_counts().reset_index()
    pattern_counts.columns = ['Pattern', 'Count']
    pattern_counts['Percent'] = 100 * pattern_counts['Count'] / len(patterns)
    
    pattern_counts_to_plot = pattern_counts.head(25)
    title_str = title_suffix
    if len(pattern_counts) > 25:
        title_str += " (Top 25 Patterns Shown)"

    bars = ax.bar(pattern_counts_to_plot['Pattern'], pattern_counts_to_plot['Count'], color='skyblue')
    for bar, count, pct in zip(bars, pattern_counts_to_plot['Count'], pattern_counts_to_plot['Percent']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{count}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8)
    
    ax.set_title(title_str)
    ax.set_xlabel('Response Pattern (E: excitation, i: inhibition, x: no response)')
    ax.set_ylabel('Number of ROIs')
    ax.tick_params(axis='x', rotation=45)
    
    if not pattern_counts_to_plot.empty:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    
    key_text = "Pattern Key (Position → Stimulus):\n" + "\n".join([f"{i+1} → {lbl}" for i, lbl in enumerate(stims_to_plot)])
    ax.text(1.02, 0.5, key_text, transform=ax.transAxes, fontsize=8, va='center', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5))
    
    return patterns.reset_index(name='Pattern'), pattern_counts

def plot_pattern_overlay_logic(ax, pattern_df, pattern_counts, chemical_stim_order, ops, stat_sel, pos_map):
    """Helper function containing the core logic for pattern overlay plots."""
    if pattern_df is None or pattern_counts is None or pattern_counts.empty:
        ax.text(0.5, 0.5, "No patterns to display.", ha='center', va='center')
        return

    top_patterns = pattern_counts.head(10)
    top_map = {patt['Pattern']: i for i, patt in top_patterns.iterrows()}
    colors = plt.cm.get_cmap('tab10', max(10, len(top_map)))

    roi_to_pattern = pattern_df.set_index('ROI_ID')['Pattern']
    
    other_rois = [roi for roi, patt in roi_to_pattern.items() if patt not in top_map]
    xs_other, ys_other = [], []
    for roi_id in other_rois:
        p = pos_map.get(roi_id)
        if p is not None:
            cy, cx = stat_sel[p]['med']
            xs_other.append(cx); ys_other.append(cy)
    ax.scatter(xs_other, ys_other, s=15, c='gray', alpha=0.3, label=f"Other Patterns (N={len(other_rois)})")

    legend_handles = [Patch(color='gray', alpha=0.3, label=f"Other Patterns (N={len(other_rois)})")]
    for _, row in top_patterns.iterrows():
        patt, count = row['Pattern'], row['Count']
        color = colors(top_map[patt])
        rois_in_patt = roi_to_pattern[roi_to_pattern == patt].index
        xs, ys = [], []
        for roi_id in rois_in_patt:
            p = pos_map.get(roi_id)
            if p is not None:
                cy, cx = stat_sel[p]['med']
                xs.append(cx); ys.append(cy)
        ax.scatter(xs, ys, s=20, c=[color], edgecolors='white', linewidth=0.3, alpha=0.9)
        legend_handles.append(Patch(color=color, label=f"'{patt}' (N={count})"))

    key_text = "Pattern Key:\n" + "\n".join([f"Pos {i+1}: {lbl}" for i, lbl in enumerate(chemical_stim_order)])
    legend_title = f"Top 10 Patterns\n\n{key_text}"
    ax.legend(handles=legend_handles, title=legend_title, fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')

def plot_pattern_overlay(pdf_obj, fignum, pattern_df, pattern_counts, chemical_stim_order, ops, stat_sel, pos_map, title_suffix):
    fig, ax = plt.subplots(figsize=(10, 8))
    add_figure_title(fig, fignum, f"Spatial Overlay of Top 10 Chemical Response Patterns ({title_suffix})")
    ax.imshow(ops['meanImg'], cmap='gray'); ax.axis('off')
    
    plot_pattern_overlay_logic(ax, pattern_df, pattern_counts, chemical_stim_order, ops, stat_sel, pos_map)

    fig.tight_layout(rect=[0, 0, 0.8, 0.95])
    pdf_obj.savefig(fig); plt.close(fig)

def reformat_glm_patterns(df, stim_order):
    """
    Reformat glm_patterns dataframe into wide format with per-stim columns
    and a final combined pattern column.
    Uses OUTPUT_SYMBOLS mapping instead of +, -, 0.
    """
    if df is None or df.empty:
        return None

    wide = df.pivot(index="ROI_ID", columns="Stimulus", values="Response").reindex(columns=stim_order)
    wide = wide.replace(OUTPUT_SYMBOLS)

    # Rename columns
    col_map = {stim: f"Stim {i+1} [{stim}] Response" for i, stim in enumerate(wide.columns)}
    wide.rename(columns=col_map, inplace=True)

    # Add combined pattern column
    wide["Combined Stim Pattern"] = wide.apply(lambda row: ''.join(row.fillna("x")), axis=1)

    wide.reset_index(inplace=True)  # restore ROI_ID as a column
    return wide



def generate_pdf_report(pdf_path, run_id, data_folder, dff, ops, good_rois, stat_sel,
                        stim_frame_labels, peak_df, glm_all, glm_consensus_df,
                        exploratory_df, sparseness_ts, local_output_dir, central_csv_dir, timestamp_str, glm_method_tag):
    log("--- Generating PDF Report (dF/F Only) ---")
    stim_order = [lbl for lbl, _ in stim_frame_labels]
    chemical_stim_order = [s for s in stim_order if not s.startswith("ASW")]
    n_cells = dff.shape[0]
    pos_map = {r: i for i, r in enumerate(good_rois)}

    with PdfPages(str(pdf_path)) as PDF:
        # --- METADATA PAGES ---
        run_folder = next((p for p in data_folder.parents if "_zout_" in p.name), None)
        abs_data_folder = data_folder.resolve(); tif_filepath = "N/A"
        if run_folder:
            tif_try = run_folder / f"{run_id}_zcorrstack.tif"
            if tif_try.exists(): tif_filepath = str(tif_try.resolve())
        species_map = {'obo':'Octopus bocki', 'obocki':'Octopus bocki', 'obi':'Octopus bimaculoides', 'eberryi':'Euprymna berryi', 'ovu':'Octopus vulgaris'}
        key = re.match(r'^([a-zA-Z]+)', run_id)
        species = species_map.get(key.group(1).lower(), "Unknown") if key else "Unknown"
        sr = re.search(r'_S(\d+)R(\d+)_', run_id)
        slice_num, recording_num = (sr.group(1), sr.group(2)) if sr else ("Unknown","Unknown")
        date_md = re.search(r'_(\d{4}-\d{2}-\d{2})_', run_id)
        recording_date = date_md.group(1) if date_md else "Unknown"
        stim_summary = ", ".join(f"{lbl} (frame {fr})" for lbl,fr in stim_frame_labels)
        total_sig_dff = int(glm_all[glm_all['is_sig']].shape[0])
        
        metadata_lines_p1 = [
            "METADATA (Page 1/2)", f"Run ID:            {run_id}", f"Data Folder:       {abs_data_folder}", f"Analysis Script:   {Path(__file__).name}", "",
            "Experiment Details:", f"    Species:             {species}", f"    Slice #:             {slice_num}", f"    Recording #:         {recording_num}",
            f"    Recording Date:      {recording_date}", f"    Stimuli Applied:     {stim_summary}", "",
            "ROI & Trace Processing:", f"    Total ROIs:          {n_cells}", f"    Frame Rate (Hz):     {ops['fs']:.3f}", "",
            "Per-Cell GLM Findings (dF/F only):", f"    dF/F tests:          {len(glm_all)} (significant: {total_sig_dff})",
        ]
        add_text_page_to_pdf(PDF, "Analysis Metadata", metadata_lines_p1, fontsize=8)

        config_settings = [
            "CONFIGURATION SETTINGS (Page 2/2)",
            f"    GLM Method:                 {GLM_METHOD}",
            f"    Chem Local Baseline (s):     {CHEM_LOCAL_BASELINE_SECONDS}",
            f"    Run Peak-Threshold Analysis: {RUN_PEAK_THRESHOLD_ANALYSIS}",
            f"    LOWESS Detrending:          {APPLY_LOWESS_DETREND}",
            f"    Global Signal Reg.:         {USE_GLOBAL_REGRESSION}",
            f"    Noise-based Threshold:      {USE_NOISE_BASED_THRESHOLD}",
            f"    Noise Multiplier:           {NOISE_MULTIPLIER}",
            f"    ASW Gating:                 {USE_ASW_EFFECT_SIZE_CONTROL}",
            f"    ASW Fold Threshold:         {ASW_FOLD_THRESHOLD}",
            f"    Neuropil Subtraction:       {'on' if USE_NEUROPIL_SUBTRACTION else 'off'}",
            f"    Neuropil alpha / source:    {ops.get('neuropil_alpha_used', 'NA')}  [{ops.get('neuropil_alpha_note','NA')}]",
            "", "Stimulus Window Config:",
            f"    ASW Window (s):             {STIM_CONFIG['ASW']['window_seconds']}",
            f"    ASW Bin Size (s):           {STIM_CONFIG['ASW']['bin_size_seconds']}",                
            f"    ASW Delay (s):              {STIM_CONFIG['ASW'].get('delay_seconds', 0)}",
            f"    Default Window (s):         {STIM_CONFIG['default']['window_seconds']}",
            f"    Default Bin Size (s):       {STIM_CONFIG['default']['bin_size_seconds']}",
            f"    Default Delay (s):          {STIM_CONFIG['default'].get('delay_seconds', 0)}",
            f"    dF/F baseline frames:       {ops.get('dff_baseline_frames','Stim1→Stim2')}",
            "", "Fire Fraction Config:",
            f"    dF/F Multiplier:            {FIRE_FRACTION_DFF_MULTIPLIER}",
        ]
        add_text_page_to_pdf(PDF, "Configuration Settings", config_settings, fontsize=8)

                # --- SAVE METADATA JSON for aggregation ---
        meta = {
            "run_id": run_id,
            "data_folder": str(abs_data_folder),
            "analysis_script": Path(__file__).name,
            "species": species,
            "slice_num": slice_num,
            "recording_num": recording_num,
            "recording_date": recording_date,
            "stimuli_applied": stim_summary,
            "total_rois": int(n_cells),
            "frame_rate_hz": float(ops['fs']),
            "glm_tests_total": int(len(glm_all)),
            "glm_tests_significant": int(total_sig_dff),

            "GLM_METHOD": GLM_METHOD,
            "CHEM_LOCAL_BASELINE_SECONDS": CHEM_LOCAL_BASELINE_SECONDS,
            "RUN_PEAK_THRESHOLD_ANALYSIS": RUN_PEAK_THRESHOLD_ANALYSIS,
            "APPLY_LOWESS_DETREND": APPLY_LOWESS_DETREND,
            "USE_GLOBAL_REGRESSION": USE_GLOBAL_REGRESSION,
            "USE_NOISE_BASED_THRESHOLD": USE_NOISE_BASED_THRESHOLD,
            "NOISE_MULTIPLIER": NOISE_MULTIPLIER,
            "USE_ASW_EFFECT_SIZE_CONTROL": USE_ASW_EFFECT_SIZE_CONTROL,
            "ASW_FOLD_THRESHOLD": ASW_FOLD_THRESHOLD,
            "ASW_window_seconds": STIM_CONFIG['ASW']['window_seconds'],
            "ASW_bin_size_seconds": STIM_CONFIG['ASW']['bin_size_seconds'],
            "ASW_delay_seconds": STIM_CONFIG['ASW'].get('delay_seconds', 0),
            "default_window_seconds": STIM_CONFIG['default']['window_seconds'],
            "default_bin_size_seconds": STIM_CONFIG['default']['bin_size_seconds'],
            "default_delay_seconds": STIM_CONFIG['default'].get('delay_seconds', 0),
            "FIRE_FRACTION_DFF_MULTIPLIER": FIRE_FRACTION_DFF_MULTIPLIER,
        }
        save_json(meta, f"{run_id}_metadata.json", local_output_dir, central_csv_dir, glm_method_tag)


        # --- FIGURE 1: OVERALL ACTIVITY OVERVIEW ---
        fig1A, ax1A = plt.subplots(figsize=(12, 8))

        if FIG1A_SORT_MODE.lower() == "roi":
            # Sort rows by ROI_ID (numerically by the Suite2p ROI indices kept in good_rois)
            order = np.argsort(good_rois)
            title = "Fluorescence Intensity of All ROIs Over Time (ROI-ID Sorted)"
            ylab  = "ROI (sorted by ROI ID)"
        else:
            # Default: sort by each ROI's peak-time in its trace
            peak_times = np.argmax(dff, axis=1)
            order = np.argsort(peak_times)
            title = "Fluorescence Intensity of All ROIs Over Time (Peak-time Sorted)"
            ylab  = "ROI (sorted by peak time)"

        add_figure_title(fig1A, "1A", title)
        p5, p95 = np.percentile(dff, [5, 95])
        im = ax1A.imshow(dff[order], aspect='auto', interpolation='nearest',
                        cmap='magma', vmin=p5, vmax=p95)
        fig1A.colorbar(im, ax=ax1A, pad=0.02, label="ΔF/F")
        ax1A.set_xlabel("Frame")
        ax1A.set_ylabel(ylab)
        PDF.savefig(fig1A)
        plt.close(fig1A)


        # --- FIGURE 2: GLM-BASED ANALYSIS ---
        fig2A = plt.figure(figsize=(8.5, 5))
        ax2A = fig2A.add_subplot(111)
        plot_consensus_summary_table(ax2A, glm_all, stim_order, "2A", "GLM Method")
        PDF.savefig(fig2A); plt.close(fig2A)
        
        fig2B, ax2B = plt.subplots(figsize=(8, 6)); add_figure_title(fig2B, "2B", "ROI Response Fractions (GLM Method, dF/F)")
        cnt = glm_all.groupby(['Stimulus', 'Response']).size().unstack(fill_value=0).reindex(stim_order)
        frac = cnt.div(cnt.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(frac))
        for resp, color in zip(['activated', 'inhibited', 'unresponsive'], [COLOR_MAP["activated"], COLOR_MAP["inhibited"], COLOR_MAP["unresponsive"]]):
            if resp in frac.columns:
                bars = ax2B.bar(frac.index, frac[resp], bottom=bottom, color=color, label=resp)
                for bar, ct, pct in zip(bars, cnt[resp], frac[resp]):
                    if ct > 0: ax2B.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{int(ct)}\n({pct:.1f}%)", ha='center', va='center', fontsize=8)
                bottom += frac[resp].values
        ax2B.set_title("dF/F"); ax2B.set_ylabel("Fraction of ROIs (%)"); plt.setp(ax2B.get_xticklabels(), rotation=45, ha='right')
        fig2B.tight_layout(rect=[0, 0, 1, 0.95]); PDF.savefig(fig2B); plt.close(fig2B)

 # --- FIGURE 2C: DETAILED OVERLAYS (robust categories) ---
        for i in range(0, len(stim_order), 2):
            sub = stim_order[i:i + 2]
            fig = plt.figure(figsize=(8.5, 5 * len(sub)))
            add_figure_title(fig, "2C", "Detailed Overlays of ROI Response Categories (GLM, dF/F)")
            gs = gridspec.GridSpec(len(sub), 1, hspace=0.4)

            for j, stim in enumerate(sub):
                ax = fig.add_subplot(gs[j, 0])
                ax.imshow(ops['meanImg'], cmap='gray')
                ax.axis('off')

                # be tolerant to either 'Category' or 'Response'
                df_stim = glm_consensus_df[glm_consensus_df['Stimulus'] == stim].copy()
                if 'Category' not in df_stim.columns and 'Response' in df_stim.columns:
                    df_stim['Category'] = df_stim['Response']

                # normalize labels 
                df_stim['CategoryNorm'] = df_stim['Category'].apply(_normalize_category)

                legend_handles = []
                for cat, dfc in df_stim.groupby('CategoryNorm', dropna=True):
                    col = COLOR_MAP.get(cat, COLOR_MAP['unresponsive'])
                    xs, ys = [], []
                    for roi in dfc['ROI_ID']:
                        p = pos_map.get(roi)
                        if p is None:
                            continue
                        cy, cx = stat_sel[p]['med']
                        xs.append(cx); ys.append(cy)

                    # draw (skip if empty)
                    if xs:
                        ax.scatter(xs, ys, s=15, c=col, edgecolors='white', linewidth=0.2, alpha=0.85, zorder=3)

                    count = len(dfc)
                    percent = 100 * count / n_cells if n_cells else 0.0
                    pretty = {"activated":"Activated","inhibited":"Inhibited","unresponsive":"Unresponsive"}[cat]
                    legend_handles.append(Patch(color=col, label=f"{pretty} ({count}, {percent:.1f}%)"))

                ax.set_title(f"Responses to {stim}", y=1.05)

                if legend_handles:
                    # fixed order: Activated, Inhibited, Unresponsive
                    order = {"Activated":0, "Inhibited":1, "Unresponsive":2}
                    legend_handles.sort(key=lambda h: order.get(h.get_label().split()[0], 99))
                    ax.legend(
                        handles=legend_handles,
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.05),
                        ncol=3,
                        fontsize="small",
                        frameon=False
                    )

            fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])
            PDF.savefig(fig)
            plt.close(fig)


        fig2D, ax2D = plt.subplots(1, 1, figsize=(18, 6)); fig2D.subplots_adjust(top=0.85, right=0.8)
        add_figure_title(fig2D, "2D", "Cell Counts by Response Pattern (GLM Method, dF/F, All Stimuli)")
        glm_patt_all, _ = plot_response_patterns(ax2D, glm_all, stim_order, title_suffix="All Stimuli")
        PDF.savefig(fig2D); plt.close(fig2D)
        
        fig2E, ax2E = plt.subplots(1, 1, figsize=(18, 6)); fig2E.subplots_adjust(top=0.85, right=0.8)
        add_figure_title(fig2E, "2E", "Cell Counts by Response Pattern (GLM, dF/F, Chemicals Only)")
        glm_patt_chem, glm_patt_chem_counts = plot_response_patterns(ax2E, glm_all, stim_order, title_suffix="Chemicals Only", chemical_only=True)
        PDF.savefig(fig2E); plt.close(fig2E)
        
        if glm_patt_all is not None:
            reformatted_all = reformat_glm_patterns(glm_all, stim_order)
            if reformatted_all is not None:
                save_df(reformatted_all, f"{run_id}_glm_patterns_all_stim.csv", local_output_dir, central_csv_dir, glm_method_tag)

        if glm_patt_chem is not None:
            chem_order = [s for s in stim_order if not s.startswith("ASW")]
            reformatted_chem = reformat_glm_patterns(glm_all[glm_all['Stimulus'].isin(chem_order)], chem_order)
            if reformatted_chem is not None:
                save_df(reformatted_chem, f"{run_id}_glm_patterns_chemical.csv", local_output_dir, central_csv_dir, glm_method_tag)

        plot_pattern_overlay(PDF, "2F", glm_patt_chem, glm_patt_chem_counts, chemical_stim_order, ops, stat_sel, pos_map, "GLM Method")
        

        ff_rows = calculate_fire_fraction_summary(dff, stim_frame_labels)  # list of [Epoch, "X (Y%)"]

        parsed = []
        for epoch, txt in ff_rows:
            m = re.match(r'\s*(\d+)\s*\(\s*([\d.]+)%\s*\)\s*', str(txt))
            count = int(m.group(1)) if m else None
            pct = float(m.group(2)) if m else None
            parsed.append({"epoch": epoch,
                        "active_cells_count_dff": count,
                        "active_cells_pct_dff": pct,
                        "active_cells_display_dff": str(txt)})

        ff_df = pd.DataFrame(parsed, columns=["epoch","active_cells_count_dff","active_cells_pct_dff","active_cells_display_dff"])
        save_df(ff_df, f"{run_id}_fire_fraction_summary.csv", local_output_dir, central_csv_dir, glm_method_tag)

        fig3A, ax3A = plt.subplots(figsize=(8.5, 2.5)); add_figure_title(fig3A, "3A", "Summary of Active Cell Fractions (dF/F only)")
        tbl3A = ax3A.table(cellText=[[r["epoch"], r["active_cells_display_dff"]] for r in parsed],
                        colLabels=['Epoch', 'Active Cells (dF/F)'], loc='center', cellLoc='center')
        ax3A.axis('off'); PDF.savefig(fig3A); plt.close(fig3A)

        
        window, step = int(30 * ops['fs']), int(10 * ops['fs'])
        t_axis, dff_tc = [], []
        baseline_end_frame = stim_frame_labels[1][1]
        
        sigma_dff_tc = FIRE_FRACTION_DFF_MULTIPLIER * dff[:, :baseline_end_frame].std(axis=1)
        for t in range(0, dff.shape[1] - window + 1, step):
            t_axis.append(t/ops['fs']); dff_tc.append(_compute_frac_dff(dff, sigma_dff_tc, np.arange(t,t+window)))
        
        fig3B, ax3B = plt.subplots(figsize=(12, 4)); add_figure_title(fig3B, "3B", "Fire-Fraction Time-course (dF/F only)")
        ax3B.plot(t_axis,dff_tc); ax3B.set_title("dF/F"); ax3B.set_ylabel("Fraction Active"); ax3B.set_xlabel("Time (s)")
        for _,fr in stim_frame_labels: ax3B.axvline(fr/ops['fs'],color='r',ls='--',lw=1,alpha=0.7)
        PDF.savefig(fig3B); plt.close(fig3B)
        
        fig3C = plt.figure(figsize=(12, 4)); add_figure_title(fig3C, "3C", "Population Sparseness Over Time")
        plt.plot(np.arange(len(sparseness_ts)) / ops['fs'], sparseness_ts, lw=1); plt.xlabel("Time (s)"); plt.ylabel("Sparseness"); plt.ylim(0, 1)
        fig3C.tight_layout(rect=[0.05, 0.05, 1, 0.95])
        PDF.savefig(fig3C); plt.close(fig3C)

        

        # --- FIGURE 4: Post-Stimulus 5-minute Activity (Bars + Table) ---
        try:
            activity_long, activity_wide = classify_activity_windows(dff, stim_frame_labels, ops['fs'], good_rois)

            # proportions table
            order_classes = ["oscillatory", "sporadic", "stable"]
            tbl = (activity_long.groupby(["WindowLabel", "Class"]).size().unstack(fill_value=0))
            for cc in order_classes:
                if cc not in tbl.columns: tbl[cc] = 0
            # Order windows by earliest start frame (occurrence order)
            window_order = list(activity_long.groupby('WindowLabel')["WindowStartFrame"].min().sort_values().index)
            tbl = tbl.reindex(window_order)
            props = (tbl.T / tbl.sum(axis=1)).T.fillna(0.0) * 100.0

            import matplotlib.gridspec as _gs
            fig_ps = plt.figure(figsize=(10.5, 9.5))  
            add_figure_title(fig_ps, "4A", "Post-Stimulus 5-minute Activity (ROI % per window)")
            gs = _gs.GridSpec(3, 1, height_ratios=[2, 1, 2])  


            # Row 1: stacked bars
            ax_bar = fig_ps.add_subplot(gs[0, 0])
            bottoms = np.zeros(len(props))
            x = np.arange(len(props))
            labels = list(props.index)
            for cls in order_classes:
                vals = props[cls].values
                ax_bar.bar(x, vals, bottom=bottoms, label=cls.capitalize())
                bottoms += vals
            ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, rotation=30, ha='right')
            ax_bar.set_ylabel("ROIs (%)")
            ax_bar.legend(loc='upper right', fontsize=8)

            # Row 2: table (counts + percents)
            ax_tbl = fig_ps.add_subplot(gs[1, 0]); ax_tbl.axis('off')
            disp = pd.concat([tbl.add_suffix(" (n)"), props.round(1).add_suffix(" (%)")], axis=1)
            table = ax_tbl.table(cellText=disp.values, colLabels=disp.columns, rowLabels=disp.index,
                                 loc='center', cellLoc='center')
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.3)
            pos_map = {int(r): i for i, r in enumerate(good_rois)}
            cls_colors = {"oscillatory": "#ff8c00", "sporadic": "#6a5acd", "stable": "#555555"}  # NEW: local palette
            sub = _gs.GridSpecFromSubplotSpec(1, len(window_order), subplot_spec=gs[2, 0], wspace=0.05)
            for j, win_lbl in enumerate(window_order):
                ax_win = fig_ps.add_subplot(sub[0, j])
                ax_win.imshow(ops['meanImg'], cmap='gray')
                ax_win.set_title(win_lbl, fontsize=8, pad=4)
                ax_win.axis('off')
                dfw = activity_long[activity_long['WindowLabel'] == win_lbl]
                for cls_name in ["oscillatory", "sporadic", "stable"]:
                    dfc = dfw[dfw['Class'] == cls_name]
                    if dfc.empty:
                        continue
                    xs, ys = [], []
                    for roi in dfc['ROI_ID'].astype(int):
                        idx = pos_map.get(int(roi))
                        if idx is None:
                            continue
                        cy, cx = stat_sel[idx]['med']  # (y, x)
                        xs.append(cx); ys.append(cy)
                    if xs:
                        ax_win.scatter(xs, ys, s=10, c=cls_colors[cls_name],
                                       edgecolors='white', linewidth=0.2, alpha=0.85, zorder=3)
            legend_handles = [Patch(color=cls_colors["oscillatory"], label="Oscillatory"),
                              Patch(color=cls_colors["sporadic"], label="Sporadic"),
                              Patch(color=cls_colors["stable"], label="Stable")]
            fig_ps.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8,
                          bbox_to_anchor=(0.5, 0.02), frameon=False)

            fig_ps.tight_layout(rect=[0, 0.03, 1, 0.94])  
            PDF.savefig(fig_ps); plt.close(fig_ps)

            save_df(activity_long, f"{run_id}_poststim_activity_windows_long.csv", local_output_dir, central_csv_dir, glm_method_tag)
            save_df(activity_wide.reset_index(), f"{run_id}_poststim_activity_windows_wide.csv", local_output_dir, central_csv_dir, glm_method_tag)
            window_order = list(activity_long.groupby('WindowLabel')["WindowStartFrame"]
                                .min().sort_values().index)
            summary_4c, ordered_patterns = _compute_state_combinations(
                activity_long, window_order, use_first_n_windows=3
            )


            save_df(summary_4c, f"{run_id}_Fig4C_state_combinations.csv",
                    local_output_dir, central_csv_dir, glm_method_tag)

            # --- Fig 4C: Bar graph of pattern percentages 
            fig4C, ax4C = plt.subplots(figsize=(12, 4))
            add_figure_title(
                fig4C, "4C",
                "State combinations across first three 5-min windows (n=stable, o=oscillatory, s=sporadic)"
            )

            bars = ax4C.bar(summary_4c['Pattern'], summary_4c['Percent'])
            ax4C.set_ylabel("ROIs (%)")
            ax4C.set_xlabel("Pattern (three windows)")
            plt.setp(ax4C.get_xticklabels(), rotation=45, ha='right')

            for bar, pct in zip(bars, summary_4c['Percent']):
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax4C.text(x, y + max(0.2, 0.01 * max(1.0, summary_4c['Percent'].max())),
                        f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)
            ax4C.set_ylim(0, max(5.0, summary_4c['Percent'].max() * 1.18))

            fig4C.tight_layout(rect=[0, 0, 1, 0.95])
            PDF.savefig(fig4C)
            plt.close(fig4C)


            # --- Fig 4D: Table of counts & percents 
            disp_4d = summary_4c.copy()
            disp_4d['Percent'] = disp_4d['Percent'].map(lambda v: f"{v:.1f}")

            rows_per_page = 15  
            n_rows = len(disp_4d)
            n_pages = int(np.ceil(n_rows / rows_per_page)) if n_rows else 1

            for p in range(n_pages):
                chunk = disp_4d.iloc[p * rows_per_page:(p + 1) * rows_per_page]

                fig4D, ax4D = plt.subplots(figsize=(11, 8.5))
                title_suffix = "" if p == 0 else f" (cont. {p})"
                add_figure_title(fig4D, f"4D{title_suffix}", "State combination table (first three windows)")

                ax4D.axis('off')

                table = ax4D.table(
                    cellText=chunk[['Pattern', 'Count', 'Percent']].values,
                    colLabels=['Pattern', 'Count', 'Percent'],
                    loc='center',
                    cellLoc='center'
                )

                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.1, 1.25)

                for (r, c), cell in table.get_celld().items():
                    if r == 0:
                        cell.set_text_props(weight='bold')
                        cell.set_height(0.06)
                    else:
                        cell.set_height(0.055)

                fig4D.tight_layout(rect=[0, 0, 1, 0.95])
                PDF.savefig(fig4D)
                plt.close(fig4D)


        except Exception as _e:
            log(f"Warning: could not render Fig 4C/4D state combinations: {_e}")
        except Exception as _e:
            log(f"Warning: could not render post-stim activity summary: {_e}")


        all_embedding_dfs = []

        fig4A, axes4A = plt.subplots(1, 3, figsize=(15, 4)); add_figure_title(fig4A, "5A", "Signal Properties and Embeddings")
        axes4A[0].hist(exploratory_df['dff_snr'].dropna(), bins=30); axes4A[0].set_title("ΔF/F SNR")
        axes4A[0].set_xlabel("Signal-to-Noise Ratio (dF/F)"); axes4A[0].set_ylabel("Number of ROIs")
        sc1 = axes4A[1].scatter(exploratory_df['pca_coord_1'], exploratory_df['pca_coord_2'], s=10, c=exploratory_df['dff_snr'], cmap='viridis'); axes4A[1].set_title("PCA of all Traces"); fig4A.colorbar(sc1, ax=axes4A[1], label="dF/F SNR")
        sc2 = axes4A[2].scatter(exploratory_df['tsne_coord_1'], exploratory_df['tsne_coord_2'], s=10, c=exploratory_df['dff_snr'], cmap='viridis'); axes4A[2].set_title("t-SNE of all Traces"); fig4A.colorbar(sc2, ax=axes4A[2], label="dF/F SNR")
        fig4A.tight_layout(rect=[0,0,1,0.95]); PDF.savefig(fig4A); plt.close(fig4A)

        df_4a_embed = exploratory_df[['ROI_ID', 'pca_coord_1', 'pca_coord_2', 'tsne_coord_1', 'tsne_coord_2']].copy()
        df_4a_embed.columns = ['ROI_ID', 'pca_dff_all_traces_1', 'pca_dff_all_traces_2', 'tsne_dff_all_traces_1', 'tsne_dff_all_traces_2']
        all_embedding_dfs.append(df_4a_embed.set_index('ROI_ID'))

        fig4B = plt.figure(figsize=(18, 6)); add_figure_title(fig4B, "5B", "Dimensionality Reduction of Response Fingerprints (GLM Betas, dF/F)")
        gs = gridspec.GridSpec(1, 3)
        feature_vecs, roi_order_for_betas = [], []
        for roi in good_rois:
            betas = glm_all[(glm_all.ROI_ID == roi)].sort_values('Stimulus')['betas'].tolist()
            if betas:
                feature_vecs.append(np.hstack([b for b in betas]))
                roi_order_for_betas.append(roi)
        
        X = np.nan_to_num(np.array(feature_vecs))
        resp_counts = glm_all[glm_all['is_sig']].groupby('ROI_ID').size().reindex(roi_order_for_betas, fill_value=0)
        
        if X.shape[0] > 2 and X.shape[1] > 2:
            pca_coords = PCA(n_components=2).fit_transform(X)
            axP = fig4B.add_subplot(gs[0, 0]); sc = axP.scatter(pca_coords[:,0], pca_coords[:,1], s=10, c=resp_counts, cmap='plasma', vmin=0, vmax=len(stim_order)); axP.set_title("PCA"); fig4B.colorbar(sc, ax=axP, label="# Sig Responses")
            
            tsne_coords = TSNE(n_components=2, perplexity=min(30, X.shape[0] - 1), random_state=0).fit_transform(X)
            axT = fig4B.add_subplot(gs[0, 1]); sc = axT.scatter(tsne_coords[:,0], tsne_coords[:,1], s=10, c=resp_counts, cmap='plasma', vmin=0, vmax=len(stim_order)); axT.set_title("t-SNE"); fig4B.colorbar(sc, ax=axT, label="# Sig Responses")
            
            umap_coords = umap.UMAP(random_state=42).fit_transform(X)
            axU = fig4B.add_subplot(gs[0, 2]); sc = axU.scatter(umap_coords[:,0], umap_coords[:,1], s=10, c=resp_counts, cmap='plasma', vmin=0, vmax=len(stim_order)); axU.set_title("UMAP"); fig4B.colorbar(sc, ax=axU, label="# Sig Responses")

            df_pca = pd.DataFrame(pca_coords, index=roi_order_for_betas, columns=['pca_dff_glm_betas_1', 'pca_dff_glm_betas_2'])
            df_tsne = pd.DataFrame(tsne_coords, index=roi_order_for_betas, columns=['tsne_dff_glm_betas_1', 'tsne_dff_glm_betas_2'])
            df_umap = pd.DataFrame(umap_coords, index=roi_order_for_betas, columns=['umap_dff_glm_betas_1', 'umap_dff_glm_betas_2'])
            all_embedding_dfs.extend([df_pca, df_tsne, df_umap])

        fig4B.tight_layout(rect=[0,0,1,0.96]); PDF.savefig(fig4B); plt.close(fig4B)
        
        if all_embedding_dfs:
           cell_embeddings_df = pd.concat(all_embedding_dfs, axis=1).reset_index().rename(columns={'index': 'ROI_ID'})
           cell_embeddings_df = cell_embeddings_df.merge(glm_consensus_df[['ROI_ID', 'Category']].rename(columns={'Category': 'glm_category'}), on='ROI_ID', how='left')
           cell_embeddings_df = cell_embeddings_df.merge(exploratory_df[['ROI_ID', 'dff_snr']], on='ROI_ID', how='left')
           
           region_file = local_output_dir / f"{run_id}_anatomical_regions_{glm_method_tag}.csv"
           if region_file.exists():
               region_df = pd.read_csv(region_file)
               cell_embeddings_df = cell_embeddings_df.merge(region_df[['ROI_ID', 'region']], on='ROI_ID', how='left')
    
           save_df(cell_embeddings_df, f"{run_id}_cell_embeddings.csv", local_output_dir, central_csv_dir, glm_method_tag)

           

        # Save the counts/percent table to CSV
        fig4_csv = disp.reset_index()
        save_df(fig4_csv,
                f"{run_id}_Fig4_poststim_activity_summary.csv",
                local_output_dir, central_csv_dir, glm_method_tag)


        # --- FIGURE 5: FUNCTIONAL CLUSTERING ---
        cluster_df = plot_functional_clusters(PDF, glm_all, stim_order, stat_sel, good_rois, ops, fignum="6", local_output_dir=local_output_dir, central_csv_dir=central_csv_dir, run_id=run_id, timestamp_str=timestamp_str, glm_method_tag=glm_method_tag)
        
        # --- FIGURE 6: PRE-STIMULUS OSCILLATIONS ---
        fig6, ax6 = plt.subplots(figsize=(8, 6)); add_figure_title(fig6, "7", "Distribution of Pre-Stimulus Oscillation Index")
        pre = dff[:, :stim_frame_labels[0][1]]; osc = []
        if pre.shape[1] > 20:
            for row in pre:
                freqs, psd = welch(row, fs=ops['fs'], nperseg=min(256, len(row)))
                bw = (freqs >= 0.01) & (freqs <= 0.1)
                if psd.sum() > 0: osc.append(psd[bw].sum() / psd.sum())
            if osc: ax6.hist(osc, bins=30)
        ax6.set_xlabel("Oscillation Index (Power in 0.01-0.1 Hz band)"); ax6.set_ylabel("Number of ROIs")
        PDF.savefig(fig6); plt.close(fig6)

        # --- FIGURE 7 & 8: ANATOMICAL ANALYSES ---
        analyze_anatomical_regions(PDF, data_folder, glm_consensus_df, stat_sel, good_rois, ops, glm_patt_chem, glm_patt_chem_counts, chemical_stim_order, fignum_base=8, local_output_dir=local_output_dir, central_csv_dir=central_csv_dir, run_id=run_id, stim_order=stim_order, timestamp_str=timestamp_str, glm_method_tag=glm_method_tag)
        analyze_outer_inner_regions(
            PDF, data_folder, glm_consensus_df, stat_sel, good_rois, ops,
            glm_patt_chem, glm_patt_chem_counts, chemical_stim_order,
            fignum_base=9,
            local_output_dir=local_output_dir,
            central_csv_dir=central_csv_dir,
            run_id=run_id,
            timestamp_str=timestamp_str,
            stim_order=stim_order,
            glm_method_tag=glm_method_tag
        )

        # --- APPENDIX A1: PEAK-THRESHOLD ANALYSIS (CONDITIONAL) ---
        if RUN_PEAK_THRESHOLD_ANALYSIS:
            figA1A = plt.figure(figsize=(8.5, 5))
            axA1A = figA1A.add_subplot(111)
            plot_consensus_summary_table(axA1A, peak_df, stim_order, "Appendix A1A", "Peak-Threshold Method")
            PDF.savefig(figA1A); plt.close(figA1A)
            
            plot_peak_threshold_bar_charts(PDF, peak_df, stim_order, fignum="Appendix A1B")
            
            for i in range(0, len(stim_order), 2):
                sub_stims = stim_order[i:i+2]
                fig = plt.figure(figsize=(8.5, 5 * len(sub_stims))); add_figure_title(fig, "Appendix A1C", "Detailed Overlays (Peak-Threshold Method)")
                gs = gridspec.GridSpec(len(sub_stims), 1, hspace=0.4)
                for j, stim in enumerate(sub_stims):
                    ax = fig.add_subplot(gs[j, 0]); ax.imshow(ops['meanImg'], cmap='gray'); ax.axis('off')
                    stim_df = peak_df[peak_df['Stimulus'] == stim]
                    legend_handles = []
                    for cat, group in stim_df.groupby('Response'):
                        color = COLOR_MAP.get(cat, 'gray'); xs, ys = [], []
                        for roi_id in group['ROI_ID']:
                            p = pos_map.get(roi_id)
                            if p is not None: cy, cx = stat_sel[p]['med']; xs.append(cx); ys.append(cy)
                        ax.scatter(xs, ys, s=15, c=color, edgecolors='white', linewidth=0.2, alpha=0.8)
                        count = len(group); percent = 100 * count / n_cells
                        legend_handles.append(Patch(color=color, label=f"{cat} ({count}, {percent:.1f}%)"))
                    ax.set_title(f"Responses to {stim}", y=1.05)
                    ax.legend(handles=sorted(legend_handles, key=lambda h: h.get_label()), loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize="small", frameon=False)
                fig.tight_layout(rect=[0.05, 0.15, 0.95, 0.95]); PDF.savefig(fig); plt.close(fig)

            figA1D, axA1D = plt.subplots(1, 1, figsize=(18, 6)); figA1D.subplots_adjust(top=0.85, right=0.8)
            add_figure_title(figA1D, "Appendix A1D", "Cell Counts by Response Pattern (Peak-Threshold, Chemicals Only)")
            peak_patt_chem, peak_patt_chem_counts = plot_response_patterns(axA1D, peak_df, stim_order, title_suffix="Chemicals Only", chemical_only=True)
            PDF.savefig(figA1D); plt.close(figA1D)
            if peak_patt_chem is not None: save_df(peak_patt_chem, f"{run_id}_peak_threshold_chemical_patterns.csv", local_output_dir, central_csv_dir, glm_method_tag)

            plot_pattern_overlay(PDF, "Appendix A1E", peak_patt_chem, peak_patt_chem_counts, chemical_stim_order, ops, stat_sel, pos_map, "Peak-Threshold Method")



# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------
def analyze_single_session(data_folder_path, summary_csv_path):
    data_folder = Path(data_folder_path)
    log(f"--- Processing session: {data_folder.parent.name} ---")
    timestamp_str = datetime.now().strftime("_%Y%m%d-%H%M%S")
    run_id, stim_frame_labels, _ = extract_stims(data_folder, summary_csv_path)
    
    glm_method_tag = (
        "GLM-Standard" if GLM_METHOD == "full_trace_intercept"
        else "GLM-Contrast-2drugs" if GLM_METHOD == "explicit_baseline_contrast_2drugs"
        else "GLM-Contrast"
    )

    
    local_output_dir = data_folder.parent.parent / f"CELLanalysis_{run_id}_{glm_method_tag}{timestamp_str}"
    central_csv_dir = Path("/Volumes/DataHD2025/CalciumAnalysis/Results_CSV_CELL") / f"{run_id}_{glm_method_tag}{timestamp_str}"
    central_pdf_dir = Path("/Volumes/DataHD2025/CalciumAnalysis/RESULTS_PDFS_CELL")
    
    dff, good_rois, ops, stat_sel = load_data(data_folder, stim_frame_labels)
    if MANUAL_FRAMERATE_HZ: ops['fs'] = MANUAL_FRAMERATE_HZ
    if APPLY_LOWESS_DETREND: dff = detrend_traces(dff)
    if USE_GLOBAL_REGRESSION: dff = global_regression(dff)

    if RUN_PEAK_THRESHOLD_ANALYSIS:
        log("--- Running Peak-Threshold Analysis Pipeline (dF/F only) ---")
        peak_df = classify_responses_by_peak(dff, stim_frame_labels, ops['fs'], good_rois)
    else:
        log("--- Skipping Peak-Threshold Analysis Pipeline ---")
        peak_df = None

    log("--- Running GLM Analysis Pipeline (dF/F only) ---")
    exploratory_df, sparseness_ts = calculate_exploratory_metrics(dff, good_rois)
    
    if GLM_METHOD == "explicit_baseline_contrast":
        log("--- GLM Stage 1: Analyzing ASW vs. Pre-Stimulus Baseline ---")
        dm_asw = make_design_matrix_for_asw(stim_frame_labels, dff.shape[1], ops['fs'])
        glm_asw = run_narrow_glm(dff, "dF/F", dm_asw, good_rois, baseline_col='pre_asw_baseline')

        log("--- GLM Stage 2: Analyzing Chemicals vs. ASW-Period Baseline ---")
        dm_chem = make_design_matrix_for_chemicals(stim_frame_labels, dff.shape[1], ops['fs'])
        glm_chem = run_narrow_glm(dff, "dF/F", dm_chem, good_rois, baseline_col='baseline')
        
        glm_all_raw = pd.concat([glm_asw, glm_chem], ignore_index=True)

    elif GLM_METHOD == "explicit_baseline_contrast_2drugs":
        log("Using GLM Method: Explicit Baseline Contrast (first chem vs ASW; subsequent chems vs local pre-stim)")

        # --- Stage 1: ASW vs pre-ASW baseline (unchanged) ---
        log("--- GLM Stage 1: Analyzing ASW vs. Pre-Stimulus Baseline ---")
        dm_asw = make_design_matrix_for_asw(stim_frame_labels, dff.shape[1], ops['fs'])
        glm_asw = run_narrow_glm(dff, "dF/F", dm_asw, good_rois, baseline_col='pre_asw_baseline')

        # Ordered chemical stimuli
        chem_stims = [s for s in stim_frame_labels if not s[0].startswith("ASW")]
        asw_stim = next((s for s in stim_frame_labels if s[0].startswith("ASW")), None)

        glm_chunks = []

        if chem_stims:
            # --- Stage 2a: FIRST chemical vs ASW-period baseline (like original contrast) ---
            first_label, first_frame = chem_stims[0]
            if asw_stim is not None:
                asw_label, asw_fr = asw_stim
                # Baseline is ASW-period: from ASW onset to the next stimulus (or end)
                try:
                    idx_asw = [lbl for lbl, _ in stim_frame_labels].index(asw_label)
                    baseline_end = stim_frame_labels[idx_asw + 1][1] if (idx_asw + 1) < len(stim_frame_labels) else dff.shape[1]
                except Exception:
                    baseline_end = dff.shape[1]
                baseline_start = asw_fr
            else:
                # No ASW present → use the local baseline knob for the first chemical
                pre_frames = int(CHEM_LOCAL_BASELINE_SECONDS * ops['fs'])
                baseline_start = max(0, first_frame - pre_frames)
                baseline_end   = first_frame

            dm_firstchem = make_design_matrix_for_single_chemical_vs_baseline(
                first_label, first_frame, dff.shape[1], ops['fs'],
                baseline_start_frame=baseline_start,
                baseline_end_frame=baseline_end
            )
            glm_firstchem = run_narrow_glm(dff, "dF/F", dm_firstchem, good_rois, baseline_col='baseline')
            glm_chunks.append(glm_firstchem)

            # --- Stage 2b: SUBSEQUENT chemicals vs local pre-stim baseline (configurable seconds) ---
            pre_frames = int(CHEM_LOCAL_BASELINE_SECONDS * ops['fs'])
            for label, frame_start in chem_stims[1:]:
                baseline_start = max(0, frame_start - pre_frames)
                baseline_end   = frame_start

                dm_local = make_design_matrix_for_single_chemical_vs_baseline(
                    label, frame_start, dff.shape[1], ops['fs'],
                    baseline_start_frame=baseline_start,
                    baseline_end_frame=baseline_end
                )
                glm_local = run_narrow_glm(dff, "dF/F", dm_local, good_rois, baseline_col='baseline')
                glm_chunks.append(glm_local)

        glm_chem = pd.concat(glm_chunks, ignore_index=True) if glm_chunks else pd.DataFrame()
        glm_all_raw = pd.concat([glm_asw, glm_chem], ignore_index=True)

        
    elif GLM_METHOD == "full_trace_intercept":
        log("Using GLM Method: Full Trace Intercept")
        design_matrix = make_design_matrix(stim_frame_labels, dff.shape[1], ops['fs'])
        glm_all_raw = run_glm(dff, "dF/F", design_matrix, good_rois)
    else:
        raise ValueError(f"Unknown GLM_METHOD: {GLM_METHOD}")
    
    glm_all_gated = gate_glm_responses_by_effect_size(glm_all_raw, dff, stim_frame_labels, ops['fs'], good_rois)
    glm_all_gated = calculate_temporal_properties(glm_all_gated, stim_frame_labels)
    glm_consensus_df = create_glm_consensus_responses(glm_all_gated)

    stim_order = [lbl for lbl, _ in stim_frame_labels]
    save_all_csv_outputs(exploratory_df, glm_all_gated, glm_consensus_df, peak_df,
                         stim_order, local_output_dir, central_csv_dir, run_id, timestamp_str, glm_method_tag)

    pdf_name = f"{run_id}_{glm_method_tag}{timestamp_str}.pdf"
    pdf_path = local_output_dir / pdf_name
    generate_pdf_report(
        pdf_path=pdf_path, run_id=run_id, data_folder=data_folder, dff=dff,
        ops=ops, good_rois=good_rois, stat_sel=stat_sel, stim_frame_labels=stim_frame_labels,
        peak_df=peak_df, glm_all=glm_all_gated,
        glm_consensus_df=glm_consensus_df, exploratory_df=exploratory_df,
        sparseness_ts=sparseness_ts, local_output_dir=local_output_dir,
        central_csv_dir=central_csv_dir, timestamp_str=timestamp_str, glm_method_tag=glm_method_tag
    )
    
    central_pdf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(pdf_path), str(central_pdf_dir / pdf_name))
    log(f"✅ Session complete. Report at: {pdf_path}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} sessions_to_analyze.txt")
        sys.exit(1)
    input_list_path = Path(sys.argv[1])
    if not input_list_path.is_file():
        log(f"Error: Input file not found at {input_list_path}")
        sys.exit(1)
    summary_csv_path = "/Volumes/DataHD2025/CalciumAnalysis/zdrift_new_stim_frames.csv"
    sessions = [line.strip() for line in input_list_path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    log(f"Found {len(sessions)} sessions to analyze.")
    for path in sessions:
        try:
            analyze_single_session(path, summary_csv_path)
        except Exception as e:
            log(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log(f"!!! CRITICAL FAILURE on session: {path}")
            log(f"!!! Error: {e}")
            import traceback
            traceback.print_exc()
            log(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue
    log("=== Batch analysis complete ===")

if __name__ == "__main__":
    main()