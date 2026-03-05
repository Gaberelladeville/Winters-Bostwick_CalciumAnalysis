#!/usr/bin/env python3

import sys
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import json

# ------------------------ CONFIG  ------------------------ #
MANUAL_FRAMERATE_HZ = 0.68

# Neuropil options 
USE_NEUROPIL_SUBTRACTION = False      # False => alpha = 0.0 (skip neuropil)
FORCE_NEUROPIL_ALPHA = 0.2           # used only if USE_NEUROPIL_SUBTRACTION == True
NEUROPIL_ALPHA_FALLBACK = 0.5

# Fire-fraction threshold multiplier 
FIRE_FRACTION_DFF_MULTIPLIER = 3.0

# Stim summary CSV 
SUMMARY_CSV_PATH = Path("/Volumes/. . .csv")

# Output directory for fire-fraction summaries
CENTRAL_FF_DIR = Path("/Volumes/. . .")


# -------------------------- HELPER / LOGGING ---------------------------- #

def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def get_neuropil_alpha_and_tag(ops):

    if not USE_NEUROPIL_SUBTRACTION:
        return 0.0, "off"

    if FORCE_NEUROPIL_ALPHA is not None:
        return float(FORCE_NEUROPIL_ALPHA), f"forced={FORCE_NEUROPIL_ALPHA:.3f}"

    alpha = float(ops.get("neucoeff", float("nan")))
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = float(NEUROPIL_ALPHA_FALLBACK)
        return alpha, f"fallback={alpha:.3f}"

    return alpha, f"ops.neucoeff={alpha:.3f}"


# ------------------------ DATA LOADING (MATCHED) ------------------------ #

def load_data(sess_dir: Path, stim_frames):
    """
    Load Suite2p data and compute dF/F exactly as in CellAnalysis11_5.py:

      - load F, Fneu, iscell, ops, stat
      - restrict to good_rois (iscell[:,0] == 1)
      - optional neuropil subtraction
      - baseline for dF/F is 20th percentile between Stim1 and Stim2
    """
    log("Loading Suite2p data...")

    F = np.load(sess_dir / "F.npy")
    Fneu = np.load(sess_dir / "Fneu.npy")
    iscell = np.load(sess_dir / "iscell.npy", allow_pickle=True)
    good_rois = np.where(iscell[:, 0] == 1)[0]

    ops = np.load(sess_dir / "ops.npy", allow_pickle=True).item()
    stat_all = np.load(sess_dir / "stat.npy", allow_pickle=True)

    F_good = F[good_rois]
    Fneu_good = Fneu[good_rois]
    stat_sel = stat_all[good_rois]

    log(f"Found {len(good_rois)} manually curated cells.")

    # --- Neuropil subtraction (same logic) ---
    alpha, alpha_tag = get_neuropil_alpha_and_tag(ops)
    if alpha > 0:
        log(f"Neuropil subtraction ON ({alpha_tag})")
        Fcorr = F_good - alpha * Fneu_good
    else:
        log("Neuropil subtraction OFF (alpha=0.0)")
        Fcorr = F_good

    ops["neuropil_alpha_used"] = float(alpha)
    ops["neuropil_alpha_note"] = str(alpha_tag)

    # dF/F baseline: frames between Stim 1 and Stim 2  →  [start_of_stim1 : start_of_stim2)
    baseline_start_frame = int(stim_frames[0][1])  # Stim 1 onset
    baseline_end_frame = int(stim_frames[1][1])    # Stim 2 onset

    nF = Fcorr.shape[1]
    lo = max(0, min(baseline_start_frame, nF))
    hi = max(0, min(baseline_end_frame, nF))

    # Fallback if slice invalid
    if hi <= lo:
        lo, hi = 0, max(0, min(baseline_end_frame, nF))

    ops["dff_baseline_frames"] = f"{lo}-{hi}"

    F0 = np.percentile(Fcorr[:, lo:hi], 20, axis=1, keepdims=True)
    F0[F0 <= 0] = 1e-6
    dff = (Fcorr - F0) / F0

    # Set fs if you want, though not strictly needed for frame-based epochs
    if MANUAL_FRAMERATE_HZ:
        ops["fs"] = MANUAL_FRAMERATE_HZ

    return dff, good_rois, ops, stat_sel


def extract_stims(sess_dir: Path, summary_csv_path: Path):
    """

      - infer run_id from parent folder containing '_zout_'
      - look up stim frames in summary CSV
      - decode labels from x / z codes (with zN override to ASW1/CHEM1/CHEM2)
    """
    log("Extracting stimulus frames/labels...")
    summary = pd.read_csv(summary_csv_path)

    run_folder = next((p for p in sess_dir.parents if "_zout_" in p.name), None)
    if run_folder is None:
        raise ValueError(f"Could not find a '_zout_' parent folder for {sess_dir}")

    m = re.match(
        r"^(.*)_zout_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(t\S+)$",
        run_folder.name,
    )
    if not m:
        raise ValueError(f"Cannot parse run_id from folder name: {run_folder.name}")
    run_id = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"

    row = summary[summary.run_id.str.strip().str.lower() == run_id.lower()]
    if row.empty:
        raise ValueError(f"No entry found in summary CSV for run_id={run_id}")
    row = row.iloc[0]

    stim_cols = sorted(
        [c for c in row.index if re.match(r"stim\d+_frame", c) and pd.notna(row[c])],
        key=lambda s: int(re.search(r"\d+", s).group()),
    )
    frames = [int(row[c]) for c in stim_cols]

    neuro_map = {
        "g": "GABA",
        "a": "ACh",
        "d": "DA",
        "o": "Oct",
        "s": "5HT",
        "i": "INK",
        "f": "FOOD",
        "l": "Glut",
        "w": "ASW",
        "v": "Vibr",
        "n": "NoStim",
        "p": "FLRIamide",
    }

    code_x = re.search(r"_x([A-Za-z]+)x", run_id)
    code_z = re.search(r"_z([A-Za-z]+)(?:x|_|$)", run_id)

    # zN override: first is mechanical control (ASW1), next two are treated as CHEM1, CHEM2
    if code_z and code_z.group(1).lower() == "n":
        frames = [202, 404, 606]
        raw_labels = ["ASW1", "CHEM1", "CHEM2"]
    elif code_x:
        raw_labels = ["ASW"] + [neuro_map.get(l, l) for l in code_x.group(1).lower()]
    elif code_z:
        raw_labels = [neuro_map.get(l, l) for l in code_z.group(1).lower()]
    else:
        raw_labels = []

    if len(raw_labels) != len(frames):
        log("Warning: mismatch between labels and frames; using generic labels.")
        raw_labels = [f"Stim{i+1}" for i in range(len(frames))]

    labels = []
    counts = {}
    for L in raw_labels:
        counts[L] = counts.get(L, 0) + 1
        labels.append(f"{L}_{counts[L]}" if counts[L] > 1 else L)

    stim_frame_labels = list(zip(labels, frames))
    return run_id, stim_frame_labels, row


# ---------------------- FIRE-FRACTION (UPDATED BASELINE) ------------------- #

def compute_fire_fraction_for_epochs(dff: np.ndarray, stims, epochs_dict):
    """
    Fire-fraction logic with UPDATED baseline for activation threshold.

        baseline definition for sigma_dff:
        baseline_start_frame = stims[0][1]   # ASW onset
        baseline_end_frame   = stims[1][1]   # Stim 1 onset
        sigma_dff = std(dff[:, baseline_start:baseline_end], per cell)
        threshold = FIRE_FRACTION_DFF_MULTIPLIER * sigma_dff
        active if any(dF/F >= threshold in that epoch)

    Returns list of dicts with:
        epoch, active_cells_count_dff, active_cells_pct_dff, active_cells_display_dff
    """
    n_cells, n_frames = dff.shape

    # --- UPDATED baseline frames: ASW -> Stim 1 ---
    if stims and len(stims) >= 2:
        baseline_start_frame = int(stims[0][1])
        baseline_end_frame = int(stims[1][1])
    elif stims and len(stims) == 1:
        # fallback: start -> first stim
        baseline_start_frame = 0
        baseline_end_frame = int(stims[0][1])
    else:
        # fallback: first 100 frames or whole recording if shorter
        baseline_start_frame = 0
        baseline_end_frame = min(100, n_frames)

    # Clamp to valid range
    baseline_start_frame = max(0, min(baseline_start_frame, n_frames))
    baseline_end_frame = max(0, min(baseline_end_frame, n_frames))

    # If invalid slice, fallback to first 100 frames
    if baseline_end_frame <= baseline_start_frame:
        baseline_start_frame = 0
        baseline_end_frame = min(100, n_frames)

    sigma_dff = dff[:, baseline_start_frame:baseline_end_frame].std(axis=1)

    dff_abs_thresh = FIRE_FRACTION_DFF_MULTIPLIER * sigma_dff

    rows = []
    for name, idx in epochs_dict.items():
        idx = np.asarray(idx, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_frames)]
        if idx.size == 0:
            continue

        active_mask = (dff[:, idx] >= dff_abs_thresh[:, np.newaxis]).any(axis=1)
        active_count = int(active_mask.sum())
        active_pct = 100.0 * active_count / n_cells if n_cells > 0 else 0.0
        display = f"{active_count} ({active_pct:.1f}%)"

        rows.append(
            {
                "epoch": name,
                "active_cells_count_dff": active_count,
                "active_cells_pct_dff": active_pct,
                "active_cells_display_dff": display,
            }
        )

    return rows


def define_epochs(n_frames: int, stims):
    """
    Define epochs:

      - Pre-Stim (1st 100 frames)
      - Full Recording
      - Post-Stim (last 100 frames)
      - First Half
      - Second Half
      - Frame 1-ASW (NEW)
      - 8 equal segments: Epoch 1/8 ... Epoch 8/8
      - PLUS stimulus-associated epochs (if enough stimuli are present):
          * "ASW-Stim 1"    : from stim[0] onset to stim[1] onset
          * "Stim 1-Stim 2" : from stim[1] onset to stim[2] onset
          * "Stim 2 onward" : from stim[2] onset (or stim[1] if only 2 stims)
                               to end of recording
    """
    from collections import OrderedDict

    epochs = OrderedDict()

    # Original three
    epochs["Pre-Stim (1st 100 frames)"] = np.arange(0, min(100, n_frames))
    epochs["Full Recording"] = np.arange(0, n_frames)
    epochs["Post-Stim (last 100 frames)"] = np.arange(
        max(0, n_frames - 100), n_frames
    )

    # NEW: Frame 1 -> ASW (start of recording to first stimulus onset)
    if stims and len(stims) >= 1:
        s0 = int(stims[0][1])
        start = 0
        end = max(0, min(s0, n_frames))
        if end > start:
            epochs["Frame 1-ASW"] = np.arange(start, end)

    # New epochs
    epochs["First Half"] = np.arange(0, n_frames // 2)
    epochs["Second Half"] = np.arange(n_frames // 2, n_frames)

    # Stimulus-associated epochs (only if we have enough stim onsets)
    if stims and len(stims) >= 2:
        s1 = int(stims[0][1])
        s2 = int(stims[1][1])

        # Clamp to valid range and ensure order
        start = max(0, min(s1, n_frames))
        end = max(0, min(s2, n_frames))
        if end > start:
            epochs["ASW-Stim 1"] = np.arange(start, end)

    if stims and len(stims) >= 3:
        s2 = int(stims[1][1])
        s3 = int(stims[2][1])

        start = max(0, min(s2, n_frames))
        end = max(0, min(s3, n_frames))
        if end > start:
            epochs["Stim 1-Stim 2"] = np.arange(start, end)

    # Stim 2 onward: from stim 3 if present, otherwise stim 2
    if stims and len(stims) >= 2:
        if len(stims) >= 3:
            onward_start = int(stims[2][1])
        else:
            onward_start = int(stims[1][1])

        start = max(0, min(onward_start, n_frames))
        end = n_frames
        if end > start:
            epochs["Stim 2 onward"] = np.arange(start, end)

    # 8 equal epochs
    for i in range(8):
        start = int(i * n_frames / 8)
        end = int((i + 1) * n_frames / 8)
        epochs[f"Epoch {i+1}/8"] = np.arange(start, end)

    return epochs


# --------------------------- PER-SESSION RUN ---------------------------- #

def analyze_single_session(sess_dir: Path, summary_csv_path: Path):
    log(f"--- Processing session: {sess_dir} ---")

    run_id, stim_frame_labels, _ = extract_stims(sess_dir, summary_csv_path)

    dff, good_rois, ops, stat_sel = load_data(sess_dir, stim_frame_labels)

    n_cells, n_frames = dff.shape
    log(f"dF/F matrix shape: {n_cells} cells x {n_frames} frames")

    epochs = define_epochs(n_frames, stim_frame_labels)

    rows = compute_fire_fraction_for_epochs(dff, stim_frame_labels, epochs)
    ff_df = pd.DataFrame(rows)

    CENTRAL_FF_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CENTRAL_FF_DIR / f"{run_id}_fire_fraction_summary_GLM-Contrast.csv"

    ff_df.to_csv(out_csv, index=False)
    log(f"Saved fire-fraction summary to: {out_csv}")


# ------------------------------ MAIN ------------------------------------ #

def main():
    if len(sys.argv) != 2:
        print(
            f"Usage: python {Path(sys.argv[0]).name} sessions_to_analyze.txt\n"
            f"  (each line: path to a Suite2p plane folder, e.g. .../plane0)"
        )
        sys.exit(1)

    list_path = Path(sys.argv[1])
    if not list_path.is_file():
        log(f"Error: sessions list file not found: {list_path}")
        sys.exit(1)

    sessions = [
        line.strip()
        for line in list_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    log(f"Found {len(sessions)} sessions to analyze.")

    for sess in sessions:
        try:
            analyze_single_session(Path(sess), SUMMARY_CSV_PATH)
        except Exception as e:
            log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            log(f"!!! CRITICAL FAILURE on session: {sess}")
            log(f"!!! Error: {e}")
            import traceback

            traceback.print_exc()
            log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            continue

    log("=== Fire-fraction epoch analysis complete ===")


if __name__ == "__main__":
    main()
