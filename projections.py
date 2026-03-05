#!/usr/bin/env python3
"""
Create colored difference images & overlays from saved projections (max or mean).

Unified palette (applies to both MAX and MEAN):
* EXC:  (Stim1−ASW) pinkish-red #ff3366, (Stim2−Stim1) periwinkle #6f7ffc
* INH:  (ASW−Stim1) green #05fa9c,       (Stim1−Stim2) orange #ff7f00

All output filenames include the source tag:
  *_MAX.png/.tif or *_MEAN.png/.tif
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import tifffile as tiff

# ---------- Paths ----------
DEFAULT_REP_ROOT = Path("/Volumes/DataHD2025/Manuscript/ManuscriptImages/representatives")

# ---------- IO helpers ----------
def load_projection(proj_dir: Path, prefix: str, label: str, source_type: str = "max") -> np.ndarray:
    """
    Load a projection as float32 grayscale.
    source_type = 'max'  -> {prefix}_{label}_MAX.tif/png
                 'mean' -> {prefix}_{label}_MEAN.tif/png
    """
    suffix = "_MEAN" if source_type == "mean" else "_MAX"
    tiff_path = proj_dir / f"{prefix}_{label}{suffix}.tif"
    png_path  = proj_dir / f"{prefix}_{label}{suffix}.png"

    if tiff_path.exists():
        arr = tiff.imread(str(tiff_path))
    elif png_path.exists():
        arr = np.array(Image.open(png_path))
    else:
        raise FileNotFoundError(
            f"Missing {source_type.upper()} projection for label '{label}': "
            f"{tiff_path.name} or {png_path.name}"
        )
    if arr.ndim == 3:
        arr = arr.max(axis=2)  # collapse any RGB just in case
    return arr.astype(np.float32)

def normalize_to_uint8(img: np.ndarray, clip=(1.0, 99.9)) -> np.ndarray:
    lo, hi = np.percentile(img, clip)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi <= lo:
            hi = lo + 1.0
    imgc = np.clip(img, lo, hi)
    scaled = (imgc - lo) / (hi - lo)
    return (scaled * 255.0 + 0.5).astype(np.uint8)

def scale_to_uint16(img: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint16)
    scaled = (img - lo) / (hi - lo)
    return (scaled * 65535.0 + 0.5).astype(np.uint16)

def save_tiff_gray(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, scale_to_uint16(arr))

def save_tiff_rgb(rgb8: np.ndarray, path: Path):
    """Save an RGB uint8 image as RGB uint16 TIFF (0..255 -> 0..65535)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb8.dtype != np.uint8 or rgb8.ndim != 3 or rgb8.shape[2] != 3:
        raise ValueError("save_tiff_rgb expects an HxWx3 uint8 RGB array.")
    rgb16 = (rgb8.astype(np.uint16) * 257)  # 255 -> 65535
    tiff.imwrite(path, rgb16)

# ---------- Colors / overlay (unified for MAX & MEAN) ----------
# Singles (per-diff images; mapped to unified overlay hues)
COL_S1_MINUS_BASE = ImageColor.getrgb("#ff3366")  # Stim1−ASW  (pinkish-red)
COL_S2_MINUS_S1   = ImageColor.getrgb("#6f7ffc")  # Stim2−Stim1 (periwinkle)
COL_BASE_MINUS_S1 = ImageColor.getrgb("#05fa9c")  # ASW−Stim1   (green)
COL_S1_MINUS_S2   = ImageColor.getrgb("#ff7f00")  # Stim1−Stim2 (orange)

# Overlay palettes (same hues as singles)
COL_EXC_1 = COL_S1_MINUS_BASE  # Stim1−ASW
COL_EXC_2 = COL_S2_MINUS_S1    # Stim2−Stim1
COL_INH_1 = COL_BASE_MINUS_S1  # ASW−Stim1
COL_INH_2 = COL_S1_MINUS_S2    # Stim1−Stim2

# ---------- Colorize / blend / legends ----------
def colorize_uint8(gray: np.ndarray, rgb: Tuple[int,int,int]) -> np.ndarray:
    r, g, b = rgb
    w = np.array([r, g, b], dtype=np.float32)/255.0
    g8 = gray.astype(np.float32)[:, :, None]
    return np.clip(g8 * w[None, None, :], 0, 255).astype(np.uint8)

def blend_layers(layers: List[np.ndarray], mode: str="max") -> np.ndarray:
    if not layers:
        raise ValueError("No layers to blend.")
    acc = layers[0].astype(np.uint16)
    if mode == "max":
        for L in layers[1:]:
            acc = np.maximum(acc, L.astype(np.uint16))
        return acc.astype(np.uint8)
    elif mode == "additive":
        for L in layers[1:]:
            acc = acc + L.astype(np.uint16)
        return np.clip(acc, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Blend must be 'max' or 'additive'.")

def add_legend(img: Image.Image, labels: List[str], colors: List[Tuple[int,int,int]], pad=10):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    items = []
    for lab, col in zip(labels, colors):
        l, t, r, b = draw.textbbox((0,0), lab, font=font)
        items.append((lab, col, r-l, b-t))
    x0, y0 = pad, pad
    box_w = max(18 + w + 6 for (_,_,w,_) in items) + 2*pad
    box_h = sum(max(12,h) + 6 for (_,_,_,h) in items) + 2*pad
    draw.rectangle([x0-2, y0-2, x0+box_w, y0+box_h], fill=(0,0,0))
    y = y0
    for (lab, col, w, h) in items:
        draw.rectangle([x0, y, x0+12, y+12], fill=col)
        draw.text((x0+18, y-1), lab, fill=(255,255,255), font=font)
        y += max(12, h) + 6

def stamp_text(img: Image.Image, text: str, pad=8):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    l, t, r, b = draw.textbbox((0,0), text, font=font)
    w, h = r - l, b - t
    draw.rectangle([pad-4, pad-4, pad + w + 4, pad + h + 4], fill=(0,0,0))
    draw.text((pad, pad), text, fill=(255,255,255), font=font)

# ---------- Diffs ----------
def diff_positive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (b - a) with negatives clipped to 0 (what increased)."""
    d = b - a
    np.maximum(d, 0.0, out=d)
    return d

def diff_signed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return signed difference (b - a) as float32."""
    return (b - a).astype(np.float32, copy=False)

def threshold_dpos(dpos: np.ndarray, A: np.ndarray, B: np.ndarray, frac: float) -> np.ndarray:
    """If frac>0, zero out positive diffs below frac × mean((A,B))."""
    if frac is None or frac <= 0.0:
        return dpos
    m = 0.5 * (float(np.mean(A)) + float(np.mean(B)))
    t = frac * m
    if t <= 0:
        return dpos
    out = dpos.copy()
    out[out < t] = 0.0
    return out

def threshold_signed(diff: np.ndarray, A: np.ndarray, B: np.ndarray, frac: float) -> np.ndarray:
    """If frac>0, zero signed diffs where |diff| < frac × mean((A,B))."""
    if frac is None or frac <= 0.0:
        return diff
    m = 0.5 * (float(np.mean(A)) + float(np.mean(B)))
    t = frac * m
    if t <= 0:
        return diff
    out = diff.copy()
    mask = np.abs(out) < t
    out[mask] = 0.0
    return out

def bipolar_rgb_from_signed(diff: np.ndarray,
                            pos_rgb: Tuple[int,int,int],
                            neg_rgb: Tuple[int,int,int],
                            clip_percentile: float = 99.5) -> np.ndarray:
    """
    Map a signed diff to an RGB image: positives -> pos_rgb, negatives -> neg_rgb.
    Symmetric robust scaling using the given percentile of |diff|.
    """
    mag = np.abs(diff)
    vmax = np.percentile(mag, clip_percentile)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(mag.max()) if mag.size else 1.0
        if vmax <= 0: vmax = 1.0

    pos = np.clip(diff, 0, None) / vmax
    neg = np.clip(-diff, 0, None) / vmax
    pos = np.clip(pos, 0, 1)[:, :, None]
    neg = np.clip(neg, 0, 1)[:, :, None]

    pr = np.array(pos_rgb, dtype=np.float32)/255.0
    nr = np.array(neg_rgb, dtype=np.float32)/255.0

    pos_col = pos * pr[None,None,:]
    neg_col = neg * nr[None,None,:]

    rgb = (pos_col + neg_col) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Make colored diffs & overlays from projections (max or mean).")
    # Either provide proj_dir + --prefix, or just --tag
    ap.add_argument("proj_dir", nargs="?", type=Path,
                    help="Directory with {prefix}_{label}[_MEAN/_MAX].tif/.png projections (ignored if --tag is used)")
    ap.add_argument("--prefix", help="Filename prefix (ignored if --tag is used)")
    ap.add_argument("--tag", help="Shortcut: resolves proj_dir and prefix automatically")

    ap.add_argument("--labels", nargs=3, required=True, metavar=("BASELINE", "STIM1", "STIM2"),
                    help="Exactly three labels, in order: BASELINE STIM1 STIM2 (e.g., ASW Stim1 Stim2)")
    ap.add_argument("--source-type", choices=["max", "mean"], default="max",
                    help="Which projections to use: max -> {prefix}_{label}_MAX, mean -> {prefix}_{label}_MEAN")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="Output directory (default: <proj_dir>/<prefix>_diffs_all_{source-type})")
    ap.add_argument("--blend", choices=["max","additive"], default="max", help="Blend mode for overlays")
    ap.add_argument("--legend", action="store_true", help="Add legends (keys) to overlays")
    ap.add_argument("--save-diff-tiff", action="store_true", help="Also save grayscale TIFF for each diff")
    ap.add_argument("--save-overlay-tiff", action="store_true",
                    help="Also save EXC/INH/ALL overlays as 16-bit RGB TIFFs")

    # Bipolar options
    ap.add_argument("--bipolar", action="store_true", help="Also save bipolar (signed) PNGs for each diff")
    ap.add_argument("--bipolar-pos-color", default="#ff00ff", help="Color for positive values (default magenta)")
    ap.add_argument("--bipolar-neg-color", default="#00ffff", help="Color for negative values (default cyan)")
    ap.add_argument("--bipolar-clip", type=float, default=99.5, help="Percentile for symmetric robust scaling (default 99.5)")
    ap.add_argument("--stamp", action="store_true", help="Stamp each bipolar image with a caption showing color meaning")
    ap.add_argument("--bipolar-min-abs-frac", type=float, default=0.0,
                    help="If >0, zero |B−A| below frac × mean((A,B)) before bipolar mapping.")

    # Independent positive-difference thresholds for EXC vs INH classes
    ap.add_argument("--exc-min-diff-frac", type=float, default=0.0,
                    help="If >0, EXC (Stim1−ASW, Stim2−Stim1): keep (B−A)+ only where > frac × mean((A,B)).")
    ap.add_argument("--inh-min-diff-frac", type=float, default=0.0,
                    help="If >0, INH (ASW−Stim1, Stim1−Stim2): keep (B−A)+ only where > frac × mean((A,B)).")

    args = ap.parse_args()

    # Resolve tag → proj_dir + prefix if provided
    if args.tag:
        prefix = args.tag
        proj_dir = DEFAULT_REP_ROOT / prefix / f"{prefix}_outputs"
    else:
        if args.proj_dir is None or args.prefix is None:
            raise SystemExit("Provide either --tag TAG  OR  positional proj_dir plus --prefix PREFIX.")
        proj_dir = args.proj_dir
        prefix = args.prefix

    base_lab, s1_lab, s2_lab = args.labels

    outdir = args.outdir or (proj_dir / f"{prefix}_diffs_all_{args.source_type}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Tag to stamp onto filenames
    SRC_TAG = "MAX" if args.source_type == "max" else "MEAN"

    # Load projections according to source-type
    BASE = load_projection(proj_dir, prefix, base_lab, source_type=args.source_type)
    S1   = load_projection(proj_dir, prefix, s1_lab,  source_type=args.source_type)
    S2   = load_projection(proj_dir, prefix, s2_lab,  source_type=args.source_type)

    # -------- Directional single diffs (B-A clipped to 0) --------
    # Apply EXC threshold to EXC pairs, INH threshold to INH pairs.
    directional = [
        (f"{s1_lab}_minus_{base_lab}", BASE, S1, COL_S1_MINUS_BASE, args.exc_min_diff_frac),   # EXC
        (f"{s2_lab}_minus_{s1_lab}",   S1,   S2, COL_S2_MINUS_S1,   args.exc_min_diff_frac),   # EXC
        (f"{base_lab}_minus_{s1_lab}", S1,   BASE, COL_BASE_MINUS_S1, args.inh_min_diff_frac), # INH
        (f"{s1_lab}_minus_{s2_lab}",   S2,   S1, COL_S1_MINUS_S2,   args.inh_min_diff_frac),   # INH
    ]

    colored_layers: List[np.ndarray] = []
    layer_labels:   List[str]        = []
    layer_colors:   List[Tuple[int,int,int]] = []

    for name, A, B, col, frac in directional:
        dpos = diff_positive(A, B)                     # B-A, negatives->0
        dpos = threshold_dpos(dpos, A, B, frac)        # apply class-specific threshold
        g8   = normalize_to_uint8(dpos)                # for visualization
        rgb  = colorize_uint8(g8, col)
        Image.fromarray(rgb, mode="RGB").save(outdir / f"{prefix}_DIFF_{name}_{SRC_TAG}.png")
        print(f"[Saved] {prefix}_DIFF_{name}_{SRC_TAG}.png")
        if args.save_diff_tiff:
            save_tiff_gray(dpos, outdir / f"{prefix}_DIFF_{name}_{SRC_TAG}.tif")
            print(f"[Saved] {prefix}_DIFF_{name}_{SRC_TAG}.tif")
        colored_layers.append(rgb)
        layer_labels.append(name)
        layer_colors.append(col)

    # -------- Overlays (using unified palette) --------
    # EXC: Stim1−ASW (pinkish-red), Stim2−Stim1 (periwinkle)
    dpos_exc1 = diff_positive(BASE, S1)
    dpos_exc1 = threshold_dpos(dpos_exc1, BASE, S1, args.exc_min_diff_frac)
    dpos_exc2 = diff_positive(S1,   S2)
    dpos_exc2 = threshold_dpos(dpos_exc2, S1,   S2, args.exc_min_diff_frac)

    exc1_rgb  = colorize_uint8(normalize_to_uint8(dpos_exc1), COL_EXC_1)
    exc2_rgb  = colorize_uint8(normalize_to_uint8(dpos_exc2), COL_EXC_2)
    exc_overlay = blend_layers([exc1_rgb, exc2_rgb], mode=args.blend)
    exc_img = Image.fromarray(exc_overlay, mode="RGB")
    if args.legend:
        add_legend(exc_img,
                   [f"{s1_lab}_minus_{base_lab}", f"{s2_lab}_minus_{s1_lab}"],
                   [COL_EXC_1, COL_EXC_2], pad=10)
    exc_img.save(outdir / f"{prefix}_DIFF_overlay_EXC_{args.blend}_{SRC_TAG}.png")
    print(f"[Saved] {prefix}_DIFF_overlay_EXC_{args.blend}_{SRC_TAG}.png")
    if args.save_overlay_tiff:
        save_tiff_rgb(exc_overlay, outdir / f"{prefix}_DIFF_overlay_EXC_{args.blend}_{SRC_TAG}.tif")
    # INH: ASW−Stim1 (green), Stim1−Stim2 (orange)
    dpos_inh1 = diff_positive(S1, BASE)  # ASW−Stim1
    dpos_inh1 = threshold_dpos(dpos_inh1, S1, BASE, args.inh_min_diff_frac)
    dpos_inh2 = diff_positive(S2, S1)    # Stim1−Stim2
    dpos_inh2 = threshold_dpos(dpos_inh2, S2, S1, args.inh_min_diff_frac)

    inh1_rgb  = colorize_uint8(normalize_to_uint8(dpos_inh1), COL_INH_1)
    inh2_rgb  = colorize_uint8(normalize_to_uint8(dpos_inh2), COL_INH_2)
    inh_overlay = blend_layers([inh1_rgb, inh2_rgb], mode=args.blend)
    inh_img = Image.fromarray(inh_overlay, mode="RGB")
    if args.legend:
        add_legend(inh_img,
                   [f"{base_lab}_minus_{s1_lab}", f"{s1_lab}_minus_{s2_lab}"],
                   [COL_INH_1, COL_INH_2], pad=10)
    inh_img.save(outdir / f"{prefix}_DIFF_overlay_INH_{args.blend}_{SRC_TAG}.png")
    print(f"[Saved] {prefix}_DIFF_overlay_INH_{args.blend}_{SRC_TAG}.png")
    if args.save_overlay_tiff:
        save_tiff_rgb(inh_overlay, outdir / f"{prefix}_DIFF_overlay_INH_{args.blend}_{SRC_TAG}.tif")

    # ALL: overlay all four singles (uses singles' colors)
    all_overlay = blend_layers(colored_layers, mode=args.blend)
    all_img     = Image.fromarray(all_overlay, mode="RGB")
    if args.legend:
        add_legend(all_img, layer_labels, layer_colors, pad=10)
    all_img.save(outdir / f"{prefix}_DIFF_overlay_ALL_{args.blend}_{SRC_TAG}.png")
    print(f"[Saved] {prefix}_DIFF_overlay_ALL_{args.blend}_{SRC_TAG}.png")
    if args.save_overlay_tiff:
        save_tiff_rgb(all_overlay, outdir / f"{prefix}_DIFF_overlay_ALL_{args.blend}_{SRC_TAG}.tif")

    # ----- Optional: bipolar (signed) maps for each diff -----
    # Apply optional magnitude threshold BEFORE bipolar mapping (unlike EXC/INH which are positive-only).
    if args.bipolar:
        pos_rgb = ImageColor.getrgb(args.bipolar_pos_color)
        neg_rgb = ImageColor.getrgb(args.bipolar_neg_color)
        for name, A, B, _col, _frac in directional:
            d_signed = diff_signed(A, B)  # B - A, signed
            d_signed = threshold_signed(d_signed, A, B, args.bipolar_min_abs_frac)
            rgb = bipolar_rgb_from_signed(d_signed, pos_rgb, neg_rgb, clip_percentile=args.bipolar_clip)
            img = Image.fromarray(rgb, mode="RGB")
            if args.stamp:
                caption = f"{name}: +={args.bipolar_pos_color}  -= {args.bipolar_neg_color}"
                stamp_text(img, caption)
            img.save(outdir / f"{prefix}_BIPOLAR_{name}_{SRC_TAG}.png")
            print(f"[Saved] {prefix}_BIPOLAR_{name}_{SRC_TAG}.png")

    print("Done.")

if __name__ == "__main__":
    main()
