#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate paper figures for TANET 2025 from available logs and data.
- If models/lstm_training_log.csv exists, plot training/val curves.
- Always plot data distribution from flights*.geojson (trajectory length histogram).

Outputs PNGs into paper/figures/.
"""
from __future__ import annotations
import os, json, glob, csv
from typing import List
import math
try:
    import numpy as np  # optional
    HAS_NP = True
except Exception:
    np = None  # type: ignore
    HAS_NP = False
try:
    import matplotlib
    matplotlib.use('Agg')  # use non-GUI backend for headless/save-only
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Project root is three levels up from this file: figures -> paper -> <project root>
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, 'paper', 'figures')
MODEL_DIR = os.path.join(ROOT, 'models')
LOG_PATH = os.path.join(MODEL_DIR, 'lstm_training_log.csv')

os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH_FILE = os.path.join(OUT_DIR, 'generate_figures.log')


def log(msg: str):
    try:
        print(msg)
    finally:
        try:
            with open(LOG_PATH_FILE, 'a', encoding='utf-8') as lf:
                lf.write(msg + "\n")
        except Exception:
            pass


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float('nan')
    if HAS_NP:
        return float(np.percentile(values, p))
    s = sorted(values)
    k = (len(s)-1) * (p/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return float(d0 + d1)


def histogram(values: List[float], bins: int, vmin: float, vmax: float):
    if HAS_NP:
        return np.histogram(values, bins=bins, range=(vmin, vmax))
    if vmin >= vmax:
        vmin, vmax = (min(values), max(values)) if values else (0.0, 1.0)
    width = (vmax - vmin) / bins if bins > 0 else 1.0
    edges = [vmin + i*width for i in range(bins+1)]
    hist = [0]*bins
    for v in values:
        if v < vmin or v > vmax:
            continue
        idx = bins-1 if v == vmax else int((v - vmin) / width)
        if 0 <= idx < bins:
            hist[idx] += 1
    return hist, edges


def find_geojson_files(root: str) -> List[str]:
    pats = [os.path.join(root, 'flights*.geojson'), os.path.join(root, 'flights.geojson')]
    files: List[str] = []
    for p in pats:
        files += sorted(glob.glob(p))
    # filter out stitched files by default
    files = [f for f in files if 'stitched' not in os.path.basename(f).lower()]
    # de-dup keep order
    seen = set(); uniq = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f); uniq.append(f)
    return uniq


def lengths_from_geojson(paths: List[str]) -> List[int]:
    lens: List[int] = []
    for fp in paths:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            feats = data.get('features') or []
            for feat in feats:
                geom = feat.get('geometry') or {}
                gtype = (geom.get('type') or '').lower()
                if gtype not in ('linestring', 'multilinestring'):
                    continue
                if gtype == 'linestring':
                    coords = geom.get('coordinates') or []
                else:
                    lines = geom.get('coordinates') or []
                    coords = [pt for line in lines for pt in line]
                lens.append(len(coords))
        except Exception:
            continue
    return lens


def _save_hist_svg(values: List[float], out_svg: str, title: str, x_label: str, y_label: str, bins: int = 40):
    if not values:
        return False
    width, height, pad = 800, 450, 50
    max_x = np.percentile(values, 99)
    vals = [v for v in values if v <= max_x]
    if not vals:
        return False
    hist, edges = np.histogram(vals, bins=bins, range=(min(vals), max_x))
    max_h = max(hist) if len(hist) else 1
    # Build SVG
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />")
    # axes
    x0, y0 = pad, height - pad
    x1, y1 = width - pad, pad
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='black' stroke-width='1' />")
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='black' stroke-width='1' />")
    # bars
    plot_w = x1 - x0
    plot_h = y0 - y1
    bw = plot_w / bins
    for i, h in enumerate(hist):
        bar_h = 0 if max_h == 0 else (h / max_h) * plot_h
        x = x0 + i * bw
        y = y0 - bar_h
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bw-1:.1f}' height='{bar_h:.1f}' fill='#1f77b4' opacity='0.85' />")
    # labels
    parts.append(f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-size='16' fill='black'>{title}</text>")
    parts.append(f"<text x='{(x0+x1)/2:.1f}' y='{height-10}' text-anchor='middle' font-size='12' fill='black'>{x_label}</text>")
    parts.append(f"<text x='20' y='{(y0+y1)/2:.1f}' text-anchor='middle' font-size='12' fill='black' transform='rotate(-90 20 {(y0+y1)/2:.1f})'>{y_label}</text>")
    parts.append("</svg>")
    with open(out_svg, 'w', encoding='utf-8') as f:
        f.write("\n".join(parts))
    return True


def plot_length_histogram(lengths: List[int]):
    if not lengths:
        log('No GeoJSON lengths to plot.')
        return
    if HAS_MPL:
        plt.figure(figsize=(7,4))
        mx = percentile(lengths, 99)
        plt.hist([l for l in lengths if l <= mx], bins=40, color='#1f77b4', alpha=0.85)
        plt.xlabel('Points per trajectory')
        plt.ylabel('Count')
        plt.title('Trajectory length distribution (trimmed 99th percentile)')
        plt.tight_layout()
        out = os.path.join(OUT_DIR, 'fig_traj_length_hist.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log(f'Saved {out}')
    else:
        out_svg = os.path.join(OUT_DIR, 'fig_traj_length_hist.svg')
        ok = _save_hist_svg(lengths, out_svg, 'Trajectory length distribution (trimmed 99th percentile)', 'Points per trajectory', 'Count')
        if ok:
            log(f'Saved {out_svg}')
        else:
            log('Failed to create histogram (SVG)')


def _save_lines_svg(xs: List[float], series: List[List[float]], labels: List[str], colors: List[str], out_svg: str, title: str, x_label: str, y_label: str):
    if not xs or not any(series):
        return False
    width, height, pad = 800, 450, 50
    # compute bounds
    x_min, x_max = min(xs), max(xs)
    y_vals = [v for s in series for v in s if np.isfinite(v)]
    if not y_vals:
        return False
    y_min, y_max = min(y_vals), max(y_vals)
    if y_min == y_max:
        y_min -= 1e-6
        y_max += 1e-6
    def sx(x):
        return pad + (x - x_min) / (x_max - x_min) * (width - 2*pad)
    def sy(y):
        return height - pad - (y - y_min) / (y_max - y_min) * (height - 2*pad)
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"]
    parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />")
    x0, y0 = pad, height - pad
    x1, y1 = width - pad, pad
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x1}' y2='{y0}' stroke='black' stroke-width='1' />")
    parts.append(f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='black' stroke-width='1' />")
    # draw lines
    for s, lbl, col in zip(series, labels, colors):
        pts = [f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, s) if np.isfinite(y)]
        if len(pts) >= 2:
            parts.append(f"<polyline fill='none' stroke='{col}' stroke-width='2' points='" + ' '.join(pts) + "' />")
    # legend
    lx, ly = x0 + 10, y1 + 10
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        parts.append(f"<rect x='{lx}' y='{ly + i*18}' width='14' height='4' fill='{col}' />")
        parts.append(f"<text x='{lx + 20}' y='{ly + i*18 + 6}' font-size='12' fill='black'>{lbl}</text>")
    # labels
    parts.append(f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-size='16' fill='black'>{title}</text>")
    parts.append(f"<text x='{(x0+x1)/2:.1f}' y='{height-10}' text-anchor='middle' font-size='12' fill='black'>{x_label}</text>")
    parts.append(f"<text x='20' y='{(y0+y1)/2:.1f}' text-anchor='middle' font-size='12' fill='black' transform='rotate(-90 20 {(y0+y1)/2:.1f})'>{y_label}</text>")
    parts.append("</svg>")
    with open(out_svg, 'w', encoding='utf-8') as f:
        f.write("\n".join(parts))
    return True


def plot_training_log(csv_path: str):
    if not os.path.exists(csv_path):
        log('Training log not found, skip.')
        return
    epochs, train_mse, val_mse = [], [], []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        # expect columns: epoch,train_mse,val_mse or similar
        for row in rdr:
            try:
                e = int(row.get('epoch') or row.get('Epoch') or row.get('ep') or 0)
                tr = float(row.get('train_mse') or row.get('train') or row.get('train_loss') or 'nan')
                va = float(row.get('val_mse') or row.get('val') or row.get('val_loss') or 'nan')
            except Exception:
                continue
            if e > 0:
                epochs.append(e); train_mse.append(tr); val_mse.append(va)
    if not epochs:
        log('Empty training log, skip.')
        return
    if HAS_MPL:
        plt.figure(figsize=(7,4))
        plt.plot(epochs, train_mse, '-o', label='Train MSE', color='#2ca02c')
        if any(math.isfinite(v) for v in val_mse):
            plt.plot(epochs, val_mse, '-o', label='Val MSE', color='#d62728')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('LSTM training curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, 'fig_lstm_training.png')
        plt.savefig(out, dpi=150)
        plt.close()
        log(f'Saved {out}')
    else:
        labels = ['Train MSE'] + (['Val MSE'] if any(math.isfinite(v) for v in val_mse) else [])
        series = [train_mse] + ([val_mse] if any(math.isfinite(v) for v in val_mse) else [])
        colors = ['#2ca02c'] + (['#d62728'] if len(series) > 1 else [])
        out_svg = os.path.join(OUT_DIR, 'fig_lstm_training.svg')
        ok = _save_lines_svg(epochs, series, labels, colors, out_svg, 'LSTM training curve', 'Epoch', 'MSE')
        if ok:
            log(f'Saved {out_svg}')
        else:
            log('Failed to create training curve (SVG)')


if __name__ == '__main__':
    try:
        # clear log
        with open(LOG_PATH_FILE, 'w', encoding='utf-8') as lf:
            lf.write('')
        # Length histogram
        log(f"Project ROOT: {ROOT}")
        files = find_geojson_files(ROOT)
        log(f"Found {len(files)} GeoJSON files under ROOT")
        lens = lengths_from_geojson(files)
        log(f"Collected {len(lens)} trajectory lengths")
        plot_length_histogram(lens)
        # Optional training curve
        log(f"Training log path: {LOG_PATH}")
        plot_training_log(LOG_PATH)
        log('Done.')
    except Exception as e:
        try:
            log(f'Error: {e}')
        finally:
            raise
