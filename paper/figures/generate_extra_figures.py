#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速補齊：
- 系統架構圖（簡易框圖，SVG）
- 多模型整合流程圖（檢索→共識→LSTM，SVG）
- 以 quick CSV 繪製預測誤差比較（ADE/RMSE 條圖，PNG）
- 以 quick CSV 產生 ROC 佔位（無標註資料，繪製示意曲線與說明，PNG）
- 產出簡易地圖視覺化（若有中心點與偏移，示意路徑，SVG）
"""
import os, csv, math, random, time
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from figure_style import get_palette, get_hatches, annotate_bars

# Optional requests for live API usage
try:
    import requests  # type: ignore
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.constrained_layout.use'] = True

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def write_svg(path: str, parts: List[str], width=1000, height=600):
    header = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
              "<rect x='0' y='0' width='100%' height='100%' fill='white' />"]
    footer = ["</svg>"]
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(header + parts + footer))

def draw_box(x, y, w, h, label, color='#e3f2fd'):
    return [
        f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' ry='8' fill='{color}' stroke='#1976d2' stroke-width='2' />",
        f"<text x='{x + w/2}' y='{y + h/2 + 4}' text-anchor='middle' font-size='16' fill='#0d47a1'>{label}</text>"
    ]

def arrow(x1,y1,x2,y2,label=None):
    parts = [f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#424242' stroke-width='2' marker-end='url(#arrow)' />"]
    if label:
        parts.append(f"<text x='{(x1+x2)/2}' y='{(y1+y2)/2 - 6}' text-anchor='middle' font-size='14' fill='#424242'>{label}</text>")
    return parts

def system_architecture():
    parts = [
        "<defs><marker id='arrow' markerWidth='10' markerHeight='7' refX='10' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='#424242'/></marker></defs>"
    ]
    parts += draw_box(60, 240, 200, 120, '前端 UI\n(Leaflet)')
    parts += draw_box(360, 80, 240, 120, 'Flask API\n/identify, /forecast, /predict')
    parts += draw_box(360, 280, 240, 120, '演算法層\nDTW/LCSS/Fréchet/EDR/ERP/Euclid')
    parts += draw_box(700, 80, 240, 120, '資料層\nflights_*.geojson\n(OpenFlights 可選)')
    parts += draw_box(700, 280, 240, 120, '模型層\n輕量 LSTM (可選)')
    parts += arrow(260, 300, 360, 300, '查詢/結果')
    parts += arrow(480, 200, 480, 280, '呼叫相似度/預測')
    parts += arrow(600, 120, 700, 120, '載入/掃描')
    parts += arrow(600, 340, 700, 340, '載入/推論')
    out = os.path.join(FIG_DIR, 'fig_system_architecture.svg')
    write_svg(out, parts)
    print(f"✅ Saved: {out}")

def integration_flow():
    parts = [
        "<defs><marker id='arrow' markerWidth='10' markerHeight='7' refX='10' refY='3.5' orient='auto'><polygon points='0 0, 10 3.5, 0 7' fill='#424242'/></marker></defs>"
    ]
    parts += draw_box(80, 80, 220, 100, '步驟 1\n使用者繪製片段')
    parts += draw_box(360, 80, 260, 100, '步驟 2\nidentify SUBSEQ_DTW')
    parts += draw_box(680, 80, 240, 100, '步驟 3\n共識續行 + 外插')
    parts += draw_box(360, 260, 260, 100, '替代路徑\nLSTM 直接滾動預測')
    parts += arrow(300, 130, 360, 130, 'query_len, stride, fast')
    parts += arrow(620, 130, 680, 130, 'TopN IDs')
    parts += arrow(300, 130, 360, 310, '可切 LSTM')
    out = os.path.join(FIG_DIR, 'fig_integration_flow.svg')
    write_svg(out, parts)
    print(f"✅ Saved: {out}")

def read_quick(path):
    rows = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    return rows

def safe_float(s):
    try:
        return float(s)
    except:
        return float('nan')

def error_comparison(prelim_csv):
    rows = read_quick(prelim_csv)
    if not rows:
        print(f"Skip error_comparison: {prelim_csv} not found or empty")
        return
    methods = [r['method'] for r in rows]
    ade = np.array([safe_float(r.get('ade_km_mean')) for r in rows])
    rmse = np.array([safe_float(r.get('rmse_km_mean')) for r in rows])
    # CI error bars if available
    ade_lo = np.array([safe_float(r.get('ade_km_ci95_lo')) for r in rows])
    ade_hi = np.array([safe_float(r.get('ade_km_ci95_hi')) for r in rows])
    rmse_lo = np.array([safe_float(r.get('rmse_km_ci95_lo')) for r in rows])
    rmse_hi = np.array([safe_float(r.get('rmse_km_ci95_hi')) for r in rows])
    ade_err = np.vstack([(ade - ade_lo).clip(min=0), (ade_hi - ade).clip(min=0)]) if np.isfinite(ade_lo).any() else None
    rmse_err = np.vstack([(rmse - rmse_lo).clip(min=0), (rmse_hi - rmse).clip(min=0)]) if np.isfinite(rmse_lo).any() else None
    x = np.arange(len(methods))
    w = 0.35
    plt.figure(figsize=(9,4.8), constrained_layout=True)
    colors = get_palette(2); h = get_hatches(2)
    b1 = plt.bar(x - w/2, ade, w, yerr=ade_err, label='ADE km', color=colors[0], ecolor='#555', capsize=3, edgecolor='#333', linewidth=0.5)
    b2 = plt.bar(x + w/2, rmse, w, yerr=rmse_err, label='RMSE km', color=colors[1], ecolor='#555', capsize=3, edgecolor='#333', linewidth=0.5)
    for b in b1: b.set_hatch(h[0])
    for b in b2: b.set_hatch(h[1])
    plt.xticks(x, methods, rotation=30, ha='right')
    plt.ylabel('km')
    plt.title('誤差比較 REAL quick 95% CI', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9); plt.grid(axis='y', alpha=0.3)
    out = os.path.join(FIG_DIR, 'fig_error_comparison_REAL.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {out}")

def latency_breakdown(prelim_csv):
    rows = read_quick(prelim_csv)
    if not rows:
        return
    methods = [r['method'] for r in rows]
    lat = np.array([safe_float(r.get('latency_ms_mean')) for r in rows])
    lat_lo = np.array([safe_float(r.get('latency_ms_ci95_lo')) for r in rows])
    lat_hi = np.array([safe_float(r.get('latency_ms_ci95_hi')) for r in rows])
    lat_err = np.vstack([(lat - lat_lo).clip(min=0), (lat_hi - lat).clip(min=0)]) if np.isfinite(lat_lo).any() else None
    plt.figure(figsize=(9,4.2), constrained_layout=True)
    palette = get_palette(len(methods)); hlat = get_hatches(len(methods))
    bars = plt.bar(methods, lat, yerr=lat_err, color=palette, ecolor='#555', capsize=3, edgecolor='#333', linewidth=0.5)
    for b, hh in zip(bars, hlat): b.set_hatch(hh)
    plt.ylabel('延遲 ms')
    plt.title('延遲比較 REAL quick 95% CI', fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.3)
    annotate_bars(plt.gca(), bars, fmt='{:.0f}')
    out = os.path.join(FIG_DIR, 'fig_latency_breakdown_REAL.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {out}")

def example_predictions(server: str, k: int = 4, qlen: int = 4, horizon: int = 6):
    if not HAS_REQUESTS:
        print('Skip example_predictions: requests not installed')
        return
    try:
        r = requests.get(server.rstrip('/') + '/api/flights-with-paths?limit=800', timeout=5)
        r.raise_for_status()
        flights = r.json()
    except Exception as e:
        print(f'Skip example_predictions (API unavailable): {e}')
        return
    cand = [f for f in flights if isinstance(f.get('path'), list) and len(f['path']) >= (qlen + horizon + 1)]
    if not cand:
        print('No sufficient flights for examples')
        return
    samples = random.sample(cand, k=min(k, len(cand)))
    n = len(samples)
    cols = 2
    rows = int(math.ceil(n / cols))
    plt.figure(figsize=(12, 4.5*rows), constrained_layout=True)
    plt.suptitle('範例：共識與 LSTM 預測（REAL）', fontsize=14, fontweight='bold')
    for i, rec in enumerate(samples):
        path = rec['path']
        query = path[:qlen]
        truth = path[qlen:qlen+horizon]
        ax = plt.subplot(rows, cols, i+1)
        # plot observed
        qx = [p['lng'] for p in query]; qy = [p['lat'] for p in query]
        ax.plot(qx, qy, 'ko-', label='觀測片段')
        # truth future
        tx = [p['lng'] for p in truth]; ty = [p['lat'] for p in truth]
        ax.plot(tx, ty, 'g.-', label='真值續行')
        # consensus
        try:
            url = server.rstrip('/') + '/api/forecast-consensus?algo=SUBSEQ_DTW&subseq=true&fast=true&directional=true&topN=5&horizon=' + str(horizon)
            rc = requests.post(url, json=query, timeout=5)
            rc.raise_for_status()
            cons = rc.json().get('consensus') or []
        except Exception:
            cons = []
        if cons:
            cx = [p['lng'] for p in cons]; cy = [p['lat'] for p in cons]
            ax.plot(cx, cy, 'b.-', label='共識預測')
            # endpoint error
            def geod(a,b):
                R=6371.0088
                lat1,lon1 = math.radians(a[0]), math.radians(a[1])
                lat2,lon2 = math.radians(b[0]), math.radians(b[1])
                dlat=lat2-lat1; dlon=lon2-lon1
                s=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                return 2*R*math.asin(math.sqrt(max(0.0,min(1.0,s))))
            e_cons = geod((truth[-1]['lat'], truth[-1]['lng']), (cons[-1]['lat'], cons[-1]['lng'])) if truth and cons else float('nan')
        else:
            e_cons = float('nan')
        # LSTM
        try:
            url = server.rstrip('/') + '/api/predict-trajectory?model=lstm&horizon=' + str(horizon)
            rl = requests.post(url, json=query, timeout=5)
            rl.raise_for_status()
            pred = rl.json().get('predicted_trajectory') or []
        except Exception:
            pred = []
        if pred:
            px = [p['lng'] for p in pred]; py = [p['lat'] for p in pred]
            ax.plot(px, py, 'r.-', label='LSTM 預測')
            def geod2(a,b):
                R=6371.0088
                lat1,lon1 = math.radians(a[0]), math.radians(a[1])
                lat2,lon2 = math.radians(b[0]), math.radians(b[1])
                dlat=lat2-lat1; dlon=lon2-lon1
                s=math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
                return 2*R*math.asin(math.sqrt(max(0.0,min(1.0,s))))
            e_lstm = geod2((truth[-1]['lat'], truth[-1]['lng']), (pred[-1]['lat'], pred[-1]['lng'])) if truth and pred else float('nan')
        else:
            e_lstm = float('nan')
        ax.set_title(f"id={rec.get('id','?')}  e_cons={e_cons:.1f}km  e_lstm={e_lstm:.1f}km", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, 'fig_example_predictions_REAL.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {out}")

def roc_placeholder(prelim_csv):
    """無標註則產生『未計算』的中性佈局，避免誤導。
    - 顯示 ROC/AUPRC 坐標軸與對角線，置中標註 Not computed (no labeled anomalies)
    """
    # ROC not computed
    plt.figure(figsize=(5.4,5.4), constrained_layout=True)
    plt.plot([0,1],[0,1],'k--', alpha=0.4)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC 未計算 無標註', fontsize=12, fontweight='bold')
    plt.text(0.5,0.5,'未計算\n無標註', ha='center', va='center', fontsize=12, color='#444', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    out = os.path.join(FIG_DIR, 'fig_roc_placeholder.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {out}")

    # AUPRC not computed
    plt.figure(figsize=(5.4,5.4), constrained_layout=True)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('PR 未計算 無標註', fontsize=12, fontweight='bold')
    plt.text(0.5,0.5,'未計算\n無標註', ha='center', va='center', fontsize=12, color='#444', transform=plt.gca().transAxes)
    plt.grid(True, alpha=0.3)
    out_pr = os.path.join(FIG_DIR, 'fig_pr_placeholder.png')
    plt.savefig(out_pr, dpi=150, bbox_inches='tight'); plt.close()
    print(f"✅ Saved: {out_pr}")

def simple_map_placeholder():
    parts = []
    parts.append("<rect x='50' y='50' width='900' height='500' fill='#e8f5e9' stroke='#1b5e20' stroke-width='2' />")
    # 拿幾條示意折線代表航跡
    polylines = [
        [(100,480),(220,420),(350,390),(540,300),(700,260),(880,200)],
        [(120,460),(240,410),(360,360),(520,280),(680,250),(860,210)],
        [(140,470),(260,430),(380,380),(560,310),(740,280),(900,240)],
    ]
    colors = ['#1565c0','#2e7d32','#ad1457']
    for line, col in zip(polylines, colors):
        pts = ' '.join([f"{x},{y}" for x,y in line])
        parts.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2' opacity='0.9' />")
    parts.append("<text x='500' y='35' text-anchor='middle' font-size='18' fill='#1b5e20'>地圖視覺化（示意）</text>")
    out = os.path.join(FIG_DIR, 'fig_map_visualization_placeholder.svg')
    write_svg(out, parts)
    print(f"✅ Saved: {out}")

def main():
    system_architecture()
    integration_flow()
    prelim = os.path.join(os.path.dirname(FIG_DIR), 'preliminary_table_REAL_quick_filled.csv')
    error_comparison(prelim)
    latency_breakdown(prelim)
    # Try to render real example predictions from live API if available
    example_predictions('http://localhost:5000', k=4, qlen=4, horizon=6)
    roc_placeholder(prelim)
    simple_map_placeholder()
    print("=== Extra figures generated ===")

if __name__ == '__main__':
    main()
