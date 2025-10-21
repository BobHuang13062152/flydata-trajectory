#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
為審稿委員 2 的要求生成圖表：
1. 最小對照表的長條圖（Top-1命中率、ADE、RMSE、延遲）
2. 消融實驗影響圖（bbox/subsequence/directional對精度與速度的影響）
"""
import os
import csv
import matplotlib
matplotlib.use('Agg')  # 無頭模式，適用於Python 3.13
import matplotlib.pyplot as plt
import numpy as np
from figure_style import get_palette, get_hatches, annotate_bars

# 設定中文字型（Windows）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def read_csv_as_dict(csv_path):
    """讀取CSV並轉為字典列表"""
    rows = []
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return rows
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def plot_comparison_table(csv_path, output_dir):
    """繪製最小對照表的長條圖"""
    rows = read_csv_as_dict(csv_path)
    if not rows:
        print(f"No data in {csv_path}")
        return
    
    methods = [r['method'] for r in rows]
    
    # 提取數值（處理可能的空值）
    def safe_float(s):
        try:
            return float(s)
        except:
            return 0.0
    
    hit_rates = [safe_float(r.get('top1_hit_rate', 0)) for r in rows]
    ade_means = [safe_float(r.get('ade_km_mean', 0)) for r in rows]
    rmse_means = [safe_float(r.get('rmse_km_mean', 0)) for r in rows]
    p50s = [safe_float(r.get('end_km_p50', 0)) for r in rows]
    p90s = [safe_float(r.get('end_km_p90', 0)) for r in rows]
    latencies = [safe_float(r.get('latency_ms_mean', 0)) for r in rows]
    
    # 改為 1x3 子圖：不顯示 Top-1，直接聚焦於 ADE/RMSE、端點分位數、延遲
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5))
    fig.suptitle('最小對照表 各方法表現比較', fontsize=16, fontweight='bold')

    # 1. ADE vs RMSE
    ax = axes[0]
    x = np.arange(len(methods))
    width = 0.36
    colors = get_palette(2); h = get_hatches(2)
    bars1 = ax.bar(x - width/4, ade_means, width/2, label='ADE km', color=colors[0], hatch=h[0], edgecolor='#333', linewidth=0.5)
    bars2 = ax.bar(x + width/4, rmse_means, width/2, label='RMSE km', color=colors[1], hatch=h[1], edgecolor='#333', linewidth=0.5)
    ax.set_ylabel('誤差 km', fontsize=11)
    ax.set_title('ADE vs RMSE', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. 端點誤差 P50/P90
    ax = axes[1]
    bars1 = ax.bar(x - width/4, p50s, width/2, label='P50', color=colors[0], hatch=h[0], edgecolor='#333', linewidth=0.5)
    bars2 = ax.bar(x + width/4, p90s, width/2, label='P90', color=colors[1], hatch=h[1], edgecolor='#333', linewidth=0.5)
    ax.set_ylabel('端點誤差 km', fontsize=11)
    ax.set_title('端點誤差分位數', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. 單查詢延遲
    ax = axes[2]
    palette = get_palette(len(methods)); hlat = get_hatches(len(methods))
    bars = ax.bar(methods, latencies, color=palette, edgecolor='#333', linewidth=0.5)
    for b, hh in zip(bars, hlat): b.set_hatch(hh)
    ax.set_ylabel('延遲 ms', fontsize=11)
    ax.set_title('單次查詢延遲', fontsize=12, fontweight='bold')
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    annotate_bars(ax, bars, fmt='{:.0f}')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig_comparison_table.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def plot_ablation_study(csv_path, output_dir):
    """繪製消融實驗影響圖"""
    rows = read_csv_as_dict(csv_path)
    if not rows:
        print(f"No data in {csv_path}")
        return
    
    settings = [r['setting'] for r in rows]
    
    def safe_float(s):
        try:
            return float(s)
        except:
            return 0.0
    
    hit_rates = [safe_float(r.get('top1_hit_rate', 0)) for r in rows]
    ade_means = [safe_float(r.get('ade_km_mean', 0)) for r in rows]
    rmse_means = [safe_float(r.get('rmse_km_mean', 0)) for r in rows]
    p90s = [safe_float(r.get('end_km_p90', 0)) for r in rows]
    latencies = [safe_float(r.get('latency_ms_mean', 0)) for r in rows]
    
    # 簡化標籤
    labels = []
    for s in settings:
        if 'bbox_on+subseq_on+dir_on' in s:
            labels.append('完整\n(baseline)')
        elif 'bbox_off' in s:
            labels.append('無bbox\n預篩')
        elif 'subseq_off' in s:
            labels.append('無子序列\n視窗')
        elif 'dir_off' in s:
            labels.append('無方向\n檢查')
        else:
            labels.append(s.replace('+', '\n'))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('消融實驗 加速策略對精度與延遲', fontsize=16, fontweight='bold')
    
    # 1. 精度指標
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.25
    bars1 = ax.bar(x - width, hit_rates, width, label='命中率', color=get_palette(1)[0], hatch=get_hatches(1)[0], edgecolor='#333', linewidth=0.5)
    # 正規化 ADE 與 RMSE 到 0-1 以便視覺化
    max_ade = max(ade_means) if max(ade_means) > 0 else 1
    max_rmse = max(rmse_means) if max(rmse_means) > 0 else 1
    ade_norm = [a/max_ade for a in ade_means]
    rmse_norm = [r/max_rmse for r in rmse_means]
    bars2 = ax.bar(x, ade_norm, width, label=f'ADE/最大 {max_ade:.1f}', color=get_palette(1)[0], alpha=0.5, edgecolor='#333', linewidth=0.5)
    bars3 = ax.bar(x + width, rmse_norm, width, label=f'RMSE/最大 {max_rmse:.1f}', color=get_palette(1)[0], alpha=0.2, edgecolor='#333', linewidth=0.5)
    ax.set_ylabel('正規化 / 命中率', fontsize=11)
    ax.set_title('精度指標變化', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=hit_rates[0] if hit_rates else 0, color='#0072B2', linestyle='--', alpha=0.5, label='基準命中率')
    
    # 2. 延遲變化
    ax = axes[1]
    colors_lat = ['green' if lat <= latencies[0]*1.1 else 'orange' if lat <= latencies[0]*1.5 else 'red' for lat in latencies]
    bars = ax.bar(labels, latencies, color=colors_lat, edgecolor='#333', linewidth=0.5)
    ax.set_ylabel('延遲 ms', fontsize=11)
    ax.set_title('單次查詢延遲變化', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    if latencies:
        ax.axhline(y=latencies[0], color='#0072B2', linestyle='--', linewidth=2, label='基準延遲')
        ax.legend()
    annotate_bars(ax, bars, fmt='{:.0f}')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'fig_ablation_study.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def main():
    # 設定路徑
    paper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(paper_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    preliminary_csv = os.path.join(paper_dir, 'preliminary_table.csv')
    ablation_csv = os.path.join(paper_dir, 'ablation.csv')
    
    print("=== 開始生成審稿回應圖表 ===")
    
    # 1. 最小對照表
    if os.path.exists(preliminary_csv):
        print(f"\n[1/2] 繪製最小對照表...")
        plot_comparison_table(preliminary_csv, output_dir)
    else:
        print(f"\n[1/2] SKIP: {preliminary_csv} 不存在")
    
    # 2. 消融實驗
    if os.path.exists(ablation_csv):
        print(f"\n[2/2] 繪製消融實驗...")
        plot_ablation_study(ablation_csv, output_dir)
    else:
        print(f"\n[2/2] SKIP: {ablation_csv} 不存在")
    
    print("\n=== 完成！圖表已儲存至", output_dir, "===")

if __name__ == '__main__':
    main()
