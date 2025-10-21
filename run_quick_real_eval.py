#!/usr/bin/env python3
"""Direct runner for quick REAL evaluation."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from tools.evaluate_methods import evaluate
import argparse

args = argparse.Namespace(
    base_url="http://localhost:5000",
    queries=3,          # ultra-fast
    pool=300,           # small candidate pool
    query_len=2,        # short-trajectory friendly
    horizon=3,          # short horizon for speed
    topN=5,
    stride=2,
    end_threshold_km=100.0,  # relaxed for short noisy segments
    time_budget_s=20,
    methods=["DTW", "SUBSEQ_DTW", "EUCLIDEAN", "CONSENSUS"],  # run classic baselines; LSTM added by eval_lstm_quick
    out_csv="paper/preliminary_table_REAL_quick.csv",
    with_ci=True,
    bootstrap=200,
    ablation_out="paper/ablation_REAL_quick.csv",
    seed=42,
    # enable frequent checkpoints so we can plot partials immediately
    checkpoint_every=1,
    checkpoint_ablation=False,
    # use bbox prefilter in identify to improve stability/latency
    use_bbox=True
)

print(f"Quick REAL evaluation starting...")
print(f"  Server: {args.base_url}")
print(f"  Queries: {args.queries}, Pool: {args.pool}, Query Len: {args.query_len}, Horizon: {args.horizon}")
print(f"  Output: {args.out_csv}, {args.ablation_out}")
print()

try:
    evaluate(args)
    print("\n✅ Evaluation complete!")
    print(f"   Check: {args.out_csv}")
    print(f"          {args.ablation_out}")
except Exception as e:
    print(f"\n❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
