from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import math
import time
from glob import glob
import numpy as np
from geopy.distance import geodesic

app = Flask(__name__)
CORS(app)

# 全域儲存與狀態
flight_database = []
progress_data = {'task': None, 'done': True, 'processed': 0, 'total': 0, 'message': '', 'eta_seconds': None, 'start_time': None, 'version': None}
# 支援多任務並行的進度登錄表：task_name -> progress dict
progress_registry = {}

def progress_update(payload: dict):
    """更新全域與每任務的進度，避免多任務互相覆蓋導致前端進度亂跳。
    需求欄位：task(optional), processed, total(optional), done(optional), eta_seconds(optional), message(optional)
    """
    try:
        # 先更新全域（保留相容性）
        progress_data.update(payload)
        task_name = payload.get('task') or progress_data.get('task') or 'default'
        # 合併到任務專屬資料
        cur = dict(progress_registry.get(task_name) or {})
        cur.update({'task': task_name})
        cur.update(payload)
        # 確保必要鍵存在
        cur.setdefault('processed', 0)
        cur.setdefault('total', 0)
        cur.setdefault('done', False)
        cur.setdefault('message', '')
        progress_registry[task_name] = cur
    except Exception:
        # 靜默失敗避免影響主流程
        pass
openflights_index = {}
openflights_routes = {}

ENHANCED_ALGORITHMS_AVAILABLE = False
LSTM_PREDICTOR_AVAILABLE = False
BAYESIAN_OPTIMIZER_AVAILABLE = False

# Algorithms whose distance scores are in kilometers (approx.)
KM_BASED_ALGOS = {'DTW', 'SUBSEQ_DTW', 'EUCLIDEAN', 'FRECHET', 'HAUSDORFF', 'ERP'}

# === 基礎工具與距離函式 ===

def _to_latlng_tuple(p):
    """Accepts dict with lat/lng or tuple/list [lng,lat] and returns (lat, lng)."""
    if p is None:
        return (0.0, 0.0)
    if isinstance(p, dict):
        lat = p.get('lat', p.get('latitude'))
        lng = p.get('lng', p.get('longitude'))
        if lat is None or lng is None:
            # fallback if dict shaped like GeoJSON point: {'type': 'Point', 'coordinates':[lng,lat]}
            coords = p.get('coordinates')
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return float(coords[1]), float(coords[0])
        return float(lat), float(lng)
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        # assume GeoJSON [lng, lat]
        return float(p[1]), float(p[0])
    return (0.0, 0.0)


def _downsample_query_path(path, factor=2, max_points=200):
    try:
        pts = list(path) if isinstance(path, list) else []
        if factor and factor > 1:
            pts = pts[::int(factor)]
        if len(pts) > max_points:
            step = max(1, len(pts) // max_points)
            pts = pts[::step]
        if pts and path and pts[-1] is not path[-1]:
            pts.append(path[-1])
        return pts
    except Exception:
        return path


def _bearing_deg(lat1, lon1, lat2, lon2):
    try:
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360.0) % 360.0
    except Exception:
        return 0.0


def _heading_diff_deg(a, b):
    d = ((a - b + 540.0) % 360.0) - 180.0
    return abs(d)


def _point_in_bbox(lat, lng, bbox):
    try:
        return (bbox['south'] <= lat <= bbox['north']) and (bbox['west'] <= lng <= bbox['east'])
    except Exception:
        return True


def _bbox_intersects(b1, b2):
    try:
        # two axis-aligned rectangles intersect if not separated
        return not (b1['east'] < b2['west'] or b1['west'] > b2['east'] or b1['north'] < b2['south'] or b1['south'] > b2['north'])
    except Exception:
        return True


def _flight_bbox(coords):
    try:
        lats = []
        lngs = []
        for c in coords:
            lat, lng = _to_latlng_tuple(c)
            lats.append(lat); lngs.append(lng)
        if not lats:
            return None
        return {
            'north': max(lats), 'south': min(lats),
            'east': max(lngs), 'west': min(lngs)
        }
    except Exception:
        return None


def _parse_bbox_from_args(args):
    try:
        north = args.get('north', type=float)
        south = args.get('south', type=float)
        east = args.get('east', type=float)
        west = args.get('west', type=float)
        if None in (north, south, east, west):
            return None
        return {'north': float(north), 'south': float(south), 'east': float(east), 'west': float(west)}
    except Exception:
        return None


def dtw_distance(path1, path2):
    """DTW using geodesic distance between points."""
    n, m = len(path1), len(path2)
    if n == 0 or m == 0:
        return float('inf')
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        lat1, lng1 = _to_latlng_tuple(path1[i - 1])
        for j in range(1, m + 1):
            lat2, lng2 = _to_latlng_tuple(path2[j - 1])
            cost = geodesic((lat1, lng1), (lat2, lng2)).kilometers
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def euclidean_distance(path1, path2):
    """Simplified Euclidean over min-length with tail penalty per step."""
    n, m = len(path1), len(path2)
    if n == 0 or m == 0:
        return float('inf')
    L = min(n, m)
    s = 0.0
    for i in range(L):
        lat1, lng1 = _to_latlng_tuple(path1[i])
        lat2, lng2 = _to_latlng_tuple(path2[i])
        s += geodesic((lat1, lng1), (lat2, lng2)).kilometers
    tail = abs(n - m)
    return float(s + 0.5 * tail)


def lcss_distance(path1, path2, epsilon_km=1.0):
    n, m = len(path1), len(path2)
    if n == 0 or m == 0:
        return float('inf')
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        lat1, lng1 = _to_latlng_tuple(path1[i - 1])
        for j in range(1, m + 1):
            lat2, lng2 = _to_latlng_tuple(path2[j - 1])
            if geodesic((lat1, lng1), (lat2, lng2)).kilometers <= epsilon_km:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    # convert to a distance-like measure: fewer matches => larger value
    lcss_len = dp[n, m]
    return float(max(n, m) - lcss_len)


def frechet_distance(path1, path2):
    """Discrete Fréchet distance using geodesic metric."""
    n, m = len(path1), len(path2)
    if n == 0 or m == 0:
        return float('inf')
    ca = np.full((n, m), -1.0)

    def dist(i, j):
        lat1, lng1 = _to_latlng_tuple(path1[i])
        lat2, lng2 = _to_latlng_tuple(path2[j])
        return geodesic((lat1, lng1), (lat2, lng2)).kilometers

    def _c(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = dist(0, 0)
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), dist(i, 0))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), dist(0, j))
        elif i > 0 and j > 0:
            ca[i, j] = max(min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), dist(i, j))
        else:
            ca[i, j] = float('inf')
        return ca[i, j]

    return float(_c(n - 1, m - 1))


def hausdorff_distance(path1, path2):
    """Undirected Hausdorff distance between two point sets (geodesic)."""
    if not path1 or not path2:
        return float('inf')
    def directed(A, B):
        max_min = 0.0
        for a in A:
            latA, lngA = _to_latlng_tuple(a)
            best = float('inf')
            for b in B:
                latB, lngB = _to_latlng_tuple(b)
                d = geodesic((latA, lngA), (latB, lngB)).kilometers
                if d < best:
                    best = d
            if best > max_min:
                max_min = best
        return max_min
    return float(max(directed(path1, path2), directed(path2, path1)))


def compute_dynamic_horizon(path, min_steps=1, max_steps=1000, base_steps=6):
    """Heuristic dynamic horizon: proportional to path length, clamped.
    Uses half of current path length as baseline, at least base_steps.
    """
    try:
        L = len(path) if path else 0
        h = max(int(base_steps or 0), int(round(L * 0.5)))
        return max(int(min_steps or 1), min(int(max_steps or 1000), h))
    except Exception:
        return max(int(min_steps or 1), min(int(max_steps or 1000), int(base_steps or 6)))


def edr_distance(path1, path2, epsilon_km=1.0):
    n, m = len(path1), len(path2)
    if n == 0 or m == 0:
        return float('inf')
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        lat1, lng1 = _to_latlng_tuple(path1[i - 1])
        for j in range(1, m + 1):
            lat2, lng2 = _to_latlng_tuple(path2[j - 1])
            d = geodesic((lat1, lng1), (lat2, lng2)).kilometers
            sub_cost = 0 if d <= epsilon_km else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub_cost)
    return float(dp[n][m])


def erp_distance(path1, path2, gap_penalty_km=1.0):
    n, m = len(path1), len(path2)
    if n == 0 or m == 0:
        return float('inf')
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty_km
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty_km
    for i in range(1, n + 1):
        lat1, lng1 = _to_latlng_tuple(path1[i - 1])
        for j in range(1, m + 1):
            lat2, lng2 = _to_latlng_tuple(path2[j - 1])
            match = geodesic((lat1, lng1), (lat2, lng2)).kilometers + dp[i - 1, j - 1]
            gap_i = gap_penalty_km + dp[i - 1, j]
            gap_j = gap_penalty_km + dp[i, j - 1]
            dp[i, j] = min(match, gap_i, gap_j)
    return float(dp[n, m])


def subsequence_dtw_distance(query_path, full_coords, stride=1):
    qn = len(query_path)
    fn = len(full_coords)
    if qn < 2 or fn < qn:
        return float('inf'), -1
    best = float('inf')
    best_idx = -1
    for s in range(0, fn - qn + 1, max(1, int(stride))):
        win = full_coords[s:s + qn]
        d = dtw_distance(query_path, win)
        if d < best:
            best = d
            best_idx = s
    return float(best), int(best_idx)


def subsequence_distance_generic(query_path, full_coords, algo, stride=1):
    qn = len(query_path)
    fn = len(full_coords)
    if qn < 2 or fn < qn:
        return float('inf'), -1
    best = float('inf')
    best_idx = -1
    for s in range(0, fn - qn + 1, max(1, int(stride))):
        win = full_coords[s:s + qn]
        if algo == 'DTW':
            d = dtw_distance(query_path, win)
        elif algo == 'EUCLIDEAN':
            d = euclidean_distance(query_path, win)
        elif algo == 'LCSS':
            d = lcss_distance(query_path, win)
        elif algo == 'FRECHET':
            d = frechet_distance(query_path, win)
        elif algo == 'HAUSDORFF':
            d = hausdorff_distance(query_path, win)
        elif algo == 'EDR':
            d = edr_distance(query_path, win)
        elif algo == 'ERP':
            d = erp_distance(query_path, win)
        else:
            d = dtw_distance(query_path, win)
        if d < best:
            best = d
            best_idx = s
    return float(best), int(best_idx)


def _calibrate_similarity_percent(items, score_key='score', floor_pct=5.0, top_pct=60.0, ref_percentile=80.0):
    """Attach a calibrated similarity_percent to each dict in items based on their distance score.
    - Smaller score => higher percent. d_min -> top_pct, d_ref (percentile) -> floor_pct.
    - Clamp to [floor_pct, top_pct].
    - Does not normalize to sum 100; intended for readable display (5-60%).
    """
    try:
        if not items:
            return items
        dists = [float(x.get(score_key, float('inf'))) for x in items if np.isfinite(x.get(score_key, float('inf')))]
        if not dists:
            for x in items:
                x['similarity_percent'] = int(floor_pct)
            return items
        d0 = float(min(dists))
        try:
            d_ref = float(np.percentile(dists, max(0.0, min(100.0, float(ref_percentile)))))
        except Exception:
            d_ref = d0
        if d_ref <= d0:
            # fallback: all equal or single item
            for x in items:
                x['similarity_percent'] = int(round((top_pct + floor_pct) * 0.5))
            return items
        rng = d_ref - d0
        for x in items:
            d = float(x.get(score_key, d_ref))
            if not np.isfinite(d):
                p = floor_pct
            elif d <= d0:
                p = top_pct
            elif d >= d_ref:
                p = floor_pct
            else:
                # linear map between d0..d_ref -> top..floor
                t = (d - d0) / rng
                p = top_pct + (floor_pct - top_pct) * t
            x['similarity_percent'] = int(max(floor_pct, min(top_pct, round(p))))
        return items
    except Exception:
        # best effort
        for x in items:
            x['similarity_percent'] = int(floor_pct)
        return items


def _map_similarity_percent(items, score_key='score', mode='calibrated', floor_pct=5.0, top_pct=60.0, ref_percentile=80.0, require_zero_for_100=False, epsilon_for_perfect=1e-6):
    """General mapper for similarity_percent.
    mode:
      - 'calibrated' (default): map d_min->top_pct, d_ref(percentile)->floor_pct.
      - 'minmax': map d_min->top_pct, d_max->floor_pct (full spread 0–100 if floor=0, top=100).
    """
    mode = (mode or 'calibrated').lower()
    if not items:
        return items
    # collect finite distances
    dists = [float(x.get(score_key, float('inf'))) for x in items if np.isfinite(x.get(score_key, float('inf')))]
    if not dists:
        for x in items:
            x['similarity_percent'] = int(floor_pct)
        return items
    d0 = float(min(dists))
    if mode == 'minmax':
        d1 = float(max(dists))
        if d1 <= d0:
            # all identical
            for x in items:
                x['similarity_percent'] = int(round((top_pct + floor_pct) * 0.5))
            return items
        span = d1 - d0
        for x in items:
            d = float(x.get(score_key, d1))
            if not np.isfinite(d):
                p = floor_pct
            elif d <= d0:
                # perfect match only if distance is (near) zero, otherwise cap below top
                if require_zero_for_100 and not (abs(d0) <= epsilon_for_perfect):
                    p = min(top_pct, 99.0)
                else:
                    p = top_pct
            elif d >= d1:
                p = floor_pct
            else:
                t = (d - d0) / span  # 0..1
                p = top_pct + (floor_pct - top_pct) * t  # linear descending
            x['similarity_percent'] = int(max(min(p, top_pct), floor_pct))
        return items
    # default calibrated mode
    return _calibrate_similarity_percent(items, score_key=score_key, floor_pct=floor_pct, top_pct=top_pct, ref_percentile=ref_percentile)


def _absolute_similarity_percent(items, avg_key='avg_km_per_step', floor_pct=0.0, top_pct=100.0, clip_km_per_step=2.0, gamma=1.2, require_zero_for_100=True, epsilon=1e-6, cap_below_perfect=1.0):
    """Map an average km-per-step deviation to 0-100% absolutely, not relatively.
    avg=0 -> top_pct; avg>=clip -> floor_pct; smooth by exponent gamma.
    If require_zero_for_100, cap at 99 when avg > epsilon.
    """
    if not items:
        return items
    clip = max(1e-6, float(clip_km_per_step))
    lo = float(floor_pct); hi = float(top_pct)
    for x in items:
        avg = x.get(avg_key, None)
        if avg is None or not np.isfinite(avg):
            p = lo
        else:
            a = max(0.0, 1.0 - float(avg) / clip)
            p = lo + (hi - lo) * (a ** float(gamma))
            if require_zero_for_100 and float(avg) > epsilon:
                # prevent near-100 scores unless deviation is virtually zero
                try:
                    cap = float(cap_below_perfect)
                except Exception:
                    cap = 1.0
                p = min(p, hi - max(0.0, cap))
        x['similarity_percent'] = int(max(lo, min(hi, round(p))))
    return items


def simple_predict_trajectory(input_path, prediction_horizon=10):
    if not input_path or len(input_path) < 2:
        return []
    lat1, lng1 = _to_latlng_tuple(input_path[-2])
    lat2, lng2 = _to_latlng_tuple(input_path[-1])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    out = []
    decay = 0.9
    cur_lat, cur_lng = lat2, lng2
    vlat, vlng = dlat, dlng
    for _ in range(int(prediction_horizon)):
        cur_lat += vlat
        cur_lng += vlng
        out.append({'lat': cur_lat, 'lng': cur_lng})
        vlat *= decay
        vlng *= decay
    return out


# === 深度學習：LSTM 预测（可選） ===
def _try_import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        return torch, nn
    except Exception:
        return None, None


def _load_lstm_model(device='cpu'):
    """Lazy-load LSTM model weights from models/lstm_forecaster.pt if present.
    Returns (model, device) or (None, None) if unavailable.
    """
    torch, nn = _try_import_torch()
    if torch is None:
        return None, None
    class TinyLSTMForecast(nn.Module):
        def __init__(self, input_size=2, hidden_size=64, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 2)  # predict delta lat,lng
        def forward(self, x, h=None):
            # x: [B,T,2]
            out, h = self.lstm(x, h)
            y = self.head(out[:, -1:, :])  # [B,1,2]
            return y, h
    model = TinyLSTMForecast()
    model.to(device)
    # try load weights
    cwd = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(cwd, 'models', 'lstm_forecaster.pt')
    if os.path.exists(wpath):
        try:
            state = torch.load(wpath, map_location=device)
            model.load_state_dict(state)
            model.eval()
            return model, device
        except Exception:
            return None, None
    # weights missing
    return None, None


def lstm_predict_trajectory(input_path, prediction_horizon=10):
    """Predict future points using a tiny LSTM model if available.
    Expects a file models/lstm_forecaster.pt. Returns list of {lat,lng}.
    """
    if not input_path or len(input_path) < 3:
        return []
    torch, nn = _try_import_torch()
    if torch is None:
        raise RuntimeError('PyTorch 未安裝')
    device = 'cpu'
    model, device = _load_lstm_model(device)
    if model is None:
        raise FileNotFoundError('缺少模型權重 models/lstm_forecaster.pt')
    # prepare sequence
    coords = [(_to_latlng_tuple(p)[0], _to_latlng_tuple(p)[1]) for p in input_path]
    arr = np.array(coords, dtype=np.float32)
    # normalize to improve numerical stability
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    norm = (arr - mean) / std
    x = torch.from_numpy(norm[None, :, :]).to(device)
    out = []
    with torch.no_grad():
        h = None
        # prime hidden state using history, then iterative rollout
        _, h = model(x, h)
        last = x[:, -1:, :]
        for _ in range(int(prediction_horizon)):
            delta, h = model(last, h)  # predict delta in normalized space
            next_pos = last + delta
            denorm = next_pos.cpu().numpy()[0, 0, :] * std + mean
            out.append({'lat': float(denorm[0]), 'lng': float(denorm[1])})
            last = next_pos
    return out


def _continue_from_match(full_coords, match_start, k=15):
    """Return continuation points from matched window end forward (up to k)."""
    try:
        end_idx = match_start + k if match_start is not None and match_start >= 0 else None
        # If we have a match window, continue from its end; otherwise from path end
        start = match_start if (match_start is not None and match_start >= 0) else (len(full_coords) - 1)
        tail = full_coords[start: start + k] if start is not None else []
        return [{'lat': p[1], 'lng': p[0]} for p in tail]
    except Exception:
        return []


@app.route('/api/forecast-consensus', methods=['POST'])
def forecast_consensus():
    """Blend Top-5 continuations with heuristic for a more stable forecast.
    Body: [{lat,lng}, ...]
    Query: algo, subseq, stride, fast, directional, topN, horizon | horizon=auto | horizon=distance with step_km & distance_km
    """
    try:
        body = request.json
        # 允許 body 是 { query_path: [...], top_flights: [id...] } 或直接為路徑陣列
        if isinstance(body, dict):
            query_path = body.get('query_path') or body.get('path') or []
            limit_to_ids = set(body.get('top_flights') or [])
        else:
            query_path = body or []
            limit_to_ids = set()
        if not query_path or len(query_path) < 2:
            return jsonify({'error': '路徑數據不足'}), 400

    algo = request.args.get('algo', 'SUBSEQ_DTW').upper()
        use_subseq = request.args.get('subseq', 'true').lower() == 'true' or algo == 'SUBSEQ_DTW'
        stride = request.args.get('stride', 1, type=int)
        fast = request.args.get('fast', 'true').lower() == 'true'
        directional = request.args.get('directional', 'true').lower() == 'true'
        topN = max(1, min(5, request.args.get('topN', 5, type=int)))

        # horizon parameters
        min_steps = request.args.get('min_steps', 1, type=int)
        max_steps = request.args.get('max_steps', 1000, type=int)
        base_steps = request.args.get('base_steps', 6, type=int)
        step_km = request.args.get('step_km', None, type=float) or request.args.get('km_per_step', None, type=float)
        distance_km = request.args.get('distance_km', None, type=float)
        horizon_strategy = (request.args.get('horizon_strategy', '') or '').lower()
        min_steps = max(1, min_steps); max_steps = max(min_steps, max_steps); base_steps = max(0, base_steps)

        horizon_raw = request.args.get('horizon', 'auto')
        hmode = 'fixed'
        if (horizon_strategy == 'distance') or (isinstance(horizon_raw, str) and horizon_raw.lower() == 'distance'):
            try:
                if step_km and distance_km and step_km > 0 and distance_km > 0:
                    horizon = int(math.ceil(float(distance_km) / float(step_km)))
                else:
                    horizon = compute_dynamic_horizon(query_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
                hmode = 'distance'
            except Exception:
                horizon = compute_dynamic_horizon(query_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
                hmode = 'auto'
            horizon = max(min_steps, min(max_steps, int(horizon)))
        elif isinstance(horizon_raw, str) and horizon_raw.lower() == 'auto':
            horizon = compute_dynamic_horizon(query_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
            hmode = 'auto'
        else:
            try:
                horizon = int(horizon_raw)
            except Exception:
                horizon = 10
            if horizon <= 0:
                horizon = compute_dynamic_horizon(query_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
                hmode = 'auto'
            else:
                horizon = max(min_steps, min(max_steps, horizon))
                hmode = 'fixed'

        # Pre-filter by query bbox to reduce candidate set
        def _avg_step_km_local(path):
            try:
                if not path or len(path) < 2:
                    return None
                total = 0.0
                for i in range(1, len(path)):
                    a = _to_latlng_tuple(path[i-1]); b = _to_latlng_tuple(path[i])
                    total += geodesic(a, b).kilometers
                return total / (len(path) - 1)
            except Exception:
                return None

        # optional fast downsample of query
        if fast:
            query_path = _downsample_query_path(query_path, factor=2, max_points=200)

        qbbox = _flight_bbox(query_path)
        candidates = flight_database
        # 如有 identify 的 Top IDs 傳入，僅限縮到這些候選，避免重複掃全庫
        if limit_to_ids:
            try:
                idset = set(str(x) for x in limit_to_ids)
                candidates = [f for f in flight_database if str(f.get('id')) in idset]
            except Exception:
                candidates = flight_database
        # 若仍為全庫，使用擴展 bbox 先做粗篩
        if candidates is flight_database and qbbox:
            try:
                pad_km = 150.0
                lat_pad = pad_km / 111.0
                lng_pad = pad_km / (111.0 * max(0.2, math.cos(math.radians((qbbox['north']+qbbox['south'])/2))))
                qbbox_exp = {
                    'north': qbbox['north'] + lat_pad,
                    'south': qbbox['south'] - lat_pad,
                    'east': qbbox['east'] + lng_pad,
                    'west': qbbox['west'] - lng_pad,
                }
                candidates = [f for f in flight_database if f.get('bbox') and _bbox_intersects(f['bbox'], qbbox_exp)]
            except Exception:
                pass

        # similarity search with progress
        try:
            total = len(candidates)
            progress_update({'task': 'forecast-consensus', 'done': False, 'processed': 0, 'total': total, 'message': '共識預測：搜尋相似航班', 'start_time': time.time(), 'eta_seconds': None, 'version': time.time()})
        except Exception:
            pass
        sims = []
        processed = 0
        for f in candidates:
            coords = f.get('coordinates') or []
            if len(coords) < 2:
                continue
            try:
                processed += 1
                # Fast mode: light downsample for long candidates to speed up DTW windowing
                coords_eval = coords
                if fast and len(coords) > 1200:
                    try:
                        step_ds = 2
                        coords_eval = coords[::step_ds]
                        if coords_eval and coords_eval[-1] != coords[-1]:
                            coords_eval = coords_eval + [coords[-1]]
                    except Exception:
                        coords_eval = coords
                if use_subseq:
                    # Support generic subsequence search for multiple algorithms
                    base_algo = algo
                    if base_algo == 'SUBSEQ_DTW':
                        base_algo = 'DTW'
                    try:
                        dist, start = subsequence_distance_generic(query_path, coords_eval, base_algo, stride)
                    except Exception:
                        # Fallback to DTW-based subsequence if generic fails
                        dist, start = subsequence_dtw_distance(query_path, coords_eval, stride)
                    # In fast mode the start index is relative to coords_eval; avoid mis-slicing
                    if fast:
                        start = -1
                else:
                    start = -1
                    # Direct full-trajectory distance by selected algorithm
                    if algo == 'DTW' or algo == 'SUBSEQ_DTW':
                        dist = dtw_distance(query_path, coords_eval)
                    elif algo == 'EUCLIDEAN':
                        dist = euclidean_distance(query_path, coords_eval)
                    elif algo == 'LCSS':
                        dist = lcss_distance(query_path, coords_eval)
                    elif algo == 'FRECHET' or algo == 'DFD':
                        dist = frechet_distance(query_path, coords_eval)
                    elif algo == 'HAUSDORFF':
                        dist = hausdorff_distance(query_path, coords_eval)
                    elif algo == 'EDR':
                        dist = edr_distance(query_path, coords_eval)
                    elif algo == 'ERP':
                        dist = erp_distance(query_path, coords_eval)
                    else:
                        dist = dtw_distance(query_path, coords_eval)
                if not np.isfinite(dist) or dist < 0:
                    continue
                sims.append({'flight': f['id'], 'score': float(dist), 'start': int(start) if start is not None else -1, 'coords': coords})
                # Update progress more frequently for smoother UI (every ~100 items)
                if processed % 100 == 0 or processed == total:
                    try:
                        st = None
                        try:
                            st = (progress_registry.get('forecast-consensus') or {}).get('start_time')
                        except Exception:
                            st = None
                        elapsed = max(1e-3, time.time() - (st or time.time()))
                        rate = processed / elapsed
                        rem = max(0, total - processed)
                        eta = rem / rate if rate > 0 else None
                        progress_update({'task': 'forecast-consensus', 'processed': processed, 'eta_seconds': round(eta, 1) if eta is not None else None})
                    except Exception:
                        pass
            except Exception:
                continue

        sims.sort(key=lambda x: x['score'])
        top = sims[:topN]

        # continuations
        conts = []
        for s in top:
            tail = _continue_from_match(s['coords'], s.get('start', -1), k=horizon)
            if tail:
                conts.append({'weight': 1.0 / max(1e-6, s['score']), 'path': tail})

        heuristic = simple_predict_trajectory(query_path, prediction_horizon=horizon)
        if heuristic:
            conts.append({'weight': 0.7, 'path': heuristic})

        if not conts:
            try:
                progress_update({'task': 'forecast-consensus', 'processed': processed, 'done': True, 'eta_seconds': 0, 'message': '完成'})
            except Exception:
                pass
            return jsonify({'consensus': heuristic or [], 'topN_used': 0, 'horizon': horizon, 'horizon_mode': hmode})
        L = max(len(c['path']) for c in conts)
        out = []
        for i in range(L):
            num_lat = 0.0; num_lng = 0.0; den = 0.0
            for c in conts:
                if i < len(c['path']):
                    w = float(c['weight'])
                    num_lat += w * float(c['path'][i]['lat'])
                    num_lng += w * float(c['path'][i]['lng'])
                    den += w
            if den > 0:
                out.append({'lat': num_lat / den, 'lng': num_lng / den})
        try:
            progress_update({'task': 'forecast-consensus', 'processed': processed, 'done': True, 'eta_seconds': 0, 'message': '完成'})
        except Exception:
            pass
        return jsonify({'consensus': out, 'topN_used': len(top), 'horizon': horizon, 'horizon_mode': hmode})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def load_flight_database():
    global flight_database
    flight_database = []
    # 載入所有 flights*.geojson 檔
    cwd = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob(os.path.join(cwd, 'flights*.geojson')))
    # 僅用原始檔，忽略任何縫合/加工檔案（檔名含 stitched）
    files = [p for p in files if 'stitched' not in os.path.basename(p).lower()]
    # 兼容 demo 檔
    demo = os.path.join(cwd, 'demo_ship_trajectory.geojson')
    if os.path.exists(demo):
        files.append(demo)
    for fp in files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            feats = data.get('features') or []
            for idx, feat in enumerate(feats):
                geom = (feat.get('geometry') or {})
                props = feat.get('properties') or {}
                gtype = geom.get('type')
                if gtype == 'LineString':
                    coords = geom.get('coordinates') or []
                elif gtype == 'MultiLineString':
                    lines = geom.get('coordinates') or []
                    coords = [pt for line in lines for pt in line]
                else:
                    continue
                if not coords:
                    continue
                fid = str(props.get('id') or props.get('flight') or props.get('name') or f"{os.path.basename(fp)}#{idx}")
                seg = int(props.get('segment') or 0)
                flight_database.append({
                    'id': fid,
                    'segment': seg,
                    'coordinates': coords,
                    'bbox': _flight_bbox(coords),
                    'properties': props,
                    'times': props.get('times', []),
                    'source_file': os.path.basename(fp)
                })
        except Exception as e:
            print(f"[警告] 載入 {fp} 失敗: {e}")
    print(f"[資訊] 載入航班數據: {len(flight_database)} 筆")


def load_openflights_data():
    """Try to load OpenFlights airports and routes from workspace if available."""
    try:
        from openflights_adapter import load_airports, load_routes, build_indexes, load_airports_dafif
    except Exception as e:
        print(f"[警告] 無法載入 openflights_adapter: {e}")
        return
    global openflights_index, openflights_routes
    cwd = os.path.dirname(os.path.abspath(__file__))
    # Common filenames users might drop into this folder
    candidates_airports = [
        os.path.join(cwd, 'airports.dat'),
        os.path.join(cwd, 'airports-extended.dat'),
    os.path.join(cwd, 'airports-dafif.dat'),
    os.path.join(cwd, 'airports-dafif.csv'),
        os.path.join(cwd, 'airports.csv'),
    ]
    candidates_routes = [
        os.path.join(cwd, 'routes.dat'),
        os.path.join(cwd, 'routes.csv'),
    ]
    ap_file = next((p for p in candidates_airports if os.path.exists(p)), None)
    rt_file = next((p for p in candidates_routes if os.path.exists(p)), None)
    if not ap_file and not rt_file:
        print('[資訊] 未找到 OpenFlights 檔案（airports.dat/routes.dat），略過載入')
        return
    # Choose proper loader: DAFIF has lon,lat order and fewer columns
    if ap_file and os.path.basename(ap_file).lower().startswith('airports-dafif'):
        airports = load_airports_dafif(ap_file)
    else:
        airports = load_airports(ap_file) if ap_file else []
    openflights_index = build_indexes(airports) if airports else {'airports': [], 'by_iata': {}, 'by_icao': {}}
    openflights_routes = load_routes(rt_file) if rt_file else {}
    print(f"[資訊] OpenFlights 載入完成: 機場 {len(openflights_index.get('airports', []))} 筆，航線 {sum(len(v) for v in openflights_routes.values())} 條")


def detect_lstm_availability():
    global LSTM_PREDICTOR_AVAILABLE
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        wpath = os.path.join(cwd, 'models', 'lstm_forecaster.pt')
        LSTM_PREDICTOR_AVAILABLE = os.path.exists(wpath)
    except Exception:
        LSTM_PREDICTOR_AVAILABLE = False


def detect_enhanced_availability():
    """Set ENHANCED_ALGORITHMS_AVAILABLE if enhanced_algorithms.py exists and can be imported."""
    global ENHANCED_ALGORITHMS_AVAILABLE
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(cwd, 'enhanced_algorithms.py')
        if os.path.exists(path):
            try:
                __import__('enhanced_algorithms')
                ENHANCED_ALGORITHMS_AVAILABLE = True
                return
            except Exception:
                ENHANCED_ALGORITHMS_AVAILABLE = True  # File exists; mark available though not wired-in
                return
        ENHANCED_ALGORITHMS_AVAILABLE = False
    except Exception:
        ENHANCED_ALGORITHMS_AVAILABLE = False


def detect_bayesian_optimizer_availability():
    """Set BAYESIAN_OPTIMIZER_AVAILABLE if bayesian_optimizer.py exists in cwd."""
    global BAYESIAN_OPTIMIZER_AVAILABLE
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(cwd, 'bayesian_optimizer.py')
        BAYESIAN_OPTIMIZER_AVAILABLE = os.path.exists(path)
    except Exception:
        BAYESIAN_OPTIMIZER_AVAILABLE = False


# API 端點
@app.route('/api/flights')
def get_flights():
    limit = request.args.get('limit', 0, type=int) or len(flight_database)
    out = []
    for f in flight_database[:limit]:
        out.append({'id': f['id'], 'segment': f['segment'], 'points': len(f['coordinates']), 'source': f['source_file']})
    return jsonify(out)


@app.route('/api/flights-with-paths')
def get_flights_with_paths():
    limit = request.args.get('limit', 0, type=int) or len(flight_database)
    out = []
    for f in flight_database[:limit]:
        path = [{'lat': c[1], 'lng': c[0]} for c in f['coordinates']]
        out.append({'id': f['id'], 'segment': f['segment'], 'points': len(f['coordinates']), 'source': f['source_file'], 'path': path})
    return jsonify(out)


@app.route('/api/flight/<flight_id>')
def get_single_flight(flight_id):
    for f in flight_database:
        if f['id'] == flight_id:
            coords = [{'lat': c[1], 'lng': c[0]} for c in f['coordinates']]
            return jsonify({'id': f['id'], 'coordinates': coords, 'times': f.get('times') or [], 'segment': f['segment'], 'properties': f.get('properties') or {}, 'type': 'real'})
    return jsonify({'error': f'找不到航班 {flight_id}'}), 404


@app.route('/api/identify', methods=['POST'])
def identify_similar_flights():
    try:
        query_path = request.json or []
        if not query_path or len(query_path) < 2:
            return jsonify({'error': '路徑數據不足'}), 400

        # 基本參數
        algorithm = request.args.get('algo', 'DTW').upper()
        use_enhanced = request.args.get('enhanced', 'false').lower() == 'true'
        use_subsequence = request.args.get('subseq', 'false').lower() == 'true' or algorithm == 'SUBSEQ_DTW'
        stride = request.args.get('stride', 1, type=int)
        fast_mode = request.args.get('fast', 'false').lower() == 'true' or request.args.get('approx', 'false').lower() == 'true'
        apply_directional = request.args.get('directional', 'true').lower() == 'true'
        heading_max_diff = request.args.get('heading_max_diff', 60, type=int)

        # 顯示/校準參數
        calibrate_percent = request.args.get('calibrate_percent', 'true').lower() == 'true'
        percent_mode = (request.args.get('percent_mode', 'absolute') or 'absolute').lower()
        floor_pct_q = request.args.get('floor_pct', None, type=float)
        top_pct_q = request.args.get('top_pct', None, type=float)
        ref_pct_q = request.args.get('ref_pct', None, type=float)

        # BBOX 參數
        bbox_filter = _parse_bbox_from_args(request.args)
        bbox_strict = request.args.get('bbox_strict', 'false').lower() == 'true'
        min_span_ratio = request.args.get('min_span_ratio', None, type=float)
        bbox_mode = request.args.get('bbox_mode', 'intersect').lower()
        bbox_window_min_coverage = request.args.get('bbox_window_min_coverage', None, type=float)

        # 候選集合（先用 bbox + 距離中心粗篩，再按最接近終點先掃）
        if bbox_filter:
            candidates = [f for f in flight_database if f.get('bbox') and _bbox_intersects(f['bbox'], bbox_filter)]
        else:
            candidates = flight_database
        # 中心近鄰預挑：取查詢的中心點，先挑離中心較近的候選（減少掃描量）
        try:
            qcenter = _to_latlng_tuple(query_path[len(query_path)//2])
            def center_dist(f):
                bb = f.get('bbox')
                if not bb: return 1e9
                clat = (bb['north'] + bb['south'])/2.0
                clng = (bb['east'] + bb['west'])/2.0
                return geodesic(qcenter, (clat, clng)).kilometers
            # 預選前 N（按資料量自適應）：避免一次性全庫 DTW
            Npref = min(len(candidates), max(500, int(len(candidates)*0.2)))
            candidates = sorted(candidates, key=center_dist)[:Npref]
        except Exception:
            pass

        # 進度初始化
        try:
            progress_update({'task': 'identify', 'done': False, 'processed': 0, 'total': len(candidates), 'message': '分析中', 'start_time': time.time(), 'eta_seconds': None, 'version': time.time()})
        except Exception:
            pass

        if fast_mode:
            query_path = _downsample_query_path(query_path, factor=2, max_points=200)

        # 預計算查詢跨度與方位
        try:
            q_start = query_path[0]; q_end = query_path[-1]
            q_span_km = geodesic((_to_latlng_tuple(q_start)), (_to_latlng_tuple(q_end))).kilometers
            q_bearing = _bearing_deg(q_start['lat'], q_start['lng'], q_end['lat'], q_end['lng'])
        except Exception:
            q_span_km = None
            q_bearing = None

        sims = []
        processed = 0
        # 時間上限：避免單次分析拖超過 30s（可調）
        time_budget_s = request.args.get('time_budget_s', 30, type=int) or 30
        t0 = time.time()
        for flight in candidates:
            if time.time() - t0 > time_budget_s:
                # 標註截斷，便於前端提示
                progress_update({'task': 'identify', 'processed': processed, 'message': f'時間用盡，提前截斷（{processed}/{len(candidates)}）'})
                break
            processed += 1

            # Update progress more frequently for smoother UI (every ~100 items)
            if processed % 100 == 0 or processed == len(candidates):
                try:
                    st = None
                    try:
                        st = (progress_registry.get('identify') or {}).get('start_time')
                    except Exception:
                        st = None
                    elapsed = max(1e-3, time.time() - (st or time.time()))
                    rate = processed / elapsed
                    rem = max(0, len(candidates) - processed)
                    eta = rem / rate if rate > 0 else None
                    progress_update({'task': 'identify', 'processed': processed, 'eta_seconds': round(eta, 1) if eta is not None else None})
                except Exception:
                    pass

            coords = flight.get('coordinates') or []
            if len(coords) < 2:
                continue

            # 在 fast 模式下也對候選路徑做下採樣以提升速度
            coords_eval = coords
            if fast_mode and len(coords) > 800:
                try:
                    step_ds = 2
                    coords_eval = coords[::step_ds]
                    if coords_eval and coords_eval[-1] != coords[-1]:
                        coords_eval = coords_eval + [coords[-1]]
                except Exception:
                    coords_eval = coords

            # 粗略 bbox 過濾 + 嚴格模式 path_inside
            if bbox_filter:
                if bbox_mode == 'path_inside':
                    try:
                        if not all(_point_in_bbox(pt[1], pt[0], bbox_filter) for pt in coords):
                            continue
                    except Exception:
                        pass

            try:
                # 計算距離
                if use_enhanced and ENHANCED_ALGORITHMS_AVAILABLE:
                    dist = dtw_distance(query_path, coords)
                    start = -1
                    confidence = 0.8
                else:
                    confidence = 0.6
                    start = -1
                    if use_subsequence or algorithm == 'SUBSEQ_DTW':
                        dist, start_idx = subsequence_dtw_distance(query_path, coords_eval, stride)
                        start = start_idx
                    elif algorithm == 'DTW':
                        dist = dtw_distance(query_path, coords_eval)
                    elif algorithm == 'EUCLIDEAN':
                        dist = euclidean_distance(query_path, coords_eval)
                    elif algorithm == 'LCSS':
                        dist = lcss_distance(query_path, coords_eval)
                    elif algorithm == 'FRECHET':
                        dist = frechet_distance(query_path, coords_eval)
                    elif algorithm == 'HAUSDORFF':
                        dist = hausdorff_distance(query_path, coords_eval)
                    elif algorithm == 'EDR':
                        dist = edr_distance(query_path, coords_eval)
                    elif algorithm == 'ERP':
                        dist = erp_distance(query_path, coords_eval)
                    else:
                        dist = dtw_distance(query_path, coords_eval)

                    if not np.isfinite(dist) or dist < 0:
                        continue

                # 方向懲罰
                if (not fast_mode) and apply_directional and (use_subsequence or algorithm == 'SUBSEQ_DTW') and (start is not None and start >= 0) and (q_bearing is not None):
                    try:
                        ws = coords[start]
                        we = coords[start + len(query_path) - 1]
                        r_bear = _bearing_deg(ws[1], ws[0], we[1], we[0])
                        diff = _heading_diff_deg(q_bearing, r_bear)
                        if diff > heading_max_diff:
                            dist *= (1.0 + 0.8 * (diff / 180.0))
                    except Exception:
                        pass

                # 嚴格端點在 bbox 內（匹配窗口或整段）
                if (not fast_mode) and bbox_filter and bbox_strict:
                    try:
                        if start is not None and start >= 0 and len(query_path) >= 2:
                            ws = coords[start]; we = coords[start + len(query_path) - 1]
                        else:
                            ws = coords[0]; we = coords[-1]
                        if not (_point_in_bbox(ws[1], ws[0], bbox_filter) and _point_in_bbox(we[1], we[0], bbox_filter)):
                            continue
                    except Exception:
                        pass

                # 覆蓋率（匹配窗口在 bbox 內點比例）
                if (not fast_mode) and bbox_filter and (bbox_window_min_coverage is not None) and (start is not None and start >= 0) and len(query_path) >= 2:
                    try:
                        thr = max(0.0, min(1.0, float(bbox_window_min_coverage)))
                        win = coords[start: start + len(query_path)]
                        if win:
                            inside = sum(1 for p in win if _point_in_bbox(p[1], p[0], bbox_filter))
                            cov = inside / float(len(win))
                            if cov < thr:
                                continue
                    except Exception:
                        pass

                # 最小跨度限制
                if (not fast_mode) and (min_span_ratio is not None) and (min_span_ratio > 0) and q_span_km is not None:
                    try:
                        if start is not None and start >= 0 and len(query_path) >= 2:
                            ws = coords[start]; we = coords[start + len(query_path) - 1]
                        else:
                            ws = coords[0]; we = coords[-1]
                        span_km = geodesic((ws[1], ws[0]), (we[1], we[0])).kilometers
                        if span_km < (float(min_span_ratio) * q_span_km):
                            continue
                    except Exception:
                        pass

                # compute absolute deviation (avg km per aligned step)
                avg_dev = None
                try:
                    if (use_subsequence or algorithm == 'SUBSEQ_DTW') and (start is not None and start >= 0):
                        win = coords_eval[start: start + len(query_path)]
                        if len(win) == len(query_path):
                            total = 0.0
                            for i in range(len(query_path)):
                                qlat, qlng = _to_latlng_tuple(query_path[i])
                                wlat, wlng = _to_latlng_tuple(win[i])
                                total += geodesic((qlat, qlng), (wlat, wlng)).kilometers
                            avg_dev = total / max(1, len(query_path))
                    else:
                        L = min(len(query_path), len(coords_eval))
                        if L > 1:
                            total = 0.0
                            for i in range(L):
                                qlat, qlng = _to_latlng_tuple(query_path[i])
                                clat, clng = _to_latlng_tuple(coords_eval[i])
                                total += geodesic((qlat, qlng), (clat, clng)).kilometers
                            avg_dev = total / L
                except Exception:
                    avg_dev = None

                rec = {'flight': flight['id'], 'segment': flight.get('segment', 0), 'score': float(dist), 'points': len(coords), 'source': flight.get('source_file', ''), 'avg_km_per_step': avg_dev}
                if start is not None and start >= 0:
                    rec['match_start_index'] = int(start)
                    rec['match_window_length'] = len(query_path)
                if use_enhanced and ENHANCED_ALGORITHMS_AVAILABLE:
                    rec['confidence'] = confidence
                    rec['enhanced'] = True
                sims.append(rec)
            except Exception:
                pass

        sims.sort(key=lambda x: x['score'])
        top_similar = sims[:15]

        # 依需求附加顯示百分比（不影響排序）
        if calibrate_percent:
            def _avg_step_km_q(path):
                try:
                    if not path or len(path) < 2:
                        return None
                    total = 0.0
                    for i in range(1, len(path)):
                        a = _to_latlng_tuple(path[i-1]); b = _to_latlng_tuple(path[i])
                        total += geodesic(a, b).kilometers
                    return total / (len(path) - 1)
                except Exception:
                    return None

            avg_step = _avg_step_km_q(query_path)
            clip_guess = 1.0
            if avg_step and np.isfinite(avg_step):
                clip_guess = max(0.5, min(3.0, 2.0 * float(avg_step)))
            try:
                devs = [float(x['avg_km_per_step']) for x in top_similar if x.get('avg_km_per_step') is not None and np.isfinite(x.get('avg_km_per_step'))]
                if devs:
                    med = float(np.median(devs))
                    clip_guess = max(clip_guess, min(5.0, med * 1.2))
            except Exception:
                pass
            fp = float(floor_pct_q) if floor_pct_q is not None else 5.0
            if not np.isfinite(fp) or fp < 1.0:
                fp = 5.0
            if percent_mode == 'absolute':
                _absolute_similarity_percent(
                    top_similar,
                    avg_key='avg_km_per_step',
                    floor_pct=fp,
                    top_pct=float(top_pct_q) if top_pct_q is not None else 100.0,
                    clip_km_per_step=clip_guess,
                    gamma=1.5,
                    require_zero_for_100=True,
                )
            else:
                _map_similarity_percent(
                    top_similar,
                    score_key='score',
                    mode=percent_mode,
                    floor_pct=float(floor_pct_q) if floor_pct_q is not None else (5.0 if percent_mode!='minmax' else 0.0),
                    top_pct=float(top_pct_q) if top_pct_q is not None else (60.0 if percent_mode!='minmax' else 100.0),
                    ref_percentile=float(ref_pct_q) if ref_pct_q is not None else 80.0,
                    require_zero_for_100=True,
                )
            try:
                have_pct = any((isinstance(r.get('similarity_percent'), (int, float)) and np.isfinite(r['similarity_percent']) and r['similarity_percent'] > 0) for r in top_similar)
                all_one_or_equal = False
                try:
                    vals = [float(r.get('similarity_percent') or 0) for r in top_similar]
                    all_one_or_equal = (len(vals) > 0) and (len(set(int(v) for v in vals)) <= 1) and (int(vals[0]) <= 1)
                except Exception:
                    pass
                if (not have_pct) or all_one_or_equal:
                    scores = [float(r.get('score') or 0.0) for r in top_similar]
                    if scores:
                        mn = min(scores); mx = max(scores); spread = mx - mn
                        scale = max(spread / 3.0, 1e-6)
                        weights = [math.exp(-(s - mn)/scale) for s in scores]
                        sw = sum(weights) or 1.0
                        for i, r in enumerate(top_similar):
                            p = float(weights[i] * 100.0 / sw)
                            r['similarity_percent'] = max(5.0, min(95.0, p))
                else:
                    for r in top_similar:
                        try:
                            if not isinstance(r.get('similarity_percent'), (int, float)) or not np.isfinite(r['similarity_percent']) or r['similarity_percent'] < 5.0:
                                r['similarity_percent'] = 5.0
                            elif r['similarity_percent'] > 100.0:
                                r['similarity_percent'] = 100.0
                        except Exception:
                            r['similarity_percent'] = 5.0
            except Exception:
                pass

        try:
            progress_update({'task': 'identify', 'processed': len(candidates), 'done': True, 'eta_seconds': 0, 'message': '完成'})
        except Exception:
            pass
        return jsonify(top_similar)
    except Exception as e:
        try:
            progress_update({'task': 'identify', 'done': True, 'message': f'錯誤: {e}'})
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


@app.route('/api/identify-all', methods=['POST'])
def identify_all_algorithms():
    try:
        query_path = request.json or []
        if not query_path or len(query_path) < 2:
            return jsonify({'error': '路徑數據不足'}), 400

        # 顯示/校準參數（可選）
        calibrate_percent = request.args.get('calibrate_percent', 'true').lower() == 'true'
        percent_mode = (request.args.get('percent_mode', 'absolute') or 'absolute').lower()
        floor_pct_q = request.args.get('floor_pct', None, type=float)
        top_pct_q = request.args.get('top_pct', None, type=float)
        ref_pct_q = request.args.get('ref_pct', None, type=float)

        # 比對參數
        use_subsequence = request.args.get('subseq', 'true').lower() == 'true'
        stride = request.args.get('stride', 1, type=int)
        fast_mode = request.args.get('fast', 'false').lower() == 'true' or request.args.get('approx', 'false').lower() == 'true'
        apply_directional = request.args.get('directional', 'true').lower() == 'true'
        heading_max_diff = request.args.get('heading_max_diff', 60, type=int)

        # BBOX
        bbox_filter = _parse_bbox_from_args(request.args)
        bbox_strict = request.args.get('bbox_strict', 'false').lower() == 'true'
        min_span_ratio = request.args.get('min_span_ratio', None, type=float)
        bbox_mode = request.args.get('bbox_mode', 'intersect').lower()
        bbox_window_min_coverage = request.args.get('bbox_window_min_coverage', None, type=float)

        # 候選集合
        if bbox_filter:
            candidates = [f for f in flight_database if f.get('bbox') and _bbox_intersects(f['bbox'], bbox_filter)]
        else:
            candidates = flight_database

        # 查詢跨度/方位
        try:
            q_start = query_path[0]; q_end = query_path[-1]
            q_span_km = geodesic((_to_latlng_tuple(q_start)), (_to_latlng_tuple(q_end))).kilometers
            q_bearing = _bearing_deg(q_start['lat'], q_start['lng'], q_end['lat'], q_end['lng'])
        except Exception:
            q_span_km = None
            q_bearing = None

        try:
            progress_update({'task': 'identify-all', 'done': False, 'processed': 0, 'total': len(candidates), 'message': '分析中（多算法）', 'start_time': time.time(), 'eta_seconds': None, 'version': time.time()})
        except Exception:
            pass

        algos = ['DTW', 'SUBSEQ_DTW', 'EUCLIDEAN', 'LCSS', 'FRECHET', 'HAUSDORFF', 'EDR', 'ERP']
        results_by_algo = {a: [] for a in algos}
        processed = 0

        for flight in candidates:
            processed += 1
            coords = flight.get('coordinates') or []
            if len(coords) < 2:
                continue
            coords_eval = coords
            if fast_mode and len(coords) > 800:
                try:
                    step_ds = 2
                    coords_eval = coords[::step_ds]
                    if coords_eval and coords_eval[-1] != coords[-1]:
                        coords_eval = coords_eval + [coords[-1]]
                except Exception:
                    coords_eval = coords

            if bbox_filter and bbox_mode == 'path_inside':
                try:
                    if not all(_point_in_bbox(pt[1], pt[0], bbox_filter) for pt in coords):
                        continue
                except Exception:
                    pass

            try:
                for algo in algos:
                    if algo == 'SUBSEQ_DTW':
                        dist, start_idx = subsequence_dtw_distance(query_path, coords_eval, stride)
                        start = -1 if fast_mode else start_idx
                    else:
                        if use_subsequence:
                            dist, start_idx = subsequence_distance_generic(query_path, coords_eval, algo, stride)
                            start = -1 if fast_mode else start_idx
                        else:
                            start = -1
                            if algo == 'DTW':
                                dist = dtw_distance(query_path, coords_eval)
                            elif algo == 'EUCLIDEAN':
                                dist = euclidean_distance(query_path, coords_eval)
                            elif algo == 'LCSS':
                                dist = lcss_distance(query_path, coords_eval)
                            elif algo == 'FRECHET':
                                dist = frechet_distance(query_path, coords_eval)
                            elif algo == 'HAUSDORFF':
                                dist = hausdorff_distance(query_path, coords_eval)
                            elif algo == 'EDR':
                                dist = edr_distance(query_path, coords_eval)
                            elif algo == 'ERP':
                                dist = erp_distance(query_path, coords_eval)
                            else:
                                dist = dtw_distance(query_path, coords_eval)

                    if not np.isfinite(dist) or dist < 0:
                        continue

                    if (not fast_mode) and bbox_filter and bbox_strict:
                        try:
                            if start is not None and start >= 0 and len(query_path) >= 2:
                                ws = coords[start]; we = coords[start + len(query_path) - 1]
                            else:
                                ws = coords[0]; we = coords[-1]
                            if not (_point_in_bbox(ws[1], ws[0], bbox_filter) and _point_in_bbox(we[1], we[0], bbox_filter)):
                                continue
                        except Exception:
                            pass

                    if (not fast_mode) and bbox_filter and (bbox_window_min_coverage is not None) and (start is not None and start >= 0) and len(query_path) >= 2:
                        try:
                            thr = max(0.0, min(1.0, float(bbox_window_min_coverage)))
                            win = coords[start: start + len(query_path)]
                            if win:
                                inside = sum(1 for p in win if _point_in_bbox(p[1], p[0], bbox_filter))
                                cov = inside / float(len(win))
                                if cov < thr:
                                    continue
                        except Exception:
                            pass

                    if (not fast_mode) and (min_span_ratio is not None) and (min_span_ratio > 0) and q_span_km is not None:
                        try:
                            if start is not None and start >= 0 and len(query_path) >= 2:
                                ws = coords[start]; we = coords[start + len(query_path) - 1]
                            else:
                                ws = coords[0]; we = coords[-1]
                            span_km = geodesic((ws[1], ws[0]), (we[1], we[0])).kilometers
                            if span_km < (float(min_span_ratio) * q_span_km):
                                continue
                        except Exception:
                            pass

                    if (not fast_mode) and apply_directional and (start is not None and start >= 0) and (q_bearing is not None):
                        try:
                            ws = coords[start]
                            we = coords[start + len(query_path) - 1]
                            r_bear = _bearing_deg(ws[1], ws[0], we[1], we[0])
                            diff = _heading_diff_deg(q_bearing, r_bear)
                            if diff > heading_max_diff:
                                dist *= (1.0 + 0.8 * (diff / 180.0))
                        except Exception:
                            pass

                    avg_dev = None
                    try:
                        if (start is not None and start >= 0):
                            win = coords[start: start + len(query_path)]
                            if len(win) == len(query_path):
                                total = 0.0
                                for i in range(len(query_path)):
                                    qlat, qlng = _to_latlng_tuple(query_path[i])
                                    wlat, wlng = _to_latlng_tuple(win[i])
                                    total += geodesic((qlat, qlng), (wlat, wlng)).kilometers
                                avg_dev = total / max(1, len(query_path))
                        else:
                            L = min(len(query_path), len(coords_eval))
                            if L > 1:
                                total = 0.0
                                for i in range(L):
                                    qlat, qlng = _to_latlng_tuple(query_path[i])
                                    clat, clng = _to_latlng_tuple(coords_eval[i])
                                    total += geodesic((qlat, qlng), (clat, clng)).kilometers
                                avg_dev = total / L
                    except Exception:
                        avg_dev = None

                    rec = {'flight': flight['id'], 'segment': flight.get('segment', 0), 'score': float(dist), 'points': len(coords), 'source': flight.get('source_file', ''), 'avg_km_per_step': avg_dev}
                    if start is not None and start >= 0:
                        rec['match_start_index'] = int(start)
                        rec['match_window_length'] = len(query_path)
                    results_by_algo[algo].append(rec)
            except Exception:
                pass

            # Update progress more frequently for smoother UI (every ~100 items)
            if processed % 100 == 0 or processed == len(candidates):
                try:
                    st = None
                    try:
                        st = (progress_registry.get('identify-all') or {}).get('start_time')
                    except Exception:
                        st = None
                    elapsed = max(1e-3, time.time() - (st or time.time()))
                    rate = processed / elapsed
                    rem = max(0, len(candidates) - processed)
                    eta = rem / rate if rate > 0 else None
                    progress_update({'task': 'identify-all', 'processed': processed, 'eta_seconds': round(eta, 1) if eta else None})
                except Exception:
                    pass

        top_by_algo = {}

        def _avg_step_km_q(path):
            try:
                if not path or len(path) < 2:
                    return None
                total = 0.0
                for i in range(1, len(path)):
                    a = _to_latlng_tuple(path[i-1]); b = _to_latlng_tuple(path[i])
                    total += geodesic(a, b).kilometers
                return total / (len(path) - 1)
            except Exception:
                return None

        avg_step = _avg_step_km_q(query_path)
        clip_guess = 1.0
        if avg_step and np.isfinite(avg_step):
            clip_guess = max(0.5, min(3.0, 2.0 * float(avg_step)))

        for algo, lst in results_by_algo.items():
            lst.sort(key=lambda x: x['score'])
            topk = lst[:5]
            if calibrate_percent:
                if percent_mode == 'absolute':
                    clip_used = clip_guess
                    try:
                        devs = [float(x['avg_km_per_step']) for x in topk if x.get('avg_km_per_step') is not None and np.isfinite(x.get('avg_km_per_step'))]
                        if devs:
                            med = float(np.median(devs))
                            clip_used = max(clip_used, min(5.0, med * 1.2))
                    except Exception:
                        pass
                    fp2 = float(floor_pct_q) if floor_pct_q is not None else 5.0
                    if not np.isfinite(fp2) or fp2 < 1.0:
                        fp2 = 5.0
                    _absolute_similarity_percent(
                        topk,
                        avg_key='avg_km_per_step',
                        floor_pct=fp2,
                        top_pct=float(top_pct_q) if top_pct_q is not None else 100.0,
                        clip_km_per_step=clip_used,
                        gamma=1.5,
                        require_zero_for_100=True,
                    )
                else:
                    _map_similarity_percent(
                        topk,
                        score_key='score',
                        mode=percent_mode,
                        floor_pct=float(floor_pct_q) if floor_pct_q is not None else (10.0 if percent_mode!='minmax' else 0.0),
                        top_pct=float(top_pct_q) if top_pct_q is not None else (90.0 if percent_mode!='minmax' else 100.0),
                        ref_percentile=float(ref_pct_q) if ref_pct_q is not None else 75.0,
                        require_zero_for_100=True,
                    )
                try:
                    have_pct = any((isinstance(r.get('similarity_percent'), (int, float)) and np.isfinite(r['similarity_percent']) and r['similarity_percent'] > 0) for r in topk)
                    all_one_or_equal = False
                    try:
                        vals = [float(r.get('similarity_percent') or 0) for r in topk]
                        all_one_or_equal = (len(vals) > 0) and (len(set(int(v) for v in vals)) <= 1) and (int(vals[0]) <= 1)
                    except Exception:
                        pass
                    if (not have_pct) or all_one_or_equal:
                        scores = [float(r.get('score') or 0.0) for r in topk]
                        if scores:
                            mn = min(scores); mx = max(scores); spread = mx - mn
                            scale = max(spread / 3.0, 1e-6)
                            weights = [math.exp(-(s - mn)/scale) for s in scores]
                            sw = sum(weights) or 1.0
                            for i, r in enumerate(topk):
                                p = float(weights[i] * 100.0 / sw)
                                r['similarity_percent'] = max(5.0, min(95.0, p))
                    else:
                        for r in topk:
                            try:
                                if not isinstance(r.get('similarity_percent'), (int, float)) or not np.isfinite(r['similarity_percent']) or r['similarity_percent'] < 5.0:
                                    r['similarity_percent'] = 5.0
                                elif r['similarity_percent'] > 100.0:
                                    r['similarity_percent'] = 100.0
                            except Exception:
                                r['similarity_percent'] = 5.0
                except Exception:
                    pass
            top_by_algo[algo] = topk

        try:
            progress_update({'task': 'identify-all', 'processed': len(candidates), 'done': True, 'eta_seconds': 0, 'message': '完成'})
        except Exception:
            pass

        return jsonify({'results': top_by_algo, 'algorithms': list(results_by_algo.keys()), 'total_flights': len(flight_database)})
    except Exception as e:
        try:
            progress_update({'task': 'identify-all', 'done': True, 'message': f'錯誤: {e}'})
        except Exception:
            pass
        return jsonify({'error': str(e)}), 500


@app.route('/api/progress')
def get_progress():
    try:
        data = dict(progress_data)
        total = max(1, int(data.get('total') or 1))
        processed = int(data.get('processed') or 0)
        data['percent'] = max(0, min(100, int(processed * 100 / total)))
        # 產生每個任務的百分比，並輸出 tasks 陣列（相容舊版的單一欄位）
        tasks = []
        try:
            for tname, t in list(progress_registry.items()):
                tt = dict(t)
                t_total = max(1, int(tt.get('total') or 1))
                t_processed = int(tt.get('processed') or 0)
                tt['percent'] = max(0, min(100, int(t_processed * 100 / t_total)))
                tasks.append(tt)
        except Exception:
            tasks = []
        data['tasks'] = tasks
        return jsonify(data)
    except Exception as e:
        return jsonify({'task': None, 'done': True, 'message': f'progress error: {e}', 'percent': 100})


@app.route('/api/statistics')
def get_statistics():
    total_flights = len(flight_database)
    lengths = [len(f['coordinates']) for f in flight_database]
    avg_points = float(np.mean(lengths)) if lengths else 0.0
    sources = {}
    for f in flight_database:
        src = f.get('source_file', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    of_ap = len(openflights_index.get('airports', [])) if isinstance(openflights_index, dict) else 0
    of_routes = sum(len(v) for v in openflights_routes.values()) if isinstance(openflights_routes, dict) else 0
    return jsonify({
        'total_records': total_flights,
        'unique_flights': len({f['id'] for f in flight_database}),
        'average_points_per_flight': round(avg_points, 1),
        'source_files': sources,
        'point_range': [min(lengths) if lengths else 0, max(lengths) if lengths else 0],
        'enhancements': {
            'enhanced_algorithms': ENHANCED_ALGORITHMS_AVAILABLE,
            'lstm_predictor': LSTM_PREDICTOR_AVAILABLE,
            'bayesian_optimizer': BAYESIAN_OPTIMIZER_AVAILABLE
        },
        'openflights': {
            'airports_loaded': of_ap,
            'routes_loaded': of_routes
        }
    })


@app.route('/api/openflights/nearest-airports')
def api_nearest_airports():
    try:
        if not openflights_index or not openflights_index.get('airports'):
            return jsonify({'error': 'OpenFlights 機場資料未載入'}), 404
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        if lat is None or lng is None:
            return jsonify({'error': '缺少 lat/lng 參數'}), 400
        k = request.args.get('k', 5, type=int)
        airport_only = request.args.get('airport_only', 'true').lower() == 'true'
        require_iata = request.args.get('require_iata', 'true').lower() == 'true'
        from openflights_adapter import nearest_airports
        res = nearest_airports(openflights_index, lat, lng, k=k, airport_only=airport_only, require_iata=require_iata)
        # 若最近 IATA 機場距離異常偏大（>200km），再試一次包含非 IATA 的小機場
        if (not res) or (isinstance(res, list) and res and res[0].get('distance_km', 1e9) and float(res[0]['distance_km']) > 200):
            if require_iata:
                res2 = nearest_airports(openflights_index, lat, lng, k=k, airport_only=airport_only, require_iata=False)
                if res2:
                    res = res2
        return jsonify({'nearest': res})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/openflights/airport/<code>')
def api_airport_lookup(code):
    try:
        if not openflights_index:
            return jsonify({'error': 'OpenFlights 機場資料未載入'}), 404
        c = (code or '').upper()
        by_iata = openflights_index.get('by_iata', {})
        by_icao = openflights_index.get('by_icao', {})
        ap = by_iata.get(c) or by_icao.get(c)
        if not ap:
            return jsonify({'error': f'找不到機場代碼 {c}'}), 404
        return jsonify(ap)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/openflights/routes-from/<src_iata>')
def api_routes_from(src_iata):
    try:
        if not openflights_routes:
            return jsonify({'error': 'OpenFlights 航線資料未載入'}), 404
        src = (src_iata or '').upper()
        dsts = openflights_routes.get(src, {})
        # Expand with airport details if available
        by_iata = openflights_index.get('by_iata', {}) if openflights_index else {}
        expanded = []
        for dst_code, count in sorted(dsts.items(), key=lambda x: x[1], reverse=True):
            ap = by_iata.get(dst_code)
            rec = {'dst': dst_code, 'count': int(count)}
            if ap:
                rec.update({'name': ap.get('name'), 'city': ap.get('city'), 'country': ap.get('country'), 'lat': ap.get('lat'), 'lng': ap.get('lng')})
            expanded.append(rec)
        return jsonify({'src': src, 'destinations': expanded})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-trajectory', methods=['POST'])
def predict_trajectory():
    try:
        input_path = request.json or []
        if not input_path or len(input_path) < 2:
            return jsonify({'error': '輸入路徑點數不足，需要至少 2 個點'}), 400

        model = (request.args.get('model', '') or '').lower()

        # horizon parameters
        min_steps = request.args.get('min_steps', 1, type=int)
        max_steps = request.args.get('max_steps', 1000, type=int)
        base_steps = request.args.get('base_steps', 6, type=int)
        step_km = request.args.get('step_km', None, type=float) or request.args.get('km_per_step', None, type=float)
        distance_km = request.args.get('distance_km', None, type=float)
        horizon_strategy = (request.args.get('horizon_strategy', '') or '').lower()
        min_steps = max(1, min_steps); max_steps = max(min_steps, max_steps); base_steps = max(0, base_steps)

        horizon_raw = request.args.get('horizon', 'auto')
        if (horizon_strategy == 'distance') or (isinstance(horizon_raw, str) and horizon_raw.lower() == 'distance'):
            try:
                if step_km and distance_km and step_km > 0 and distance_km > 0:
                    horizon = int(math.ceil(float(distance_km) / float(step_km)))
                else:
                    horizon = compute_dynamic_horizon(input_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
                hmode = 'distance'
            except Exception:
                horizon = compute_dynamic_horizon(input_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
                hmode = 'auto'
            horizon = max(min_steps, min(max_steps, int(horizon)))
        elif isinstance(horizon_raw, str) and horizon_raw.lower() == 'auto':
            horizon = compute_dynamic_horizon(input_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
            hmode = 'auto'
        else:
            try:
                horizon = int(horizon_raw)
            except Exception:
                horizon = 10
            if horizon <= 0:
                horizon = compute_dynamic_horizon(input_path, min_steps=min_steps, max_steps=max_steps, base_steps=base_steps)
                hmode = 'auto'
            else:
                horizon = max(min_steps, min(max_steps, horizon))
                hmode = 'fixed'

        if model == 'lstm':
            # progress init
            try:
                progress_update({'task': 'predict-lstm', 'done': False, 'processed': 0, 'total': horizon, 'message': 'LSTM 預測', 'start_time': time.time(), 'eta_seconds': None, 'version': time.time()})
            except Exception:
                pass
            try:
                traj = lstm_predict_trajectory(input_path, prediction_horizon=horizon)
                try:
                    progress_update({'task': 'predict-lstm', 'processed': horizon, 'done': True, 'eta_seconds': 0, 'message': '完成'})
                except Exception:
                    pass
                return jsonify({'predicted_trajectory': traj, 'prediction_horizon': horizon, 'horizon_mode': hmode, 'model_status': 'lstm', 'input_points': len(input_path), 'algorithm': 'LSTM Forecaster'})
            except Exception as e:
                # fallback to heuristic if DL not available
                predicted_points = simple_predict_trajectory(input_path, prediction_horizon=horizon)
                try:
                    progress_update({'task': 'predict-lstm', 'processed': horizon, 'done': True, 'eta_seconds': 0, 'message': f'改用啟發式：{e}'})
                except Exception:
                    pass
                return jsonify({'predicted_trajectory': predicted_points, 'prediction_horizon': horizon, 'horizon_mode': hmode, 'model_status': f'fallback:{e}', 'input_points': len(input_path), 'algorithm': 'Heuristic Velocity Extrapolation'})

        predicted_points = simple_predict_trajectory(input_path, prediction_horizon=horizon)
        return jsonify({'predicted_trajectory': predicted_points, 'prediction_horizon': horizon, 'horizon_mode': hmode, 'model_status': 'heuristic', 'input_points': len(input_path), 'algorithm': 'Heuristic Velocity Extrapolation'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimize-parameters', methods=['POST'])
def optimize_parameters():
    """Kick off the real-API Bayesian optimizer in a background thread if available.
    Body (optional): { n_calls: int, eval_total: int }
    """
    try:
        if not BAYESIAN_OPTIMIZER_AVAILABLE:
            return jsonify({'error': '貝葉斯優化器不可用'}), 501
        payload = request.json or {}
        n_calls = int(payload.get('n_calls', 20))
        eval_total = int(payload.get('eval_total', 24))

        import threading, subprocess, sys
        cwd = os.path.dirname(os.path.abspath(__file__))
        def _run():
            try:
                progress_update({'task': 'optimize', 'done': False, 'processed': 0, 'total': n_calls, 'message': '參數優化中', 'start_time': time.time(), 'eta_seconds': None, 'version': time.time()})
            except Exception:
                pass
            cmd = [sys.executable, os.path.join(cwd, 'bayesian_optimizer.py'), '--n-calls', str(n_calls), '--eval-total', str(eval_total)]
            try:
                subprocess.run(cmd, cwd=cwd, check=False)
            finally:
                try:
                    progress_update({'task': 'optimize', 'done': True, 'eta_seconds': 0, 'message': '完成'})
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({'status': 'started', 'n_calls': n_calls, 'eval_total': eval_total})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/algorithm-info')
def get_algorithm_info():
    return jsonify({
        'standard_algorithms': ['DTW', 'SUBSEQ_DTW', 'EUCLIDEAN', 'LCSS', 'FRECHET', 'DFD', 'HAUSDORFF', 'EDR', 'ERP'],
        'enhanced_algorithms_available': ENHANCED_ALGORITHMS_AVAILABLE,
        'lstm_predictor_available': LSTM_PREDICTOR_AVAILABLE,
        'bayesian_optimizer_available': BAYESIAN_OPTIMIZER_AVAILABLE
    })


@app.route('/')
def serve_index():
    return send_from_directory('.', 'demo_with_real_data_fixed.html')


@app.route('/demo_with_real_data_fixed.html')
def serve_demo_fixed():
    return send_from_directory('.', 'demo_with_real_data_fixed.html')


@app.route('/test')
def test_page():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>API測試</title>
        <style>body{font-family:Arial;margin:40px;}</style>
    </head>
    <body>
        <h2>🧪 航班預測API測試</h2>
        <div>
            <h3>API端點:</h3>
            <ul>
                <li><a href="/api/flights">/api/flights</a></li>
                <li><a href="/api/statistics">/api/statistics</a></li>
            </ul>
        </div>
    </body>
    </html>
    '''


if __name__ == '__main__':
    load_flight_database()
    load_openflights_data()
    detect_lstm_availability()
    detect_enhanced_availability()
    detect_bayesian_optimizer_availability()
    print('航班預測服務啟動中... http://localhost:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)

