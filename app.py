# app.py
import pickle
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 閾值（經緯度距離，約 0.01 度 ≈ 1km）
EPS = 0.01

# -----------------------------------------------------------------------------
# 1) 載入參考庫 ref_db.pkl
#    格式：{ flight_id: { 'latlon': np.ndarray[[lat,lon],...], 'feat': np.ndarray[[z-scored features],...] } }
# -----------------------------------------------------------------------------
with open('ref_db.pkl', 'rb') as f:
    REF_DB = pickle.load(f)


# -----------------------------------------------------------------------------
# 2) 助手函式：各種相似度度量
# -----------------------------------------------------------------------------
def lsed(q, r):
    """Lock-step Euclidean (平方距離總和的平方根)"""
    m = len(q)
    if len(r) < m: return np.inf
    r0 = r[:m]
    d2 = np.sum((q - r0)**2, axis=1)
    return np.sqrt(d2.sum())

def dtw_dist(q, r):
    """Dynamic Time Warping (使用 fastdtw)"""
    return fastdtw(q, r, dist=euclidean)[0] / len(q)

def edr(q, r, eps=EPS):
    """Edit Distance on Real sequence"""
    m, n = len(q), len(r)
    D = np.zeros((m+1,n+1), dtype=int)
    D[:,0] = np.arange(m+1)
    D[0,:] = np.arange(n+1)
    for i in range(1,m+1):
        for j in range(1,n+1):
            cost = 0 if np.linalg.norm(q[i-1]-r[j-1])<=eps else 1
            D[i,j] = min(
                D[i-1,j-1] + cost,  # substitute
                D[i-1,j]   + 1,     # delete
                D[i,j-1]   + 1      # insert
            )
    return D[m,n]

def lcss(q, r, eps=EPS):
    """Longest Common Subsequence with threshold"""
    m, n = len(q), len(r)
    L = np.zeros((m+1,n+1), dtype=int)
    for i in range(1,m+1):
        for j in range(1,n+1):
            if np.linalg.norm(q[i-1]-r[j-1])<=eps:
                L[i,j] = L[i-1,j-1] + 1
            else:
                L[i,j] = max(L[i-1,j], L[i,j-1])
    # 回傳「不匹配」分數 = 1 - (matches / m)
    return 1 - L[m,n]/m

def dfd(q, r):
    """Discrete Fréchet Distance (簡易實現)"""
    m, n = len(q), len(r)
    ca = np.full((m, n), -1.0)
    def _c(i,j):
        if ca[i,j] > -1: return ca[i,j]
        d = np.linalg.norm(q[i]-r[j])
        if i==0 and j==0:
            ca[i,j] = d
        elif i>0 and j==0:
            ca[i,j] = max(_c(i-1,0), d)
        elif i==0 and j>0:
            ca[i,j] = max(_c(0,j-1), d)
        else:
            ca[i,j] = max(min(_c(i-1,j), _c(i,j-1), _c(i-1,j-1)), d)
        return ca[i,j]
    return _c(m-1,n-1)

def frechet_approx(q, r):
    """連續 Fréchet 距離近似：採樣 DFD 即可視為近似"""
    return dfd(q, r)

def l2_prefix(q, r):
    """歐式前綴距離（只取前 m 點的 L2 平均距離）"""
    m = len(q)
    if len(r)<m: return np.inf
    return np.linalg.norm(q - r[:m], axis=1).mean()

# -----------------------------------------------------------------------------
# 3) 識別最相似航線：ensemble 七種度量
# -----------------------------------------------------------------------------
@app.route('/api/identify', methods=['POST'])
def identify():
    partial = request.json
    # 1) 轉為 np.ndarray[[lat,lon],...]
    q_latlon = np.array([[p['lat'], p['lon']] for p in partial])
    # 2) 計算每條完整航線的 7 種距離
    scores = []
    for flight_id, info in REF_DB.items():
        r_latlon = info['latlon']
        dists = np.array([
            lsed(q_latlon, r_latlon),
            dtw_dist(q_latlon, r_latlon),
            edr(q_latlon, r_latlon),
            lcss(q_latlon, r_latlon),
            dfd(q_latlon, r_latlon),
            frechet_approx(q_latlon, r_latlon),
            l2_prefix(q_latlon, r_latlon)
        ], dtype=float)
        scores.append((flight_id, dists))
    # 3) min–max 正規化每個指標，越小越相似 → 統一轉成「越小越好」
    arr = np.vstack([d for _,d in scores])  # shape=(N,7)
    mn = arr.min(axis=0); mx = arr.max(axis=0)
    norm = (arr - mn) / (mx - mn + 1e-9)
    # 4) 融合：各指標等權平均
    fused = norm.mean(axis=1)
    # 5) 取前 5 名
    idx = np.argsort(fused)[:5]
    result = [{'flight': scores[i][0], 'score': float(fused[i])} for i in idx]
    return jsonify(result)


# -----------------------------------------------------------------------------
# 4) 預測未來路徑：subsequence DTW
# -----------------------------------------------------------------------------
@app.route('/api/predict/<flight_id>', methods=['POST'])
def predict(flight_id):
    partial = request.json
    q = np.array([[p['lat'], p['lon']] for p in partial])
    info = REF_DB.get(flight_id)
    if info is None:
        return jsonify([]), 404
    r = info['latlon']
    m, n = len(q), len(r)
    # subsequence DTW，找到最佳起始點
    best_d, best_i = np.inf, 0
    for i in range(n - m + 1):
        d, _ = fastdtw(q, r[i:i+m], dist=euclidean)
        if d < best_d:
            best_d, best_i = d, i
    # 接上後半段
    future = r[best_i + m : ]
    return jsonify(future.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
