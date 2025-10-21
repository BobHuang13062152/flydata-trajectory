# 可重現 API（本地端）

本文檔提供在 Windows（PowerShell）上一鍵啟動與測試 API 的步驟，讓審稿人或使用者能在本機重現論文結果的最小流程。

## 先決條件
- Windows 10/11
- Python 3.13（或 3.10+）
- PowerShell

## 安裝依賴
```powershell
py -3.13 -m pip install -r .\requirements.txt
```

## 啟動服務（Flask）
```powershell
py -3.13 .\flight_prediction_server_fixed.py
```
服務預設在 http://localhost:5000

## 範例請求

### 1) 查相似（DTW 子序列 + 方向檢查）
```powershell
$path = @(
  @{ lat = 25.08; lng = 121.23 },
  @{ lat = 25.12; lng = 121.30 },
  @{ lat = 25.18; lng = 121.38 }
) | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:5000/api/identify?algo=DTW&subseq=true&stride=3&directional=true" `
  -ContentType "application/json" -Body $path | ConvertTo-Json -Depth 6
```

### 2) 共識延續（(Algo)+CONSENSUS 管線：DTW/LCSS/Fréchet/DFD/Hausdorff/EDR/ERP/Euclidean）
```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:5000/api/forecast-consensus?algo=DTW&subseq=true&stride=3&directional=true&topN=5&horizon=auto" `
  -ContentType "application/json" -Body $path | ConvertTo-Json -Depth 6
```

其他演算法（替換 `algo=`）：`EUCLIDEAN`, `LCSS`, `FRECHET`/`DFD`, `HAUSDORFF`, `EDR`, `ERP`。
若 `subseq=true`，會對應使用泛型子序列掃描（Fréchet/LCSS/EDR/ERP/Euclidean 也可）。

### 3) LSTM 延續（DTW+LSTM 管線）
如 `models/lstm_forecaster.pt` 存在則使用 LSTM；否則自動回退啟發式。
```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:5000/api/predict-trajectory?model=lstm&horizon=auto" `
  -ContentType "application/json" -Body $path | ConvertTo-Json -Depth 6
```

### 4) 進度查詢（多任務）
```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:5000/api/progress" | ConvertTo-Json -Depth 6
```

### 5) 數據摘要
```powershell
Invoke-RestMethod -Method Get -Uri "http://localhost:5000/api/statistics" | ConvertTo-Json -Depth 6
```

## 最小數據說明
- 服務會自動載入工作目錄下的 `flights*.geojson`（排除 `stitched` 字樣的檔案）。
- 本倉庫包含 `flights_20250807_094940.geojson`，足以跑通對照表與消融的最小流程。

## 產生圖表（選用）
```powershell
# 以新的真實消融 CSV 更新圖
py -3.13 .\paper\figures\generate_real_figures.py `
  --prelim .\paper\preliminary_table_REAL_quick_filled.csv `
  --ablation .\paper\ablation_REAL_quick.csv `
  --outdir .\paper\figures `
  --exclude SUBSEQ_DTW EUCLIDEAN

# 整理海報素材到 poster_ready/
py -3.13 .\tools\prepare_poster_figs.py
```

## 方法命名（對照海報）
- (Algo)+CONSENSUS：先用指定演算法（DTW/LCSS/Fréchet/DFD/Hausdorff/EDR/ERP/Euclidean；可選子序列與方向檢查）取 Top-N，做共識延續
- DTW+LSTM：DTW 輔助定位/配對，使用輕量 LSTM 進行延續
- 單一度量基線（如 DTW）只在報表中為「基準參考」，圖上已標註為管線對比

## 授權
見 `LICENSE`（MIT）。
