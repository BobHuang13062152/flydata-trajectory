# 🛫 航班軌跡分析系統（最小運行版）

這是實際上線使用的最小可執行版本：Flask 後端 + 地圖前端 + 軌跡相似度與預測 API。

## 🚀 快速啟動（Windows PowerShell）

1) 安裝套件（Python 3.13）

```powershell
Set-Location -Path 'C:\NCHC_DATA\flydata'
py -3.13 -m pip install -r .\requirements.txt
```

2) 啟動服務

```powershell
py -3.13 .\flight_prediction_server_fixed.py
# 若上行失敗，可改用：python .\flight_prediction_server_fixed.py
```

3) 開啟前端

- 在瀏覽器打開：http://localhost:5000
- 介面檔：`demo_with_real_data_fixed.html`（由後端直接提供）

## � 目錄重點

- `flight_prediction_server_fixed.py`：主後端（API 與演算法）
- `demo_with_real_data_fixed.html`：互動地圖 UI
- `openflights_adapter.py`：OpenFlights 匯入（可選）
- `models/`：模型目錄（可選，LSTM 權重放這裡）
- `requirements.txt`：最小依賴

資料檔案：請放於根目錄，檔名為 `flights*.geojson`（自動載入；略過含 `stitched` 的檔）。

## 🌐 API 概覽

- `GET /api/statistics`：系統統計與可用功能
- `GET /api/flights`：航班摘要
- `POST /api/identify`：相似性搜尋（支援 DTW/SUBSEQ_DTW/LCSS/…）
- `POST /api/identify-all`：多演算法 Top‑k 對照
- `POST /api/forecast-consensus`：Top‑k 共識預測 + 啟發式混合
- `POST /api/predict-trajectory`：路徑預測（model=heuristic|lstm）
- `GET /api/openflights/*`：OpenFlights 查詢（若資料存在）

簡易健康檢查頁面：`/test`

## 🔮 啟用 LSTM（可選）

1) 安裝 CPU 版 PyTorch：

```powershell
py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

2) 將模型權重放到 `models/lstm_forecaster.pt`（詳見 `models/README.md`）。

3) 前端切換 LSTM 或呼叫 API：

```http
POST /api/predict-trajectory?model=lstm&horizon=20
```

若缺少模型或未安裝 PyTorch，系統會自動退回啟發式預測。

## 🛠️ 故障排除

- ImportError/ModuleNotFoundError：
	- 重新安裝依賴：`py -3.13 -m pip install -r .\requirements.txt`
- 啟動後沒有資料：
	- 確認根目錄存在 `flights*.geojson` 檔案（非 stitched）。
- LSTM 不可用：
	- 安裝 PyTorch 並放置 `models/lstm_forecaster.pt`。

## 📜 授權與貢獻

請建立 GitHub Issue 提出建議或問題。若要貢獻程式碼，請以 PR 形式提交。
