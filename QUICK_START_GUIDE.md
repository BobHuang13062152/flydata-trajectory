# 🛫 航班軌跡分析系統 - 快速啟動指南

## 🎯 您要的簡單使用方式

### 方法1: 推薦方式 (無跨域問題)
```
1. 雙擊執行: start_server.bat
2. 等待看到啟動訊息
3. 開啟瀏覽器訪問: http://localhost:5000
```

### 方法2: 本地文件方式 (需要先啟動後端)
```
1. 雙擊執行: start_server.bat
2. 等待看到 "Flask server is running" 
3. 開啟: file:///C:/NCHC_DATA/flydata/demo_with_real_data.html
```

## 🔧 解決 "後端服務離線" 問題

**問題原因**: 當您用 `file://` 開啟 HTML 時，瀏覽器安全政策限制了對 `http://localhost:5000` 的訪問。

**解決方案**: 
1. ✅ **使用 http://localhost:5000** (推薦)
2. ✅ 或者先確保 `start_server.bat` 正在執行

## 📊 系統狀態

- ✅ 最新數據: `flights_20250807_094940.geojson` (105.3 MB, 157,313 航班)
- ✅ 前端界面: `demo_with_real_data.html` (您喜歡的簡潔設計)
- ✅ 後端服務: `flight_prediction_server.py` (已配置自動載入最新數據)

## 🎉 現在開始使用

**立即執行**: `start_server.bat`
**然後訪問**: `http://localhost:5000`

這樣您就可以使用您喜歡的 `demo_with_real_data.html` UI/UX 設計，並且連接到真實的後端服務！
