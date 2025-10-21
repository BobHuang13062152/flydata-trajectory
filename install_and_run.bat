@echo off
echo 🚢 船隻軌跡預測系統 - 安裝與執行指南
echo ==========================================

echo.
echo 📦 步驟1: 安裝必要套件
echo ----------------
pip install -r requirements.txt

echo.
echo ⚠️  如果上述指令失敗，請嘗試單獨安裝：
echo pip install flask flask-cors mysql-connector-python numpy pandas scikit-learn

echo.
echo 🗄️  步驟2: 檢查資料庫連線
echo ----------------
python -c "import mysql.connector; print('MySQL connector 已安裝')"

echo.
echo 🌐 步驟3: 啟動服務
echo ----------------
echo 選擇執行模式：
echo [1] 完整模式 (需要資料庫連線): python api_server.py
echo [2] 純前端模式: 直接開啟 ship_prediction_interface.html
echo [3] 測試模式: python test_system.py

echo.
echo 📖 使用說明：
echo - 開啟 http://localhost:5000 使用完整系統
echo - 或直接開啟 ship_prediction_interface.html 使用前端功能
echo - 在地圖上手繪軌跡，系統會自動分析並預測

echo.
echo 🎯 主要功能：
echo ✅ DTW演算法軌跡匹配
echo ✅ Kalman濾波器預測
echo ✅ 氣象影響模型
echo ✅ Google Maps 視覺化
echo ✅ 多演算法比較
echo ✅ 置信度評估

pause
