@echo off
setlocal
chcp 65001 > nul
title 啟動航班軌跡分析系統

cd /d "c:\NCHC_DATA\flydata"
echo ===============================================
echo 🛫 啟動航班軌跡分析系統
echo ===============================================
echo.

echo 📦 安裝/檢查相依套件（此步驟只需偶爾執行）...
where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3.13 -m pip install --disable-pip-version-check -q -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN] pip install via py -3.13 失敗，改用 python
        python -m pip install --disable-pip-version-check -q -r requirements.txt
    )
) else (
    echo [INFO] 未找到 py 指令，使用 python
    python -m pip install --disable-pip-version-check -q -r requirements.txt
)

echo.
echo 📍 後端服務: http://localhost:5000
echo 🗺️  前端頁面: /demo_with_real_data_fixed.html
echo 🔥 正在啟動後端服務，請保持此視窗開啟...
echo.

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3.13 flight_prediction_server_fixed.py
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN] 無法使用 py -3.13 啟動，改用 python
        python flight_prediction_server_fixed.py
    )
) else (
    python flight_prediction_server_fixed.py
)

pause
endlocal
