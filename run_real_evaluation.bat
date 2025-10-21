@echo off
REM 真實評估執行腳本
REM 這個腳本會在背景執行評估,並記錄詳細進度

echo ========================================
echo 真實評估執行腳本
echo ========================================
echo.

REM 檢查伺服器是否運行
curl -s http://localhost:5000/api/statistics >nul 2>&1
if errorlevel 1 (
    echo [錯誤] Flask 伺服器未運行!
    echo.
    echo 請先在另一個終端執行:
    echo    python flight_prediction_server_fixed.py
    echo.
    pause
    exit /b 1
)

echo [確認] Flask 伺服器運行中
echo.

REM 詢問使用哪個參數方案
echo 請選擇評估參數:
echo.
echo   1. 快速測試 (5 queries, pool=100, 約 10-30 分鐘)
echo   2. 標準評估 (20 queries, pool=500, 約 1-3 小時) [建議]
echo   3. 完整評估 (50 queries, pool=1000, 約 3-6 小時)
echo   4. 自訂參數
echo.
set /p choice="請選擇 (1-4): "

if "%choice%"=="1" (
    set QUERIES=5
    set POOL=100
    set QUERY_LEN=5
    set HORIZON=10
) else if "%choice%"=="2" (
    set QUERIES=20
    set POOL=500
    set QUERY_LEN=5
    set HORIZON=10
) else if "%choice%"=="3" (
    set QUERIES=50
    set POOL=1000
    set QUERY_LEN=5
    set HORIZON=10
) else if "%choice%"=="4" (
    set /p QUERIES="查詢次數 (建議 20): "
    set /p POOL="航班池大小 (建議 500): "
    set /p QUERY_LEN="查詢長度 (建議 5): "
    set /p HORIZON="預測步數 (建議 10): "
) else (
    echo [錯誤] 無效選擇
    pause
    exit /b 1
)

echo.
echo [參數] queries=%QUERIES%, pool=%POOL%, query_len=%QUERY_LEN%, horizon=%HORIZON%
echo.
echo [開始] 正在執行評估...
echo [日誌] 進度會記錄到 paper\eval_debug.log
echo [提示] 可按 Ctrl+C 中斷 (已執行的部分會保留)
echo.

REM 設定輸出檔案
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set OUT_CSV=paper\preliminary_table_REAL_%TIMESTAMP%.csv
set ABL_CSV=paper\ablation_REAL_%TIMESTAMP%.csv

echo [輸出] %OUT_CSV%
echo [輸出] %ABL_CSV%
echo.

REM 執行評估
py -3.13 .\tools\evaluate_methods.py ^
  --out-csv "%OUT_CSV%" ^
  --ablation-out "%ABL_CSV%" ^
  --with-ci ^
  --bootstrap 1000 ^
  --queries %QUERIES% ^
  --pool %POOL% ^
  --query-len %QUERY_LEN% ^
  --horizon %HORIZON% ^
  --end-threshold-km 50.0 ^
  --time-budget-s 60

if errorlevel 1 (
    echo.
    echo [錯誤] 評估失敗,請檢查 paper\eval_debug.log
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo [完成] 真實評估執行完畢!
echo ========================================
echo.
echo 生成的檔案:
echo   - %OUT_CSV%
echo   - %ABL_CSV%
echo.
echo 下一步:
echo   1. 檢查 CSV 內容: type "%OUT_CSV%"
echo   2. 生成圖表: py -3.13 .\paper\figures\generate_reviewer_figures_REAL.py
echo   3. 更新論文中的數據和圖表
echo.
pause

