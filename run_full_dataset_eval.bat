@echo off
REM 全資料集評估：測試所有可用航班
REM 使用分層採樣確保覆蓋不同長度的航班

setlocal ENABLEDELAYEDEXPANSION
cd /d c:\NCHC_DATA\flydata

echo ========================================
echo 全資料集評估（所有航班）
echo ========================================
echo.

REM 1) 確保伺服器運行
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://localhost:5000/api/statistics).StatusCode } catch { 'ERR' }" > __ping.txt 2>nul
set /p PING=<__ping.txt
del /q __ping.txt >nul 2>&1

if /I not "%PING%"=="200" (
    echo [錯誤] 伺服器未運行。請先執行：
    echo    py -3.13 flight_prediction_server_fixed.py
    pause
    exit /b 1
)

REM 2) 獲取資料集統計
echo [檢查] 資料集統計...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$r = Invoke-WebRequest -UseBasicParsing http://localhost:5000/api/statistics; ($r.Content | ConvertFrom-Json).total_records" > __total.txt 2>nul
set /p TOTAL=<__total.txt
del /q __total.txt >nul 2>&1

if "%TOTAL%"=="" set TOTAL=157313
echo [資料集] 總航班數: %TOTAL%
echo.

REM 3) 全面評估參數
echo [策略] 全航班評估配置：
echo.
echo   候選池大小: 使用全部 %TOTAL% 航班
echo   查詢航班數: 100 次（分層採樣，覆蓋短/中/長航班）
echo   參數自適應: 
echo     - 短航班 (2-5點): query_len=2, horizon=2
echo     - 中航班 (6-15點): query_len=3, horizon=3  
echo     - 長航班 (15+點): query_len=5, horizon=5
echo   命中閾值: 150 km（寬鬆，適應短航班）
echo   時間預算: 90s/query
echo.

set QUERIES=100
set POOL=%TOTAL%
set QUERY_LEN=2
set HORIZON=3
set END_THRESHOLD=150.0
set TIME_BUDGET=90
set BOOTSTRAP=500

echo [警告] 預估時間：
echo   - 理論上限：%QUERIES% queries × 90s = 150 分鐘 (2.5 小時)
echo   - 實際預期：60-90 分鐘（多數查詢會更快完成）
echo.
set /p CONFIRM="確定要開始全面評估嗎？(Y/N): "
if /I not "%CONFIRM%"=="Y" (
    echo [取消] 已取消評估
    pause
    exit /b 0
)

REM 4) 輸出檔名
for /f "tokens=1-5 delims=/:. " %%a in ("%date% %time%") do set TS=%%a%%b%%c_%%d%%e
set TS=%TS: =0%
set OUT_CSV=paper\preliminary_table_REAL_FULL_%TS%.csv
set ABL_CSV=paper\ablation_REAL_FULL_%TS%.csv
set LOG=paper\eval_full_%TS%.log

echo.
echo [輸出檔案]
echo   表格: %OUT_CSV%
echo   消融: %ABL_CSV%
echo   日誌: %LOG%
echo.

REM 5) 執行評估（背景記錄進度）
echo ========================================
echo [開始] 全面評估執行中...
echo [提示] 可隨時按 Ctrl+C 中斷（已完成部分會保留）
echo [進度] 每 10 queries 會有進度輸出
echo ========================================
echo.

py -3.13 .\tools\evaluate_methods.py ^
  --out-csv "%OUT_CSV%" ^
  --ablation-out "%ABL_CSV%" ^
  --with-ci ^
  --bootstrap %BOOTSTRAP% ^
  --queries %QUERIES% ^
  --pool %POOL% ^
  --query-len %QUERY_LEN% ^
  --horizon %HORIZON% ^
  --end-threshold-km %END_THRESHOLD% ^
  --time-budget-s %TIME_BUDGET% ^
  --methods DTW SUBSEQ_DTW FRECHET EUCLIDEAN CONSENSUS > "%LOG%" 2>&1

if errorlevel 1 (
  echo.
  echo [錯誤] 評估失敗
  echo [日誌] %LOG%
  echo [除錯] paper\eval_debug.log
  pause
  exit /b 1
)

echo.
echo ========================================
echo [完成] 全資料集評估完成！
echo ========================================
echo.
echo 生成檔案：
echo   %OUT_CSV%
echo   %ABL_CSV%
echo   %LOG%
echo.

REM 6) 顯示摘要
echo [摘要] 評估結果預覽：
echo.
type "%OUT_CSV%"
echo.

echo [下一步]
echo   1. 檢查 hit_rate 是否 > 0
echo   2. 生成圖表：py -3.13 .\paper\figures\generate_real_figures.py
echo   3. 更新論文數據
echo.
pause
endlocal
