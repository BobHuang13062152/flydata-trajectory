@echo off
REM 非互動：完整真實評估（標準參數），自動檢查/啟動伺服器，並將結果輸出到 paper\*_REAL_*.csv

setlocal ENABLEDELAYEDEXPANSION
cd /d c:\NCHC_DATA\flydata

echo ========================================
echo 完整真實評估（標準參數）
echo ========================================
echo.

REM 0) 參數（可視情況調整）
set QUERIES=20
set POOL=2000
set QUERY_LEN=3
set HORIZON=5
set BOOTSTRAP=1000
set TIME_BUDGET=90

REM 1) 確保 Flask 伺服器運行
echo [檢查] Flask 伺服器...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://localhost:5000/api/statistics).StatusCode } catch { 'ERR' }" > __server_ping.txt 2>nul
set /p PING=<__server_ping.txt
del /q __server_ping.txt >nul 2>&1

if /I not "%PING%"=="200" (
    echo [啟動] 伺服器未就緒，嘗試啟動 flight_prediction_server_fixed.py...
    start "server" /min py -3.13 c:\NCHC_DATA\flydata\flight_prediction_server_fixed.py
    timeout /t 6 /nobreak >nul
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://localhost:5000/api/statistics).StatusCode } catch { 'ERR' }" > __server_ping2.txt 2>nul
    set /p PING2=<__server_ping2.txt
    del /q __server_ping2.txt >nul 2>&1
    if /I not "%PING2%"=="200" (
        echo [錯誤] 無法啟動或連線到 Flask 伺服器（/api/statistics 不是 200）。
        echo        請手動在另一個終端執行：py -3.13 flight_prediction_server_fixed.py
        echo        然後再執行本批次檔。
        pause
        exit /b 1
    )
)
echo [確認] 伺服器就緒（/api/statistics = 200）
echo.

REM 2) 準備輸出檔名（含時間戳）
for /f "tokens=1-5 delims=/:. " %%a in ("%date% %time%") do set TS=%%a%%b%%c_%%d%%e
set TS=%TS: =0%
set OUT_CSV=paper\preliminary_table_REAL_%TS%.csv
set ABL_CSV=paper\ablation_REAL_%TS%.csv

echo [輸出] %OUT_CSV%
echo [輸出] %ABL_CSV%

REM 3) 執行評估（記錄 eval_debug.log 與畫面輸出）
echo.
echo [開始] evaluate_methods.py ...
py -3.13 .\tools\evaluate_methods.py ^
  --out-csv "%OUT_CSV%" ^
  --ablation-out "%ABL_CSV%" ^
  --with-ci ^
  --bootstrap %BOOTSTRAP% ^
  --queries %QUERIES% ^
  --pool %POOL% ^
  --query-len %QUERY_LEN% ^
  --horizon %HORIZON% ^
  --end-threshold-km 50.0 ^
  --time-budget-s %TIME_BUDGET% ^
  --methods DTW SUBSEQ_DTW FRECHET EUCLIDEAN CONSENSUS

if errorlevel 1 (
  echo.
  echo [錯誤] 評估失敗，請查看 paper\eval_debug.log 以取得詳情。
  pause
  exit /b 1
)

echo.
echo [完成] REAL 評估已完成！
echo 生成：
echo   - %OUT_CSV%
echo   - %ABL_CSV%
echo.

REM 4) 可選：產生 REAL 圖表
if exist "%OUT_CSV%" (
  if exist "%ABL_CSV%" (
    echo [繪圖] 產生 REAL 圖表（可略過）...
    py -3.13 .\paper\figures\generate_real_figures.py >nul 2>&1
  )
)

echo [OK] 全部完成。
pause
endlocal
