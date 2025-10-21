@echo off
REM 專門針對短航班資料集的評估
REM 放寬參數以適應實際資料特性

setlocal ENABLEDELAYEDEXPANSION
cd /d c:\NCHC_DATA\flydata

echo ========================================
echo 短航班資料集評估（寬鬆參數）
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
echo [OK] 伺服器就緒
echo.

REM 2) 寬鬆參數配置
set QUERIES=30
set POOL=3000
set QUERY_LEN=2
set HORIZON=3
set END_THRESHOLD=100.0
set TIME_BUDGET=60
set BOOTSTRAP=500

echo [參數] 針對短航班優化：
echo   - queries=%QUERIES% (增加樣本數)
echo   - pool=%POOL% (擴大候選池)
echo   - query_len=%QUERY_LEN% (最短查詢)
echo   - horizon=%HORIZON% (最短預測)
echo   - end_threshold=%END_THRESHOLD% km (放寬命中標準)
echo   - time_budget=%TIME_BUDGET%s (加速評估)
echo   - bootstrap=%BOOTSTRAP% (平衡速度與精度)
echo.

REM 3) 輸出檔名
for /f "tokens=1-5 delims=/:. " %%%%a in ("%date% %time%") do set TS=%%%%a%%%%b%%%%c_%%%%d%%%%e
set TS=%TS: =0%
set OUT_CSV=paper\preliminary_table_REAL_short_%TS%.csv
set ABL_CSV=paper\ablation_REAL_short_%TS%.csv

echo [輸出] %OUT_CSV%
echo [輸出] %ABL_CSV%
echo.

REM 4) 執行評估
echo [執行中] 預估時間：15-20 分鐘...
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
  --methods DTW SUBSEQ_DTW FRECHET EUCLIDEAN CONSENSUS

if errorlevel 1 (
  echo.
  echo [錯誤] 評估失敗，檢查 paper\eval_debug.log
  pause
  exit /b 1
)

echo.
echo ========================================
echo [完成] 短航班評估完成！
echo ========================================
echo.
echo 生成檔案：
echo   %OUT_CSV%
echo   %ABL_CSV%
echo.
echo 下一步：檢查結果並生成圖表
pause
endlocal
