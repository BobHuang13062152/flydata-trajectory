@echo off
REM 實用全航班評估：平衡覆蓋率與速度
REM 策略：使用合理的候選池大小 + 多次採樣

setlocal ENABLEDELAYEDEXPANSION
cd /d c:\NCHC_DATA\flydata

echo ========================================
echo 實用全航班評估
echo ========================================
echo.

REM 1) 確保伺服器運行
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://localhost:5000/api/statistics).StatusCode } catch { 'ERR' }" > __ping.txt 2>nul
set /p PING=<__ping.txt
del /q __ping.txt >nul 2>&1

if /I not "%PING%"=="200" (
    echo [錯誤] 伺服器未運行
    pause
    exit /b 1
)

echo [OK] 伺服器就緒
echo.

REM 2) 實用參數
echo [配置] 實用全航班評估：
echo.
echo   策略: 使用可管理的候選池 + 充分採樣
echo   - queries: 50 次（足夠的統計意義）
echo   - pool: 5000 航班（可快速取得，涵蓋多樣性）
echo   - query_len: 2, horizon: 3（適應短航班）
echo   - 命中閾值: 100 km（合理平衡）
echo   - bootstrap: 1000（完整 CI）
echo.
echo   預估時間: 25-40 分鐘
echo.

set /p CONFIRM="開始評估？(Y/N): "
if /I not "%CONFIRM%"=="Y" (
    echo [取消]
    pause
    exit /b 0
)

REM 3) 輸出檔名
for /f "tokens=1-5 delims=/:. " %%a in ("%date% %time%") do set TS=%%a%%b%%c_%%d%%e
set TS=%TS: =0%
set OUT_CSV=paper\preliminary_table_REAL_practical_%TS%.csv
set ABL_CSV=paper\ablation_REAL_practical_%TS%.csv

echo.
echo [輸出]
echo   %OUT_CSV%
echo   %ABL_CSV%
echo.

REM 4) 執行評估
echo ========================================
echo [執行中] 實用評估...
echo ========================================
echo.

py -3.13 .\tools\evaluate_methods.py ^
  --out-csv "%OUT_CSV%" ^
  --ablation-out "%ABL_CSV%" ^
  --with-ci ^
  --bootstrap 1000 ^
  --queries 50 ^
  --pool 5000 ^
  --query-len 2 ^
  --horizon 3 ^
  --end-threshold-km 100.0 ^
  --time-budget-s 60 ^
  --methods DTW SUBSEQ_DTW FRECHET EUCLIDEAN CONSENSUS

if errorlevel 1 (
  echo.
  echo [錯誤] 評估失敗
  pause
  exit /b 1
)

echo.
echo ========================================
echo [完成] 實用評估完成！
echo ========================================
echo.
type "%OUT_CSV%"
echo.
pause
endlocal
