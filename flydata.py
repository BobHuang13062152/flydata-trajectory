import mysql.connector
from mysql.connector import Error, MySQLConnection
import csv
from datetime import datetime

# 資料庫設定（use_pure=True 強制純 Python driver）
DB_CONFIG = {
    'host': '140.110.93.152',
    'port': 3306,
    'user': 'project',
    'password': 'project',
    'database': 'flydata',
    'charset': 'utf8mb4',
    'use_unicode': True,
    'use_pure': True
}

# 全域連線物件
_conn: MySQLConnection | None = None

def get_connection() -> MySQLConnection:
    """
    回傳一個可用的資料庫連線；若沒有連線或已斷開，自動重連。
    """
    global _conn
    if _conn is None or not _conn.is_connected():
        _conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ 已成功連線至資料庫")
    return _conn

def query_data(**filters):
    """
    動態建構 WHERE 子句，接收任意欄位過濾條件：
      例：query_data(ident='ANA848', latest_time='2022-10-01', other_col='xxx')
    special case: 當 key='latest_time'，自動用 DATE(`latest_time`) = %s
    回傳 (rows, cols)
    """
    conn = get_connection()
    cursor = conn.cursor()
    clauses, params = [], []
    for col, val in filters.items():
        if col == 'latest_time':
            clauses.append("DATE(`latest_time`) = %s")
        else:
            clauses.append(f"`{col}` = %s")
        params.append(val)
    where = " AND ".join(clauses) if clauses else "1"
    sql = f"SELECT * FROM `2022_12` WHERE {where}"
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    cursor.close()
    return rows, cols

def export_to_files(rows, cols):
    """
    將查詢結果寫入 CSV/TXT，檔名會加上當前執行時間避免覆蓋。
    回傳 (csv_path, txt_path)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"output_{ts}.csv"
    txt_path = f"output_{ts}.txt"
    
    # CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(cols)
        writer.writerows(rows)
    print(f"✅ 已將結果寫入 {csv_path}")

    # TXT (tab 分隔)
    with open(txt_path, 'w', encoding='utf-8') as f_txt:
        f_txt.write('\t'.join(cols) + '\n')
        for row in rows:
            f_txt.write('\t'.join(str(v) for v in row) + '\n')
    print(f"✅ 已將結果寫入 {txt_path}")

    return csv_path, txt_path

if __name__ == "__main__":
    # ----------- 使用範例 -----------
    # 自訂你要過濾的欄位和值（可任意增減）
    filters = {
        #'ident': 'ANA848',
        #'latest_time': '2022-10-04',
        # 'other_col': 'some_value',
    }

    rows, cols = query_data(**filters)
    print(f"查到 {len(rows)} 筆資料，正在輸出檔案…")

    csv_file, txt_file = export_to_files(rows, cols)

    # 如果之後要關閉連線，請手動呼叫：
    # conn = get_connection()
    # conn.close()
    # print("🔒 資料庫連線已關閉")
一、規劃與需求釐清
確認目標

可視化：把已知的飛行航線疊加在 Google Maps 上，並能以滑動或點擊方式查看飛行軌跡。

預測：給定一段「已飛行」的殘缺航段，預測接下來的飛行路徑。

明確里程碑

資料蒐集與前處理

地圖可視化介面開發

歷史航線庫建置

DTW-based 預測原型

（可選）深度學習微調版模型

二、資料蒐集與前處理
資料來源

航班歷史軌跡：經緯度、時間戳（ISO‐8601）、地速（SOG）、航向（COG）、高度。

必要時抓 ADS-B API、開放資料或航管系統匯出。

前處理步驟

時間對齊：統一采樣間隔（例如每 1 分鐘、每 5 分鐘），不足用線性插值補點。

清洗雜訊：去掉異常點（如跳躍式漂移，方法可用速度閾值或地理距離閾值）。

特徵擴增：航向角轉 sin/cos 編碼，若要考慮高度，也可標準化處理。

分段標記：用 trip_id 標記「飛行段」，保持完整軌跡。

三、Google Maps 可視化
1. 選擇技術棧
前端：JavaScript + Google Maps JavaScript API

後端（可選）：Node.js／Python Flask 提供 GeoJSON 端點

2. 範例：用 Leaflet + Google Maps Tile
html
複製
編輯
<!DOCTYPE html>
<html>
<head>
  <title>Flight Tracks on Google Maps</title>
  <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY"></script>
</head>
<body>
  <div id="map" style="width:100%; height:100vh;"></div>
  <script>
    const map = new google.maps.Map(document.getElementById('map'), {
      center: { lat: 25.0, lng: 121.5 }, // 台灣附近
      zoom: 6
    });

    // 假設你有 GeoJSON 格式的飛行軌跡陣列 flights[]
    flights.forEach(flight => {
      const path = flight.coords.map(p => ({ lat: p[0], lng: p[1] }));
      new google.maps.Polyline({
        path,
        strokeColor: '#FF0000',
        strokeOpacity: 0.7,
        strokeWeight: 2,
        map
      });
    });
  </script>
</body>
</html>
後端：把處理後的 flights 以 JSON 提供給前端。

四、建立歷史航線庫
資料結構

matlab
複製
編輯
% MATLAB 版
referenceTrajectories = struct( ...
  'flightID', {1001,1002,…}, ...
  'traj',      {M1×D 矩陣, M2×D 矩陣, …} );
儲存方式

存成 mat 檔，或用 Python 存成 pickle / HDF5；

前端若要動態載入，可做成 REST API 回傳 JSON。

五、DTW-Based 預測原型
核心演算法

matlab
複製
編輯
function futurePart = predictFuture(partialTraj, refs)
  bestDist = Inf;
  for i = 1:numel(refs)
    [d, startIdx] = subseq_dtw(partialTraj, refs(i).traj);
    if d < bestDist
      bestDist = d;
      bestRef   = refs(i).traj;
      bestStart = startIdx;
    end
  end
  Np = size(partialTraj,1);
  % 截取最優匹配後的剩餘段
  futurePart = bestRef(bestStart+Np : end, :);
end
流程

對 partial 做同樣的「插值＋歸一化＋ sin/cos」處理

呼叫 predictFuture 拿到 futurePart（未經修飾）

將 futurePart 接回原有軌跡，在地圖上疊加（用不同顏色區分預測段）

六、深度學習微調（可選）
動機：直接拿未來段可能會有噪訊或長度不足，可以用 LSTM 做「殘缺→未來」的映射，提升平滑度。

做法：

訓練資料：歷史航跡切出「前 T 點」當 X，「後 U 點」當 Y。

網路：sequence-to-sequence LSTM，輸入維度 D，輸出未來 U×D。

推論：給 partial，再加上 DTW 找到的 guideSegment 作為「引導」，輸出最後預測。

七、整合部署
API 介面

/api/getFlights → 回傳可視化所需飛行列表。

/api/predict → 輸入 partial JSON，回傳 futurePart JSON。

前端

地圖上動態載入航跡與該航跡的預測段，提供滑動條控制「預測長度」。

監控與評估

可視化檢視：把預測 vs 真實後段同圖對比。

誤差統計：一段段測試誤差 RMSE、DTW 距離分佈。

小結
先完成「飛線可視化」：清洗資料 → GeoJSON → Google Maps Polyline。

再實作「DTW 預測原型」：subsequence DTW 找 guide → 截未來段 → 顯示在地圖上。

後可進階「LSTM 微調」與工程化部署。

按照這個流程，你就能先把飛行航線畫上地圖、再一步步加上預測功能。祝你順利！如果有任何技術細節需要深入，隨時再問。      

