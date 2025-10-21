
# -*- coding: utf-8 -*-

# import os
# import pandas as pd
# import numpy as np
# import json
# from datetime import datetime


# def preprocess_adsb_file(
#     txt_path,
#     sampling_freq='30s',         # 採樣頻率：30 秒
#     time_gap_thresh_s=600,       # 斷段閾值：600 秒
#     include_vertical=True
# ):
#     """
#     1) 讀取 ADS-B 原始資料 (.csv)；
#     2) 時間轉 datetime、排序，並剔除無效時間行；
#     3) 欄位重命名、heading→sin/cos 編碼；
#     4) 同 ident 分段（時間差 > time_gap_thresh_s 為新段）；
#     5) 每段以 sampling_freq 重採樣、線性插值補點；
#     6) 回傳 list of DataFrames，每個含 time + 特徵欄位。
#     """
#     df = pd.read_csv(txt_path, sep=None, engine='python')

#     df['clock_time']  = pd.to_datetime(df['clock'], unit='s', utc=True)
#     df['latest_time'] = pd.to_datetime(df['latest_time'], utc=True)
#     df = df.dropna(subset=['clock_time'])

#     needed = {'clock_time','ident','lat','lon','gs','heading','latest_time'}
#     missing = needed - set(df.columns)
#     if missing:
#         raise KeyError(f"輸入檔缺少這些欄位：{missing}")

#     df = df.sort_values(['ident','clock_time'])
#     df = df.rename(columns={
#         'lat':'latitude',
#         'lon':'longitude',
#         'gs':'ground_speed',
#         'vertRate':'vertical_rate'
#     })

#     df['heading_rad'] = np.deg2rad(df['heading'])
#     df['sin_heading'] = np.sin(df['heading_rad'])
#     df['cos_heading'] = np.cos(df['heading_rad'])

#     feats = ['latitude','longitude','ground_speed','sin_heading','cos_heading']
#     if include_vertical:
#         feats.append('vertical_rate')

#     segments = []
#     gap = pd.Timedelta(seconds=time_gap_thresh_s)

#     for ident, grp in df.groupby('ident', sort=False):
#         grp = grp.set_index('clock_time')
#         grp = grp[~grp.index.duplicated(keep='first')]

#         seg_id = 0
#         prev_time = None
#         grp['segment'] = 0

#         for t in grp.index:
#             if prev_time is None or (t - prev_time) > gap:
#                 seg_id += 1
#             grp.at[t, 'segment'] = seg_id
#             prev_time = t

#         for sid, sub in grp.groupby('segment', sort=False):
#             if sub.empty:
#                 continue
#             t0, t1 = sub.index.min(), sub.index.max()
#             if pd.isna(t0) or pd.isna(t1):
#                 continue
#             new_idx = pd.date_range(start=t0, end=t1, freq=sampling_freq)
#             sub_rs = sub.reindex(new_idx)[feats].interpolate(method='time')
#             sub_rs = sub_rs.assign(ident=ident, segment=int(sid)) \
#                             .reset_index().rename(columns={'index':'time'})
#             segments.append(sub_rs)

#     return segments


# def export_segments_to_geojson(segments, out_path):
#     features = []
#     for seg in segments:
#         coords = [
#             [float(lon), float(lat)]
#             for lon, lat in zip(seg['longitude'], seg['latitude'])
#             if not (pd.isna(lon) or pd.isna(lat))
#         ]
#         if len(coords) < 2:
#             continue
#         features.append({
#             "type": "Feature",
#             "properties": {
#                 "ident": str(seg['ident'].iat[0]),
#                 "segment": int(seg['segment'].iat[0])
#             },
#             "geometry": {
#                 "type": "LineString",
#                 "coordinates": coords
#             }
#         })

#     fc = {"type":"FeatureCollection","features":features}
#     with open(out_path, 'w', encoding='utf-8') as f:
#         json.dump(fc, f, ensure_ascii=False, indent=2)
#     print(f"[GeoJSON] 已輸出 {len(features)} 條航段到 '{out_path}'")


# if __name__ == "__main__":
#     # 1) 手動列出四個輸入檔案
#     folder = r"C:\NCHC_DATA\flydata"
#     files = [
#         os.path.join(folder, "output_20250806_154334.csv"),
#         os.path.join(folder, "output_20250806_155423.csv"),
#         os.path.join(folder, "output_20250806_155629.csv"),
#         os.path.join(folder, "output_20250806_160537.csv"),
#     ]

#     # 2) 讀取並合併
#     dfs = []
#     for fp in files:
#         if not os.path.isfile(fp):
#             raise FileNotFoundError(f"找不到檔案：{fp}")
#         print(f"讀取 {os.path.basename(fp)} …")
#         dfs.append(pd.read_csv(fp, sep=None, engine='python'))
#     df_all = pd.concat(dfs, ignore_index=True)
#     print(f"✅ 共合併 {len(dfs)} 個檔案，總筆數 {len(df_all)}")

#     # 3) 暫存合併後 CSV（可選）
#     merged_csv = os.path.join(folder, "output_all.csv")
#     df_all.to_csv(merged_csv, index=False, encoding='utf-8')
#     print(f"✅ 合併後 CSV 已寫入: {merged_csv}")

#     # 4) 前處理 & 匯出 GeoJSON
#     segments = preprocess_adsb_file(
#         txt_path=merged_csv,
#         sampling_freq="30s",
#         time_gap_thresh_s=120,
#         include_vertical=True
#     )
#     print(f"共產生 {len(segments)} 段軌跡。")

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_geojson = os.path.join(folder, f"flights_{timestamp}.geojson")
#     export_segments_to_geojson(segments, out_geojson)



import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

def preprocess_adsb_file(
    df: pd.DataFrame,
    sampling_freq: str = '30s',
    time_gap_thresh_s: int = 600,
    include_vertical: bool = True
):
    df['clock_time']  = pd.to_datetime(df['clock'], unit='s', utc=True)
    df['latest_time'] = pd.to_datetime(df['latest_time'], utc=True)
    df = df.dropna(subset=['clock_time'])

    needed = {'clock_time','ident','lat','lon','gs','heading','latest_time'}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"缺少欄位：{missing}")

    df = df.sort_values(['ident','clock_time'])
    df = df.rename(columns={
        'lat':'latitude',
        'lon':'longitude',
        'gs':'ground_speed',
        'vertRate':'vertical_rate'
    })
    df['heading_rad'] = np.deg2rad(df['heading'])
    df['sin_heading'] = np.sin(df['heading_rad'])
    df['cos_heading'] = np.cos(df['heading_rad'])

    feats = ['latitude','longitude','ground_speed','sin_heading','cos_heading']
    if include_vertical:
        feats.append('vertical_rate')

    segments = []
    gap = pd.Timedelta(seconds=time_gap_thresh_s)

    for ident, grp in df.groupby('ident', sort=False):
        grp = grp.set_index('clock_time')
        grp = grp[~grp.index.duplicated(keep='first')]
        seg_id = 0
        prev = None
        grp['segment'] = 0
        for t in grp.index:
            if prev is None or (t - prev) > gap:
                seg_id += 1
            grp.at[t,'segment'] = seg_id
            prev = t

        for sid, sub in grp.groupby('segment', sort=False):
            if sub.empty: continue
            t0, t1 = sub.index.min(), sub.index.max()
            if pd.isna(t0) or pd.isna(t1): continue
            idx = pd.date_range(t0, t1, freq=sampling_freq)
            sub_rs = sub.reindex(idx)[feats].interpolate(method='time')
            sub_rs = sub_rs.assign(ident=ident, segment=int(sid))\
                           .reset_index().rename(columns={'index':'time'})
            segments.append(sub_rs)

    return segments

def export_segments_to_geojson(segments, out_path):
    features = []
    for seg in segments:
        coords = []
        times  = []
        for lon, lat, t in zip(seg['longitude'], seg['latitude'], seg['time']):
            if pd.isna(lon) or pd.isna(lat): continue
            coords.append([float(lon), float(lat)])
            times.append(t.isoformat())
        if len(coords) < 2: continue
        features.append({
            "type":"Feature",
            "properties":{
                "ident": str(seg['ident'].iat[0]),
                "segment": int(seg['segment'].iat[0]),
                "times": times
            },
            "geometry":{
                "type":"LineString",
                "coordinates": coords
            }
        })

    fc = {"type":"FeatureCollection","features":features}
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)
    print(f"[GeoJSON] 輸出 {len(features)} 條航段到 {out_path}")

if __name__ == "__main__":
    folder = r"C:\NCHC_DATA\flydata"
    files = [
        os.path.join(folder, "output_20250806_154334.csv"),
        os.path.join(folder, "output_20250806_155423.csv"),
        os.path.join(folder, "output_20250806_155629.csv"),
        os.path.join(folder, "output_20250806_160537.csv"),
    ]
    dfs = []
    for fp in files:
        print("讀取", fp)
        dfs.append(pd.read_csv(fp, sep=None, engine='python'))
    df_all = pd.concat(dfs, ignore_index=True)
    print("合併後總筆數:", len(df_all))

    merged_csv = os.path.join(folder, "output_all_neo.csv")
    df_all.to_csv(merged_csv, index=False)
    print("寫入合併檔:", merged_csv)

    segments = preprocess_adsb_file(df_all, sampling_freq='30s', time_gap_thresh_s=120)
    print("產生 segments:", len(segments))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_geo = os.path.join(folder, f"flights_{ts}.geojson")
    export_segments_to_geojson(segments, out_geo)
