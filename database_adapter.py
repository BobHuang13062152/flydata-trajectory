"""
數據庫適配器 - 將飛行數據適配為船隻軌跡預測系統
"""

import mysql.connector
from mysql.connector import Error, MySQLConnection
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json

class FlightToShipAdapter:
    """將飛行數據適配為船隻數據的適配器"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.connection = None
        
    def connect(self):
        """連接資料庫"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            print("✅ 數據庫連接成功")
            return True
        except Error as e:
            print(f"❌ 數據庫連接失敗: {e}")
            return False
    
    def get_available_tables(self) -> List[str]:
        """獲取可用的數據表"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            cursor.close()
            return tables
        except Error as e:
            print(f"❌ 獲取表格列表失敗: {e}")
            return []
    
    def get_table_structure(self, table_name: str) -> Dict:
        """獲取數據表結構"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return {}
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"DESCRIBE `{table_name}`")
            columns = cursor.fetchall()
            
            structure = {
                'columns': [],
                'sample_data': []
            }
            
            for col in columns:
                structure['columns'].append({
                    'name': col[0],
                    'type': col[1],
                    'nullable': col[2] == 'YES',
                    'key': col[3],
                    'default': col[4]
                })
            
            # 獲取樣本數據
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 5")
            structure['sample_data'] = cursor.fetchall()
            
            cursor.close()
            return structure
            
        except Error as e:
            print(f"❌ 獲取表格結構失敗: {e}")
            return {}
    
    def analyze_flight_data(self, table_name: str = "2022_10") -> Dict:
        """分析飛行數據以了解數據格式"""
        structure = self.get_table_structure(table_name)
        
        if not structure:
            return {}
        
        analysis = {
            'table_name': table_name,
            'total_columns': len(structure['columns']),
            'columns': structure['columns'],
            'sample_data': structure['sample_data'],
            'recommended_mapping': {}
        }
        
        # 自動映射列名
        column_names = [col['name'].lower() for col in structure['columns']]
        
        # 查找可能的經緯度列
        lat_candidates = [name for name in column_names if any(keyword in name for keyword in ['lat', 'latitude', 'y'])]
        lng_candidates = [name for name in column_names if any(keyword in name for keyword in ['lng', 'lon', 'longitude', 'x'])]
        
        # 查找時間列
        time_candidates = [name for name in column_names if any(keyword in name for keyword in ['time', 'timestamp', 'date'])]
        
        # 查找識別符列
        id_candidates = [name for name in column_names if any(keyword in name for keyword in ['id', 'ident', 'callsign', 'flight'])]
        
        analysis['recommended_mapping'] = {
            'latitude_candidates': lat_candidates,
            'longitude_candidates': lng_candidates,
            'timestamp_candidates': time_candidates,
            'identifier_candidates': id_candidates
        }
        
        return analysis
    
    def get_flight_trajectories(self, 
                              table_name: str = "2022_10",
                              limit: int = 10,
                              identifier_column: str = "ident",
                              lat_column: str = "lat", 
                              lng_column: str = "lon",
                              time_column: str = "latest_time") -> Dict[str, List[Dict]]:
        """獲取飛行軌跡數據並轉換為船隻格式"""
        
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return {}
        
        try:
            cursor = self.connection.cursor()
            
            # 獲取不同的飛行器標識符
            cursor.execute(f"""
                SELECT DISTINCT `{identifier_column}` 
                FROM `{table_name}` 
                WHERE `{identifier_column}` IS NOT NULL 
                LIMIT {limit}
            """)
            
            identifiers = [row[0] for row in cursor.fetchall()]
            trajectories = {}
            
            for identifier in identifiers:
                # 獲取每個飛行器的軌跡點
                cursor.execute(f"""
                    SELECT `{lat_column}`, `{lng_column}`, `{time_column}`
                    FROM `{table_name}` 
                    WHERE `{identifier_column}` = %s 
                    AND `{lat_column}` IS NOT NULL 
                    AND `{lng_column}` IS NOT NULL
                    ORDER BY `{time_column}`
                """, (identifier,))
                
                points = []
                for row in cursor.fetchall():
                    if row[0] is not None and row[1] is not None:
                        point = {
                            'latitude': float(row[0]),
                            'longitude': float(row[1]),
                            'timestamp': row[2].isoformat() if isinstance(row[2], datetime) else str(row[2]),
                            'speed': None,  # 飛行數據可能沒有速度
                            'heading': None  # 飛行數據可能沒有航向
                        }
                        points.append(point)
                
                if len(points) >= 3:  # 只保留有足夠點的軌跡
                    trajectories[f"FLIGHT_{identifier}"] = points
            
            cursor.close()
            return trajectories
            
        except Error as e:
            print(f"❌ 獲取軌跡數據失敗: {e}")
            return {}
    
    def convert_to_geojson(self, trajectories: Dict[str, List[Dict]]) -> Dict:
        """將軌跡數據轉換為GeoJSON格式"""
        
        features = []
        
        for flight_id, points in trajectories.items():
            if len(points) < 2:
                continue
            
            # 轉換為GeoJSON LineString格式 [lng, lat]
            coordinates = [[point['longitude'], point['latitude']] for point in points]
            
            feature = {
                "type": "Feature",
                "properties": {
                    "ident": flight_id,
                    "segment": 1,
                    "point_count": len(points),
                    "start_time": points[0]['timestamp'],
                    "end_time": points[-1]['timestamp']
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            }
            
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def create_ship_trajectory_table(self):
        """創建船隻軌跡表（基於飛行數據結構）"""
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return False
        
        try:
            cursor = self.connection.cursor()
            
            # 創建船隻軌跡表
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS ship_positions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ship_id VARCHAR(50) NOT NULL,
                timestamp DATETIME NOT NULL,
                latitude DECIMAL(10, 7) NOT NULL,
                longitude DECIMAL(10, 7) NOT NULL,
                speed DECIMAL(5, 2),
                heading DECIMAL(5, 2),
                source_table VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_ship_time (ship_id, timestamp),
                INDEX idx_location (latitude, longitude)
            )
            """
            
            cursor.execute(create_table_sql)
            self.connection.commit()
            
            print("✅ 船隻軌跡表創建成功")
            cursor.close()
            return True
            
        except Error as e:
            print(f"❌ 創建船隻軌跡表失敗: {e}")
            return False
    
    def migrate_flight_to_ship_data(self, 
                                  source_table: str = "2022_10",
                                  limit: int = 1000,
                                  identifier_column: str = "ident",
                                  lat_column: str = "lat",
                                  lng_column: str = "lon", 
                                  time_column: str = "latest_time"):
        """將飛行數據遷移到船隻軌跡表"""
        
        if not self.create_ship_trajectory_table():
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # 清空目標表
            cursor.execute("TRUNCATE TABLE ship_positions")
            
            # 複製數據
            insert_sql = f"""
            INSERT INTO ship_positions (ship_id, timestamp, latitude, longitude, source_table)
            SELECT 
                CONCAT('SHIP_', `{identifier_column}`) as ship_id,
                `{time_column}` as timestamp,
                `{lat_column}` as latitude,
                `{lng_column}` as longitude,
                '{source_table}' as source_table
            FROM `{source_table}`
            WHERE `{identifier_column}` IS NOT NULL 
            AND `{lat_column}` IS NOT NULL 
            AND `{lng_column}` IS NOT NULL
            LIMIT {limit}
            """
            
            cursor.execute(insert_sql)
            self.connection.commit()
            
            # 獲取插入的記錄數
            cursor.execute("SELECT COUNT(*) FROM ship_positions")
            count = cursor.fetchone()[0]
            
            print(f"✅ 成功遷移 {count} 條記錄到船隻軌跡表")
            cursor.close()
            return True
            
        except Error as e:
            print(f"❌ 數據遷移失敗: {e}")
            return False

def main():
    """主函數 - 數據分析和遷移"""
    
    # 資料庫配置
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
    
    print("🔄 飛行數據到船隻軌跡適配器")
    print("=" * 40)
    
    adapter = FlightToShipAdapter(DB_CONFIG)
    
    # 1. 連接數據庫
    if not adapter.connect():
        return
    
    # 2. 獲取可用表格
    tables = adapter.get_available_tables()
    print(f"📊 找到 {len(tables)} 個數據表: {', '.join(tables)}")
    
    # 3. 分析主要數據表
    if "2022_10" in tables:
        print(f"\n🔍 分析表格: 2022_10")
        analysis = adapter.analyze_flight_data("2022_10")
        
        print(f"📋 列數: {analysis.get('total_columns', 0)}")
        print("🗂️ 建議的列映射:")
        mapping = analysis.get('recommended_mapping', {})
        for key, candidates in mapping.items():
            if candidates:
                print(f"  {key}: {', '.join(candidates)}")
    
    # 4. 獲取軌跡數據樣本
    print(f"\n📍 獲取軌跡數據樣本...")
    trajectories = adapter.get_flight_trajectories(limit=5)
    print(f"✅ 獲取 {len(trajectories)} 條軌跡")
    
    for flight_id, points in list(trajectories.items())[:3]:
        print(f"  {flight_id}: {len(points)} 個點")
    
    # 5. 轉換為GeoJSON
    if trajectories:
        print(f"\n🌍 轉換為GeoJSON格式...")
        geojson_data = adapter.convert_to_geojson(trajectories)
        
        # 保存GeoJSON文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adapted_ship_trajectories_{timestamp}.geojson"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ GeoJSON文件已保存: {filename}")
    
    # 6. 數據遷移（可選）
    print(f"\n🔄 是否進行數據遷移？")
    print("這將創建 ship_positions 表並遷移飛行數據...")
    
    # 自動執行遷移（生產環境中應該讓用戶選擇）
    if adapter.migrate_flight_to_ship_data(limit=500):
        print("✅ 數據遷移完成")
        
        # 驗證遷移結果
        cursor = adapter.connection.cursor()
        cursor.execute("SELECT COUNT(DISTINCT ship_id) FROM ship_positions")
        ship_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ship_positions")
        total_points = cursor.fetchone()[0]
        
        print(f"📊 遷移結果: {ship_count} 艘船隻, {total_points} 個軌跡點")
        cursor.close()
    
    # 7. 關閉連接
    if adapter.connection:
        adapter.connection.close()
        print("🔒 數據庫連接已關閉")

if __name__ == "__main__":
    main()
