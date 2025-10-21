# -*- coding: utf-8 -*-
"""
增強的相似性算法集合
學習 MATLAB 版本的高級算法實現

Author: GitHub Copilot Enhanced
Date: 2025-08-07
"""

import numpy as np
from geopy.distance import geodesic
from typing import List, Dict, Tuple, Any, Optional
import math
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import json
import time

class AdvancedSimilarityAlgorithms:
    """高級相似性算法集合 - 學習 MATLAB 版本的精密實現"""
    
    def __init__(self):
        self.algorithm_weights = {
            'DTW': 0.25,
            'LCSS': 0.20,
            'FRECHET': 0.15,
            'HAUSDORFF': 0.15,
            'EDR': 0.125,
            'ERP': 0.125
        }
        
        # 算法參數（學習 MATLAB 版本的最佳參數）
        self.params = {
            'dtw_window_ratio': 0.1,
            'dtw_step_penalty': 1.0,
            'lcss_epsilon': 0.8,
            'lcss_threshold_ratio': 0.3,
            'frechet_sample_points': 20,
            'frechet_tolerance': 2.0,
            'hausdorff_distance_factor': 1.5,
            'hausdorff_sample_density': 8,
            'edr_epsilon': 1.2,
            'edr_gap_penalty': 1.5,
            'erp_gap_penalty': 1.0,
            'erp_match_threshold': 0.8
        }
    
    def coord_from_point(self, point: Any) -> Tuple[float, float]:
        """統一座標提取"""
        if isinstance(point, dict):
            return (point.get('lat', 0.0), point.get('lng', 0.0))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            return (float(point[1]), float(point[0]))  # GeoJSON: [lng, lat]
        else:
            return (0.0, 0.0)
    
    def geographic_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """計算地理距離（公里）"""
        try:
            return geodesic(coord1, coord2).kilometers
        except:
            # 備用計算方法
            lat1, lng1 = math.radians(coord1[0]), math.radians(coord1[1])
            lat2, lng2 = math.radians(coord2[0]), math.radians(coord2[1])
            
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
            c = 2 * math.asin(math.sqrt(a))
            
            return 6371.0 * c  # 地球半徑 6371 km
    
    def enhanced_dtw_distance(self, path1: List[Any], path2: List[Any]) -> float:
        """增強的 DTW 距離 - 學習 MATLAB 版本的全局搜索"""
        if len(path1) < 2 or len(path2) < 2:
            return float('inf')
        
        n, m = len(path1), len(path2)
        
        # 窗口限制（學習 MATLAB 版本）
        window_size = max(1, int(min(n, m) * self.params['dtw_window_ratio']))
        
        # 初始化 DTW 矩陣
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(max(1, i - window_size), min(m + 1, i + window_size + 1)):
                coord1 = self.coord_from_point(path1[i-1])
                coord2 = self.coord_from_point(path2[j-1])
                
                cost = self.geographic_distance(coord1, coord2)
                
                # 學習 MATLAB 版本的步進懲罰
                step_penalty = self.params['dtw_step_penalty']
                
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j] + step_penalty,        # 插入
                    dtw_matrix[i, j-1] + step_penalty,        # 刪除
                    dtw_matrix[i-1, j-1]                      # 匹配
                )
        
        return dtw_matrix[n, m] / max(n, m)
    
    def enhanced_lcss_distance(self, path1: List[Any], path2: List[Any]) -> float:
        """增強的 LCSS 距離 - 學習 MATLAB 版本的閾值處理"""
        try:
            n, m = len(path1), len(path2)
            epsilon = self.params['lcss_epsilon']
            
            # 動態規劃表
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    coord1 = self.coord_from_point(path1[i-1])
                    coord2 = self.coord_from_point(path2[j-1])
                    
                    distance = self.geographic_distance(coord1, coord2)
                    
                    if distance <= epsilon:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcss_length = dp[n][m]
            max_length = max(n, m)
            
            # 學習 MATLAB 版本的相似度計算
            similarity_ratio = lcss_length / max_length
            threshold_ratio = self.params['lcss_threshold_ratio']
            
            if similarity_ratio < threshold_ratio:
                return 100.0  # 低相似度懲罰
            
            return (1.0 - similarity_ratio) * 100.0
            
        except Exception:
            return float('inf')
    
    def enhanced_frechet_distance(self, path1: List[Any], path2: List[Any]) -> float:
        """增強的 Fréchet 距離 - 學習 MATLAB 版本的連續性考慮"""
        try:
            # 重採樣到統一長度
            sample_points = self.params['frechet_sample_points']
            tolerance = self.params['frechet_tolerance']
            
            # 確保有足夠的點
            if len(path1) < 2 or len(path2) < 2:
                return float('inf')
            
            # 重採樣路徑
            indices1 = np.linspace(0, len(path1)-1, sample_points, dtype=int)
            indices2 = np.linspace(0, len(path2)-1, sample_points, dtype=int)
            
            coords1 = [self.coord_from_point(path1[i]) for i in indices1]
            coords2 = [self.coord_from_point(path2[i]) for i in indices2]
            
            # 計算距離矩陣
            distances = []
            for c1 in coords1:
                row_distances = []
                for c2 in coords2:
                    dist = self.geographic_distance(c1, c2)
                    row_distances.append(dist)
                distances.append(row_distances)
            
            distances = np.array(distances)
            
            # 學習 MATLAB 版本的 Fréchet 距離近似算法
            max_distance = 0.0
            
            # 計算對應點之間的最大距離
            for i in range(sample_points):
                dist = distances[i, i]
                max_distance = max(max_distance, dist)
            
            # 考慮路徑的連續性
            for i in range(sample_points - 1):
                # 檢查相鄰點的距離變化
                dist1 = distances[i, i]
                dist2 = distances[i+1, (i+1) % sample_points]
                
                continuity_penalty = abs(dist2 - dist1) * 0.1
                max_distance += continuity_penalty
            
            return min(max_distance, tolerance * 50)  # 上限控制
            
        except Exception:
            return float('inf')
    
    def enhanced_hausdorff_distance(self, path1: List[Any], path2: List[Any]) -> float:
        """增強的 Hausdorff 距離 - 學習 MATLAB 版本的雙向計算"""
        try:
            coords1 = [self.coord_from_point(p) for p in path1]
            coords2 = [self.coord_from_point(p) for p in path2]
            
            distance_factor = self.params['hausdorff_distance_factor']
            sample_density = self.params['hausdorff_sample_density']
            
            # 採樣以提高效率
            if len(coords1) > sample_density:
                step1 = len(coords1) // sample_density
                coords1 = coords1[::step1]
            
            if len(coords2) > sample_density:
                step2 = len(coords2) // sample_density
                coords2 = coords2[::step2]
            
            # 計算方向1: path1 到 path2 的最大最小距離
            max_min_dist1 = 0.0
            for c1 in coords1:
                min_dist = float('inf')
                for c2 in coords2:
                    dist = self.geographic_distance(c1, c2)
                    min_dist = min(min_dist, dist)
                max_min_dist1 = max(max_min_dist1, min_dist)
            
            # 計算方向2: path2 到 path1 的最大最小距離
            max_min_dist2 = 0.0
            for c2 in coords2:
                min_dist = float('inf')
                for c1 in coords1:
                    dist = self.geographic_distance(c1, c2)
                    min_dist = min(min_dist, dist)
                max_min_dist2 = max(max_min_dist2, min_dist)
            
            # 學習 MATLAB 版本的雙向 Hausdorff 距離
            hausdorff_dist = max(max_min_dist1, max_min_dist2)
            
            return hausdorff_dist * distance_factor
            
        except Exception:
            return float('inf')
    
    def enhanced_edr_distance(self, path1: List[Any], path2: List[Any]) -> float:
        """增強的 EDR 距離 - 學習 MATLAB 版本的編輯距離"""
        try:
            n, m = len(path1), len(path2)
            epsilon = self.params['edr_epsilon']
            gap_penalty = self.params['edr_gap_penalty']
            
            # 動態規劃表
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            
            # 初始化邊界
            for i in range(n + 1):
                dp[i][0] = i * gap_penalty
            for j in range(m + 1):
                dp[0][j] = j * gap_penalty
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    coord1 = self.coord_from_point(path1[i-1])
                    coord2 = self.coord_from_point(path2[j-1])
                    
                    distance = self.geographic_distance(coord1, coord2)
                    
                    if distance <= epsilon:
                        # 匹配情況
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        # 不匹配情況 - 選擇最小代價操作
                        dp[i][j] = min(
                            dp[i-1][j] + gap_penalty,      # 刪除
                            dp[i][j-1] + gap_penalty,      # 插入
                            dp[i-1][j-1] + gap_penalty     # 替換
                        )
            
            # 正規化到百分比
            max_ops = max(n, m)
            return (dp[n][m] / max_ops) * 100 if max_ops > 0 else 100.0
            
        except Exception:
            return float('inf')
    
    def enhanced_erp_distance(self, path1: List[Any], path2: List[Any]) -> float:
        """增強的 ERP 距離 - 學習 MATLAB 版本的實數序列編輯"""
        try:
            n, m = len(path1), len(path2)
            gap_penalty = self.params['erp_gap_penalty']
            match_threshold = self.params['erp_match_threshold']
            
            # 動態規劃表
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            
            # 初始化邊界
            for i in range(n + 1):
                dp[i][0] = i * gap_penalty
            for j in range(m + 1):
                dp[0][j] = j * gap_penalty
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    coord1 = self.coord_from_point(path1[i-1])
                    coord2 = self.coord_from_point(path2[j-1])
                    
                    distance = self.geographic_distance(coord1, coord2)
                    
                    # 學習 MATLAB 版本的距離加權
                    if distance <= match_threshold:
                        match_cost = distance * 0.5  # 好匹配的獎勵
                    else:
                        match_cost = distance  # 普通匹配
                    
                    dp[i][j] = min(
                        dp[i-1][j-1] + match_cost,     # 匹配
                        dp[i-1][j] + gap_penalty,      # 刪除
                        dp[i][j-1] + gap_penalty       # 插入
                    )
            
            # 正規化
            return dp[n][m] / max(n, m) if max(n, m) > 0 else gap_penalty
            
        except Exception:
            return float('inf')
    
    def ensemble_similarity(self, path1: List[Any], path2: List[Any]) -> Dict[str, Any]:
        """集成相似性計算 - 學習 MATLAB 版本的多算法融合"""
        results = {}
        
        try:
            # 計算所有算法的相似度
            results['DTW'] = self.enhanced_dtw_distance(path1, path2)
            results['LCSS'] = self.enhanced_lcss_distance(path1, path2)
            results['FRECHET'] = self.enhanced_frechet_distance(path1, path2)
            results['HAUSDORFF'] = self.enhanced_hausdorff_distance(path1, path2)
            results['EDR'] = self.enhanced_edr_distance(path1, path2)
            results['ERP'] = self.enhanced_erp_distance(path1, path2)
            
            # 計算加權平均分數
            weighted_score = 0.0
            total_weight = 0.0
            valid_scores = {}
            
            for algorithm, score in results.items():
                if not np.isinf(score) and not np.isnan(score) and score >= 0:
                    weight = self.algorithm_weights.get(algorithm, 0.1)
                    weighted_score += weight * score
                    total_weight += weight
                    valid_scores[algorithm] = score
            
            ensemble_score = weighted_score / total_weight if total_weight > 0 else float('inf')
            
            # 計算置信度（學習 MATLAB 版本的置信度評估）
            if len(valid_scores) >= 3:
                scores_array = np.array(list(valid_scores.values()))
                std_dev = np.std(scores_array)
                mean_score = np.mean(scores_array)
                
                # 變異係數作為置信度指標
                confidence = max(0.0, 1.0 - (std_dev / max(mean_score, 1e-6)))
            else:
                confidence = 0.5  # 低置信度
            
            return {
                'individual_scores': results,
                'valid_scores': valid_scores,
                'ensemble_score': ensemble_score,
                'confidence': confidence,
                'algorithm_count': len(valid_scores)
            }
            
        except Exception as e:
            return {
                'individual_scores': results,
                'valid_scores': {},
                'ensemble_score': float('inf'),
                'confidence': 0.0,
                'algorithm_count': 0,
                'error': str(e)
            }
    
    def update_parameters(self, new_params: Dict[str, Any]):
        """更新算法參數"""
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value
                print(f"🔄 更新參數 {key}: {value}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """獲取算法信息"""
        return {
            'available_algorithms': list(self.algorithm_weights.keys()),
            'algorithm_weights': self.algorithm_weights.copy(),
            'current_parameters': self.params.copy(),
            'description': {
                'DTW': '動態時間規整 - 適合時序相似性',
                'LCSS': '最長公共子序列 - 適合部分匹配',
                'FRECHET': 'Fréchet距離 - 適合連續路徑',
                'HAUSDORFF': 'Hausdorff距離 - 適合形狀相似性',
                'EDR': '編輯距離 - 適合序列對齊',
                'ERP': '實數編輯距離 - 適合數值序列'
            }
        }


# 使用範例和測試
if __name__ == "__main__":
    # 創建增強算法實例
    algorithms = AdvancedSimilarityAlgorithms()
    
    # 生成測試路徑
    path1 = [
        {'lat': 25.0, 'lng': 121.5},
        {'lat': 25.1, 'lng': 121.6},
        {'lat': 25.2, 'lng': 121.7},
        {'lat': 25.3, 'lng': 121.8}
    ]
    
    path2 = [
        {'lat': 25.05, 'lng': 121.55},
        {'lat': 25.15, 'lng': 121.65},
        {'lat': 25.25, 'lng': 121.75},
        {'lat': 25.35, 'lng': 121.85}
    ]
    
    # 測試集成相似性
    print("🧪 測試增強的相似性算法...")
    result = algorithms.ensemble_similarity(path1, path2)
    
    print("📊 相似性分析結果:")
    print(f"  集成分數: {result['ensemble_score']:.4f}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  有效算法數: {result['algorithm_count']}")
    print("  個別分數:")
    for algo, score in result['valid_scores'].items():
        print(f"    {algo}: {score:.4f}")
    
    print("✅ 增強算法測試完成！")
