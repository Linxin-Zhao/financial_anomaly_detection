import networkx as nx
import numpy as np
from collections import defaultdict

class FinancialGraphBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_heterogeneous_graph(self, data):
        """构建异构金融交易图"""
        # 创建节点映射
        node_mapping = {}
        node_counter = 0
        
        # 添加交易节点
        for idx in data.index:
            node_id = f"transaction_{idx}"
            node_mapping[node_id] = node_counter
            node_counter += 1
        
        # 构建多种关系的边
        edges = {
            'amount_similarity': self._build_amount_edges(data, node_mapping),
            'time_proximity': self._build_time_edges(data, node_mapping),
            'user_similarity': self._build_user_edges(data, node_mapping)
        }
        
        return edges, node_mapping
    
    def _build_time_edges(self, data, node_mapping, time_window=3600):
        """基于时间邻近性构建边（优化版本）"""
        edges = []
        
        # 对于creditcard数据集，Time列是相对时间，需要特殊处理
        if 'Time' in data.columns:
            times = data['Time'].values
            # 按时间排序获取索引
            sorted_indices = np.argsort(times)
            
            # 只连接时间窗口内的相邻交易，限制连接数量
            for i, idx in enumerate(sorted_indices):
                # 只检查后续的少量交易，避免O(n²)复杂度
                for j in range(i+1, min(i+20, len(sorted_indices))):
                    next_idx = sorted_indices[j]
                    if abs(times[idx] - times[next_idx]) <= time_window:
                        edges.append((idx, next_idx))
                    else:
                        break  # 时间差太大，后续的更不可能满足条件
        else:
            # 如果没有时间列，创建基于索引的简单连接
            for i in range(min(1000, len(data))):  # 限制处理的数据量
                for j in range(i+1, min(i+5, len(data))):
                    edges.append((i, j))
        
        return edges
    
    def _build_amount_edges(self, data, node_mapping, threshold=0.1):
        """基于交易金额相似性构建边（优化版本）"""
        edges = []
        amounts = data['Amount'].values if 'Amount' in data.columns else data['TransactionAmt'].values
        
        # 使用更高效的方法：按金额排序后只比较相邻的交易
        sorted_indices = np.argsort(amounts)
        
        for i, idx in enumerate(sorted_indices):
            # 只检查金额相近的少量交易
            for j in range(i+1, min(i+10, len(sorted_indices))):
                next_idx = sorted_indices[j]
                amt_diff = abs(amounts[idx] - amounts[next_idx]) / (amounts[idx] + amounts[next_idx] + 1e-8)
                if amt_diff < threshold:
                    edges.append((idx, next_idx))
                else:
                    break  # 金额差距太大，后续的更不可能满足条件
        
        return edges
    
    def _build_user_edges(self, data, node_mapping):
        """基于用户相似性构建边（优化版本）"""
        edges = []
        
        # 对于creditcard数据集，使用PCA特征的相似性
        if 'V1' in data.columns:
            # 选择几个主要的PCA特征进行相似性计算
            feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5']
            features = data[feature_cols].values
            
            # 使用采样方法减少计算量
            sample_size = min(5000, len(data))  # 限制样本大小
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            
            for i, idx1 in enumerate(sample_indices):
                for j in range(i+1, min(i+20, len(sample_indices))):
                    idx2 = sample_indices[j]
                    # 计算特征相似性
                    similarity = np.corrcoef(features[idx1], features[idx2])[0, 1]
                    if not np.isnan(similarity) and similarity > 0.8:
                        edges.append((idx1, idx2))
        
        return edges
    
    def create_adjacency_lists(self, edges_dict, num_nodes):
        """创建邻接表，确保每个节点至少有一个邻居"""
        adj_lists = []
        
        for edge_type, edges in edges_dict.items():
            adj_list = defaultdict(set)
            
            for src, dst in edges:
                if src < num_nodes and dst < num_nodes:
                    adj_list[src].add(dst)
                    adj_list[dst].add(src)
            
            # 转换为列表格式
            adj_array = []
            for i in range(num_nodes):
                neighbors = list(adj_list[i])
                # 如果节点没有邻居，添加自环或最近的节点
                if len(neighbors) == 0:
                    # 添加自环
                    neighbors = [i]
                    # 或者添加相邻的节点
                    if i > 0:
                        neighbors.append(i - 1)
                    if i < num_nodes - 1:
                        neighbors.append(i + 1)
                    # 去重
                    neighbors = list(set(neighbors))
                adj_array.append(neighbors)
            
            adj_lists.append(adj_array)
        
        return adj_lists