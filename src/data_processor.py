import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import defaultdict

class FinancialDataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_ieee_data(self, data_path):
        """加载IEEE欺诈检测数据集"""
        # 加载交易数据
        train_transaction = pd.read_csv(f"{data_path}/train_transaction.csv")
        train_identity = pd.read_csv(f"{data_path}/train_identity.csv")
        
        # 合并数据
        data = train_transaction.merge(train_identity, on='TransactionID', how='left')
        
        # 处理缺失值
        data = self._handle_missing_values(data)
        
        return data
    
    def load_credit_data(self, data_path):
        """加载信用卡欺诈数据集"""
        data = pd.read_csv(f"{data_path}/creditcard.csv")
        
        # 如果数据太大，进行采样
        if len(data) > 50000:
            # 保持正负样本比例的分层采样
            fraud_data = data[data['Class'] == 1]
            normal_data = data[data['Class'] == 0].sample(n=min(40000, len(data[data['Class'] == 0])))
            data = pd.concat([fraud_data, normal_data]).sample(frac=1).reset_index(drop=True)
            print(f"数据采样后大小: {len(data)}")
        
        return data
    
    def _handle_missing_values(self, data):
        """处理缺失值"""
        # 数值型特征用中位数填充
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['TransactionID', 'isFraud']:
                data[col].fillna(data[col].median(), inplace=True)
        
        # 类别型特征用众数填充
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown', inplace=True)
        
        return data
    
    def build_transaction_graph(self, data):
        """构建交易网络图"""
        user_merchant_edges = []
        user_device_edges = []
        merchant_category_edges = []
        
        # 对于creditcard数据集，我们需要基于现有特征构建图
        # creditcard数据集主要包含V1-V28的PCA特征和Amount、Time等
        for idx, row in data.iterrows():
            # 基于交易金额范围创建虚拟的用户-商户关系
            amount_range = int(row['Amount'] // 100)  # 按100为单位分组
            time_period = int(row['Time'] // 3600)    # 按小时分组
            
            user_id = f"user_{idx}"
            merchant_id = f"merchant_{amount_range}"
            time_id = f"time_{time_period}"
            
            user_merchant_edges.append((user_id, merchant_id))
            user_device_edges.append((user_id, time_id))
        
        return {
            'user_merchant': user_merchant_edges,
            'user_device': user_device_edges,
            'merchant_category': merchant_category_edges
        }
    
    def create_adjacency_lists(self, edges_dict, node_mapping):
        """创建邻接表"""
        adj_lists = []
        
        for edge_type, edges in edges_dict.items():
            adj_list = defaultdict(set)
            
            for src, dst in edges:
                if src in node_mapping and dst in node_mapping:
                    src_idx = node_mapping[src]
                    dst_idx = node_mapping[dst]
                    adj_list[src_idx].add(dst_idx)
                    adj_list[dst_idx].add(src_idx)  # 无向图
            
            # 转换为列表格式
            max_node = max(node_mapping.values()) + 1
            adj_array = [list(adj_list[i]) for i in range(max_node)]
            adj_lists.append(adj_array)
        
        return adj_lists
    
    def prepare_features(self, data):
        """准备节点特征"""
        # 对于creditcard数据集，选择所有V特征和Amount、Time
        if 'Class' in data.columns:  # creditcard数据集
            feature_cols = [col for col in data.columns 
                           if col not in ['Class'] and col.startswith(('V', 'Amount', 'Time'))]
            target_col = 'Class'
        else:  # IEEE数据集
            feature_cols = [col for col in data.columns 
                           if col not in ['TransactionID', 'isFraud', 'TransactionDT']]
            target_col = 'isFraud'
        
        # 编码类别型特征
        for col in data.select_dtypes(include=['object']).columns:
            if col in feature_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        # 标准化特征
        features = data[feature_cols].values
        features = self.scaler.fit_transform(features)
        
        return features, data[target_col].values