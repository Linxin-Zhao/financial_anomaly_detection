import numpy as np
import random
from collections import defaultdict

def pos_neg_split(idx_train, y_train):
    """分离正负样本"""
    train_pos = []
    train_neg = []
    
    for i, label in enumerate(y_train):
        if label == 1:
            train_pos.append(idx_train[i])
        else:
            train_neg.append(idx_train[i])
    
    return train_pos, train_neg

def pick_step_financial(idx_train, y_train, size=None):
    """为金融异常检测采样训练数据"""
    if size is None:
        return idx_train
    
    # 分离正负样本
    pos_indices = [i for i, label in enumerate(y_train) if label == 1]
    neg_indices = [i for i, label in enumerate(y_train) if label == 0]
    
    # 计算采样数量
    pos_size = min(len(pos_indices), size // 2)
    neg_size = size - pos_size
    
    # 采样
    sampled_pos = random.sample(pos_indices, pos_size) if pos_size > 0 else []
    sampled_neg = random.sample(neg_indices, min(neg_size, len(neg_indices))) if neg_size > 0 else []
    
    # 转换为原始索引
    sampled_indices = [idx_train[i] for i in sampled_pos + sampled_neg]
    
    return sampled_indices

def calculate_metrics(y_true, y_pred, y_prob):
    """计算评估指标"""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
    
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred)
    }

def normalize_features(features):
    """特征标准化"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def build_k_hop_subgraph(adj_list, center_nodes, k=2):
    """构建k跳子图"""
    subgraph_nodes = set(center_nodes)
    current_nodes = set(center_nodes)
    
    for hop in range(k):
        next_nodes = set()
        for node in current_nodes:
            if node < len(adj_list):
                next_nodes.update(adj_list[node])
        current_nodes = next_nodes - subgraph_nodes
        subgraph_nodes.update(current_nodes)
    
    return list(subgraph_nodes)