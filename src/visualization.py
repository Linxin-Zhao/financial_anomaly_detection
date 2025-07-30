import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

class FinancialGNNVisualizer:
    """金融GNN模型可视化工具"""
    
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_training_curves(self, train_losses, val_aucs, save_path=None):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 训练损失曲线
        ax1.plot(train_losses, 'b-', label='训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True)
        
        # 验证AUC曲线
        ax2.plot(val_aucs, 'r-', label='验证AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.set_title('验证AUC曲线')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distribution(self, features, labels, save_path=None):
        """绘制特征分布"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(min(6, features.shape[1])):
            ax = axes[i]
            normal_data = features[labels == 0, i]
            fraud_data = features[labels == 1, i]
            
            ax.hist(normal_data, bins=50, alpha=0.7, label='正常交易', density=True)
            ax.hist(fraud_data, bins=50, alpha=0.7, label='异常交易', density=True)
            ax.set_xlabel(f'特征 {i+1}')
            ax.set_ylabel('密度')
            ax.set_title(f'特征 {i+1} 分布')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_embedding_visualization(self, embeddings, labels, method='tsne', save_path=None):
        """绘制嵌入向量可视化"""
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            title = 't-SNE 嵌入向量可视化'
        else:
            reducer = PCA(n_components=2, random_state=42)
            title = 'PCA 嵌入向量可视化'
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], 
                            c='blue', alpha=0.6, label='正常交易', s=20)
        scatter = plt.scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], 
                            c='red', alpha=0.8, label='异常交易', s=20)
        
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['正常', '异常'], 
                   yticklabels=['正常', '异常'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        """绘制ROC曲线"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (FPR)')
        plt.ylabel('真正率 (TPR)')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_network_structure(self, save_path=None):
        """绘制网络结构图"""
        import networkx as nx
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加节点
        layers = [
            '输入特征\n(30维)',
            '金额关系\n聚合器',
            '时间关系\n聚合器', 
            '用户关系\n聚合器',
            '跨关系\n聚合器',
            '分类器\n(2类)'
        ]
        
        for i, layer in enumerate(layers):
            G.add_node(i, label=layer)
        
        # 添加边
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (4, 5)]
        G.add_edges_from(edges)
        
        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.9)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, arrowstyle='->')
        
        # 添加标签
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('金融GNN网络结构图', fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
