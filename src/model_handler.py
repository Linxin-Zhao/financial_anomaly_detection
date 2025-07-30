import time
import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from src.data_processor import FinancialDataProcessor
from src.graph_builder import FinancialGraphBuilder
from src.layers import FinancialInterAgg, FinancialIntraAgg, FinancialGNNLayer, FinancialGNNModel
from src.utils import pos_neg_split, pick_step_financial

class FinancialModelHandler:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and not config['no_cuda'] else 'cpu')
        
        # 添加用于可视化的数据收集
        self.train_losses = []
        self.val_aucs = []
        self.embeddings_history = []
        
        # 加载和处理数据
        self._load_data()
        self._prepare_model()
    
    def train(self):
        """训练模型"""
        print("开始训练...")
        
        best_auc = 0
        best_epoch = -1
        
        # 创建保存目录
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.config['save_dir'], f"financial_gnn_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(self.config['num_epochs']):
            # 训练一个epoch
            epoch_loss = self._train_epoch(epoch)
            self.train_losses.append(epoch_loss)  # 收集训练损失
            
            # 验证
            if epoch % self.config['valid_epochs'] == 0:
                val_metrics = self._evaluate(self.dataset['idx_valid'], self.dataset['y_valid'])
                self.val_aucs.append(val_metrics['auc'])  # 收集验证AUC
                
                print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | "
                      f"Val AUC: {val_metrics['auc']:.4f} | "
                      f"Val F1: {val_metrics['f1_macro']:.4f}")
                
                # 保存最佳模型
                if val_metrics['auc'] > best_auc:
                    best_auc = val_metrics['auc']
                    best_epoch = epoch
                    model_path = os.path.join(save_dir, 'best_model.pth')
                    torch.save(self.model.state_dict(), model_path)
                    print(f"  保存最佳模型 (AUC: {best_auc:.4f})")
        
        # 加载最佳模型进行测试
        print(f"\n加载最佳模型 (Epoch {best_epoch})")
        model_path = os.path.join(save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path))
        
        # 测试
        test_metrics = self._evaluate(self.dataset['idx_test'], self.dataset['y_test'])
        
        return (test_metrics['f1_macro'], test_metrics['precision'], 
                test_metrics['recall'], test_metrics['auc'], test_metrics['accuracy'])
    
    def get_embeddings(self, node_indices):
        """获取节点嵌入向量"""
        self.model.eval()
        with torch.no_grad():
            if self.device.type == 'cuda':
                node_indices_tensor = torch.cuda.LongTensor(node_indices)
            else:
                node_indices_tensor = torch.LongTensor(node_indices)
            
            embeddings, _ = self.model.gnn_layer(node_indices_tensor, 
                                                 torch.zeros(len(node_indices)), 
                                                 train_flag=False)
            return embeddings.cpu().numpy().T
    
    def _load_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        
        # 初始化数据处理器
        data_processor = FinancialDataProcessor(self.config)
        
        # 根据数据集类型加载数据
        if self.config['data_name'] == 'ieee':
            data = data_processor.load_ieee_data(self.config['data_dir'])
        elif self.config['data_name'] == 'credit':
            data = data_processor.load_credit_data(self.config['data_dir'])
        else:
            raise ValueError(f"不支持的数据集: {self.config['data_name']}")
        
        # 准备特征和标签
        features, labels = data_processor.prepare_features(data)
        
        # 构建图
        graph_builder = FinancialGraphBuilder(self.config)
        edges_dict, node_mapping = graph_builder.build_heterogeneous_graph(data)
        adj_lists = graph_builder.create_adjacency_lists(edges_dict, len(features))
        
        # 数据分割
        indices = list(range(len(labels)))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            indices, labels, stratify=labels, 
            train_size=self.config['train_ratio'], 
            random_state=self.config['seed']
        )
        
        idx_valid, idx_test, y_valid, y_test = train_test_split(
            idx_temp, y_temp, stratify=y_temp,
            test_size=self.config['test_ratio'],
            random_state=self.config['seed']
        )
        
        # 正负样本分割
        train_pos, train_neg = pos_neg_split(idx_train, y_train)
        
        print(f"数据集: {self.config['data_name']}")
        print(f"总样本数: {len(labels)}, 正样本数: {np.sum(labels)}")
        print(f"训练集: {len(y_train)}, 验证集: {len(y_valid)}, 测试集: {len(y_test)}")
        print(f"特征维度: {features.shape[1]}")
        
        self.dataset = {
            'features': features,
            'labels': labels,
            'adj_lists': adj_lists,
            'idx_train': idx_train,
            'idx_valid': idx_valid,
            'idx_test': idx_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test,
            'train_pos': train_pos,
            'train_neg': train_neg
        }
    
    def _prepare_model(self):
        """准备模型"""
        features = self.dataset['features']
        adj_lists = self.dataset['adj_lists']
        train_pos = self.dataset['train_pos']
        
        # 创建特征嵌入
        feature_embedding = nn.Embedding(features.shape[0], features.shape[1])
        feature_embedding.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
        
        if self.device.type == 'cuda':
            feature_embedding = feature_embedding.cuda()
        
        # 创建关系内聚合器
        intra_agg1 = FinancialIntraAgg(feature_embedding, features.shape[1], 
                                      self.config['emb_size'], train_pos, 
                                      self.config['rho'], cuda=(self.device.type == 'cuda'))
        intra_agg2 = FinancialIntraAgg(feature_embedding, features.shape[1], 
                                      self.config['emb_size'], train_pos, 
                                      self.config['rho'], cuda=(self.device.type == 'cuda'))
        intra_agg3 = FinancialIntraAgg(feature_embedding, features.shape[1], 
                                      self.config['emb_size'], train_pos, 
                                      self.config['rho'], cuda=(self.device.type == 'cuda'))
        
        # 创建GNN层
        gnn_layer = FinancialGNNLayer(
            feature_embedding, features.shape[1], self.config['emb_size'],
            train_pos, adj_lists, [intra_agg1, intra_agg2, intra_agg3],
            thresholds=self.config['thresholds'],
            inter=self.config['multi_relation'],
            cuda=(self.device.type == 'cuda')
        )
        
        # 创建完整模型
        self.model = FinancialGNNModel(2, gnn_layer, self.config['alpha'])
        
        if self.device.type == 'cuda':
            self.model = self.model.cuda()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
    
    def _train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # 采样训练数据
        sampled_idx_train = pick_step_financial(
            self.dataset['idx_train'], self.dataset['y_train'],
            size=len(self.dataset['train_pos']) * 2
        )
        
        random.shuffle(sampled_idx_train)
        num_batches = (len(sampled_idx_train) + self.config['batch_size'] - 1) // self.config['batch_size']
        
        total_loss = 0
        
        for batch in range(num_batches):
            start_idx = batch * self.config['batch_size']
            end_idx = min((batch + 1) * self.config['batch_size'], len(sampled_idx_train))
            batch_nodes = sampled_idx_train[start_idx:end_idx]
            batch_labels = self.dataset['labels'][np.array(batch_nodes)]
            
            self.optimizer.zero_grad()
            
            if self.device.type == 'cuda':
                batch_labels_tensor = Variable(torch.cuda.LongTensor(batch_labels))
            else:
                batch_labels_tensor = Variable(torch.LongTensor(batch_labels))
            
            loss = self.model.loss(batch_nodes, batch_labels_tensor, train_flag=True)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _evaluate(self, idx_eval, y_eval):
        """评估模型"""
        self.model.eval()
        
        all_probs = []
        all_preds = []
        
        with torch.no_grad():
            num_batches = (len(idx_eval) + self.config['batch_size'] - 1) // self.config['batch_size']
            
            for batch in range(num_batches):
                start_idx = batch * self.config['batch_size']
                end_idx = min((batch + 1) * self.config['batch_size'], len(idx_eval))
                batch_nodes = idx_eval[start_idx:end_idx]
                batch_labels = y_eval[start_idx:end_idx]
                
                if self.device.type == 'cuda':
                    batch_labels_tensor = Variable(torch.cuda.LongTensor(batch_labels))
                else:
                    batch_labels_tensor = Variable(torch.LongTensor(batch_labels))
                
                gnn_probs, _ = self.model.to_prob(batch_nodes, batch_labels_tensor, train_flag=False)
                
                probs = gnn_probs[:, 1].cpu().numpy()  # 正类概率
                preds = (probs > self.config['threshold']).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
        
        # 计算指标
        f1_macro = f1_score(y_eval, all_preds, average='macro')
        precision = precision_score(y_eval, all_preds)
        recall = recall_score(y_eval, all_preds)
        auc = roc_auc_score(y_eval, all_probs)
        accuracy = accuracy_score(y_eval, all_preds)
        
        return {
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'accuracy': accuracy
        }