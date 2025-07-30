import os
import yaml
import numpy as np
import torch
from src.model_handler import FinancialModelHandler
from src.visualization import FinancialGNNVisualizer
from src.data_processor import FinancialDataProcessor

def main():
    # 加载配置
    with open('config/financial_credit.yml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建模型处理器
    model_handler = FinancialModelHandler(config)
    
    # 训练模型（如果还没有训练过）
    print("开始训练模型...")
    f1_macro, precision, recall, auc, accuracy = model_handler.train()
    
    # 创建可视化器
    visualizer = FinancialGNNVisualizer(model_handler.model, model_handler)
    
    # 创建可视化结果保存目录
    vis_dir = 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    print("\n开始生成可视化结果...")
    
    # 1. 绘制训练曲线
    if model_handler.train_losses and model_handler.val_aucs:
        print("绘制训练曲线...")
        visualizer.plot_training_curves(
            model_handler.train_losses, 
            model_handler.val_aucs,
            save_path=os.path.join(vis_dir, 'training_curves.png')
        )
    
    # 2. 绘制特征分布
    print("绘制特征分布...")
    test_features = model_handler.dataset['features'][model_handler.dataset['idx_test']]
    test_labels = model_handler.dataset['y_test']
    visualizer.plot_feature_distribution(
        test_features, test_labels,
        save_path=os.path.join(vis_dir, 'feature_distribution.png')
    )
    
    # 3. 绘制嵌入向量可视化
    print("绘制嵌入向量可视化...")
    # 选择部分测试数据进行可视化（避免数据量过大）
    sample_size = min(1000, len(model_handler.dataset['idx_test']))
    sample_indices = np.random.choice(model_handler.dataset['idx_test'], sample_size, replace=False)
    sample_labels = model_handler.dataset['labels'][sample_indices]
    
    # 获取嵌入向量
    embeddings = model_handler.get_embeddings(sample_indices)
    
    # t-SNE可视化
    visualizer.plot_embedding_visualization(
        embeddings, sample_labels, method='tsne',
        save_path=os.path.join(vis_dir, 'embeddings_tsne.png')
    )
    
    # PCA可视化
    visualizer.plot_embedding_visualization(
        embeddings, sample_labels, method='pca',
        save_path=os.path.join(vis_dir, 'embeddings_pca.png')
    )
    
    # 4. 绘制网络结构图
    print("绘制网络结构图...")
    visualizer.plot_network_structure(
        save_path=os.path.join(vis_dir, 'network_structure.png')
    )
    
    # 5. 绘制混淆矩阵和ROC曲线
    print("绘制混淆矩阵和ROC曲线...")
    
    # 获取测试集预测结果
    model_handler.model.eval()
    with torch.no_grad():
        if model_handler.device.type == 'cuda':
            test_nodes = torch.cuda.LongTensor(model_handler.dataset['idx_test'])
            test_labels_tensor = torch.cuda.LongTensor(model_handler.dataset['y_test'])
        else:
            test_nodes = torch.LongTensor(model_handler.dataset['idx_test'])
            test_labels_tensor = torch.LongTensor(model_handler.dataset['y_test'])
        
        test_scores = model_handler.model(test_nodes, test_labels_tensor)
        test_probs = torch.softmax(test_scores, dim=1)[:, 1].cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
    
    # 混淆矩阵
    visualizer.plot_confusion_matrix(
        model_handler.dataset['y_test'], test_preds,
        save_path=os.path.join(vis_dir, 'confusion_matrix.png')
    )
    
    # ROC曲线
    visualizer.plot_roc_curve(
        model_handler.dataset['y_test'], test_probs,
        save_path=os.path.join(vis_dir, 'roc_curve.png')
    )
    
    print(f"\n所有可视化结果已保存到 {vis_dir} 目录")
    print("\n=== 最终测试结果 ===")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()