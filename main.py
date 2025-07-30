import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import random
import torch
import yaml  
from src.model_handler import FinancialModelHandler

def print_config(config):
    print("=" * 50)
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main(config):
    print_config(config)
    set_random_seed(config['seed'])
    
    model_handler = FinancialModelHandler(config)
    f1_macro, precision, recall, auc, accuracy = model_handler.train()
    
    print("\n=== 最终测试结果 ===")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

def get_config(config_path="config/financial_credit.yml"):  # 修改默认配置
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Financial Anomaly Detection with GNN')
    parser.add_argument('--config', type=str, default='config/financial_credit.yml',  # 修改默认配置
                       help='配置文件路径')
    args = parser.parse_args()
    
    config = get_config(args.config)
    main(config)