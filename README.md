# 基于 GNN 的金融业务异常检测

本项目实现了一个基于图神经网络的金融异常检测系统，采用Pick-Choose-Aggregate机制处理金融交易网络中的异常行为检测。

## 项目特点

- **多关系图建模**: 基于交易金额、时间邻近性和用户相似性构建异构图
- **标签感知聚合**: 结合标签信息进行邻居选择和特征聚合
- **不平衡学习**: 专门针对金融异常检测中的类别不平衡问题
- **可扩展架构**: 支持多种数据集和配置

## 项目结构

financial_anomaly_detection/
├── config/                 # 配置文件
│   └── financial_credit.yml
├── data/                   # 数据文件
│   └── creditcard.csv
├── src/                    # 源代码
│   ├── data_processor.py   # 数据处理
│   ├── graph_builder.py    # 图构建
│   ├── layers.py          # 神经网络层
│   ├── model_handler.py   # 模型处理
│   ├── utils.py           # 工具函数
│   └── visualization.py   # 可视化
├── models/                # 保存的模型
├── visualizations/        # 可视化结果
├── main.py               # 主程序
├── visualize_results.py  # 可视化脚本
├── requirements.txt      # 依赖包
└── README.md            # 项目说明


## 数据集

下载链接：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 安装依赖

```bash
pip install -r requirements.txt
