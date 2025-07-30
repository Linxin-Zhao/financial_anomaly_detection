# 基于 GNN 的金融业务异常检测

本项目实现了一个基于图神经网络的金融异常检测系统，采用Pick-Choose-Aggregate机制处理金融交易网络中的异常行为检测。

## 项目特点

- **多关系图建模**: 基于交易金额、时间邻近性和用户相似性构建异构图
- **标签感知聚合**: 结合标签信息进行邻居选择和特征聚合
- **不平衡学习**: 专门针对金融异常检测中的类别不平衡问题
- **可扩展架构**: 支持多种数据集和配置

## 数据集

下载链接：https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## 安装依赖

```bash
pip install -r requirements.txt
