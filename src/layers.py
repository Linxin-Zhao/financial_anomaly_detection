import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from operator import itemgetter
import math

"""
    Financial GNN Layers
    Adapted from PC-GNN for Financial Anomaly Detection
    Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
"""


class FinancialInterAgg(nn.Module):
    """Inter-relation aggregator for financial anomaly detection"""
    
    def __init__(self, features, feature_dim, embed_dim, 
                 train_pos, adj_lists, intraggs, inter='GNN', cuda=True):
        """
        Initialize the inter-relation aggregator for financial data
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the embed dimension
        :param train_pos: positive samples (fraudulent transactions) in training set
        :param adj_lists: a list of adjacency lists for each relation graph
        :param intraggs: the intra-relation aggregators for each relation
        :param inter: aggregator type (default: 'GNN')
        :param cuda: whether to use GPU
        """
        super(FinancialInterAgg, self).__init__()
        
        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]  # Amount similarity relation
        self.intra_agg2 = intraggs[1]  # Time proximity relation
        self.intra_agg3 = intraggs[2]  # User similarity relation
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.cuda = cuda
        self.train_pos = train_pos
        
        # Set cuda for intra-aggregators
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda
        
        # Initial filtering thresholds for each relation
        self.thresholds = [0.5, 0.5, 0.5]  # [amount, time, user]
        
        # Parameter for transforming embeddings before inter-relation aggregation
        self.weight = nn.Parameter(torch.FloatTensor(
            self.embed_dim * len(intraggs) + self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)
        
        # Label predictor for financial anomaly scoring
        self.label_clf = nn.Linear(self.feat_dim, 2)
        
        # Logs for analysis
        self.weights_log = []
        self.thresholds_log = [self.thresholds]
        self.relation_score_log = []
    
    def forward(self, nodes, labels, train_flag=True):
        """Forward pass through inter-aggregator"""
        # Get all unique nodes (including center nodes and neighbors)
        to_neighs = []
        for relation in range(len(self.adj_lists)):
            to_neighs.append([self.adj_lists[relation][int(node)] for node in nodes])
        
        # Collect all unique nodes including center nodes
        unique_nodes_set = set(nodes.cpu().numpy() if hasattr(nodes, 'cpu') else nodes)
        for relation_neighs in to_neighs:
            for neighs in relation_neighs:
                unique_nodes_set.update(neighs)
        
        unique_nodes = list(unique_nodes_set)
        
        # Get features and scores for all unique nodes
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
            pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))
            pos_features = self.features(torch.LongTensor(list(self.train_pos)))
        
        batch_scores = self.label_clf(batch_features)
        pos_scores = self.label_clf(pos_features)
        id_mapping = {node_id: index for node_id, index in 
                     zip(unique_nodes, range(len(unique_nodes)))}
        
        # Anomaly scores for current batch - 确保所有节点都在映射中
        center_indices = [id_mapping[int(node)] for node in nodes]
        center_scores = batch_scores[center_indices, :]
        
        # Get neighbor lists for each relation
        amount_list = [list(to_neigh) for to_neigh in to_neighs[0]]  # Amount similarity
        time_list = [list(to_neigh) for to_neigh in to_neighs[1]]    # Time proximity
        user_list = [list(to_neigh) for to_neigh in to_neighs[2]]    # User similarity
        
        # Assign scores to neighbors for each relation
        amount_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) 
                        for to_neigh in amount_list]
        time_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) 
                      for to_neigh in time_list]
        user_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) 
                      for to_neigh in user_list]
        
        # Calculate sampling numbers based on thresholds
        amount_sample_nums = [math.ceil(len(neighs) * self.thresholds[0]) 
                             for neighs in amount_list]
        time_sample_nums = [math.ceil(len(neighs) * self.thresholds[1]) 
                           for neighs in time_list]
        user_sample_nums = [math.ceil(len(neighs) * self.thresholds[2]) 
                           for neighs in user_list]
        
        # Intra-aggregation for each financial relation
        amount_feats, amount_scores_out = self.intra_agg1.forward(
            nodes, labels, amount_list, center_scores, amount_scores, 
            pos_scores, amount_sample_nums, train_flag)
        
        time_feats, time_scores_out = self.intra_agg2.forward(
            nodes, labels, time_list, center_scores, time_scores, 
            pos_scores, time_sample_nums, train_flag)
        
        user_feats, user_scores_out = self.intra_agg3.forward(
            nodes, labels, user_list, center_scores, user_scores, 
            pos_scores, user_sample_nums, train_flag)
        
        # Get self features
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features(index)
        
        # Concatenate features from all relations
        cat_feats = torch.cat((self_feats, amount_feats, time_feats, user_feats), dim=1)
        
        # Final transformation
        combined = F.relu(cat_feats.mm(self.weight).t())
        
        return combined, center_scores


class FinancialIntraAgg(nn.Module):
    """Intra-relation aggregator for financial relations"""
    
    def __init__(self, features, feat_dim, embed_dim, train_pos, rho, cuda=False):
        """
        Initialize the intra-relation aggregator for financial data
        :param features: input node features
        :param feat_dim: input feature dimension
        :param embed_dim: embedding dimension
        :param train_pos: positive (fraudulent) samples in training
        :param rho: oversampling ratio for minority class
        :param cuda: whether to use GPU
        """
        super(FinancialIntraAgg, self).__init__()
        
        self.features = features
        self.cuda = cuda
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.train_pos = train_pos
        self.rho = rho
        
        # Transformation weight
        self.weight = nn.Parameter(torch.FloatTensor(2 * self.feat_dim, self.embed_dim))
        init.xavier_uniform_(self.weight)
    
    def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, 
                neigh_scores, pos_scores, sample_list, train_flag):
        """
        Forward pass for intra-relation aggregation
        :param nodes: batch node ids
        :param batch_labels: batch node labels
        :param to_neighs_list: neighbor lists for each batch node
        :param batch_scores: anomaly scores for batch nodes
        :param neigh_scores: anomaly scores for neighbors
        :param pos_scores: scores for positive samples
        :param sample_list: number of neighbors to sample
        :param train_flag: training or testing mode
        :return: aggregated features and scores
        """
        
        # Filter neighbors based on training/testing mode
        if train_flag:
            samp_neighs, samp_scores = financial_choose_step_neighs(
                batch_scores, batch_labels, neigh_scores, to_neighs_list, 
                pos_scores, self.train_pos, sample_list, self.rho)
        else:
            samp_neighs, samp_scores = financial_choose_step_test(
                batch_scores, neigh_scores, to_neighs_list, sample_list)
        
        # Find unique nodes for efficient computation
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        
        # Create aggregation mask
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        
        if self.cuda:
            mask = mask.cuda()
        
        # Mean aggregation
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        
        # Get features
        if self.cuda:
            self_feats = self.features(torch.LongTensor(nodes).cuda())
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            self_feats = self.features(torch.LongTensor(nodes))
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        
        # Aggregate and transform
        agg_feats = mask.mm(embed_matrix)
        cat_feats = torch.cat((self_feats, agg_feats), dim=1)
        to_feats = F.relu(cat_feats.mm(self.weight))
        
        return to_feats, samp_scores


def financial_choose_step_neighs(center_scores, center_labels, neigh_scores, 
                                neighs_list, minor_scores, minor_list, 
                                sample_list, sample_rate):
    """
    Choose neighbors for financial anomaly detection training
    :param center_scores: anomaly scores of batch nodes
    :param center_labels: labels of batch nodes
    :param neigh_scores: scores of neighbors
    :param neighs_list: neighbor node lists
    :param minor_scores: scores of minority class nodes
    :param minor_list: minority class node list
    :param sample_list: number of neighbors to sample
    :param sample_rate: oversampling rate for minority class
    :return: sampled neighbors and score differences
    """
    samp_neighs = []
    samp_score_diff = []
    
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]
        
        # Compute L1-distance for financial similarity
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(
            score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()
        
        # Top-p sampling based on financial similarity
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]
        
        # Oversample for fraudulent transactions (minority class)
        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(
                center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(
                score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])
        
        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)
    
    return samp_neighs, samp_score_diff


def financial_choose_step_test(center_scores, neigh_scores, neighs_list, sample_list):
    """
    Choose neighbors for financial anomaly detection testing
    :param center_scores: anomaly scores of batch nodes
    :param neigh_scores: scores of neighbors
    :param neighs_list: neighbor node lists
    :param sample_list: number of neighbors to sample
    :return: sampled neighbors and scores
    """
    samp_neighs = []
    samp_scores = []
    
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]
        
        # Compute financial similarity scores
        score_diff = torch.abs(center_score - neigh_score).squeeze()
        sorted_scores, sorted_indices = torch.sort(score_diff, dim=0, descending=False)
        selected_indices = sorted_indices.tolist()
        
        # Sample based on similarity ranking
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_indices[:num_sample]]
            selected_scores = sorted_scores.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_scores = score_diff.tolist()
            if isinstance(selected_scores, float):
                selected_scores = [selected_scores]
        
        samp_neighs.append(set(selected_neighs))
        samp_scores.append(selected_scores)
    
    return samp_neighs, samp_scores


# 在文件末尾添加以下类定义

class FinancialGNNLayer(nn.Module):
    """Financial GNN Layer combining inter and intra aggregators"""
    
    def __init__(self, features, feature_dim, embed_dim, train_pos, adj_lists, 
                 intraggs, thresholds, inter='GNN', cuda=True):
        super(FinancialGNNLayer, self).__init__()
        
        self.features = features
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.train_pos = train_pos
        self.adj_lists = adj_lists
        self.thresholds = thresholds
        self.inter = inter
        self.cuda = cuda
        
        # Create inter-aggregator
        self.inter_agg = FinancialInterAgg(
            features, feature_dim, embed_dim, train_pos, 
            adj_lists, intraggs, inter, cuda
        )
    
    def forward(self, nodes, labels, train_flag=True):
        """Forward pass through the GNN layer"""
        return self.inter_agg.forward(nodes, labels, train_flag)


class FinancialGNNModel(nn.Module):
    """Complete Financial GNN Model"""
    
    def __init__(self, num_classes, gnn_layer, alpha=0.2):
        super(FinancialGNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.gnn_layer = gnn_layer
        self.alpha = alpha
        
        # Classification layer
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, gnn_layer.embed_dim))
        init.xavier_uniform_(self.weight)
        
        # Loss function
        self.xent = nn.CrossEntropyLoss()
    
    def forward(self, nodes, labels, train_flag=True):
        """Forward pass"""
        embeds, center_scores = self.gnn_layer(nodes, labels, train_flag)
        scores = self.weight.mm(embeds).t() 
        return scores
    
    def to_prob(self, nodes, labels, train_flag=True):
        """Convert scores to probabilities"""
        scores = self.forward(nodes, labels, train_flag)
        probs = torch.softmax(scores, dim=1)
        return probs, scores
    
    def loss(self, nodes, labels, train_flag=True):
        """Compute loss"""
        gnn_scores = self.forward(nodes, labels, train_flag)
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        return gnn_loss