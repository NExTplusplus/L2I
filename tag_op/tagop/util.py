import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tools import allennlp as util

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm=True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)


class IntermediateLayer(nn.Module):
    def  __init__(self, input_dim, intermediate_dim):
        super(IntermediateLayer, self).__init__()
        self.dense = nn.Linear(input_dim, intermediate_dim)
        
    def forward(self, input):
        input = self.dense(input)
        input = gelu(input)
        return input

class OutPutLayer(nn.Module):
    def __init__(self, intermediate_dim, output_dim, dropout_rate):
        super(OutPutLayer, self).__init__()
        self.dense = nn.Linear(intermediate_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln = nn.LayerNorm(output_dim)
        
    def forward(self, input, residual_tensor):
        input = self.dense(input)
        input = self.dropout(input)
        input = self.ln(input + residual_tensor)
        return input

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, Q, K, V, K_mask):
        dk = Q.size()[-1]
        attention_scores = torch.bmm(Q, K.transpose(1, 2))/math.sqrt(dk)
        attention_weights = self.dropout(util.masked_softmax(attention_scores, K_mask))
        return torch.bmm(attention_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, hidden_size, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        assert hidden_size % num_head == 0
        self.WQ = nn.Linear(hidden_size, hidden_size)
        self.WK = nn.Linear(hidden_size, hidden_size)
        self.WV = nn.Linear(hidden_size, hidden_size)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.dropout_rate)
        self.output_layer = OutPutLayer(hidden_size, hidden_size, dropout_rate)
    
    def forward(self, Q, K, V, K_mask):
        Q_ = self.WQ(Q)
        K_ = self.WK(K)
        V_ = self.WV(V)
        Q_ = torch.cat(torch.split(Q_, self.hidden_size // self.num_head, dim=2), dim=0)
        K_ = torch.cat(torch.split(K_, self.hidden_size // self.num_head, dim=2), dim=0)
        V_ = torch.cat(torch.split(V_, self.hidden_size // self.num_head, dim=2), dim=0)
        K_mask = K_mask.repeat(self.num_head, 1)
        output = self.scaled_dot_product_attention(Q_, K_, V_, K_mask)
        new_bs = output.size()[0]
        output = torch.cat(torch.split(output, new_bs// self.num_head, dim=0), dim=2)
        output = self.output_layer(output, Q)
        return output


class SimpleCrossAttention(nn.Module):
    def __init__(self, num_head, hidden_size, dropout_rate):
        super(SimpleCrossAttention, self).__init__()
        self.cross_multi_head_attention_1 = MultiHeadAttention(num_head, hidden_size, dropout_rate)
        self.cross_multi_head_attention_2 = MultiHeadAttention(num_head, hidden_size, dropout_rate)
        self.intermediate_1 = IntermediateLayer(hidden_size, hidden_size)
        self.output_1 = OutPutLayer(hidden_size, hidden_size, dropout_rate)
        self.intermediate_2 = IntermediateLayer(hidden_size, hidden_size)
        self.output_2 = OutPutLayer(hidden_size, hidden_size, dropout_rate)
        
    def forward(self, Q, K, K_mask, Q_mask):
        Q_ = self.cross_multi_head_attention_1(Q, K, K, K_mask)
        K_ = self.cross_multi_head_attention_2(K, Q, Q, Q_mask)
        Q_ = self.output_1(self.intermediate_1(Q_), Q_)
        K_ = self.output_2(self.intermediate_2(K_), K_)
        return Q_, K_

class CrossAttention(nn.Module):
    def __init__(self, num_head, hidden_size, dropout_rate):
        super(CrossAttention, self).__init__()
        self.cross_multi_head_attention_1 = MultiHeadAttention(num_head, hidden_size, dropout_rate)
        self.cross_multi_head_attention_2 = MultiHeadAttention(num_head, hidden_size, dropout_rate)
        self.self_multi_head_attention_1 = MultiHeadAttention(num_head, hidden_size, dropout_rate)
        self.self_multi_head_attention_2 = MultiHeadAttention(num_head, hidden_size, dropout_rate)
        self.intermediate_1 = IntermediateLayer(hidden_size, hidden_size)
        self.output_1 = OutPutLayer(hidden_size, hidden_size, dropout_rate)
        self.intermediate_2 = IntermediateLayer(hidden_size, hidden_size)
        self.output_2 = OutPutLayer(hidden_size, hidden_size, dropout_rate)
        
    def forward(self, Q, K, K_mask, Q_mask):
        Q_ = self.cross_multi_head_attention_1(Q, K, K, K_mask)
        K_ = self.cross_multi_head_attention_2(K, Q, Q, Q_mask)
        Q_ = self.self_multi_head_attention_1(Q_, Q_, Q_, Q_mask)
        K_ = self.self_multi_head_attention_2(K_, K_, K_, K_mask)
        Q_ = self.output_1(self.intermediate_1(Q_), Q_)
        K_ = self.output_2(self.intermediate_2(K_), K_)
        return Q_, K_
        

class GCN(nn.Module):

    def __init__(self, node_dim, extra_factor_dim=0, iteration_steps=1):
        super(GCN, self).__init__()

        self.node_dim = node_dim
        self.iteration_steps = iteration_steps

        self._node_weight_fc = torch.nn.Linear(node_dim + extra_factor_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)
        self._dd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_left = torch.nn.Linear(node_dim, node_dim, bias=False)

        self._dd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._dq_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)
        self._qd_node_fc_right = torch.nn.Linear(node_dim, node_dim, bias=False)

    def forward(self, d_node, q_node, d_node_mask, q_node_mask, graph, extra_factor=None):

        d_node_len = d_node.size(1)
        q_node_len = q_node.size(1)

        diagmat = torch.diagflat(torch.ones(d_node.size(1), dtype=torch.long, device=d_node.device))
        diagmat = diagmat.unsqueeze(0).expand(d_node.size(0), -1, -1)
        dd_graph = d_node_mask.unsqueeze(1) * d_node_mask.unsqueeze(-1) * (1 - diagmat)
        dd_graph_left = dd_graph * graph[:, :d_node_len, :d_node_len]
        dd_graph_right = dd_graph * (1 - graph[:, :d_node_len, :d_node_len])

        diagmat = torch.diagflat(torch.ones(q_node.size(1), dtype=torch.long, device=q_node.device))
        diagmat = diagmat.unsqueeze(0).expand(q_node.size(0), -1, -1)
        qq_graph = q_node_mask.unsqueeze(1) * q_node_mask.unsqueeze(-1) * (1 - diagmat)
        qq_graph_left = qq_graph * graph[:, d_node_len:, d_node_len:]
        qq_graph_right = qq_graph * (1 - graph[:, d_node_len:, d_node_len:])

        dq_graph = d_node_mask.unsqueeze(-1) * q_node_mask.unsqueeze(1)
        dq_graph_left = dq_graph * graph[:, :d_node_len, d_node_len:]
        dq_graph_right = dq_graph * (1 - graph[:, :d_node_len, d_node_len:])

        qd_graph = q_node_mask.unsqueeze(-1) * d_node_mask.unsqueeze(1)
        qd_graph_left = qd_graph * graph[:, d_node_len:, :d_node_len]
        qd_graph_right = qd_graph * (1 - graph[:, d_node_len:, :d_node_len])


        d_node_neighbor_num = dd_graph_left.sum(-1) + dd_graph_right.sum(-1) + dq_graph_left.sum(-1) + dq_graph_right.sum(-1)
        d_node_neighbor_num_mask = (d_node_neighbor_num >= 1).long()
        d_node_neighbor_num = util.replace_masked_values(d_node_neighbor_num.float(), d_node_neighbor_num_mask, 1)

        q_node_neighbor_num = qq_graph_left.sum(-1) + qq_graph_right.sum(-1) + qd_graph_left.sum(-1) + qd_graph_right.sum(-1)
        q_node_neighbor_num_mask = (q_node_neighbor_num >= 1).long()
        q_node_neighbor_num = util.replace_masked_values(q_node_neighbor_num.float(), q_node_neighbor_num_mask, 1)


        all_d_weight, all_q_weight = [], []
        for step in range(self.iteration_steps):
            if extra_factor is None:
                d_node_weight = torch.sigmoid(self._node_weight_fc(d_node)).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(q_node)).squeeze(-1)
            else:
                d_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((d_node, extra_factor), dim=-1))).squeeze(-1)
                q_node_weight = torch.sigmoid(self._node_weight_fc(torch.cat((q_node, extra_factor), dim=-1))).squeeze(-1)

            all_d_weight.append(d_node_weight)
            all_q_weight.append(q_node_weight)

            self_d_node_info = self._self_node_fc(d_node)
            self_q_node_info = self._self_node_fc(q_node)

            dd_node_info_left = self._dd_node_fc_left(d_node)
            qd_node_info_left = self._qd_node_fc_left(d_node)
            qq_node_info_left = self._qq_node_fc_left(q_node)
            dq_node_info_left = self._dq_node_fc_left(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_left,
                    0)

            qd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph_left,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph_left,
                    0)

            dq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph_left,
                    0)

            dd_node_info_left = torch.matmul(dd_node_weight, dd_node_info_left)
            qd_node_info_left = torch.matmul(qd_node_weight, qd_node_info_left)
            qq_node_info_left = torch.matmul(qq_node_weight, qq_node_info_left)
            dq_node_info_left = torch.matmul(dq_node_weight, dq_node_info_left)


            dd_node_info_right = self._dd_node_fc_right(d_node)
            qd_node_info_right = self._qd_node_fc_right(d_node)
            qq_node_info_right = self._qq_node_fc_right(q_node)
            dq_node_info_right = self._dq_node_fc_right(q_node)

            dd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dd_graph_right,
                    0)

            qd_node_weight = util.replace_masked_values(
                    d_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qd_graph_right,
                    0)

            qq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, q_node_len, -1),
                    qq_graph_right,
                    0)

            dq_node_weight = util.replace_masked_values(
                    q_node_weight.unsqueeze(1).expand(-1, d_node_len, -1),
                    dq_graph_right,
                    0)

            dd_node_info_right = torch.matmul(dd_node_weight, dd_node_info_right)
            qd_node_info_right = torch.matmul(qd_node_weight, qd_node_info_right)
            qq_node_info_right = torch.matmul(qq_node_weight, qq_node_info_right)
            dq_node_info_right = torch.matmul(dq_node_weight, dq_node_info_right)


            agg_d_node_info = (dd_node_info_left + dd_node_info_right + dq_node_info_left + dq_node_info_right) / d_node_neighbor_num.unsqueeze(-1)
            agg_q_node_info = (qq_node_info_left + qq_node_info_right + qd_node_info_left + qd_node_info_right) / q_node_neighbor_num.unsqueeze(-1)

            d_node = F.relu(self_d_node_info + agg_d_node_info)
            q_node = F.relu(self_q_node_info + agg_q_node_info)


        all_d_weight = [weight.unsqueeze(1) for weight in all_d_weight]
        all_q_weight = [weight.unsqueeze(1) for weight in all_q_weight]

        all_d_weight = torch.cat(all_d_weight, dim=1)
        all_q_weight = torch.cat(all_q_weight, dim=1)

        return d_node, q_node, all_d_weight, all_q_weight # d_node_weight, q_node_weight
