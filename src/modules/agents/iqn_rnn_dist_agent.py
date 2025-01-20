import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import math


def get_activation_func(name, hidden_dim):
    """
    'relu'
    'tanh'
    'leaky_relu'
    'elu'
    'prelu'
    :param name:
    :return:
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif name == "elu":
        return nn.ELU(alpha=1., inplace=True)
    elif name == 'prelu':
        return nn.PReLU(num_parameters=hidden_dim, init=0.25)


class Hypernet(nn.Module):
    def __init__(self, input_dim, hidden_dim, main_input_dim, main_output_dim, activation_func, n_heads):
        super(Hypernet, self).__init__()

        self.n_heads = n_heads
        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim

        self.multihead_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation_func(activation_func, hidden_dim),
            nn.Linear(hidden_dim, output_dim * self.n_heads),
        )

    def forward(self, x):
        # [...,  main_output_dim + main_output_dim + ... + main_output_dim]
        # [bs, main_input_dim, n_heads * main_output_dim]
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.main_output_dim * self.n_heads])


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)



class IQNRNNDistAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(IQNRNNDistAgent, self).__init__()
        self.args = args

        self.quantile_embed_dim = args.quantile_embed_dim
        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_approx_quantiles = args.n_approx_quantiles
        self.n_heads = args.hpn_head_num
        
        self.prev_return_distribution = None

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.hyper_input_w = Hypernet(
            input_dim=args.n_quantiles, hidden_dim=args.hpn_hyper_dim,
            main_input_dim=args.rnn_hidden_dim, main_output_dim=args.rnn_hidden_dim,
            activation_func=args.hpn_hyper_activation, n_heads=self.n_heads
        )
        self.unify_input_heads = Merger(self.n_heads, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.phi = nn.Linear(args.quantile_embed_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        self.prev_return_distribution = None

    def forward(self, inputs, hidden_state=None, forward_type=None):
        b, a, e = inputs.size()
        # c[:, :, torch.randperm(self.args.rnn_hidden_dim)[:self.n_quantiles]]
        
        if self.prev_return_distribution == None or self.prev_return_distribution.shape[0] != b*a:
            self.prev_return_distribution = th.ones(b * a, self.n_quantiles)
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        
        self.prev_return_distribution = self.prev_return_distribution.to(device=inputs.device)
        # [bs * n_agents, n_quantiles] -> [bs * n_agents, rnn_hidden_dim, rnn_hidden_dim * n_heads]
        input_w = self.hyper_input_w(self.prev_return_distribution)
        # [bs * n_agents, 1, rnn_hidden_dim] * [bs * n_agents, rnn_hidden_dim, rnn_hidden_dim * n_heads] = [bs * n_agents, 1, rnn_hidden_dim * n_heads]
        embedding = th.matmul(x.unsqueeze(1), input_w).view(
            b * a, self.n_heads, self.args.rnn_hidden_dim
        )  # [bs * n_agents, n_heads, rnn_hidden_dim]
        # [bs * n_agents, rnn_hidden_dim]
        embedding = self.unify_input_heads(embedding)
        x = F.relu(embedding)
        
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        if forward_type == "approx":
            n_rnd_quantiles = self.n_approx_quantiles
        elif forward_type == "policy":
            n_rnd_quantiles = self.n_quantiles
        elif forward_type == "target":
            n_rnd_quantiles = self.n_target_quantiles
        else:
            raise ValueError("Unknown forward_type")
        
        shape = h.shape
        batch_size = shape[0] # batch_size = bs * n_agents
        h2 = h.reshape(batch_size, 1, self.args.rnn_hidden_dim).expand(-1, n_rnd_quantiles, -1).reshape(-1, self.args.rnn_hidden_dim)
        
        shape = h2.shape
        # Generate random quantiles
        if self.args.name == "diql":
            rnd_quantiles = th.rand(batch_size * n_rnd_quantiles).to(device=inputs.device)
            batch_size_grouped = batch_size
        else:
            # Same quantiles for optimizing quantile mixture
            batch_size_grouped = batch_size // self.args.n_agents
            rnd_quantiles = th.rand(batch_size_grouped, 1, n_rnd_quantiles).to(device=inputs.device)
            rnd_quantiles = rnd_quantiles.reshape(-1)
        assert rnd_quantiles.shape == (batch_size_grouped * n_rnd_quantiles,)
        
        # Expand quantiles to cosine features
        quantiles = rnd_quantiles.view(batch_size_grouped * n_rnd_quantiles, 1).expand(-1, self.quantile_embed_dim)
        feature_id = th.arange(0, self.quantile_embed_dim).to(device=inputs.device)
        feature_id = feature_id.view(1, -1).expand(batch_size_grouped * n_rnd_quantiles, -1)
        assert feature_id.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
        cos = th.cos(math.pi * feature_id * quantiles).to(device=inputs.device)
        assert cos.shape == (batch_size_grouped * n_rnd_quantiles, self.quantile_embed_dim)
        
        # Quantile embedding network (phi)
        q_phi = F.relu(self.phi(cos))
        assert q_phi.shape == (batch_size_grouped * n_rnd_quantiles, self.args.rnn_hidden_dim)
        if self.args.name != "diql":
            q_phi = q_phi.view(batch_size_grouped, n_rnd_quantiles, self.args.rnn_hidden_dim)
            q_phi = q_phi.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1).contiguous().view(-1, self.args.rnn_hidden_dim)
        assert q_phi.shape == (batch_size * n_rnd_quantiles, self.args.rnn_hidden_dim)
        
        q_vals = self.fc2(h2 * q_phi)
        q_vals = q_vals.view(-1, n_rnd_quantiles, self.args.n_actions)
        assert q_vals.shape == (batch_size, n_rnd_quantiles, self.args.n_actions)
        q_vals = q_vals.permute(0, 2, 1)
        assert q_vals.shape == (batch_size, self.args.n_actions, n_rnd_quantiles)
        
        rnd_quantiles = rnd_quantiles.view(batch_size_grouped, n_rnd_quantiles)
        
        # Setting the previous return distribution
        max_indices = q_vals.mean(dim=-1).max(dim=-1)[1]
        max_indices = max_indices.unsqueeze(1).unsqueeze(2).expand(-1, -1, n_rnd_quantiles)
        self.prev_return_distribution = th.gather(q_vals, dim=1, index=max_indices).squeeze(1)
        del max_indices
        
        sorted_qvals = th.sort(q_vals, dim=2)[0]
        
        return sorted_qvals, h, rnd_quantiles
        # return q_vals, h, rnd_quantiles
