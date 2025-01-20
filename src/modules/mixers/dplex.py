import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dmaq_si_weight import DMAQ_SI_Weight


class DPLEXMixer(nn.Module):
    def __init__(self, args):
        super(DPLEXMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * args.n_actions
        # self.state_action_dim = self.state_dim + self.action_dim + 1

        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_actions = args.n_actions

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.hypernet_embed

            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))

            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim, 1))
        
        self.si_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        return v_tot
    
    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.args.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=1)
        return adv_tot
    
    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, target, actions=None, max_q_i=None, is_v=False):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        if agent_qs.shape[2] == 1:
            agent_qs = agent_qs.expand(-1, -1, self.n_agents, -1)
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, n_rnd_quantiles)
        q_mixture = agent_qs.sum(dim=2, keepdim=True)
        assert q_mixture.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_vals_expected = agent_qs.mean(dim=3, keepdim=True)
        q_vals_sum = q_vals_expected.sum(dim=2, keepdim=True)
        assert q_vals_expected.shape == (batch_size, episode_length, self.n_agents, 1)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, 1)

        # Factorization network
        q_joint_expected, weight_mag_regs = self.forward_qmix(q_vals_expected, states, target, actions, max_q_i, is_v)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, 1)

        # Shape network
        q_vals_sum = q_vals_sum.expand(-1, -1, -1, n_rnd_quantiles)
        q_joint_expected = q_joint_expected.expand(-1, -1, -1, n_rnd_quantiles)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        if is_v:
            q_joint = q_mixture - q_vals_sum + q_joint_expected # q_joint_expected
        else:
            q_joint = q_mixture - q_vals_sum + q_joint_expected 
        assert q_joint.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        return q_joint, weight_mag_regs

    def forward_qmix(self, agent_qs, states, target, actions=None, max_q_i=None, is_v=False):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, 1)
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        assert agent_qs.shape == (batch_size * episode_length, 1, self.n_agents)
        assert states.shape == (batch_size, episode_length, self.state_dim)
        states = states.reshape(-1, self.state_dim)
        assert states.shape == (batch_size * episode_length, self.state_dim)
        
        # agent_qs = agent_qs.view(-1, self.n_agents)
        
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        # hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, self.n_agents) + 1e-10
        
        # State-dependent bias
        v = self.V(states)
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents
        
        if self.args.weighted_head:
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)
            agents_qs = th.bmm(hidden, w_final).reshape(-1, self.n_agents) + v
        if not is_v:
            if target:
                max_q_i = max_q_i.mean(dim=-1)
                actions = actions.mean(dim=-1)
            max_q_i = max_q_i.view(-1, 1, self.n_agents)
            if self.args.weighted_head:
                hidden = F.elu(th.bmm(max_q_i, w1) + b1)
                max_q_i = th.bmm(hidden, w_final).reshape(-1, self.n_agents) + v
        
        # Calculate y
        y = self.calc(agents_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)

        # Regularize magnitude of weights
        weight_mag_regs = 0.001 * ((w1**2).mean() + (b1**2).mean() + (w_final**2).mean())
        
        # Reshape and return
        v_tot = y.view(bs, -1, 1)
        v_tot = v_tot.unsqueeze(3)
        return v_tot.to(device=agent_qs.device), weight_mag_regs.to(device=agent_qs.device)
