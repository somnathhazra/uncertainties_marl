import os

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NIQLDistMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NIQLDistMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self.args.agent == "iqn_rnn" or self.args.agent == "iqn_rnn_dist":
            agent_outputs, rnd_quantiles = self.forward(ep_batch, t_ep, forward_type="approx")
        else:
            agent_outputs = self.forward(ep_batch, t_ep, forward_type=test_mode)
        if self.args.agent in ["iqn_rnn", "iqn_rnn_dist", "iqn_rnn_dist2", "iqn_rnn_dist3", "iqn_rnn_dist4", "qrdqn_rnn_dist"]:
            agent_outputs = agent_outputs.view(ep_batch.batch_size, self.n_agents, self.args.n_actions, -1).mean(dim=3)
        
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, forward_type=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if self.args.agent == "iqn_rnn" or self.args.agent == "iqn_rnn_dist4":
            agent_outs, self.hidden_states, rnd_quantiles = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type)
        elif self.args.agent == "iqn_rnn_dist" or self.args.agent == "iqn_rnn_dist2" or self.args.agent == "iqn_rnn_dist3":
            agent_outs, self.hidden_states, rnd_quantiles, dist_loss = self.agent(agent_inputs, self.hidden_states, forward_type=forward_type)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not forward_type:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        
        if self.args.agent == "iqn_rnn" or self.args.agent == "iqn_rnn_dist":
            return agent_outs, rnd_quantiles
        elif self.args.agent == "qrdqn_rnn_dist":
            return agent_outs
        else:
            return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
