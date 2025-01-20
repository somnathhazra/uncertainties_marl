import copy
import time

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
from torch.autograd import Variable

from components.episode_buffer import EpisodeBatch
from modules.mixers.dmix import DMixer
from modules.mixers.dplex import DPLEXMixer

from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets, build_distributional_td_lambda_targets
from utils.th_utils import get_parameters_num
from utils.pcgrad import PCGrad

import numpy as np

def calculate_target_q(agent, target_mac, batch, enable_parallel_computing=False, thread_num=4, n_agents=1, n_actions=1, n_target_quantiles=8):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if agent == "qrdqn_rnn_dist":
                target_agent_outs = target_mac.forward(batch, t=t, forward_type="target")
            else:
                target_agent_outs, _ = target_mac.forward(batch, t=t, forward_type="target")
            assert target_agent_outs.shape == (batch.batch_size * n_agents, n_actions, n_target_quantiles)
            target_agent_outs = target_agent_outs.view(batch.batch_size, n_agents, n_actions, n_target_quantiles)
            assert target_agent_outs.shape == (batch.batch_size, n_agents, n_actions, n_target_quantiles)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        del target_agent_outs
        # assert target_mac_out.shape == (batch.batch_size, episode_length, n_agents, n_actions, n_target_quantiles)
        return target_mac_out


def calculate_n_step_td_target(target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda, 
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    episode_length = rewards.shape[1]
    
    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_max_qvals = target_mixer(target_max_qvals, batch["state"], target=True)

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards.unsqueeze(3), terminated.unsqueeze(3), mask.unsqueeze(3).expand(-1, -1, -1, 32), target_max_qvals, qvals, gamma, td_lambda)
        else:
            # targets = build_td_lambda_targets(rewards.unsqueeze(3), terminated.unsqueeze(3), mask.unsqueeze(3).expand(-1, -1, -1, 32), target_max_qvals, gamma, td_lambda)
            targets = build_distributional_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
            # targets = rewards.unsqueeze(3) + gamma * (1 - terminated.unsqueeze(3)) * target_max_qvals[:, 1:]
        return targets.detach()

def calculate_n_step_td_target_dplex(target_mixer, target_max_qvals, batch, actions, max_q_i, n_target_quantiles, rewards, terminated, mask,
                               gamma, td_lambda, enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    episode_length = rewards.shape[1]
    
    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_chosen, _ = target_mixer(target_max_qvals, batch["state"], target=True, is_v=True)
        target_adv, _ = target_mixer(target_max_qvals, batch["state"], target=True,
                                  actions=actions, max_q_i=max_q_i, is_v=False)
        
        target_max_qvals = target_chosen + target_adv
        assert target_max_qvals.shape == (batch.batch_size, episode_length+1, 1, n_target_quantiles)

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards.unsqueeze(3), terminated.unsqueeze(3), mask.unsqueeze(3).expand(-1, -1, -1, n_target_quantiles), target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards.unsqueeze(3), terminated.unsqueeze(3), mask.unsqueeze(3).expand(-1, -1, -1, n_target_quantiles), target_max_qvals, gamma, td_lambda)
            # targets = build_distributional_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
            # targets = rewards.unsqueeze(3) + gamma * (1 - terminated.unsqueeze(3)) * target_max_qvals[:, 1:]
        return targets.detach()


class IQNLearnerDist:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.mac_params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmix":
                self.mixer = DMixer(args)
            elif args.mixer == "dplex":
                self.mixer = DPLEXMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if args.optimizer == "RMSProp":
            self.optimiser = PCGrad(RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps))
        elif args.optimizer == "Adam":
            self.optimiser = PCGrad(Adam(params=self.params, lr=args.lr, eps=args.optim_eps))
        else:
            raise ValueError("Unknown Optimizer")

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)

    def train(self, batch: EpisodeBatch, on_batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        episode_length = rewards.shape[1]
        assert rewards.shape == (batch.batch_size, episode_length, 1)
        actions = batch["actions"][:, :-1]
        assert actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        assert mask.shape == (batch.batch_size, episode_length, 1)
        avail_actions = batch["avail_actions"]
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)
        actions_onehot = batch["actions_onehot"][:, :-1]
        
        # Filtering dead allies information from the rewards
        dead_agents = on_batch["dead_allies"][:, :-1]
        dead_agents = dead_agents.int()
        # dead_agents = dead_agents.sum(dim=1) # better to calculate deaths at each time-step
        # Calculating dead allies at each step
        dead_agents_per_step = th.zeros(dead_agents.shape).to(dead_agents.device)
        for i in range(1, dead_agents.shape[1]):
            dead_agents_per_step[:,i,:] = th.max(dead_agents[:,i,:] - dead_agents[:,i-1,:], dead_agents_per_step[:,i,:])
        # dead_agents_per_step.shape = [bs, time_steps, 1]
        
        # Calculating the CBF cost value matrix
        for i in range(dead_agents.shape[1]-2, -1, -1):
            dead_agents_per_step[:,i,:] = th.round(dead_agents_per_step[:,i,:] + dead_agents_per_step[:,i+1,:] * 0.4, decimals=4)
        # dead_agents_per_step now contains cbf values at each step
        # if dead_agents=self.args.n_agents, battle_lost
        # CBF loss enforced only when battle is lost
        dead_agents = dead_agents.sum(dim=1)
        # Removing this part for testing (did not work)
        for i in range(dead_agents.shape[0]):
            if dead_agents[i] < self.args.n_agents:
                dead_agents_per_step[i] = dead_agents_per_step[i] - dead_agents_per_step[i]

        # Calculate change in cbf
        cbf_change = th.zeros(dead_agents_per_step.shape).to(dead_agents.device)
        for i in range(1, dead_agents_per_step.shape[1]):
            cbf_change[:,i,:] = th.clamp(dead_agents_per_step[:,i,:] - dead_agents_per_step[:,i-1,:], min=0.)
        del dead_agents
        
        # Calculating final loss due to cbf
        # adding the alpha term to the cbf loss
        cbf_change = cbf_change + dead_agents_per_step + 10e-2
        cbf_loss = cbf_change # .mean(dim=1).mean(dim=0)
        cbf_loss = cbf_loss * mask # mask[:, :cbf_loss.shape[1]]
        cbf_loss = cbf_loss.sum() / mask.sum() # mask[:, :cbf_loss.shape[1]].sum()
        cbf_loss = Variable(cbf_loss, requires_grad=True)
        del dead_agents_per_step
        del cbf_change

        # done later aleady
        if self.enable_parallel_computing:
            target_mac_out = self.pool.apply_async(
                calculate_target_q,
                (self.args.agent, self.target_mac, batch, True, self.args.thread_num, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            )
            assert target_mac_out.shape == (batch.batch_size, episode_length, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)

        # Mix
        if self.mixer is not None:
            # Same quantile for quantile mixture
            n_quantile_groups = 1
        else:
            n_quantile_groups = self.args.n_agents

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        rnd_quantiles = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if self.args.agent != "qrdqn_rnn_dist":
                agent_outs, agent_rnd_quantiles = self.mac.forward(batch, t=t, forward_type="policy")
                assert agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_quantiles)
                assert agent_rnd_quantiles.shape == (batch.batch_size * n_quantile_groups, self.n_quantiles)
                agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
                rnd_quantiles.append(agent_rnd_quantiles)
            else:
                agent_outs = self.mac.forward(batch, t=t, forward_type="policy")
                assert agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_quantiles)

            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)

        del agent_outs
        del agent_rnd_quantiles

        mac_out = th.stack(mac_out, dim=1) # Concat over time
        if self.args.agent != "qrdqn_rnn_dist":
            rnd_quantiles = th.stack(rnd_quantiles, dim=1) # Concat over time
        else:
            rnd_quantiles = th.ones(1, self.args.n_quantiles).cuda() * (1 / self.args.n_quantiles)
            rnd_quantiles = rnd_quantiles[None, None, ...]
            rnd_quantiles = rnd_quantiles.expand(batch.batch_size, episode_length+1, n_quantile_groups, -1)
        assert mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length+1, n_quantile_groups, self.n_quantiles)
        rnd_quantiles = rnd_quantiles[:,:-1]
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)

        # Pick the Q-Values for the actions taken by each agent
        actions_for_quantiles = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
        chosen_action_qvals = th.gather(mac_out[:,:-1], dim=3, index=actions_for_quantiles).squeeze(3)  # Remove the action dim
        del actions_for_quantiles
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_quantiles)
        if self.args.mixer == "dplex":
            assert mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            x_mac_out = mac_out.clone().detach()
            x_mac_out[avail_actions == 0] = -9999999
            max_action_qvals, max_action_index = x_mac_out[:, :-1].mean(dim=4).max(dim=3)
            max_action_index = max_action_index.detach().unsqueeze(3)
        del actions

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = target_mac_out.get()
            else:
                target_mac_out = calculate_target_q(self.args.agent, self.target_mac, batch, n_agents=self.args.n_agents, n_actions=self.args.n_actions, n_target_quantiles=self.n_target_quantiles)
                assert target_mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            
            # Mask out unavailable actions
            assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)
            target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_mac_out[target_avail_actions == 0] = -9999999
            avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
            
            # Max over target Q-Values
            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach.mean(dim=4).max(dim=3, keepdim=True)[1]
                del mac_out_detach
                assert cur_max_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, 1)
                cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            else:
                # [0] is for max value; [1] is for argmax
                cur_max_actions = target_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1]
                assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
                cur_max_actions_ = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions_).squeeze(3)
        
        if self.args.mixer == "dplex":
            # cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.args.n_actions,)).cuda()
            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.args.n_actions,), device=target_max_qvals.device)
            cur_max_actions_onehot = cur_max_actions_onehot.reshape(batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, -1)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
            del cur_max_actions

        q_attend_regs = 0
        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        if self.args.mixer == "dplex":
            ans_chosen, q_attend_regs = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False, is_v=True)
            ans_adv, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False, actions=actions_onehot,
                    max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)

        with th.no_grad():
            assert getattr(self.args, 'q_lambda', False) == False
            if self.args.mixer.find("dmix") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(
                    calculate_n_step_td_target,
                    (self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            elif self.args.mixer == "dmix":
                targets = calculate_n_step_td_target(
                    self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )
            elif self.args.mixer.find("dplex") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(
                    calculate_n_step_td_target_dplex,
                    (self.target_mixer, target_max_qvals, batch, cur_max_actions_onehot,
                     target_max_qvals, self.n_target_quantiles, rewards, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets = calculate_n_step_td_target_dplex(
                    self.target_mixer, target_max_qvals, batch, cur_max_actions_onehot,
                     target_max_qvals, self.n_target_quantiles, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )
            
            del target_mac_out
            assert target_max_qvals.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.n_target_quantiles)

            if self.args.mixer.find("dmix") != -1 and self.enable_parallel_computing:
                targets = targets.get()
            elif self.args.mixer.find("dplex") != -1 and self.enable_parallel_computing:
                targets = targets.get()
        
        # Quantile Huber loss
        assert targets.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)
        targets = targets.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1)
        assert targets.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        chosen_action_qvals = chosen_action_qvals.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # u is the signed distance matrix
        u = targets.detach() - chosen_action_qvals
        del targets
        del chosen_action_qvals
        assert u.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert tau.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # The abs term in quantile huber loss
        abs_weight = th.abs(tau - u.le(0.).float())
        del tau
        assert abs_weight.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Huber loss
        loss = F.smooth_l1_loss(u, th.zeros(u.shape).cuda(), reduction='none')
        del u
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Quantile Huber loss
        loss = (abs_weight * loss).mean(dim=4).sum(dim=3)
        del abs_weight
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups)

        assert mask.shape == (batch.batch_size, episode_length, 1)
        mask = mask.expand_as(loss)

        # 0-out the targets that came from padded data
        loss = loss * mask

        loss = loss.sum() / mask.sum() + q_attend_regs
        assert loss.shape == ()
        cbf_loss = cbf_loss

        # Optimise
        self.optimiser.pc_backward([loss, cbf_loss])
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
