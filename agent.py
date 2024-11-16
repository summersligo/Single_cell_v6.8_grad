import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model_pn import Actor, Critic, Critic_2
from replay_buffer import ReplayBuffer
import time
from scipy import *

from scipy.optimize import minimize

from torch.autograd import Variable

class Agent:
    def __init__(self, args):
        self.args = args
        # self.buffer_size = 20000
        # self.batch_size = 25
        self.replay_buffer = ReplayBuffer(self.args)
        self.antenna_number = self.args.user_antennas
        self.user_number = self.args.user_numbers
        self.bs_antenna_number = self.args.bs_antennas
        self.device = 'cuda' if self.args.cuda else 'cpu'
        # self.device = 'cpu'
        self.writer = self.args.writer
        self.policy_net = Actor(self.args).to(self.device)
        self.critic_net = Critic(self.args).to(self.device)
        self.critic_net_2 = Critic_2(self.args).to(self.device)
        # 定义两个active网络学习率的decay系数
        self.learning_rate_policy_net = self.args.actor_lr
        self.learning_rate_critic_net = self.args.critic_lr
        self.decay_rate_policy_net = self.args.actor_lr_decay
        self.decay_rate_critic_net = self.args.critic_lr_decay
        
        # discount ratio
        self.gamma = self.args.gamma
        # GAE factor
        self.GAE_factor = self.args.GAE_factor
        # 定义loss function
        self.loss_function = nn.MSELoss()
        # 定义优化器
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate_policy_net)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate_critic_net)
        self.optimizer_critic_2 = optim.Adam(self.critic_net_2.parameters(), lr=self.learning_rate_critic_net)
        
        self.update_value_net_count = 0
        self.update_value_net_2_count = 0
        self.update_policy_net_count = 0
        self.MAX_ITER = 250
        self.STOP_CRIT = 1e-5
        self.weight_vector = [1,0]
        np.save('./Exp/init_weight',self.weight_vector)

    def Pick_action(self, channel_matrix, user_reward,noise,sample=True):  #Actor
        # 将channel_matrix, user_reward转换成tensor
        channel_matrix = torch.FloatTensor(channel_matrix).to(self.device)
        user_reward = torch.FloatTensor(user_reward).to(self.device).unsqueeze(-1)
        noise=torch.FloatTensor(noise).to(self.device)
        prob_value,schedule_result = self.policy_net(channel_matrix, user_reward,noise,sample)
        return schedule_result, prob_value

    def Update_value_net(self, target_value, channel_matrix, user_reward):   #update critic

        for i in range(1):
            self.update_value_net_count += 1
            self.optimizer_critic.zero_grad()
            approximate_value = self.critic_net(channel_matrix, user_reward)
            loss = self.loss_function(approximate_value, target_value)
            loss.backward()
            self.optimizer_critic.step()
            self.writer.add_scalar('Loss/critic_loss', loss.item(), self.update_value_net_count)

    def Update_value_net_2(self, target_value, channel_matrix, user_fairness_reward):   #update critic

        for i in range(1):
            self.update_value_net_2_count += 1
            self.optimizer_critic_2.zero_grad()
            approximate_value = self.critic_net_2(channel_matrix, user_fairness_reward)
            loss = self.loss_function(approximate_value, target_value)
            loss.backward()
            self.optimizer_critic_2.step()
            self.writer.add_scalar('Loss/critic_loss', loss.item(), self.update_value_net_2_count)

    def solve_qp(self,grad1,grad2):

        grad1 = grad1.cpu().numpy()
        grad1 = np.array(grad1, dtype=np.float64)
        grad2 = grad2.cpu().numpy()
        grad2 = np.array(grad2, dtype=np.float64)

        def objective(w):
            w1, w2 = w
            combined_grad = w1 * grad1 + w2 * grad2
            return np.linalg.norm(combined_grad) ** 2

        def constraint(w):
            return w[0] + w[1] - 1

        bounds = [(0, None), (0, None)]

        # 初始化猜测
        w0 = [0.3, 0.7]
        constraints = {'type': 'eq', 'fun': constraint}

        # 求解
        result = minimize(objective, w0, bounds=bounds, constraints=constraints)
        return result.x
    def Training(self):
        # 这个地方将进行两个网络的训练
        Trajectories = self.replay_buffer.sample()
        channel_matrix = Trajectories['Channel']
        instant_capacity_reward = Trajectories['instant_capacity_reward']
        instant_fairness_reward = Trajectories['instant_fairness_reward']
        user_fairness_reward = Trajectories['Average_fairness_reward']
        mask = Trajectories['terminate']
        probs = Trajectories['prob']
        noise = Trajectories['noise']

        channel_matrix = torch.FloatTensor(channel_matrix).to(self.device).reshape(-1, self.args.channel_dim1, self.args.channel_dim2)
        user_fairness_reward = torch.FloatTensor(user_fairness_reward).to(self.device).reshape(-1, self.args.channel_dim1,1)
        instant_capacity_reward = torch.FloatTensor(instant_capacity_reward).to(self.device).reshape(-1)
        instant_fairness_reward = torch.FloatTensor(instant_fairness_reward).to(self.device).reshape(-1)
        mask = torch.FloatTensor(mask).to(self.device).reshape(-1)
        Prob = torch.stack([torch.stack(probs[i], 0) for i in range(self.args.episodes)], 0).reshape(-1)
        noise = torch.FloatTensor(noise).to(self.device).squeeze()

        values_c = self.critic_net(channel_matrix, user_fairness_reward).detach()    #critic
        returns_c = torch.zeros(channel_matrix.shape[0],1).to(self.device)  #target
        deltas_c = torch.Tensor(channel_matrix.shape[0],1).to(self.device)  #target - critic
        advantages_c = torch.Tensor(channel_matrix.shape[0],1).to(self.device)

        values_f = self.critic_net_2(channel_matrix, user_fairness_reward).detach()
        returns_f = torch.zeros(channel_matrix.shape[0],1).to(self.device)
        deltas_f = torch.Tensor(channel_matrix.shape[0],1).to(self.device)
        advantages_f = torch.Tensor(channel_matrix.shape[0],1).to(self.device)

        prev_return_c = 0
        prev_value_c = 0
        prev_advantage_c = 0

        prev_return_f = 0
        prev_value_f = 0     
        prev_advantage_f = 0

        for i in reversed(range(instant_capacity_reward.shape[0])):
            returns_c[i] = instant_capacity_reward[i] + self.gamma * prev_return_c * mask[i]
            deltas_c[i] = instant_capacity_reward[i] + self.gamma * prev_value_c * mask[i] - values_c.data[i]
            advantages_c[i] = deltas_c[i] + self.gamma * self.GAE_factor * prev_advantage_c * mask[i]

            returns_f[i] = instant_fairness_reward[i] + self.gamma * prev_return_f * mask[i]
            deltas_f[i] = instant_fairness_reward[i] + self.gamma * prev_value_f * mask[i] - values_f.data[i]
            advantages_f[i] = deltas_f[i] + self.gamma * self.GAE_factor * prev_advantage_f * mask[i]

            prev_return_c = returns_c[i, 0]
            prev_value_c = values_c.data[i, 0]
            prev_advantage_c = advantages_c[i, 0]

            prev_return_f = returns_f[i, 0]
            prev_value_f = values_f.data[i, 0]
            prev_advantage_f = advantages_f[i, 0]

        self.Update_value_net(returns_c, channel_matrix, user_fairness_reward)
        self.Update_value_net_2(returns_f, channel_matrix, user_fairness_reward)

        ###################################################### update weight_vector
        advantages = [advantages_c, advantages_f]
        weights,nd = self.get_pareto_weight(Prob, advantages)
        w1 = weights[0]
        w2 = weights[1]
        print(w1, w2)
        advantage = w1 * advantages_c + w2 * advantages_f
        policy_net_loss = - torch.mean(Prob * advantage)
        self.update_policy_net_count += 1
        self.optimizer_policy.zero_grad()
        policy_net_loss.backward()
        self.optimizer_policy.step()
        #advantages=torch.cat((advantages_c,advantages_f),1)
        # policy_net_loss = - torch.mean(Prob* (torch.sum(torch.multiply(advantages,noise[0]),1)))
        # advantages = w1*advantages_c + w2*advantages_f
        # policy_net_loss = - torch.mean(Prob * advantages.squeeze())
        # self.writer.add_scalar('Loss/actor_loss', torch.mean(torch.exp(Prob) * advantages.squeeze()), self.update_policy_net_count)
        self.learning_rate_critic_net = (1-self.decay_rate_critic_net) * self.learning_rate_critic_net
        self.learning_rate_policy_net = (1-self.decay_rate_policy_net) * self.learning_rate_policy_net
        self.replay_buffer.reset_buffer()
        return w1,w2
    def Store_transition(self, batch):
        self.replay_buffer.store_episode(batch)

    def get_pareto_weight(self, Prob, advantages):
        grads = []
        for advantage in advantages:
            loss = -torch.mean(Prob * advantage)
            self.policy_net.zero_grad()
            loss.backward(retain_graph=True)
            g = []
            for name, p in self.policy_net.named_parameters():
                if p.requires_grad:
                    grad = p.grad.data.clone().flatten()
                    g += list(grad.cpu().numpy())
            grads.append(g)
        grads = torch.FloatTensor(grads)
        dps = {}
        init_sol, dps = self._min_norm_2d(grads, dps)
        n = len(grads)
        sol_vec = torch.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec, init_sol[2]

        iter_count = 0
        grad_mat = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < self.MAX_ITER:
            t_iter = torch.argmin(torch.matmul(grad_mat, sol_vec)).item()
            v1v1 = torch.dot(sol_vec, torch.matmul(grad_mat, sol_vec)).item()
            v1v2 = torch.dot(sol_vec, grad_mat[:, t_iter]).item()
            v2v2 = grad_mat[t_iter, t_iter].item()

            nc, nd = self._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec.clone()
            new_sol_vec[t_iter] += 1 - nc

            change = torch.sum(torch.abs(new_sol_vec - sol_vec)).item()
            if change < self.STOP_CRIT:
                return sol_vec, nd

            sol_vec = new_sol_vec
            iter_count += 1

        return sol_vec, nd

    def _min_norm_element_from2(self, v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(self, vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c, d = self._min_norm_element_from2(dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps
