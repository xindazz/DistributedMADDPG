import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id]).astype(np.float32)
        else:
            if self.args.use_gpu and self.args.gpu:
                inputs = torch.tensor(o, dtype=torch.float32).cuda().unsqueeze(0)
            else:
                inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, self.args.low_action, self.args.high_action).astype(np.float32)
        return u.copy()

    def learn(self, transitions, u_next):
        self.policy.train(transitions, u_next)

