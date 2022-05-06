import numpy as np
import torch
import os
from maddpg.maddpg_distributed import MADDPG


class Agent:
    def __init__(self, agent_id, args, num_actors, obs_shape, action_shape):
        self.args = args
        self.agent_id = agent_id
        self.num_actors = num_actors

        self.actor_action_shape = action_shape[agent_id]

        self.policy = MADDPG(args, agent_id, num_actors, obs_shape, action_shape)

    def select_action(self, o, noise_rate, epsilon):
        # take a random action with a small probability
        if np.random.uniform() < epsilon:
            u = np.random.uniform(
                -self.args.high_action,
                self.args.high_action,
                self.actor_action_shape,
            )
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = (
                noise_rate * self.args.high_action * np.random.randn(*u.shape)
            )  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, u_next):
        self.policy.train(transitions, u_next)
