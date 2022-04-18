import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # actions = self.max_action * torch.tanh(self.action_out(x))
        actions = self.max_action * torch.sigmoid(self.action_out(x)) # Min is 0

        return actions


class Critic(nn.Module):
    def __init__(self, args, agent_id):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.agent_id = agent_id
        # Adversary with DDPG
        if args.adversary_alg == "DDPG" and agent_id >= args.n_agents:
            self.input_dim = sum(args.obs_shape[args.n_agents:]) + sum(args.action_shape[args.n_agents:])
        # Input dim is all states and actions
        else: 
            self.input_dim = sum(args.obs_shape) + sum(args.action_shape)
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
