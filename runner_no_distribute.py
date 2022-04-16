from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.adversary_ids = range(self.args.n_agents, self.args.n_players)
        self.agents, self.adversaries = self._init_agents()
        self.buffer_agents = Buffer(
            args.buffer_size,
            args.n_agents,
            args.obs_shape[: args.n_agents],
            args.action_shape[: args.n_agents],
        )
        self.buffer_adversaries = Buffer(
            args.buffer_size,
            args.num_adversaries,
            args.obs_shape[args.n_agents :],
            args.action_shape[args.n_agents :],
        )
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        # separate agents observation shape and adversaries observation shape
        obs_shape_agents = self.args.obs_shape[: self.args.n_agents]
        obs_shape_adversaries = self.args.obs_shape[self.args.n_agents: ]

        # separate agents action shape and adversaries action shape
        action_shape_agents = self.args.action_shape[: self.args.n_agents]
        action_shape_adversaries = self.args.action_shape[self.args.n_agents :]

        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)

        adversaries = []
        for i in range(self.adversary_ids):
            adversary = Agent(i, self.args)
            adversaries.append(adversary)
        return agents, adversaries

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
            # split state into those observable by agents and those observable by adversaries
            s_agents = s[: self.args.n_agents]
            s_adversaries = s[self.args.n_agents :]
            # actions contains actions of all agents in the environment
            actions_agents = []
            actions_adversaries = []

            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    actions_agents.append(action)
                
            for i in range(self.args.n_agents, self.args.n_players):
                for adversary_id, adversary in enumerate(self.adversaries):
                    adversary_id += self.n_agents
                    action = adversary.select_action(s[adversary_id], self.noise, self.epsilon)
                    actions_adversaries.append(action)

            s_next, r, done, info = self.env.step(actions_agents+actions_adversaries)
            s_next_agents = s_next[: self.args.n_agents]
            s_next_adversaries = s_next[self.args.n_agents :]
            r_agents = r[: self.args.n_agents]
            r_adversaries = r[self.args.n_agents :]
            self.buffer_agents.store_episode(s_agents, actions_agents, r_agents, s_next_agents)
            self.buffer_adversaries.store_episode(s_adversaries, actions_adversaries, r_adversaries, s_next_adversaries)
            s_agents = s_next_agents
            s_adversaries = s_next_adversaries

            if self.buffer_agents.current_size >= self.args.batch_size:
                transitions_agents = self.buffer_agents.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions_agents, other_agents)
            
            if self.buffer_adversaries.current_size >= self.args.batch_size:
                transitions_adversaries = self.buffer_adversaries.sample(self.args.batch_size)
                for adversary in self.adversaries:
                    other_adversaries = self.adversaries.copy()
                    other_adversaries.remove(adversary)
                    agent.learn(transitions_adversaries, other_adversaries)
            
            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.savefig(self.save_path + '/plt.png', format='png')
                np.save(self.save_path + '/returns.pkl', returns)
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)
            # np.save(self.save_path + '/returns.pkl', returns)

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            s_agents = s[:self.args.n_agents]
            s_adversaries = s[self.args.n_agents:]
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if self.args.render:
                    self.env.render()
                # actions contains actions of all agents in the environment
                actions_agents = []
                actions_adversaries = []

                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions_agents.append(action)
                    
                for i in range(self.args.n_agents, self.args.n_players):
                    for adversary_id, adversary in enumerate(self.adversaries):
                        adversary_id += self.n_agents
                        action = adversary.select_action(s[adversary_id], 0, 0)
                        actions_adversaries.append(action)
                
                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
