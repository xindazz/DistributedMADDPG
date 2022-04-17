from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer_agents = Buffer(args, False)

        if self.args.train_adversaries:
            self.adversary_ids = range(self.args.n_agents, self.args.n_players)
            self.adversaries = self._init_adversaries()
            self.buffer_adversaries = Buffer(args, True)
        
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def _init_adversaries(self):
        adversaries = []
        for i in self.adversary_ids:
            adversary = Agent(i, self.args)
            adversaries.append(adversary)
        return adversaries

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()

            # train adversaries
            if self.args.train_adversaries:
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
                    
                    for adversary_id, adversary in enumerate(self.adversaries):
                        adversary_id += self.args.n_agents
                        action = adversary.select_action(s[adversary_id], self.noise, self.epsilon)
                        actions_adversaries.append(action)

                s_next, r, done, info = self.env.step(actions_agents + actions_adversaries)

                s_next_agents, r_agents = s_next[: self.args.n_agents], r[: self.args.n_agents]
                s_next_adversaries, r_adversaries = s_next[self.args.n_agents :], r[self.args.n_agents :]            
                self.buffer_agents.store_episode(s_agents, actions_agents, r_agents, s_next_agents)
                self.buffer_adversaries.store_episode(s_adversaries, actions_adversaries, r_adversaries, s_next_adversaries)
                
                s = s_next

                if self.buffer_agents.current_size >= self.args.batch_size:
                    transitions_agents = self.buffer_agents.sample(self.args.batch_size)
                    u_next = []
                    with torch.no_grad():
                        for agent_id in range(self.args.n_agents):
                            o_next = torch.tensor(transitions_agents['o_next_%d' % agent_id], dtype=torch.float)
                            u_next.append(self.agents[agent_id].policy.actor_target_network(o_next))
                    for agent in self.agents:
                        agent.learn(transitions_agents, u_next)
                
                if self.buffer_adversaries.current_size >= self.args.batch_size:
                    transitions_adversaries = self.buffer_adversaries.sample(self.args.batch_size)
                    u_next = []
                    with torch.no_grad():
                        for adversary_id, adversary in enumerate(self.adversaries):
                            adversary_id += self.args.n_agents
                            o_next = torch.tensor(transitions_adversaries['o_next_%d' % adversary_id], dtype=torch.float)
                            u_next.append(adversary.policy.actor_target_network(o_next))
                    for agent in self.adversaries:
                        agent.learn(transitions_adversaries, u_next)
            
            # Random adversary
            else:
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                s_next, r, done, info = self.env.step(actions)
                self.buffer_agents.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
                s = s_next
                if self.buffer_agents.current_size >= self.args.batch_size:
                    transitions = self.buffer_agents.sample(self.args.batch_size)
                    # Parse transitions into o_next
                    u_next = []
                    with torch.no_grad():
                        for agent_id in range(self.args.n_agents):
                            o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float)
                            u_next.append(self.agents[agent_id].policy.actor_target_network(o_next))

                    for agent in self.agents:
                        agent.learn(transitions, u_next)

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


    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for time_step in range(self.args.evaluate_episode_len):
                if self.args.render:
                    self.env.render()
                    time.sleep(.05)
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)

                    if self.args.train_adversaries:
                        for adversary_id, adversary in enumerate(self.adversaries):
                            adversary_id += self.args.n_agents
                            action = adversary.select_action(s[adversary_id], 0, 0)
                            actions.append(action)
                    else:
                        for i in range(self.args.n_agents, self.args.n_players):
                            actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
