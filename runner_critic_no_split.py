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
        self.agents = self._init_agents()
        self.buffer = Buffer(args, False)

        if self.args.train_adversaries:
            self.adversary_ids = range(self.args.n_agents, self.args.n_players)
        
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        if self.args.train_adversaries:
            for i in range(self.args.n_players):
                agent = Agent(i, self.args)
                agents.append(agent)
        else:
            for i in range(self.args.n_agents):
                agent = Agent(i, self.args)
                agents.append(agent)
        return agents

    def run(self):
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()

            # train adversaries
            if self.args.train_adversaries:
                # actions contains actions of all agents in the environment
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        actions.append(action)
                s_next, r, done, info = self.env.step(actions)
                self.buffer.store_episode(s, actions, r, s_next)
                s = s_next

                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)                        

                    if self.args.adversary_alg == "MADDPG":
                        u_next = []
                        with torch.no_grad():
                            for agent_id in range(self.args.n_players):
                                o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float)
                                u_next.append(self.agents[agent_id].policy.actor_target_network(o_next))

                        for agent_id, agent in enumerate(self.agents):
                            agent.learn(transitions, u_next)
                    else:
                        u_next = []
                        with torch.no_grad():
                            for agent_id in range(self.args.n_players):
                                o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float)
                                u_next.append(self.agents[agent_id].policy.actor_target_network(o_next))
                        # Agents still train with everyone's states and actions
                        for agent_id in range(self.args.n_players):
                            self.agents[agent_id].learn(transitions, u_next)
                     
                
                
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
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for key in transitions.keys():
                        transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
                    o, u, o_next = [], [], []
                    for agent_id in range(self.args.n_agents):
                        o.append(transitions['o_%d' % agent_id])
                        u.append(transitions['u_%d' % agent_id])
                        o_next.append(transitions['o_next_%d' % agent_id])
                    # Parse transitions into o_next
                    u_next = []
                    with torch.no_grad():
                        for agent_id in range(self.args.n_agents):
                            u_next.append(self.agents[agent_id].policy.actor_target_network(o_next[agent_id]))

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
                actions = []
                with torch.no_grad():
                    if self.args.train_adversaries:
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                    else:
                        for agent_id, agent in enumerate(self.agents):
                            action = agent.select_action(s[agent_id], 0, 0)
                            actions.append(action)
                        for i in range(self.args.n_agents, self.args.n_players):
                            actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
