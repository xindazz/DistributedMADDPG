from concurrent.futures import process
from re import A
from tqdm import tqdm
from copy import deepcopy
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

from agent import Agent
from common.replay_buffer import Buffer
import time

import torch.multiprocessing as mp

env = None
args = None
buffer = None
batch_size = None
agents = None
def init_worker(env_input, args_input, buffer_input, batch_size_input, agents_input):
    global env, args, buffer, batch_size, agents
    env = env_input
    args = args_input
    buffer = buffer_input
    batch_size = batch_size_input
    agents = agents_input

def run(id):
    returns = []
    print("agents in train before", [a.agent_id for a in agents])
    for time_step in tqdm(range(1000000)):
        if id == args.n_agents:
            # reset the environment
            if time_step % args.max_episode_len == 0:
                s = env.reset()
                # print("Episode", time_step / self.episode_limit)
            
            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(s[agent_id], args.noise_ratee, args.epsilon)
                    u.append(action)
                    actions.append(action)
            for i in range(args.n_agents, args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
            s_next, r, done, info = env.step(actions)
            buffer.store_episode(s[:args.n_agents], u, r[:args.n_agents], s_next[:args.n_agents])
            s = s_next

            if time_step > 0 and time_step % args.evaluate_rate == 0:
                evaluate_returns = []
                for episode in range(args.evaluate_episodes):
                    # reset the environment
                    s = env.reset()
                    rewards = 0
                    for _ in range(args.evaluate_episode_len):
                        if args.render:
                            env.render()
                        actions = []
                        with torch.no_grad():
                            for agent_id, agent in enumerate(agents):
                                action = agent.select_action(s[agent_id], 0, 0)
                                actions.append(action)

                            # if args.train_adversaries:
                            #     for adversary_id, adversary in enumerate(adversaries):
                            #         adversary_id += args.n_agents
                            #         action = adversary.select_action(s[adversary_id], 0, 0)
                            #         actions.append(action)
                            # else:
                            for i in range(args.n_agents, args.n_players):
                                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                        s_next, r, done, info = env.step(actions)
                        rewards += r[0]
                        s = s_next
                    evaluate_returns.append(rewards)
                    print('Returns is', rewards)
                returns.append(sum(evaluate_returns) / args.evaluate_episodes)

                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel('episode * ' + str(args.evaluate_rate / args.max_episode_len))
                plt.ylabel('average returns')
                save_path = args.save_dir + '/' + args.scenario_name
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(save_path + '/plt.png', format='png')
                np.save(save_path + '/returns.pkl', returns)
        else:
            transitions = buffer.sample(batch_size)
            # Parse transitions into o_next
            u_next = []
            with torch.no_grad():
                for agent_id in range(len(agents)):
                    o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float)
                    u_next.append(agents[agent_id].policy.actor_target_network(o_next))
            print("agents in train", [a.agent_id for a in agents])
            for agent_id in range(len(agents)):
                agents[agent_id].learn(transitions, u_next)
 


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer_agents = Buffer(args, False)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents


    def run(self):
        # returns = []
        # processes_started = False
        # for time_step in tqdm(range(self.args.time_steps)):
        #     # reset the environment
        #     if time_step % self.episode_limit == 0:
        #         s = self.env.reset()
        #         # print("Episode", time_step / self.episode_limit)
            
        #     u = []
        #     actions = []
        #     with torch.no_grad():
        #         for agent_id, agent in enumerate(self.agents):
        #             action = agent.select_action(s[agent_id], self.noise, self.epsilon)
        #             u.append(action)
        #             actions.append(action)
        #     for i in range(self.args.n_agents, self.args.n_players):
        #         actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
        #     s_next, r, done, info = self.env.step(actions)
        #     self.buffer_agents.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
        #     s = s_next

            
        #     if processes_started == False and self.buffer_agents.current_size >= self.args.batch_size:
        #         pool = mp.Pool
        #         (processes=3, initializer=init_worker, initargs=(self.buffer_agents, self.args.batch_size, self.agents))   
        #         print("self.agents", self.agents)                 
        #         pool.map(train, [1,2,0])
        #         pool.close()
        #         processes_started = True
        #         print("Passed")
            
        #     if time_step > 0 and time_step % self.args.evaluate_rate == 0:
        #         returns.append(self.evaluate())
        #         plt.figure()
        #         plt.plot(range(len(returns)), returns)
        #         plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
        #         plt.ylabel('average returns')
        #         plt.savefig(self.save_path + '/plt.png', format='png')
        #         np.save(self.save_path + '/returns.pkl', returns)


        #     self.noise = max(0.05, self.noise - 0.0000005)
        #     self.epsilon = max(0.05, self.epsilon - 0.0000005)

        pool = mp.Pool(processes=4, initializer=init_worker, initargs=(self.env, self.args, self.buffer_agents, self.args.batch_size, self.agents))   
        print("self.agents", self.agents)                 
        pool.map(run, [3, 0, 1, 2])
        pool.close()
        pool.join()


    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = self.env.reset()
            rewards = 0
            for _ in range(self.args.evaluate_episode_len):
                if self.args.render:
                    self.env.render()
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
