from concurrent.futures import process
from re import A
from tqdm import tqdm
from copy import deepcopy
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from common.replay_buffer import Buffer
import time

import torch.multiprocessing as mp


# def get_action(queue, agent, s, args):
#     with torch.no_grad():
#         action = agent.select_action(s, args.noise_rate, args.epsilon)
#         queue.put(action)

# def get_next_action(queue, agent, s):
#     with torch.no_grad():
#         action_next = agent.policy.actor_target_network(torch.tensor(s, dtype=torch.float))
#         queue.put(action_next)

def train(agent, transitions, u_next):
    agent.learn(transitions, u_next)


def full_train(args, agent_id, train_step, transitions, u_next, actor_network, critic_network, critic_target_network, 
                                    actor_optim, critic_optim):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for i in args.n_agents:
            o.append(transitions['o_%d' % i])
            u.append(transitions['u_%d' % i])
            o_next.append(transitions['o_next_%d' % i])

        # calculate the target Q value function
        with torch.no_grad():
            q_next = critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + args.gamma * q_next).detach()

        # the q loss
        q_value = critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        if agent_id < args.n_agents:
            u[agent_id] = actor_network(o[agent_id])
        else:
            u[agent_id - args.n_agents] = actor_network(o[agent_id - args.n_agents])
        actor_loss = - critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # update the network
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        for target_param, param in zip(critic_target_network.parameters(), critic_network.parameters()):
            target_param.data.copy_((1 - args.tau) * target_param.data + args.tau * param.data)

        if train_step > 0 and train_step % args.save_rate == 0:
            num = str(train_step // args.save_rate)
            model_path = os.path.join(args.save_dir, args.scenario_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, 'agent_%d' % agent_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
            torch.save(critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')


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
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                print("Episode", time_step / self.episode_limit)
            
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
                # print("U_next", u_next)
                # Send u_next to each agent to train
                processes = []
                for agent_id in range(self.args.n_agents):
                    # task = app.send_task("worker.get_target_next_action", queue='q' + str(agent_id), kwargs={"agent_id": agent_id, "args": vars(self.args), "s": o_next[agent_id].tolist()}, cls=NumpyEncoder)
                    # p = mp.Process(target=full_train, args=(self.args, agent_id, time_step, transitions, u_next, self.agents[agent_id].policy.actor_network, self.agents[agent_id].policy.critic_network, self.agents[agent_id].policy.critic_target_network, 
                    #                         self.agents[agent_id].policy.actor_optim, self.agents[agent_id].policy.critic_optim))
                    p = mp.Process(target=train, args=(self.agents[agent_id], transitions, u_next))
                    p.start()
                    processes.append(p)
                for agent_id in range(self.args.n_agents):
                    processes[agent_id].join()
                
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
