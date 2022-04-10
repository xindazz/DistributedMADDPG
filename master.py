from tqdm import tqdm
from copy import deepcopy
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import celery

from agent_distributed import Agent
from common.replay_buffer import Buffer
from worker import app, NumpyEncoder
import time


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        # self.agents = self._init_agents()
        self._init_agents()
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        # upload params to agents, queue name is "q{agent_id}:
        print("Start init all agents")

        tasks = []
        # for i in range(2):
        for i in range(self.args.n_agents):
            task = app.send_task("worker.init_agent", queue='q' + str(i), kwargs={"agent_id": i, "args": vars(self.args)}, cls=NumpyEncoder)
            tasks.append(task)

        print("Response: ")
        for task in tasks:
            print(task.get())


    def run(self):
        # pass
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()

            # u contains the actions of the agents that we are training
            u = []
            # actions contains actions of all agents in the environment, including those on the opposing team
            actions = []

            # Get action from every agent
            tasks = []
            for agent_id in range(self.args.n_agents):
                task = app.send_task("worker.get_action", queue='q' + str(agent_id), kwargs={"agent_id": i, "args": vars(self.args), "s": s[agent_id].tolist(), "evaluate": False}, cls=NumpyEncoder)
                tasks.append(task)
            for task in tasks:
                result = json.loads(task.get())
                u.append(result["action"])
                actions.append(result["action"])

            # Aversary agent acts randomly
            for i in range(self.args.n_agents, self.args.n_players):
                actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
            s = s_next

            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)

                # Parse transitions into o_next
                o_next = []
                for agent_id in range(self.args.n_agents):
                    o_next.append(transitions['o_next_%d' % agent_id])

                # Send o_next to each agent and get their target network's next action u_next
                tasks = []
                for agent_id in range(self.args.n_agents):
                    task = app.send_task("worker.get_target_next_action", queue='q' + str(agent_id), kwargs={"agent_id": i, "args": vars(self.args), "s": o_next[agent_id].tolist()}, cls=NumpyEncoder)
                    tasks.append(task)
                u_next = []
                for task in tasks:
                    result = json.loads(task.get())
                    u_next.append(result["action"])

                # Send u_next to each agent to train
                tasks = []
                for agent_id in range(self.args.n_agents):
                    data = json.loads(json.dumps({"agent_id": i, "args": vars(self.args), "transitions": transitions, "u_next": u_next}, cls=NumpyEncoder))
                    task = app.send_task("worker.train", queue='q' + str(agent_id), kwargs=data)
                    tasks.append(task)
                u_next = []
                for task in tasks:
                    result = task.get()
                
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

                # Get action from every agent
                tasks = []
                for agent_id in range(self.args.n_agents):
                    task = app.send_task("worker.get_action", queue='q' + str(agent_id), kwargs={"agent_id": i, "args": vars(self.args), "s": s[agent_id].tolist(), "evaluate": True}, cls=NumpyEncoder)
                    tasks.append(task)
                actions = []
                for task in tasks:
                    result = json.loads(task.get())
                    actions.append(result["action"])

                # Aversary agent acts randomly
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print('Returns is', rewards)
        return sum(returns) / self.args.evaluate_episodes
