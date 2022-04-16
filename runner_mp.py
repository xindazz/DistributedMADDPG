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

from mpi4py import MPI


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
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
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        s = [np.empty(self.args.obs_shape[agent_id], dtype=np.float64) for agent_id in range(self.args.n_agents)]
        o_next = [np.empty(self.args.obs_shape[agent_id], dtype=np.float64) for agent_id in range(self.args.n_agents)]
        u = [np.empty(self.args.action_shape[agent_id], dtype=np.float64) for agent_id in range(self.args.n_agents)]
        u_next = [np.empty(self.args.action_shape[agent_id], dtype=np.float64) for agent_id in range(self.args.n_agents)]

        # Master
        if rank == 0:
            returns = []
            for time_step in tqdm(range(self.args.time_steps)):
                # reset the environment
                if time_step % self.episode_limit == 0:
                    s = self.env.reset()
                
                # actions contains actions of all agents in the environment, including those on the opposing team
                actions = []

                # Get action from every agent
                for agent_id in range(self.args.n_agents):
                    # task = app.send_task("worker.get_action", queue='q' + str(agent_id), kwargs={"agent_id": agent_id, "args": vars(self.args), "s": s[agent_id].tolist(), "evaluate": False}, cls=NumpyEncoder)
                    task = comm.isend(s[agent_id], dest=agent_id+1, tag=0)
                requests = []
                for agent_id in range(self.args.n_agents):
                    req = comm.irecv(u[agent_id], source=agent_id+1, tag=1)
                    requests.append(req)
                for req in requests:
                    req.wait()

                for action in u:
                    actions.append(action)

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
                    for agent_id in range(self.args.n_agents):
                        # task = app.send_task("worker.get_target_next_action", queue='q' + str(agent_id), kwargs={"agent_id": agent_id, "args": vars(self.args), "s": o_next[agent_id].tolist()}, cls=NumpyEncoder)
                        req = comm.isend(o_next[agent_id], dest=agent_id+1, tag=2)
                    requests = []
                    for agent_id in range(self.args.n_agents):
                        req = comm.irecv(u_next[agent_id], source=agent_id+1, tag=3)
                        requests.append(req)
                    for req in requests:
                        req.wait()

                    # Send u_next to each agent to train
                    u_next = np.array(u_next)
                    requests = []
                    for agent_id in range(self.args.n_agents):
                        req =comm.isend(u_next, dest=agent_id+1, tag=4)
                        requests.append(req)
                    for req in requests:
                        req.wait()
                    
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
        # Worker
        else:   
            if rank in [1, 2, 3]:
                agent_id = rank - 1

                req = comm.irecv(s[agent_id], source=0, tag=0)
                req.wait()
                
                u[agent_id] = self.agents[agent_id].select_action(s, self.args.noise_rate, self.args.epsilon)
                req = comm.isend(u[agent_id], dest=0, tag=1)
                
                req = comm.irecv(o_next[agent_id], source=0, tag=2)
                req.wait()

                u_next[agent_id] = self.agents[agent_id].policy.actor_target_network(torch.tensor(o_next[agent_id], dtype=torch.float))
                req = comm.isend(u_next[agent_id], dest=0, tag=3)

                u_next = np.array(u_next)

                req = comm.irecv(u_next, source=0, tag=4)
                req.wait()
                self.agents[agent_id].learn(transitions, torch.tensor(u_next, dtype=torch.float))



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
                    task = app.send_task("worker.get_action", queue='q' + str(agent_id), kwargs={"agent_id": agent_id, "args": vars(self.args), "s": s[agent_id].tolist(), "evaluate": True}, cls=NumpyEncoder)
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
