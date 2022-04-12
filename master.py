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
        self.adversary_ids = range(self.args.n_agents, self.args.n_players)

        self._init_agents()

        # initialize buffer for agents and buffer for adversaries
        # first n_agents are agents
        self.buffer_agents = Buffer(
            args.buffer_size,
            args.n_agents,
            args.obs_shape[: args.n_agents],
            args.action_shape[: args.n_agents],
        )
        # rest are adversaries
        self.buffer_adversaries = Buffer(
            args.buffer_size,
            args.num_adversaries,
            args.obs_shape[args.n_agents :],
            args.action_shape[args.n_agents :],
        )

        self.save_path = self.args.save_dir + "/" + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        # upload params to agents
        print("Start initializing all agents...")

        # separate agents observation shape and adversaries observation shape
        obs_shape_agents = self.args.obs_shape[: self.args.n_agents]
        obs_shape_adversaries = self.args.obs_shape[self.args.n_agents: ]

        # separate agents action shape and adversaries action shape
        action_shape_agents = self.args.action_shape[: self.args.n_agents]
        action_shape_adversaries = self.args.action_shape[self.args.n_agents :]
        
        tasks = []
        # initialize agents, with queue name q{agent_id}
        # make sure to send the right observation and action shapes
        for agent_id in range(self.args.n_agents):
            task = app.send_task(
                "worker.init_agent",
                queue="q" + str(agent_id),
                kwargs={
                    "agent_id": agent_id,
                    "args": vars(self.args),
                    "obs_shape": obs_shape_agents,
                    "action_shape": action_shape_agents,
                },
                cls=NumpyEncoder,
            )
            tasks.append(task)

        for task in tasks:
            print(task.get())

        tasks = []
        # initialize adversaries, with queue name a{agent_id}
        # make sure to send the right observation and action shapes
        for agent_id in range(self.args.num_adversaries):
            task = app.send_task(
                "worker.init_agent",
                queue="a" + str(agent_id),
                kwargs={
                    "agent_id": agent_id,
                    "args": vars(self.args),
                    "obs_shape": obs_shape_adversaries,
                    "action_shape": action_shape_adversaries,
                },
                cls=NumpyEncoder,
            )
            tasks.append(task)
        print("Init:", obs_shape_adversaries, action_shape_adversaries)
        for task in tasks:
            print(task.get())

    def run(self):
        # pass
        returns = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()

            # split state into those observable by agents and those observable by adversaries
            s_agents = s[: self.args.n_agents]
            s_adversaries = s[self.args.n_agents :]

            # u contains the actions of the agents that we are training
            u = []
            # actions contains actions of all agents in the environment, including those of the adversaries
            actions = []

            # get actions from agents
            actions_agents = []
            tasks = []
            for agent_id in range(self.args.n_agents):
                task = app.send_task(
                    "worker.get_action",
                    queue="q" + str(agent_id),
                    kwargs={
                        "agent_id": agent_id,
                        "args": vars(self.args),
                        "s": s_agents[agent_id].tolist(),
                        "evaluate": False,
                    },
                    cls=NumpyEncoder,
                )
                tasks.append(task)

            for task in tasks:
                result = json.loads(task.get())
                actions_agents.append(result["action"])
                actions.append(result["action"])

            # get actions from adversaries
            actions_adversaries = []
            tasks = []
            for agent_id in range(self.args.num_adversaries):
                task = app.send_task(
                    "worker.get_action",
                    queue="a" + str(agent_id),
                    kwargs={
                        "agent_id": agent_id,
                        "args": vars(self.args),
                        "s": s_adversaries[agent_id].tolist(),
                        "evaluate": False,
                    },
                    cls=NumpyEncoder,
                )
                tasks.append(task)
                print("State:", s_adversaries[agent_id].tolist(), s_adversaries[agent_id].shape)

            for task in tasks:
                result = json.loads(task.get())
                actions_adversaries.append(result["action"])
                actions.append(result["action"])

            # # could maybe change? not sure
            # for taskidx in range(len(tasks)):
            #     task = task[taskidx]
            #     result = json.loads(task.get())
            #     if taskidx not in self.adversary_ids:
            #         u.append(result["action"])
            #         actions.append(result["action"])
            #     else:
            #         actions.append(result["action"])

            # get environment for the next time step
            s_next, r, done, info = self.env.step(actions)

            s_next_agents = s_next[: self.args.n_agents]
            s_next_adversaries = s_next[self.args.n_agents :]

            r_agents = r[: self.args.n_agents]
            r_adversaries = r[self.args.n_agents :]

            self.buffer_agents.store_episode(
                s_agents,
                actions_agents,
                r_agents,
                s_next_agents,
            )
            self.buffer_adversaries.store_episode(
                s_adversaries,
                actions_adversaries,
                r_adversaries,
                s_next_adversaries,
            )

            # update the state of the environment
            s = s_next

            # sample transitions and train agents
            if self.buffer_agents.current_size >= self.args.batch_size:
                transitions_agent = self.buffer_agents.sample(self.args.batch_size)
                # Parse transitions into o_next
                o_next = []
                for agent_id in range(self.args.n_agents):
                    o_next.append(transitions_agent["o_next_%d" % agent_id])

                # Send o_next to each agent and get their target network's next action u_next
                tasks = []
                for agent_id in range(self.args.n_agents):
                    task = app.send_task(
                        "worker.get_target_next_action",
                        queue="q" + str(agent_id),
                        kwargs={
                            "agent_id": agent_id,
                            "args": vars(self.args),
                            "s": o_next[agent_id].tolist(),
                        },
                        cls=NumpyEncoder,
                    )
                    tasks.append(task)
                u_next = []

                for task in tasks:
                    result = json.loads(task.get())
                    u_next.append(result["action"])

                # Send u_next to each agent to train
                tasks = []
                for agent_id in range(self.args.n_agents):
                    data = json.loads(
                        json.dumps(
                            {
                                "agent_id": agent_id,
                                "args": vars(self.args),
                                "transitions": transitions_agent,
                                "u_next": u_next,
                            },
                            cls=NumpyEncoder,
                        )
                    )
                    task = app.send_task(
                        "worker.train", queue="q" + str(agent_id), kwargs=data
                    )
                    tasks.append(task)

                for task in tasks:
                    result = task.get()

            # sample transitions and train adversaries
            if self.buffer_adversaries.current_size >= self.args.batch_size:
                transitions_adversary = self.buffer_adversaries.sample(
                    self.args.batch_size
                )
                # Parse transitions into o_next
                o_next = []
                for agent_id in range(self.args.num_adversaries):
                    o_next.append(transitions_adversary["o_next_%d" % agent_id])

                # Send o_next to each agent and get their target network's next action u_next
                tasks = []
                for agent_id in range(self.args.num_adversaries):
                    task = app.send_task(
                        "worker.get_target_next_action",
                        queue="a" + str(agent_id),
                        kwargs={
                            "agent_id": agent_id,
                            "args": vars(self.args),
                            "s": o_next[agent_id].tolist(),
                        },
                        cls=NumpyEncoder,
                    )
                    tasks.append(task)

                u_next = []
                for task in tasks:
                    result = json.loads(task.get())
                    u_next.append(result["action"])

                # Send u_next to each agent to train
                tasks = []
                for agent_id in range(self.args.num_adversaries):
                    data = json.loads(
                        json.dumps(
                            {
                                "agent_id": agent_id,
                                "args": vars(self.args),
                                "transitions": transitions_adversary,
                                "u_next": u_next,
                            },
                            cls=NumpyEncoder,
                        )
                    )
                    task = app.send_task(
                        "worker.train", queue="a" + str(agent_id), kwargs=data
                    )
                    tasks.append(task)

                for task in tasks:
                    result = task.get()

                #     data = json.loads(
                #         json.dumps(
                #             {
                #                 "agent_id": agent_id,
                #                 "args": vars(self.args),
                #                 "transitions": transitions_adversary,
                #             },
                #             cls=NumpyEncoder,
                #         )
                #     )
                #     task = app.send_task(
                #         "worker.train_single_adversary",
                #         queue="q" + str(agent_id),
                #         kwargs=data,
                #     )
                #     tasks.append(task)
                # for task in tasks:
                #     result = task.get()

            # # for multiple adversarys
            # if self.buffer_adversarys.current_size >= self.args.batch_size:
            #     transitions_adversary = self.buffer_adversarys.sample(self.args.batch_size)
            #     # Parse transitions into o_next
            #     o_next = []
            #     for agent_id in range(self.args.n_agents, self.args.n_players):
            #         o_next.append(transitions_adversary['o_next_%d' % agent_id])
            #     # Send o_next to each agent and get their target network's next action u_next
            #     tasks = []
            #     for agent_id in range(self.args.n_agents, self.args.n_players):
            #         task = app.send_task("worker.get_target_next_action", queue='q' + str(agent_id), kwargs={"agent_id": agent_id, "args": vars(self.args), "s": o_next[agent_id].tolist()}, cls=NumpyEncoder)
            #         tasks.append(task)
            #     u_next = []
            #     for task in tasks:
            #         result = json.loads(task.get())
            #         u_next.append(result["action"])
            #     # Send u_next to each agent to train
            #     tasks = []
            #     for agent_id in range(self.args.n_agents, self.args.n_players):
            #         data = json.loads(json.dumps({"agent_id": agent_id, "args": vars(self.args), "transitions": transitions_adversary, "u_next": u_next}, cls=NumpyEncoder))
            #         task = app.send_task("worker.train", queue='q' + str(agent_id), kwargs=data)
            #         tasks.append(task)
            #     for task in tasks:
            #         result = task.get()

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())
                plt.figure()
                plt.plot(range(len(returns)), returns)
                plt.xlabel(
                    "episode * " + str(self.args.evaluate_rate / self.episode_limit)
                )
                plt.ylabel("average returns")
                plt.savefig(self.save_path + "/plt.png", format="png")
                np.save(self.save_path + "/returns.pkl", returns)

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
                    task = app.send_task(
                        "worker.get_action",
                        queue="q" + str(agent_id),
                        kwargs={
                            "agent_id": agent_id,
                            "args": vars(self.args),
                            "s": s[agent_id].tolist(),
                            "evaluate": True,
                        },
                        cls=NumpyEncoder,
                    )
                    tasks.append(task)
                actions = []
                for task in tasks:
                    result = json.loads(task.get())
                    actions.append(result["action"])

                # Aversary agent acts randomly
                for i in range(self.args.n_agents, self.args.n_players):
                    actions.append(
                        [0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]
                    )

                s_next, r, done, info = self.env.step(actions)
                rewards += r[0]
                s = s_next
            returns.append(rewards)
            print("Returns is", rewards)
        return sum(returns) / self.args.evaluate_episodes
