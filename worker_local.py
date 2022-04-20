from tqdm import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time


from agent import Agent
from common.replay_buffer import Buffer
from common.utils import make_env


def worker_loop(input_queue, output_queue):
    env = None
    args = None
    id = None
    buffer = None
    worker_id = None
    noise = None
    epsilon = None
    episode_limit = None
    save_path = None
    agents = None
    returns = []
    returns_adv = []

    while True:
        input = input_queue.get()

        task_name = input[0]

        if task_name == "init":
            print("Initializing worker...")

            args = input[1]
            id = input[2]

            # create environment
            env, args = make_env(args)

            worker_id = id
            args.worker_id = id

            noise = args.noise_rate
            epsilon = args.epsilon
            episode_limit = args.max_episode_len

            buffer = Buffer(args)

            save_path = args.save_dir + "/" + args.scenario_name + "/worker_" + str(id)
            args.save_path = save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # initalize agents and adversaries
            agents = []
            if args.train_adversaries:
                for i in range(args.n_players):
                    agent = Agent(i, args)
                    agents.append(agent)
            else:
                for i in range(args.n_agents):
                    agent = Agent(i, args)
                    agents.append(agent)

            print("Initialization complete.")

            # return an acknowledgement back to master
            output_queue.put(True)

        elif task_name == "train":
            print("Training...")

            # reset environment before training
            s = list(env.reset().values())

            for time_step in range(episode_limit):
                # train adversaries
                if args.train_adversaries:
                    # actions contains actions of all agents in the environment
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(agents):
                            action = agent.select_action(s[agent_id], noise, epsilon)
                            actions.append(action)
                    actions_dict = {
                        agent_name: actions[agent_id]
                        for agent_id, agent_name in enumerate(env.agents)
                    }
                    s_next, r, done, info = env.step(actions_dict)
                    s_next, r = list(s_next.values()), list(r.values())
                    buffer.store_episode(s, actions, r, s_next)
                    s = s_next

                    if buffer.current_size >= args.batch_size:
                        transitions = buffer.sample(args.batch_size)

                        u_next = []
                        with torch.no_grad():
                            for agent_id in range(args.n_players):
                                if args.use_gpu and args.gpu:
                                    o_next = torch.tensor(
                                        transitions["o_next_%d" % agent_id],
                                        dtype=torch.float32,
                                    ).cuda()
                                else:
                                    o_next = torch.tensor(
                                        transitions["o_next_%d" % agent_id],
                                        dtype=torch.float32,
                                    )
                                u_next.append(
                                    agents[agent_id].policy.actor_target_network(o_next)
                                )

                        for agent_id, agent in enumerate(agents):
                            agent.learn(transitions, u_next)

                # random adversaries
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id, agent in enumerate(agents):
                            action = agent.select_action(s[agent_id], noise, epsilon)
                            actions.append(action)
                    for i in range(args.n_agents, args.n_players):
                        # actions.append(np.array([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]))
                        actions.append(
                            np.array(
                                [0, np.random.rand(), 0, np.random.rand(), 0],
                                dtype=np.float32,
                            )
                        )
                    actions_dict = {
                        agent_name: actions[agent_id]
                        for agent_id, agent_name in enumerate(env.agents)
                    }
                    s_next, r, done, info = env.step(actions_dict)
                    s_next, r = list(s_next.values()), list(r.values())
                    buffer.store_episode(s, actions, r, s_next)
                    s = s_next

                    if buffer.current_size >= args.batch_size:
                        transitions = buffer.sample(args.batch_size)
                        u_next = []
                        with torch.no_grad():
                            for agent_id in range(args.n_agents):
                                if args.use_gpu and args.gpu:
                                    o_next = torch.tensor(
                                        transitions["o_next_%d" % agent_id],
                                        dtype=torch.float32,
                                    ).cuda()
                                else:
                                    o_next = torch.tensor(
                                        transitions["o_next_%d" % agent_id],
                                        dtype=torch.float32,
                                    )
                                action_next = agents[
                                    agent_id
                                ].policy.actor_target_network(o_next)
                                u_next.append(action_next)
                            for i in range(args.n_agents, args.n_players):
                                action_next = []
                                for _ in range(args.batch_size):
                                    # action_next.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                                    action_next.append(
                                        [0, np.random.rand(), 0, np.random.rand(), 0]
                                    )
                                if args.use_gpu and args.gpu:
                                    action_next = torch.tensor(
                                        action_next, dtype=torch.float32
                                    ).cuda()
                                else:
                                    action_next = torch.tensor(
                                        action_next, dtype=torch.float32
                                    )
                                u_next.append(action_next)
                        for agent in agents:
                            agent.learn(transitions, u_next)

            # evaluate
            evaluate_returns = []
            evaluate_returns_adv = []
            for episode in range(args.evaluate_episodes):
                # reset the environment
                s = list(env.reset().values())
                rewards = 0
                rewards_adv = 0
                for time_step in range(args.evaluate_episode_len):
                    if args.render:
                        env.render()
                        time.sleep(0.05)
                    actions = []
                    with torch.no_grad():
                        if args.train_adversaries:
                            for agent_id, agent in enumerate(agents):
                                action = agent.select_action(s[agent_id], 0, 0)
                                actions.append(action)
                        else:
                            for agent_id, agent in enumerate(agents):
                                action = agent.select_action(s[agent_id], 0, 0)
                                actions.append(action)
                            for i in range(args.n_agents, args.n_players):
                                actions.append(
                                    np.array(
                                        [0, np.random.rand(), 0, np.random.rand(), 0],
                                        dtype=np.float32,
                                    )
                                )
                    actions_dict = {
                        agent_name: actions[agent_id]
                        for agent_id, agent_name in enumerate(env.agents)
                    }
                    s_next, r, done, info = env.step(actions_dict)
                    s_next, r = list(s_next.values()), list(r.values())
                    for i in range(args.n_agents):
                        rewards += r[i]
                    if args.train_adversaries:
                        for i in range(args.n_agents, args.n_players):
                            rewards_adv += r[i]

                    s = s_next
                evaluate_returns.append(rewards)
                if args.train_adversaries:
                    evaluate_returns_adv.append(rewards_adv)
                    print("Returns is", rewards, " Adversary return is", rewards_adv)
                else:
                    print("Returns is", rewards)

                return_agent, return_agent_adv = (
                    sum(evaluate_returns) / args.evaluate_episodes,
                    sum(evaluate_returns_adv) / args.evaluate_episodes,
                )

            # collect model parameters
            actor_data, critic_data = [], []
            for agent_id, agent in enumerate(agents):
                actor_data.append(agent.policy.actor_network.state_dict())
                critic_data.append(agent.policy.critic_network.state_dict())

            # output graphs
            s = list(env.reset().values())
            returns.append(return_agent)
            plt.figure()
            plt.plot(range(len(returns)), returns, label="Agent")
            if args.train_adversaries:
                returns_adv.append(return_agent_adv)
                plt.plot(range(len(returns_adv)), returns_adv, label="Adversary")
                np.save(save_path + "/returns_adv.pkl", returns_adv)
            plt.xlabel("episode * " + str(args.evaluate_rate / episode_limit))
            plt.ylabel("average returns")
            plt.legend()
            plt.savefig(save_path + "/plt.png", format="png")
            np.save(save_path + "/returns.pkl", returns)

            # reduce noise and epsilon after each training epoch
            noise = max(0.05, noise - 0.0000005)
            epsilon = max(0.05, epsilon - 0.0000005)

            # send model parameters back to master
            output_queue.put((worker_id, return_agent, actor_data, critic_data))

        elif task_name == "update_model":
            print("Updating models...")

            # get the new target networks (in state_dict format)
            actor = input[1]
            critic = input[2]

            # update each agent's actor critic network
            for i in range(args.n_agents):
                curr_agent = agents[i]
                curr_actor_network = actor[i]
                curr_critic_network = critic[i]
                curr_agent.policy.actor_network.load_state_dict(curr_actor_network)
                curr_agent.policy.critic_network.load_state_dict(curr_critic_network)

            # return an acknowledgement back to master
            output_queue.put(True)

        elif task_name == "close":
            print("Stopping worker process...")
            # stop the process loop
            return