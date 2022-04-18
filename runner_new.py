from tqdm import tqdm
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time


from agent import Agent
from common.replay_buffer import Buffer
from common.utils import make_env


class Runner:
    def __init__(self, args, id):
        args.gpu = torch.cuda.is_available()
        print("Using GPU:", args.gpu)

        self.env, self.args = make_env(args)
        self.worker_id = id
        self.args.worker_id = id
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name + "/worker_" + str(id)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        # initalize agents on local
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


    def run(self, q):
        returns = []
        returns_adv = []
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = list(self.env.reset().values())

            # train adversaries
            if self.args.train_adversaries:
                # actions contains actions of all agents in the environment
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        actions.append(action)
                actions_dict = {agent_name: actions[agent_id] for agent_id, agent_name in enumerate(self.env.agents)}
                s_next, r, done, info = self.env.step(actions_dict)
                s_next, r = list(s_next.values()), list(r.values())
                self.buffer.store_episode(s, actions, r, s_next)
                s = s_next

                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)                        

                    u_next = []
                    with torch.no_grad():
                        for agent_id in range(self.args.n_players):
                            if self.args.use_gpu and self.args.gpu:
                                o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32).cuda()
                            else:
                                o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32)
                            u_next.append(self.agents[agent_id].policy.actor_target_network(o_next))

                    for agent_id, agent in enumerate(self.agents):
                        agent.learn(transitions, u_next)
                    
            # Random adversary
            else:
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        actions.append(action)
                for i in range(self.args.n_agents, self.args.n_players):
                    # actions.append(np.array([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]))
                    actions.append(np.array([0, np.random.rand(), 0, np.random.rand(), 0], dtype=np.float32))
                actions_dict = {agent_name: actions[agent_id] for agent_id, agent_name in enumerate(self.env.agents)}
                s_next, r, done, info = self.env.step(actions_dict)
                s_next, r = list(s_next.values()), list(r.values())
                self.buffer.store_episode(s, actions, r, s_next)
                s = s_next

                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    u_next = []
                    with torch.no_grad():
                        for agent_id in range(self.args.n_agents):
                            if self.args.use_gpu and self.args.gpu:
                                o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32).cuda()
                            else:
                                o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32)
                            action_next = self.agents[agent_id].policy.actor_target_network(o_next)
                            u_next.append(action_next)
                        for i in range(self.args.n_agents, self.args.n_players):
                            action_next = []
                            for _ in range(self.args.batch_size):
                                # action_next.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                                action_next.append([0, np.random.rand(), 0, np.random.rand(), 0])
                            if self.args.use_gpu and self.args.gpu:
                                action_next = torch.tensor(action_next, dtype=torch.float32).cuda()
                            else:
                                action_next = torch.tensor(action_next, dtype=torch.float32)
                            u_next.append(action_next)
                    for agent in self.agents:
                        agent.learn(transitions, u_next)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                return_agent, return_adv = self.evaluate()

                # Send to central queue
                actor_data, critic_data = [], []
                for agent_id, agent in enumerate(self.agents):
                    actor_data = []
                    for param in agent.policy.actor_network.parameters():
                        actor_data.append(param.data.tolist())
                    critic_data = []
                    for param in agent.policy.critic_network.parameters():
                        critic_data.append(param.data.tolist())

                q.put((self.worker_id, return_agent, actor_data, critic_data))

                s = list(self.env.reset().values())
                returns.append(return_agent)
                plt.figure()
                plt.plot(range(len(returns)), returns, label="Agent")
                if self.args.train_adversaries:
                    returns_adv.append(return_adv)
                    plt.plot(range(len(returns_adv)), returns_adv, label="Adversary")
                    np.save(self.save_path + '/returns_adv.pkl', returns_adv)
                plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.episode_limit))
                plt.ylabel('average returns')
                plt.legend()
                plt.savefig(self.save_path + '/plt.png', format='png')
                np.save(self.save_path + '/returns.pkl', returns)
                

            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)


    def evaluate(self):
        returns = []
        returns_adv = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            s = list(self.env.reset().values())
            rewards = 0
            rewards_adv = 0
            for time_step in range(self.args.evaluate_episode_len):
                if self.args.render:
                    self.env.render()
                    time.sleep(0.05)
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
                            actions.append(np.array([0, np.random.rand(), 0, np.random.rand(), 0], dtype=np.float32))
                actions_dict = {agent_name: actions[agent_id] for agent_id, agent_name in enumerate(self.env.agents)}
                s_next, r, done, info = self.env.step(actions_dict)
                s_next, r = list(s_next.values()), list(r.values())
                for i in range(self.args.n_agents):
                    rewards += r[i]
                if self.args.train_adversaries:
                    for i in range(self.args.n_agents, self.args.n_players):
                        rewards_adv += r[i]
    
                s = s_next
            returns.append(rewards)
            if self.args.train_adversaries:
                returns_adv.append(rewards_adv)
                print('Returns is', rewards, " Adversary return is", rewards_adv)
            else:
                print('Returns is', rewards)

            avg_returns, avg_returns_adv = sum(returns) / self.args.evaluate_episodes, sum(returns_adv) / self.args.evaluate_episodes

        return avg_returns, avg_returns_adv





def run(queue, args, id):
    args.gpu = torch.cuda.is_available()
    print("Using GPU:", args.gpu)

    env, args = make_env(args)
    worker_id = id
    args.worker_id = id
    noise = args.noise_rate
    epsilon = args.epsilon
    episode_limit = args.max_episode_len
    buffer = Buffer(args)
    
    save_path = args.save_dir + '/' + args.scenario_name + "/worker_" + str(id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # initalize agents on local
    agents = []
    if args.train_adversaries:
        for i in range(args.n_players):
            agent = Agent(i, args)
            agents.append(agent)
    else:
        for i in range(args.n_agents):
            agent = Agent(i, args)
            agents.append(agent)


    returns = []
    returns_adv = []
    for time_step in tqdm(range(args.time_steps)):
        # reset the environment
        if time_step % episode_limit == 0:
            s = list(env.reset().values())

        # train adversaries
        if args.train_adversaries:
            # actions contains actions of all agents in the environment
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(s[agent_id], noise, epsilon)
                    actions.append(action)
            actions_dict = {agent_name: actions[agent_id] for agent_id, agent_name in enumerate(env.agents)}
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
                            o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32).cuda()
                        else:
                            o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32)
                        u_next.append(agents[agent_id].policy.actor_target_network(o_next))

                for agent_id, agent in enumerate(agents):
                    agent.learn(transitions, u_next)
                
        # Random adversary
        else:
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(s[agent_id], noise, epsilon)
                    actions.append(action)
            for i in range(args.n_agents, args.n_players):
                # actions.append(np.array([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]))
                actions.append(np.array([0, np.random.rand(), 0, np.random.rand(), 0], dtype=np.float32))
            actions_dict = {agent_name: actions[agent_id] for agent_id, agent_name in enumerate(env.agents)}
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
                            o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32).cuda()
                        else:
                            o_next = torch.tensor(transitions['o_next_%d' % agent_id], dtype=torch.float32)
                        action_next = agents[agent_id].policy.actor_target_network(o_next)
                        u_next.append(action_next)
                    for i in range(args.n_agents, args.n_players):
                        action_next = []
                        for _ in range(args.batch_size):
                            # action_next.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                            action_next.append([0, np.random.rand(), 0, np.random.rand(), 0])
                        if args.use_gpu and args.gpu:
                            action_next = torch.tensor(action_next, dtype=torch.float32).cuda()
                        else:
                            action_next = torch.tensor(action_next, dtype=torch.float32)
                        u_next.append(action_next)
                for agent in agents:
                    agent.learn(transitions, u_next)

        if time_step > 0 and time_step % args.evaluate_rate == 0:
            # Evaluate
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
                                actions.append(np.array([0, np.random.rand(), 0, np.random.rand(), 0], dtype=np.float32))
                    actions_dict = {agent_name: actions[agent_id] for agent_id, agent_name in enumerate(env.agents)}
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
                    print('Returns is', rewards, " Adversary return is", rewards_adv)
                else:
                    print('Returns is', rewards)

                return_agent, return_agent_adv = sum(evaluate_returns) / args.evaluate_episodes, sum(evaluate_returns_adv) / args.evaluate_episodes

            # Send to central queue
            actor_data, critic_data = [], []
            for agent_id, agent in enumerate(agents):
                actor_data = []
                for param in agent.policy.actor_network.parameters():
                    actor_data.append(param.data.tolist())
                critic_data = []
                for param in agent.policy.critic_network.parameters():
                    critic_data.append(param.data.tolist())

            queue.put((worker_id, return_agent, actor_data, critic_data))

            s = list(env.reset().values())
            returns.append(return_agent)
            plt.figure()
            plt.plot(range(len(returns)), returns, label="Agent")
            if args.train_adversaries:
                returns_adv.append(return_adv)
                plt.plot(range(len(returns_adv)), returns_adv, label="Adversary")
                np.save(save_path + '/returns_adv.pkl', returns_adv)
            plt.xlabel('episode * ' + str(args.evaluate_rate / episode_limit))
            plt.ylabel('average returns')
            plt.legend()
            plt.savefig(save_path + '/plt.png', format='png')
            np.save(save_path + '/returns.pkl', returns)
            

        noise = max(0.05, noise - 0.0000005)
        epsilon = max(0.05, epsilon - 0.0000005)