from tqdm import tqdm
from copy import deepcopy
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import celery
import time

from common.arguments import get_args
from common.utils import make_env
from maddpg.actor_critic import Actor, Critic
from worker_new import app, NumpyEncoder


NUM_WORKERS = 4
MAX_STEPS = 10000

def init():
    # initialize args
    args = get_args()
    env, args = make_env(args)

    # initialize workers
    tasks = []
    for i in range(NUM_WORKERS):
        data = {"id": i, "args": vars(args)}
        task = app.send_task("worker_new.init", queue="q" + str(i), kwargs=data)
        tasks.append(task)
    print("Initialized workers")
    
    
    time_step = 0
    while True:
        if time_step >= MAX_STEPS:
            break

        time.sleep(60)
        tasks = []
        for i in range(NUM_WORKERS):
            data = {"id": i, "args": vars(args)}
            task = app.send_task("worker_new.get_avg_reward", queue="q" + str(i), kwargs=data)
            tasks.append(task)
        print("Sent to all workers task Get_avg_reward")

        avg_rewards = []
        actors = []
        critics = []
        for task in tasks:
            result = json.loads(task.get())
            avg_reward, actor, critic = result["avg_reward"], result["actor_data"], result["critic_data"]
            avg_rewards.append(avg_reward)
            actors.append(actor)
            critics.append(critic)
        print("Got back result from all workers for task Get_avg_reward")

        avg_rewards = np.array(avg_rewards)
        best_worker = np.argmin(avg_rewards)

        tasks = []
        for i in range(NUM_WORKERS):
            data = {"id": i, "args": vars(args), "best_actor_target": actors[best_worker], "best_critic_target": critics[best_worker]}
            task = app.send_task("worker_new.update_target_networks", queue="q" + str(i), kwargs=data, cls=NumpyEncoder)
            tasks.append(task)
        print("Sent to all workers task Update_target_networks")

        for agent_id in range(args.n_agents):
            actor_network = Actor(args, agent_id)
            critic_network = Critic(args, agent_id)

        actor = []
        critic = []
        for agent_id in range(args.n_agents):
            target_actor_params = []
            for actor_data in actors[best_worker]:
                target_actor_params.append(torch.tensor(actor_data, dtype=torch.float32))
            actor.append(target_actor_params)
            
            target_critic_params = []
            for critic_data in critics[best_worker]:
                target_critic_params.append(torch.tensor(critic_data, dtype=torch.float32))
            critic.append(target_critic_params)


        for agent_id in range(args.n_agents):
            actor_network = Actor(args, agent_id)
            critic_network = Critic(args, agent_id)

            for param, target_actor_param in actor_network.parameters(), actor[agent_id]:
                param.data.copy_(target_actor_param)
            for param, target_critic_param in critic_network.parameters(), critic[agent_id]:
                param.data.copy_(target_critic_param)

            model_path = os.path.join(args.save_dir, args.scenario_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, 'agent_%d' % agent_id)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(actor_network.state_dict(), model_path + '/'  + 'actor_params.pkl')
            torch.save(critic_network.state_dict(),  model_path + '/' + 'critic_params.pkl')
        print("Successfully saved networks")
                

init()