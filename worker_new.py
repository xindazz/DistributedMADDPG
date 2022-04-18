import celery
import numpy as np
from copy import deepcopy
import json
import random
import matplotlib.pyplot as plt
import os
import torch
import pickle

from common.utils import make_env
from runner_new import Runner
from master_new import master_app


# run this code with "celery -A worker worker --loglevel=info --concurrency=1" on the 4 worker machines


# Make sure that the 'myguest' user exists with 'myguestpwd' on the RabbitMQ server and your load balancer has been set up correctly.
# My load balancer address is'RabbitMQLB-8e09cd48a60c9a1e.elb.us-east-2.amazonaws.com'.
# Below you will need to change it to your load balancer's address.

app = celery.Celery(
    "worker",
    broker="amqp://myguest:myguestpwd@RabbitMQLB-5103314cb3c8cc94.elb.us-east-2.amazonaws.com",
    backend="rpc://myguest:myguestpwd@RabbitMQLB-5103314cb3c8cc94.elb.us-east-2.amazonaws.com",
)

worker_id = None
args = None
runner = None

@app.task
def init(**kwargs):
    global worker_id, args, runner

    worker_id = kwargs["id"]
    myargs = kwargs["args"]

    print("Worker id", worker_id, "received args:", myargs)

    args = parse_args(myargs)
    env, args = make_env(args)
    runner = Runner(args, env)
    runner.run()


@app.task
def get_avg_reward(**kwargs):
    global runner

    returns, returns_adv = runner.evaluate()

    actor_data, critic_data = [], []
    for agent_id, agent in enumerate(runner.agents):
        actor_data = []
        for param in agent.policy.actor_network.parameters():
            actor_data.append(param.data.tolist())
        critic_data = []
        for param in agent.policy.critic_network.parameters():
            critic_data.append(param.data.tolist())

    return json.dumps({"avg_reward": returns, "actor_data": actor_data, "critic_data": critic_data})


@app.task
def update_target_networks(**kwargs):
    best_actor_target = kwargs["best_actor_target"]
    best_critic_target = kwargs["best_critic_target"]
    
    actor = []
    critic = []
    for agent_id, agent in enumerate(runner.agents):
        target_actor_params = []
        for actor_data in best_actor_target:
            target_actor_params.append(torch.tensor(actor_data, dtype=torch.float32))
        actor.append(target_actor_params)
        
        target_critic_params = []
        for critic_data in best_critic_target:
            target_critic_params.append(torch.tensor(critic_data, dtype=torch.float32))
        critic.append(target_critic_params)


    for agent_id, agent in enumerate(runner.agents):
        for param, target_actor_param in agent.policy.actor_target_network.parameters(), actor[agent_id]:
            param.data.copy_(target_actor_param)
        for param, target_critic_param in agent.policy.critic_target_network.parameters(), critic[agent_id]:
            param.data.copy_(target_critic_param)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class Arguments:
    def __init__(
        self,
        scenario_name,
        max_episode_len,
        time_steps,
        num_adversaries,
        lr_actor,
        lr_critic,
        epsilon,
        noise_rate,
        gamma,
        tau,
        buffer_size,
        batch_size,
        save_dir,
        save_rate,
        model_dir,
        evaluate_episodes,
        evaluate_episode_len,
        evaluate,
        evaluate_rate,
        render,
    ):
        self.scenario_name = scenario_name
        self.max_episode_len = max_episode_len
        self.time_steps = time_steps
        self.num_adversaries = num_adversaries
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.epsilon = epsilon
        self.noise_rate = noise_rate
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.save_rate = save_rate
        self.model_dir = model_dir
        self.evaluate_episodes = evaluate_episodes
        self.evaluate_episode_len = evaluate_episode_len
        self.evaluate = evaluate
        self.evaluate_rate = evaluate_rate
        self.render = render


def parse_args(myargs):
    scenario_name = myargs["scenario_name"]
    max_episode_len = myargs["max_episode_len"]
    time_steps = myargs["time_steps"]
    num_adversaries = myargs["num_adversaries"]
    lr_actor = myargs["lr_actor"]
    lr_critic = myargs["lr_critic"]
    epsilon = myargs["epsilon"]
    noise_rate = myargs["noise_rate"]
    gamma = myargs["gamma"]
    tau = myargs["tau"]
    buffer_size = myargs["buffer_size"]
    batch_size = myargs["batch_size"]
    save_dir = myargs["save_dir"]
    save_rate = myargs["save_rate"]
    model_dir = myargs["model_dir"]
    evaluate_episodes = myargs["evaluate_episodes"]
    evaluate_episodes_len = myargs["evaluate_episode_len"]
    evaluate = myargs["evaluate"]
    evaluate_rate = myargs["evaluate_rate"]
    render = myargs["render"]

    args = Arguments(
        scenario_name,
        max_episode_len,
        time_steps,
        num_adversaries,
        lr_actor,
        lr_critic,
        epsilon,
        noise_rate,
        gamma,
        tau,
        buffer_size,
        batch_size,
        save_dir,
        save_rate,
        model_dir,
        evaluate_episodes,
        evaluate_episodes_len,
        evaluate,
        evaluate_rate,
        render,
    )

    return args