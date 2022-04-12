import celery
import numpy as np
from copy import deepcopy
import json
import random
import matplotlib.pyplot as plt
import os
import torch

from agent_distributed import Agent
from common.utils import make_env


# run this code with "celery -A worker worker --loglevel=info --concurrency=1 --queues=q2" on the 4 worker machines


# Make sure that the 'myguest' user exists with 'myguestpwd' on the RabbitMQ server and your load balancer has been set up correctly.
# My load balancer address is'RabbitMQLB-8e09cd48a60c9a1e.elb.us-east-2.amazonaws.com'.
# Below you will need to change it to your load balancer's address.

app = celery.Celery(
    "worker",
    broker="amqp://myguest:myguestpwd@RabbitMQLB-5103314cb3c8cc94.elb.us-east-2.amazonaws.com",
    backend="rpc://myguest:myguestpwd@RabbitMQLB-5103314cb3c8cc94.elb.us-east-2.amazonaws.com",
)

agent_id = None
args = None
agent = None
env = None
time_step = None
save_path = None


@app.task
def init_agent(**kwargs):
    global agent_id, args, agent, time_step, save_path

    agent_id = kwargs["agent_id"]
    myargs = kwargs["args"]
    print("Agent", agent_id, "received args:", myargs)

    args = parse_args(myargs)
    _, args = make_env(args)

    # initialize agent
    agent = Agent(agent_id, args)

    time_step = 0
    save_path = args.save_dir + "/" + args.scenario_name

    return "Successfully initialized agent " + agent_id + "."


@app.task
def get_action(**kwargs):
    global agent_id, args, agent, time_step, save_path

    agent_id = kwargs["agent_id"]
    myargs = kwargs["args"]
    s = np.asarray(kwargs["s"])
    evaluate = kwargs["evaluate"]
    print("Agent", agent_id, "received state:")

    if agent == None:
        args = parse_args(myargs)
        _, args = make_env(args)
        agent = Agent(agent_id, args)

    with torch.no_grad():
        if evaluate:
            action = agent.select_action(s, 0, 0)
        else:
            action = agent.select_action(s, args.noise_rate, args.epsilon)

        return json.dumps({"action": action}, cls=NumpyEncoder)


@app.task
def get_target_next_action(**kwargs):
    global agent_id, args, agent, time_step, save_path

    agent_id = kwargs["agent_id"]
    myargs = kwargs["args"]
    s = np.asarray(kwargs["s"])
    print("Agent", agent_id, "received next state:")

    if agent == None:
        args = parse_args(myargs)
        _, args = make_env(args)
        agent = Agent(agent_id, args)

    with torch.no_grad():
        action = agent.policy.actor_target_network(torch.tensor(s, dtype=torch.float))

        return json.dumps({"action": action.numpy()}, cls=NumpyEncoder)


@app.task
def train(**kwargs):
    global agent_id, args, agent, time_step, save_path

    agent_id = kwargs["agent_id"]
    myargs = kwargs["args"]
    transitions = kwargs["transitions"]
    u_next = np.asarray(kwargs["u_next"])
    print("Agent", agent_id, "received transitions and u_next", transitions, u_next)

    if agent == None:
        args = parse_args(myargs)
        _, args = make_env(args)
        agent = Agent(agent_id, args)

    # Transform u_next into list of tensors
    u_next_tensors = []
    for a in u_next:
        u_next_tensors.append(torch.tensor(a, dtype=torch.float))

    agent.learn(transitions, u_next_tensors)

    time_step += 1

    return "One time step finished"


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