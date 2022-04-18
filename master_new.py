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
from worker_new import app, NumpyEncoder


master_app = celery.Celery(
    "worker",
    broker="amqp://myguest:myguestpwd@RabbitMQLB-5103314cb3c8cc94.elb.us-east-2.amazonaws.com",
    backend="rpc://myguest:myguestpwd@RabbitMQLB-5103314cb3c8cc94.elb.us-east-2.amazonaws.com",
)

NUM_WORKERS = 4

def init():
    # initialize args
    args = get_args()

    # initialize workers
    tasks = []
    for i in range(NUM_WORKERS):
        task = app.send_task("worker.init", queue="q" + str(i), kwargs={"id": i, "args": vars(args)}, cls=NumpyEncoder)
        tasks.append(task)
    
    while True:
        time.sleep(60)
        tasks = []
        for i in range(NUM_WORKERS):
            task = app.send_task("worker.get_avg_reward", queue="q" + str(i), kwargs={}, cls=NumpyEncoder)
            tasks.append(task)
        avg_rewards = []
        actor_targets = []
        critic_targets = []
        for task in tasks:
            avg_reward, actor_target, critic_target = task.get()
            avg_rewards.append(avg_reward)
            actor_targets.append(actor_target)
            critic_targets.append(critic_target)

        avg_rewards = np.array(avg_rewards)
        best_worker = np.argmin(avg_rewards)

        tasks = []
        for i in range(NUM_WORKERS):
            data = {"best_actor_target": actor_targets[best_worker], "best_critic_target": critic_targets[best_worker]}
            task = app.send_task("worker.update_target_networks", queue="q" + str(i), kwargs=data, cls=NumpyEncoder)
            tasks.append(task)