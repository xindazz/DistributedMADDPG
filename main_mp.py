import numpy as np
import matplotlib.pyplot as plt
import torch
import multiprocessing as mp
import os
from tqdm import tqdm

from common.arguments import get_args
from common.utils import make_env, save_args_to_file

from worker_local import worker_loop


NUM_WORKERS = 4

if __name__ == "__main__":

    # create input and output queues
    input_queues = [mp.Queue() for _ in range(NUM_WORKERS)]
    output_queues = [mp.Queue() for _ in range(NUM_WORKERS)]

    # create workers (processes)
    processes = []
    for i in range(NUM_WORKERS):
        process = mp.Process(
            target=worker_loop,
            args=(
                input_queues[i],
                output_queues[i],
            ),
        )

        # This is critical! The consumer function has an infinite loop
        # Which means it will never exit unless we set daemon to true
        process.daemon = True
        processes.append(process)

    # start all workers
    for process in processes:
        process.start()

    # get the params
    args = get_args()
    _, args = make_env(args)
    save_args_to_file(args, args.save_dir)

    # send params to workers as initialization
    for i in range(NUM_WORKERS):
        params = ("init", args, i)
        input_queues[i].put(params)

    # wait for workers to finish initialization
    for i in range(NUM_WORKERS):
        output_queues[i].get()

    num_epochs = int(args.time_steps / args.evaluate_rate)
    # determine how many epochs to train
    for epoch in tqdm(range(num_epochs)):
        params = ("train",)
        # request worker to start training
        for i in range(NUM_WORKERS):
            input_queues[i].put(params)

        avg_rewards = [None for _ in range(NUM_WORKERS)]
        critics = [None for _ in range(NUM_WORKERS)]
        target_critics = [None for _ in range(NUM_WORKERS)]

        # wait for worker to send params
        if (epoch + 1) % (args.sync_target_rate // args.evaluate_rate) == 0:
            for i in range(NUM_WORKERS):
                worker_id, avg_reward, critic, target_critic = output_queues[i].get()
                avg_rewards[worker_id] = avg_reward
                critics[worker_id] = critic
                target_critics[worker_id] = target_critic

            # find the best model
            avg_rewards = np.array(avg_rewards)
            best_worker, best_reward = np.argmax(avg_rewards), np.max(avg_rewards)
            print(
                "----------------------Got best worker",
                best_worker,
                "with reward",
                np.max(avg_rewards),
            )

            best_critic = critics[best_worker]
            best_target_ciritc = target_critics[best_worker]

            params = (
                "update_model",
                best_critic,
                best_target_ciritc,
            )
            # send models back to workers to update
            for i in range(NUM_WORKERS):
                input_queues[i].put(params)

            # wait for workers to finish updating model
            for i in range(NUM_WORKERS):
                output_queues[i].get()
        else:
            # wait for workers to finish a evaluation loop
            for i in range(NUM_WORKERS):
                output_queues[i].get()

    # shut down all workers
    args = ("close",)
    for i in range(NUM_WORKERS):
        input_queues[i].put(args)

    for p in processes:
        p.join()