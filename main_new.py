# from runner import Runner
import numpy as np
import random
import torch
import multiprocessing as mp
import os

from common.arguments import get_args
from common.utils import make_env
from maddpg.actor_critic import Actor, Critic
from runner_new import Runner, run



NUM_WORKERS = 4
MAX_STEPS = 100000

if __name__ == '__main__':
    # get the params
    args = get_args()
    _, args = make_env(args)
    # env, args = make_env(args)

    # runners = []
    # for i in range(NUM_WORKERS):
    #     runner = Runner(args, i)
    #     runners.append(runner)

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    q = mp.Queue(maxsize=NUM_WORKERS)
    processes = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=run, args=(q, args, i))
        p.start()
        processes.append(p)

    steps = 0
    while True:
        if steps >= MAX_STEPS:
            break

        if q.full():
            avg_rewards = [None for _ in range(NUM_WORKERS)]
            actors = [None for _ in range(NUM_WORKERS)]
            critics = [None for _ in range(NUM_WORKERS)]
            for i in range(NUM_WORKERS):
                worker_id, avg_reward, actor, critic = q.get()
                avg_rewards[worker_id] = avg_reward
                actors[worker_id] = actor
                critics[worker_id] = critic

            avg_rewards = np.array(avg_rewards)
            best_worker = np.argmin(avg_rewards)
            print("----------------------Got best worker", best_worker, "with reward", np.min(avg_rewards))

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

                for param, target_actor_param in zip(actor_network.parameters(), actor[agent_id]):
                    param.data.copy_(target_actor_param)
                for param, target_critic_param in zip(critic_network.parameters(), critic[agent_id]):
                    param.data.copy_(target_critic_param)

                for worker_id in range(NUM_WORKERS):
                    model_path = args.save_dir + '/' + args.scenario_name + "/worker_" + str(worker_id) + "/"
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    model_path = os.path.join(model_path, 'agent_%d' % agent_id)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(actor_network.state_dict(), model_path + '/temp_actor_target_params.pkl')
                    torch.save(critic_network.state_dict(),  model_path + '/temp_critic_target_params.pkl')
            print("---------------------Successfully saved networks")

    for p in processes:
        p.join()


    # if args.evaluate:
    #     returns = runner.evaluate()
    #     print('Average returns is', returns)
    # else:
    #     runner.run()
