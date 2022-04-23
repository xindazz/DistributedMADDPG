import argparse
from xmlrpc.client import boolean

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--train-adversaries", type=bool, default=False, help="whether to train adversaries or perform random actions")
    parser.add_argument("--adversary-alg", type=str, default="MADDPG", help="adversary's algorithm")

    # GPU
    parser.add_argument("--use-gpu", type=bool, default=False, help="use gpu or not")

    # MP
    parser.add_argument("--mp", type=bool, default=True, help="do multiprocessing or not")

    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-3, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-4, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--soft-update-rate", type=int, default=10, help="number of timesteps to update target network")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")
    parser.add_argument("--start-timestep", type=int, default=1000000, help="start timestep to help save model params")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=20, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=50, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--sync-target-rate", type=int, default=5000, help="how often to sync target network to that of best worker")
    
    # Render
    parser.add_argument("--render", type=bool, default=False, help="whether to render environment during evaluation")
    parser.add_argument("--save-render", type=bool, default=False, help="whether to save render results as gif")
    
    args = parser.parse_args()
    print(args)
    args.worker_id=0
    
    return args