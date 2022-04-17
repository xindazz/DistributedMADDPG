from runner_mp import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import multiprocessing


if __name__ == '__main__':
    # get the params
    print("Started main")
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
