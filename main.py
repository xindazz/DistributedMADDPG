# from runner import Runner
from runner import Runner
from common.arguments import get_args
from common.utils import make_env, save_args_to_file
import numpy as np
import random
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    save_args_to_file(args, args.save_dir)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
