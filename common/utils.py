import numpy as np
import inspect
import functools


def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(zip(argspec.args[-len(argspec.defaults) :], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):

    from pettingzoo.mpe import simple_tag_v2
    
    num_good = 1
    num_adversaries = 3
    env = simple_tag_v2.parallel_env(num_good=num_good, num_adversaries=num_adversaries, num_obstacles=2, max_cycles=25, continuous_actions=True)
    s, reward, done, info = env.reset()

    args.n_players = num_good + num_adversaries
    args.n_agents = num_adversaries
    args.obs_shape = [env.observation_spaces[agent_name].shape[0] for agent_name in env.agents]
    args.action_shape = [env.action_spaces[agent_name].shape[0] for agent_name in env.agents]
    args.high_action = 1
    args.low_action = 0

    return env, args
