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
    # from multiagent.environment import MultiAgentEnv
    # import multiagent.scenarios as scenarios

    # # load scenario from script
    # scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # # create world
    # world = scenario.make_world()
    # # create multiagent environment
    # env = MultiAgentEnv(
    #     world, scenario.reset_world, scenario.reward, scenario.observation
    # )
    # # env = MultiAgentEnv(world)
    # args.n_players = env.n  # 包含敌人的所有玩家个数
    # args.n_agents = env.n - args.num_adversaries  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法

    # # include obs_shape for both agents and adversaries
    # args.obs_shape = [
    #     env.observation_space[i].shape[0] for i in range(args.n_players)
    # ]  # 每一维代表该agent的obs维度

    # # include action_shape for both agents and adversaties
    # args.action_shape = [content.n for content in env.action_space]

    # # for content in env.action_space:
    # #     action_shape.append(content.n)

    # # args.action_shape = action_shape
    # # args.action_shape = action_shape[: args.n_agents]  # 每一维代表该agent的act维度
    # args.high_action = 1
    # args.low_action = -1

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
