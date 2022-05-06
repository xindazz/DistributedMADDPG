# Distributed Genetic MADDPG

This is a distributed implementation of Multi-agent Deep Deterministic Policy Gradient taking ideas from genetic algorithms. Read more on implementation details and experiment results in project.pdf. 

The original paper of MADDPG can be found here: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275). Pytorch code referenced from (https://github.com/starry-sky6688/MADDPG).

## Requirements

- python=3.6.5
- torch=1.1.0
- pettingzoo[mpe]

## Quick Start 

To run Distributed Genetic MADDPG using 4 multi-processing workers:
```shell
$ python main_mp.py
```

To run default MADDPG:
```shell
$ python main.py
```

Read more about specifiable command line arguments in common/arguments.py.
