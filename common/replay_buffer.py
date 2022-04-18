import threading
import numpy as np


class Buffer:
    def __init__(self, args, is_adversary):
        self.args = args
        self.size = args.buffer_size
        # self.is_adversary = is_adversary

        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        # if is_adversary:
        #     self.agent_ids = range(self.args.n_agents, self.args.n_players)
        # else:
            # self.agent_ids = range(self.args.n_agents)
        self.agent_ids = range(self.args.n_players)

        for i in self.agent_ids:
            self.buffer["o_%d" % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer["u_%d" % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer["r_%d" % i] = np.empty([self.size])
            self.buffer["o_next_%d" % i] = np.empty([self.size, self.args.obs_shape[i]])
       
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in self.agent_ids:
            # if self.is_adversary:
            #     idx = i - self.args.n_agents
            # else:
            #     idx = i
            idx = i
                
            with self.lock:
                self.buffer["o_%d" % i][idxs] = o[idx]
                self.buffer["u_%d" % i][idxs] = u[idx]
                self.buffer["r_%d" % i][idxs] = r[idx]
                self.buffer["o_next_%d" % i][idxs] = o_next[idx]

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
