import torch
import os
from maddpg.actor_critic import Actor, Critic


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args, agent_id)

        # build up the target network
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args, agent_id)

        if self.args.use_gpu and self.args.gpu:
            fn = lambda x: x.cuda()
            self.actor_network = fn(self.actor_network)
            self.critic_network = fn(self.critic_network)
            self.actor_target_network = fn(self.actor_target_network)
            self.critic_target_network = fn(self.critic_target_network)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=self.args.lr_actor
        )
        self.critic_optim = torch.optim.Adam(
            self.critic_network.parameters(), lr=self.args.lr_critic
        )

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)

        # path to save the model
        print("USING MP", self.args.mp)
        if self.args.mp == False:
            self.model_path = self.args.save_dir + "/" + self.args.scenario_name
        else:
            self.model_path = (
                self.args.save_dir
                + "/"
                + self.args.scenario_name
                + "/worker_"
                + str(self.args.worker_id)
            )
        if not os.path.exists(self.model_path + "/"):
            os.makedirs(self.model_path + "/")
        self.model_path = self.model_path + "/" + "agent_%d" % agent_id
        if not os.path.exists(self.model_path + "/"):
            os.makedirs(self.model_path + "/")

        # Load model
        if os.path.exists(self.model_path + "/" + str(self.args.load_num) + "_actor_params.pkl"):
            self.actor_network.load_state_dict(
                torch.load(self.model_path + "/" + str(self.args.load_num) + "_actor_params.pkl")
            )
            self.critic_network.load_state_dict(
                torch.load(self.model_path + "/" + str(self.args.load_num) + "_critic_params.pkl")
            )
            print("Agent {} successfully loaded actor_network: {}".format(
                    self.agent_id, self.model_path + "/" + str(self.args.load_num) + "_actor_params.pkl"
            ))
            print("Agent {} successfully loaded critic_network: {}".format(
                    self.agent_id, self.model_path + "/" + str(self.args.load_num) + "_critic_params.pkl"
            ))

        elif os.path.exists(self.model_path + "/actor_params.pkl"):
            self.actor_network.load_state_dict(
                torch.load(self.model_path + "/actor_params.pkl")
            )
            self.critic_network.load_state_dict(
                torch.load(self.model_path + "/critic_params.pkl")
            )
            print("Agent {} successfully loaded actor_network: {}".format(
                    self.agent_id, self.model_path + "/actor_params.pkl"
            ))
            print("Agent {} successfully loaded critic_network: {}".format(
                    self.agent_id, self.model_path + "/critic_params.pkl"
            ))
     

    # soft update
    def _soft_update_target_network(self):
        # for param in self.actor_network.parameters():
        #     print("Actor params: ", param.data.shape)
        # print("Done")
        for target_param, param in zip(
            self.actor_target_network.parameters(), self.actor_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * param.data
            )

        for target_param, param in zip(
            self.critic_target_network.parameters(), self.critic_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * param.data
            )

    # update the network
    def train(self, transitions, u_next):
        # self.check_load_temp_target_model()

        for key in transitions.keys():
            if self.args.use_gpu and self.args.gpu:
                if (torch.is_tensor(transitions[key]) == False):
                    transitions[key] = torch.from_numpy(transitions[key]).type(torch.float32).cuda()
                else:
                    transitions[key] = transitions[key].cuda()
            else:
                if (torch.is_tensor(transitions[key]) == False):
                    transitions[key] = torch.from_numpy(transitions[key]).type(torch.float32)
                
        r = transitions["r_%d" % self.agent_id]
        o, u, o_next = [], [], []
        for agent_id in range(self.args.n_players):
            o.append(transitions["o_%d" % agent_id])
            u.append(transitions["u_%d" % agent_id])
            o_next.append(transitions["o_next_%d" % agent_id])

        # If is adversary and algorithm is DDPG, state and action only of adversary
        if self.agent_id >= self.args.n_agents and self.args.adversary_alg == "DDPG":
            o, u, o_next, u_next = (
                o[self.args.n_agents :],
                u[self.args.n_agents :],
                o_next[self.args.n_agents :],
                u_next[self.args.n_agents :],
            )

        # calculate the target Q value function
        with torch.no_grad():
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变

        if self.agent_id >= self.args.n_agents and self.args.adversary_alg == "DDPG":
            u[self.agent_id - self.args.n_agents] = self.actor_network(
                o[self.agent_id - self.args.n_agents]
            )
        else:
            u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = -self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        # print('agent {}: critic_loss is {}, actor_loss is {}'.format(self.agent_id, critic_loss, actor_loss))
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.train_step > 0 and self.train_step % self.args.soft_update_rate == 0:
            self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)

        self.train_step += 1
        

        # self.save_temp_model()


    def save_model(self, train_step):
        num = str((train_step + self.args.start_timestep) // self.args.save_rate)
        torch.save(
            self.actor_network.state_dict(),
            self.model_path + "/" + num + "_actor_params.pkl",
        )
        torch.save(
            self.critic_network.state_dict(),
            self.model_path + "/" + num + "_critic_params.pkl",
        )


    def save_temp_model(self):
        torch.save(
            self.actor_network.state_dict(), 
            self.model_path + "/temp_actor_params.pkl"
        )
        torch.save(
            self.critic_network.state_dict(),
            self.model_path + "/temp_critic_params.pkl",
        )


    def check_load_temp_target_model(self):
        if os.path.exists(self.model_path + "/temp_actor_target_params.pkl"):
            self.actor_target_network.load_state_dict(
                torch.load(self.model_path + "/temp_actor_target_params.pkl")
            )
            self.critic_target_network.load_state_dict(
                torch.load(self.model_path + "/temp_critic_target_params.pkl")
            )
            print(
                "Agent {} successfully loaded {}".format(
                    self.agent_id, self.model_path + "/temp_actor_target_params.pkl"
                )
            )
            print(
                "Agent {} successfully loaded {}".format(
                    self.agent_id, self.model_path + "/temp_critic_target_params.pkl"
                )
            )

            # Remove so that won't load every time step
            os.remove(self.model_path + "/temp_actor_target_params.pkl")
            os.remove(self.model_path + "/temp_critic_target_params.pkl")
