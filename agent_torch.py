from agent import Agent

import numpy as np
import cPickle as pickle
from gomoku import GomokuEnv
from datetime import datetime
import torch.optim as optim
import torch_rl.environments
import torch_rl.learners as learners
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times
from torch_rl.policies import DiscreteModelPolicy
from torch_model import PGModel
import torch

def mapping_function(I):
    I = np.subtract(I[0, :, :], I[1, :, :])
    return torch.Tensor(I.astype(float).ravel())


class TorchAgent(Agent):
    def __init__(self, board_size=9):
        Agent.__init__(self, board_size)

    def learn(self, learning_rate=0.01, stdv=0.01):
        self.log("start learning with torch policy gradient - " + str(datetime.now()))

        env = self.create_env()

        mapped_env = torch_rl.environments.MappedEnv(env, mapping_function)

        #Creation of the policy
        A = env.action_space.n
        print("Number of Actions is: %d" % A)
        model = PGModel(81, 200, A, stdv)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #policy=DiscreteModelPolicy(env.action_space,model)
        learning_algorithm = learners.LearnerPolicyGradient(action_space=env.action_space, average_reward_window=10,
                                                            torch_model=model, optimizer=optimizer)
        learning_algorithm.reset()
        while True:
            learning_algorithm.step(env=mapped_env, discount_factor=0.9, maximum_episode_length=100)

            policy = learning_algorithm.get_policy(stochastic=True)
            r = rl_evaluate_policy_multiple_times(mapped_env, policy, 100, 1.0, 10)
            print("Evaluation avg reward = %f " % r)

            print("Reward = %f" % learning_algorithm.log.get_last_dynamic_value("total_reward"))
