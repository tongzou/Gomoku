import numpy as np
import cPickle as pickle
import gym
from gomoku import GomokuEnv
import random
from datetime import datetime
import os
import torch.optim as optim
import torch_rl.environments
import torch_rl.learners as learners
from torch_rl.tools import rl_evaluate_policy, rl_evaluate_policy_multiple_times
from torch_rl.policies import DiscreteModelPolicy
import opponent
from torch_model import PGModel
import torch

gym.envs.registration.register(
    id='Gomoku9x9-v0',
    entry_point='gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        #'opponent': opponent.make_opponent_policy(9, 0.001),
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    }
)

gym.envs.registration.register(
    id='Gomoku15x15-v0',
    entry_point='gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        #'opponent': opponent.make_opponent_policy(15, 1),
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 15,
    }
)

def cls():
    os.system('cls')  # For Windows
    os.system('clear')  # For Linux/OS X


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def prepro(I, color=GomokuEnv.BLACK):
    """ prepro N x N x 3 uint8 Gomoku board into N x N 1D float vector """
    #I = np.subtract(I[0, :, :], I[1, :, :])
    if color == GomokuEnv.BLACK:
        I = np.subtract(I[0, :, :], I[1, :, :])
    else:
        I = np.subtract(I[1, :, :], I[0, :, :])
    return I.astype(np.float).ravel()


def mapping_function(I):
    I = np.subtract(I[0, :, :], I[1, :, :])
    return torch.Tensor(I.astype(float).ravel())


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = softmax(logp)
    return p, h


def policy_backward(model, epx, eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).T
    dh = np.dot(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


class Agent:
    '''
        hidden:  # number of hidden layer neurons
    '''
    def __init__(self, board_size=9, hidden=200, resume=True):
        self.N = board_size
        self.H = hidden
        self.D = self.N * self.N
        self.file_name = 'save' + str(self.N) + '.p'

        self.model = None
        self.logger = None

        # load the saved model first.
        if resume:
            try:
                self.log('using saved model.')
                self.model = pickle.load(open('save' + str(self.N) + '.p', 'rb'))
            except Exception as e:
                self.log('no saved model.')
                self.model = None

    def log(self, message):
        if self.logger is None:
            self.logger = open("log.txt", 'ab')
        self.logger.write(message + '\n')
        self.logger.flush()

    def create_env(self, color=GomokuEnv.BLACK):
        if self.N == 15:
            env = gym.make("Gomoku15x15-v0")
        else:
            env = gym.make("Gomoku9x9-v0")

        env.player_color = color
        return env

    # return a random model
    def get_random_model(self):
        return {'W1': np.random.randn(self.H, self.D) / np.sqrt(self.D),
                'W2': np.random.randn(self.D + 1, self.H) / np.sqrt(self.H)}


    def get_opponent_policy(self, model, color):
        if model is None:
            # try to load it from file.
            try:
                model = pickle.load(open('opponent' + str(self.N) + '.p', 'rb'))
                self.log('using saved opponent model:\n' + str(model))
            except Exception as e:
                model = self.get_random_model()
                self.log('using random opponent model:\n' + str(model))
        else:
            self.log('using current opponent model:\n' + str(model))
            model = model.copy()

        def opponent_policy(curr_state, prev_state, prev_action):
            action, _, _, _ = self.choose_move(curr_state, model, color)
            return action

        return opponent_policy


    '''
        Choose a move based on the current board, trained model and player color
    '''
    def choose_move(self, observation, model, color):
        # preprocess the observation
        x = prepro(observation, color)

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(model, x)
        possible_moves = GomokuEnv.get_possible_actions(observation)
        if len(possible_moves) == 0:
            return self.D, x, aprob, h  # resign

        mask = np.zeros(self.D + 1, dtype=int)
        mask[possible_moves] = 1
        newprob = np.multiply(aprob, mask)
        max = newprob.max()
        if max == 0:
            self.log('all probabilies are zero!!!!')
            action = random.choice(possible_moves)
        else:
            action = np.random.choice(np.where(newprob == max)[0])
        #aprob = np.multiply(aprob, mask)
        #action = np.random.choice(np.where(aprob == aprob.max())[0])
        return action, x, aprob, h

    '''
        batch_size:  # every how many episodes to do a param update?
        learning_rate: 1e-4
        gamma:  # discount factor for reward
        decay_rate:  # decay factor for RMSProp leaky sum of grad^2
    '''
    def learn(self, render=False, model_threshold=0.8, min_episodes=100, batch_size=10, learning_rate=1e-4, gamma=0.99, decay_rate=0.99):
        # setup logging
        self.log("start learning - " + str(datetime.now()))

        env = self.create_env()
        observation = env.reset()
        xs, hs, dlogps, drs = [], [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0

        if self.model is None:
            self.model = self.get_random_model()

        # initialize component for self play.
        env.opponent_policy = self.get_opponent_policy(None, GomokuEnv.WHITE)
        opponent_episode_number = 0

        grad_buffer = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  # update buffers that add up gradients over a batch
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  # rmsprop memory

        while True:
            if render:
                env.render()

            action, x, aprob, h = self.choose_move(observation, self.model, GomokuEnv.BLACK)

            # record various intermediates (needed later for backprop)
            xs.append(x)  # observation
            hs.append(h)  # hidden state

            y = np.zeros(aprob.size)
            y[action] = 1
            dlogps.append(np.subtract(y, aprob))  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

            # step the environment and get new measurements
            observation, reward, done, info = env.step(action)
            reward_sum += reward

            drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

            if done:  # an episode finished
                if render:
                    env.render()
                episode_number += 1
                opponent_episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)

                xs, hs, dlogps, drs = [], [], [], []  # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr, gamma)

                # standardize the rewards to be unit normal
                discounted_epr = np.subtract(discounted_epr, np.mean(discounted_epr))
                discounted_epr = np.divide(discounted_epr, np.std(discounted_epr))

                epdlogp = np.multiply(epdlogp, discounted_epr)  # modulate the gradient with advantage (PG magic happens right here.)
                grad = policy_backward(self.model, epx, eph, epdlogp)
                for k in self.model:
                    grad_buffer[k] += grad[k]  # accumulate grad over batch

                # perform rmsprop parameter update every batch_size episodes
                if episode_number % batch_size == 0:
                    for k, v in self.model.iteritems():
                        g = grad_buffer[k] # gradient
                        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                        self.model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)

                # replace the opponent model once our running_reward is over the threshold and min_episodes is met
                if running_reward > model_threshold and opponent_episode_number > min_episodes:
                    self.log('replace opponent model now: ep ' + str(episode_number) + '\n' +
                              'running mean: ' + str(running_reward))
                    # save the opponent model
                    pickle.dump(self.model, open('opponent' + str(self.N) + '.p', 'wb'))
                    env.opponent_policy = self.get_opponent_policy(self.model, GomokuEnv.WHITE)
                    opponent_episode_number = 0
                    running_reward = 0

                if episode_number % 100 == 0:
                    message = 'ep: %d, running mean: %f' % (episode_number, running_reward)
                    self.log(message)
                    print message
                    pickle.dump(self.model, open('save' + str(self.N) + '.p', 'wb'))

                reward_sum = 0
                observation = env.reset()  # reset env

            if reward != 0:  # Gomoku has either +1 or -1 reward exactly when game ends.
                print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

    def learn_with_torch_pg(self, learning_rate=0.01, stdv=0.01):
        self.log("start learning with torch policy gradient - " + str(datetime.now()))

        env = self.create_env()
        env.opponent_policy = self.get_opponent_policy(None, GomokuEnv.WHITE)

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

    def trained_play(self, env):
        observation = env.reset()
        count = 0
        while True:
            # preprocess the observation
            x = prepro(observation)

            # forward the policy network and sample an action from the returned probability
            aprob, h = policy_forward(x)
            mask = np.zeros(self.D, dtype=int)
            mask[GomokuEnv.get_possible_actions(observation)] = 1
            aprob = np.multiply(aprob, mask)
            action = np.random.choice(np.where(aprob==aprob.max())[0])
            observation, reward, done, info = env.step(action)
            print 'reward: ' + str(reward) + ';done:' + str(done)
            env.render()
            if done:
                if reward == -1:
                    observation = env.reset()
                    count += 1
                else:
                    print count
                    exit()

    def play(self, color):
        if self.model is None:
            self.model = self.get_random_model()

        env = self.create_env(1 - color)
        env.opponent_policy = self.get_opponent_policy(self.model, color)
        observation = env.reset()
        env.render()

        while True:
            cls()
            env.render()
            action = input('Please enter your move in the form (x, y):')
            action = [action[0] - 1, action[1] - 1]
            action = GomokuEnv.coordinate_to_action(observation, action)
            observation, reward, done, info = env.step(action)
            if done:
                cls()
                env.render()
                if reward == 1:
                    print "You Win!"
                else:
                    print "You Lost!"
                return
