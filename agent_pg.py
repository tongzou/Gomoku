import numpy as np
import cPickle as pickle
from gomoku import GomokuEnv
import random
from agent import Agent
from datetime import datetime


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def prepro(I, color=GomokuEnv.BLACK):
    """ prepro N x N x 3 uint8 Gomoku board into N x N 1D float vector """
    I = np.subtract(I[0, :, :], I[1, :, :])
    '''
    if color == GomokuEnv.BLACK:
        I = np.subtract(I[0, :, :], I[1, :, :])
    else:
        I = np.subtract(I[1, :, :], I[0, :, :])
    '''
    return I.astype(np.float).ravel()


def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros(r.shape, dtype=float)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(model, x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    return softmax(logp), h


def policy_backward(model, epx, eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).T
    dh = np.dot(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


class PGAgent(Agent):
    '''
        hidden:  # number of hidden layer neurons
    '''
    def __init__(self, board_size=9, hidden=200, resume=True):
        Agent.__init__(self, board_size)
        self.H = hidden

        # load the saved model first.
        if resume:
            try:
                self.log('using saved model.')
                self.model = pickle.load(open(self.get_model_file_name(), 'rb'))
            except:
                self.log('no saved model.')
                self.model = None

        if self.model is None:
            self.model = self.get_random_model()

    def get_model_file_name(self):
        return 'pg_%s_%s.p' % (self.N, self.H)

    def get_opponent_model_file_name(self):
        return 'pg_opponent%s_%s.p' % (self.N, self.H)

    # return a random model
    def get_random_model(self):
        return {'W1': np.random.randn(self.H, self.D) / np.sqrt(self.D),
                'W2': np.random.randn(self.D + 1, self.H) / np.sqrt(self.H)}

    def get_policy(self, model, color):
        if model is None:
            # try to load it from file.
            try:
                model = pickle.load(open(self.get_opponent_model_file_name(), 'rb'))
            except:
                model = self.get_random_model()
        else:
            model = model.copy()

        def opponent_policy(curr_state, prev_state, prev_action):
            return self.choose_move(curr_state, model, color)

        return opponent_policy

    def choose_move(self, observation, model, color):
        action, _, _, _ = self._choose_move(observation, model, color)
        return action

    '''
        Choose a move based on the current board, trained model and player color
    '''
    def _choose_move(self, observation, model, color):
        # preprocess the observation
        x = prepro(observation, color)

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(model, x)
        possible_moves = GomokuEnv.get_possible_actions(observation)
        if len(possible_moves) == 0:
            return self.D, x, aprob, h  # resign

        newprob = np.array([aprob[k] for k in possible_moves])
        #mask = np.zeros(self.D + 1, dtype=int)
        #mask[possible_moves] = 1
        #newprob = np.multiply(aprob, mask)

        if newprob.max() == 0:
            # self.log('all probabilies are zero!!!!')
            action = random.choice(possible_moves)
        else:
            newprob = softmax(newprob)
            action = np.random.choice(possible_moves, 1, p=newprob)[0]
            #action = np.random.choice(np.where(newprob == newprob.max())[0])

        return action, x, aprob, h

    '''
        batch_size:  # every how many episodes to do a gradient calculation?
        update_per_batch:  # every how many batches to do a param update?
        learning_rate: 1e-4
        gamma:  # discount factor for reward
        decay_rate:  # decay factor for RMSProp leaky sum of grad^2
        opponent: if None, will use self play. If not none, will use that as the opponent.
    '''
    def learn(self, render=False, model_threshold=0.9, min_episodes=100, batch_size=10, update_per_batch=5,
              learning_rate=1e-3, gamma=0.99, decay_rate=0.99, opponent=None):
        # setup logging
        self.log("start learning - " + str(datetime.now()))

        env = self.create_env()
        observation = env.reset()
        xs, hs, dlogps, drs = [], [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0

        # initialize component for self play.
        if opponent is None:
            self.set_opponent_policy(env, self.get_policy(None, GomokuEnv.WHITE))
        else:
            self.set_opponent_policy(env, opponent)

        opponent_episode_number = 0

        grad_buffer = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  # update buffers that add up gradients over a batch
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.iteritems()}  # rmsprop memory

        while True:
            if render:
                env.render()

            action, x, aprob, h = self._choose_move(observation, self.model, GomokuEnv.BLACK)

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

                print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !')

                # only do the gradient for every batch_size episodes
                if episode_number % batch_size == 0:
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
                    if episode_number % (update_per_batch * batch_size) == 0:
                        print 'update params'
                        for k, v in self.model.iteritems():
                            g = grad_buffer[k]  # gradient
                            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                            self.model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                            grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                    print 'episode reward total was %f. running mean: %f' % (reward_sum, running_reward)

                    # replace the opponent model once our running_reward is over the threshold and min_episodes is met
                    '''
                    if opponent is None and running_reward > model_threshold and opponent_episode_number > min_episodes:
                        self.log('replace opponent model now: ep ' + str(episode_number) + '\n' +
                                  'running mean: ' + str(running_reward))
                        # save the opponent model
                        pickle.dump(self.model, open(self.get_opponent_model_file_name(), 'wb'))
                        self.set_opponent_policy(env, self.get_policy(self.model, GomokuEnv.WHITE))
                        opponent_episode_number = 0
                        running_reward = 0
                    '''
                    if episode_number % 100 == 0:
                        message = 'ep: %d, running mean: %f' % (episode_number, running_reward)
                        self.log(message)
                        pickle.dump(self.model, open(self.get_model_file_name(), 'wb'))

                    reward_sum = 0

                observation = env.reset()  # reset env
