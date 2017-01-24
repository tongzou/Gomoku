import numpy as np
import cPickle as pickle
import gym
from gym.utils import seeding
import opponent
from gomoku import GomokuEnv

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
        'opponent': opponent.make_opponent_policy(15, 1),
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 15,
    }
)

env = gym.make("Gomoku9x9-v0")
#env = gym.make("Gomoku15x15-v0")

'''
    Training Policy Gradient Learning
'''
# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = True  # resume from previous checkpoint?
render = False

# model initialization
N = 9
D = N * N  # input dimensionality: N x N grid
if resume:
    model = pickle.load(open('save.p', 'rb'))
    print model
else:
    model = {'W1': np.random.randn(H, D) / np.sqrt(D), 'W2': np.random.randn(D, H) / np.sqrt(H)}

grad_buffer = {k: np.zeros_like(v) for k, v in model.iteritems()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.iteritems()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    """ prepro N x N x 3 uint8 Gomoku board into N x N 1D float vector """
    I = np.subtract(I[0, :, :], I[1, :, :])
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h


def policy_backward(epx, eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).T
    dh = np.dot(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


def train():
    observation = env.reset()
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    while True:
        if render:
            env.render()

        # preprocess the observation
        x = prepro(observation)

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = np.random.choice(np.where(aprob==aprob.max())[0])

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
            episode_number += 1

            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)

            xs, hs, dlogps, drs = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)

            # standardize the rewards to be unit normal
            discounted_epr = np.subtract(discounted_epr, np.mean(discounted_epr))
            discounted_epr = np.divide(discounted_epr, np.std(discounted_epr))

            epdlogp = np.multiply(epdlogp, discounted_epr)  # modulate the gradient with advantage (PG magic happens right here.)
            grad = policy_backward(epx, eph, epdlogp)
            for k in model:
                grad_buffer[k] += grad[k]  # accumulate grad over batch

            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                for k, v in model.iteritems():
                    g = grad_buffer[k] # gradient
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                    grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
            if episode_number % 100 == 0:
                pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()  # reset env

        if reward != 0:  # Gomoku has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')


'''
    Some testing functions
'''
def test_diag(start, dir):
    # test diagonal win
    env.reset()
    step = 8 if dir == 'forward' else 10
    for i in range(5):
        observation, reward, done, info = env.step(start + i * step)

    env.render()
    assert reward == 1

#test_diag(35, 'forward')
#test_diag(36, 'backward')

def random_play():
    np_random, seed = seeding.np_random(42)
    observation = env.reset()
    env.render()
    while True:
        possible_moves = GomokuEnv.get_possible_actions(observation)
        a = possible_moves[np_random.randint(len(possible_moves))]
        observation, reward, done, info = env.step(a)
        print 'reward: ' + str(reward) + ';done:' + str(done)
        env.render()
        if done:
            exit()


def trained_play():
    observation = env.reset()
    while True:
        # preprocess the observation
        x = prepro(observation)

        # forward the policy network and sample an action from the returned probability
        aprob, h = policy_forward(x)
        action = np.random.choice(np.where(aprob==aprob.max())[0])
        observation, reward, done, info = env.step(action)
        print 'reward: ' + str(reward) + ';done:' + str(done)
        env.render()
        if done:
            observation = env.reset()

#train()
trained_play()