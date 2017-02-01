import gym
from gomoku import GomokuEnv
import os

gym.envs.registration.register(
    id='Gomoku9x9-v0',
    entry_point='gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
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
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 15,
    }
)

def cls():
    os.system('cls')  # For Windows
    os.system('clear')  # For Linux/OS X


class Agent:
    def __init__(self, board_size=9, log_file="log.txt"):
        self.N = board_size
        self.D = self.N * self.N
        self.log_file = log_file

        self.model = None
        self.logger = None

    def log(self, message):
        if self.logger is None:
            self.logger = open(self.log_file, 'ab')
        self.logger.write(message + '\n')
        self.logger.flush()

    def create_env(self, color=GomokuEnv.BLACK):
        if self.N == 15:
            env = gym.make("Gomoku15x15-v0")
        else:
            env = gym.make("Gomoku9x9-v0")

        env.player_color = color
        return env

    def set_opponent_policy(self, env, model, color):
        # default set the random policy
        env.opponent = "random"
        env._seed()

    def learn(self):
        pass

    '''
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
    '''

    def play(self, color):
        env = self.create_env(1 - color)
        self.set_opponent_policy(env, self.model, color)
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
