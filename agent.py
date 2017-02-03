import gym
from gym import error
from gomoku import GomokuEnv
import opponent
import os
import time

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

    def get_policy(self, model, color):
        return "random"

    def set_opponent_policy(self, env, policy):
        if policy is None:
            policy = "random"

        if isinstance(policy, str):
            if policy == 'ai':
                env.opponent_policy = opponent.get_ai_policy(self.N, 0.001)
            elif policy == 'naive':
                env.opponent_policy = opponent.get_naive_policy(self.N)
            elif policy == 'random':
                env.opponent = "random"
                env._seed()
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(policy))
        else:
            env.opponent_policy = policy

    def choose_move(self, observation, model, color):
        return self.D  # default return resign move

    def learn(self):
        pass

    def play(self, color, opponent='human'):
        if self.model is None:
            raise BaseException("This agent is not learned.")

        if opponent == 'human':
            env = self.create_env(1 - color)
            self.set_opponent_policy(env, self.get_policy(self.model, color))
        else:
            env = self.create_env(color)
            self.set_opponent_policy(env, opponent)

        observation = env.reset()

        while True:
            cls()
            env.render()
            if opponent == 'human':
                action = input('Please enter your move in the form (x, y):')
                action = [action[0] - 1, action[1] - 1]
                action = GomokuEnv.coordinate_to_action(observation, action)
            else:
                time.sleep(1)
                action = self.choose_move(observation, self.model, color)

            observation, reward, done, info = env.step(action)
            if done:
                cls()
                env.render()
                if reward == 1:
                    if opponent == 'human':
                        print 'You Win!'
                    else:
                        print 'Agent Win!'
                else:
                    if opponent == 'human':
                        print 'You Lost!'
                    else:
                        print 'Agent Lost!'
                return
