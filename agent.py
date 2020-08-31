import gym
from gym import error
from gym_gomoku.envs import GomokuEnv
import opponent
import os
import time


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


class Agent:
    def __init__(self, board_size=9, win_len=5, log_file="log.txt"):
        assert isinstance(board_size, int) and board_size >= 3, 'Invalid board size: {}'.format(board_size)
        self.N = board_size
        self.L = win_len
        self.D = self.N * self.N
        self.log_file = log_file

        self.model = None
        self.logger = None

    def log(self, message):
        if self.logger is None:
            self.logger = open(os.path.join('logs', self.log_file), 'a')
        self.logger.write(message + '\n')
        self.logger.flush()

    def create_env(self, color=GomokuEnv.BLACK):
        id = 'Gomoku' + str(self.N) + 'x' + str(self.N) + '_' + str(self.L) + '-v0'
        try:
            gym.envs.registration.spec(id)
        except gym.error.UnregisteredEnv:
            print('registering new gym env id: ' + id)
            gym.envs.registration.register(
                id=id,
                entry_point='gym_gomoku.envs:GomokuEnv',
                kwargs={
                    'player_color': 'black',
                    'opponent': 'random',
                    'observation_type': 'numpy3c',
                    'illegal_move_mode': 'lose',
                    'board_size': self.N,
                    'win_len': self.L
                }
            )

        env = gym.make(id)
        env.player_color = color
        return env

    def get_policy(self, model, color):
        return "random"

    def set_opponent_policy(self, env, policy):
        if policy is None:
            policy = "random"

        if isinstance(policy, str):
            if policy == 'ai':
                assert self.L == 5, 'Only works if the winning length is 5, current winning length is : {}'.format(self.L)
                env.opponent_policy = opponent.make_ai_policy(self.N, 0.001)
            else:
                env.opponent = policy
                env.seed()
        else:
            env.opponent_policy = policy

    def choose_move(self, observation, model, color):
        return self.D  # default return resign move

    def train(self):
        pass

    def test(self, color=GomokuEnv.BLACK, opponent="naive3", render=False, size=100):
        if self.model is None:
            raise error.Error("This agent is not trained.")

        env = self.create_env(color)
        self.set_opponent_policy(env, opponent)

        observation = env.reset()
        win = 0
        episode_number = 0

        while episode_number < size:
            if render:
                env.render()

            action = self.choose_move(observation, self.model, color)
            observation, reward, done, info = env.step(action)
            if reward == 1:
                win += 1

            if done:
                if render:
                    env.render()
                episode_number += 1
                print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !'))
                observation = env.reset()  # reset env

        print('Total wins for the agent is %f, the winning ratio for the agent is: %f' % (win, win/size))

    def play(self, color, opponent='human'):
        if self.model is None:
            raise BaseException("This agent is not trained.")

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
                try:
                    action = eval(input('Please enter your move in the form (x, y):'))
                except SyntaxError:
                    continue
                except NameError:
                    return

                action = [action[0] - 1, action[1] - 1]
                action = GomokuEnv.coordinate_to_action(self.N, action)
            else:
                time.sleep(1)
                action = self.choose_move(observation, self.model, color)

            observation, reward, done, info = env.step(action)
            if done:
                cls()
                env.render()
                if reward == 1:
                    if opponent == 'human':
                        print('You Win!')
                    else:
                        print('Agent Win!')
                else:
                    if opponent == 'human':
                        print('You Lost!')
                    else:
                        print('Agent Lost!')
                return
