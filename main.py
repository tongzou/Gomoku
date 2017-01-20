import gym
from gym.utils import seeding
import opponent
import numpy as np
from gomoku import GomokuEnv

gym.envs.registration.register(
    id='Gomoku9x9-v0',
    entry_point='gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': opponent.make_opponent_policy(9, 1),
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

