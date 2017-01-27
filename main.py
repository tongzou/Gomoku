from agent import Agent

import numpy as np
import gym
from gym.utils import seeding
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
        #'opponent': opponent.make_opponent_policy(15, 1),
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 15,
    }
)


'''
    Some testing functions
'''
def test_diag(env, start, dir):
    # test diagonal win
    env.reset()
    step = 8 if dir == 'forward' else 10
    for i in range(5):
        observation, reward, done, info = env.step(start + i * step)

    env.render()
    assert reward == 1

#test_diag(35, 'forward')
#test_diag(36, 'backward')

def random_play(env):
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

agent = Agent()

print "#" * 50 + "\n Welcome to Gomoku!\n By Tong Zou\n" + "#"*50 + "\n"
while True:
    try:
        mode = input('What would you like the Gomoku Agent to do?\n' + ' ' * 5 +
                     '1. Self Train\n' + ' ' * 5 +
                     '2. Agent Vs Human\n' + ' ' * 5 +
                     '3. Human Vs Agent\n')
        if mode != 1 and mode != 2 and mode != 3:
            raise ValueError()
        break
    except:
        print "Please enter a valid choice."

if mode == 1:
    agent.learn(render=False)
elif mode == 2:
    agent.play(GomokuEnv.BLACK)
else:
    agent.play(GomokuEnv.WHITE)
