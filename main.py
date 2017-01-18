import gym

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

env = gym.make("Gomoku9x9-v0")
env.reset()

def step(action):
    observation, reward, done, info = env.step(action)
    env.render()
    print 'reward:' + str(reward) + ';done:' + str(done)

step(5)
step(13)
step(21)
step(29)
step(45)
step(37)
