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

def test_env(start, dir):
    # test diagonal win
    env.reset()
    if dir == 'forward':
        for i in range(5):
            observation, reward, done, info = env.step(start + i * 8)
    else:
        for i in range(5):
            observation, reward, done, info = env.step(start + i * 10)
    env.render()
    assert reward == 1

test_env(35, 'forward')
test_env(36, 'backward')


