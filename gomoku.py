"""
Game of Gomoku
"""
import sys
import numpy as np
import re
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO

def shift(xs, n):
    if n == 0:
        return xs[:]
    e = np.zeros(xs.shape, dtype=int)
    if n > 0:
        e[:n] = 0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0
        e[:n] = xs[-n:]
    return e


# Adversary policies #
def make_random_policy(np_random):
    def random_policy(curr_state, prev_state, prev_action):
        possible_moves = GomokuEnv.get_possible_actions(curr_state)
        # No moves left
        if len(possible_moves) == 0:
            return None
        a = np_random.randint(len(possible_moves))
        return possible_moves[a]
    return random_policy


class GomokuEnv(gym.Env):
    """
    Gomoku environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
            board_size: size of the Go board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': GomokuEnv.BLACK,
            'white': GomokuEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign
        self.action_space = spaces.Discrete(self.board_size ** 2 + 1)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))

        self._seed()
        self.prev_move = -1

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def _reset(self):
        self.state = np.zeros((3, self.board_size, self.board_size), dtype=int)
        self.state[2, :, :] = 1.0
        self.to_play = GomokuEnv.BLACK
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state, None, None)
            GomokuEnv.make_move(self.state, a, GomokuEnv.BLACK)
            self.to_play = GomokuEnv.WHITE
            self.prev_move = a
        return self.state

    def _step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}

        prev_state = self.state
        if GomokuEnv.resign_move(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not GomokuEnv.valid_move(self.state, action):
            if self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))
        else:
            GomokuEnv.make_move(self.state, action, self.player_color)
            self.prev_move = action

        # Opponent play
        a = self.opponent_policy(self.state, prev_state, action)

        # Making move if there are moves left
        if a is not None:
            if GomokuEnv.resign_move(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            else:
                GomokuEnv.make_move(self.state, a, 1 - self.player_color)

        reward = GomokuEnv.game_finished(self.state, self.player_color)
        self.done = reward != 0

        # check to see if we need to roll back opponent move if we have won already.
        if reward == 1 and a is not None:
            GomokuEnv.revert_move(self.state, a, 1 - self.player_color)
            pass
        else:
            self.prev_move = a
        return self.state, reward, self.done, {'state': self.state}

    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write('To play: ')
        outfile.write('black' if self.to_play == GomokuEnv.BLACK else 'white')
        outfile.write('\n')
        d = board.shape[1]
        if d > 9:
            outfile.write(' ' * 24)
            for j in range(10, d + 1):
                outfile.write(' ' + str(j/10))
            outfile.write('\n')
        outfile.write(' ' * 6)
        for j in range(d):
            #outfile.write(' ' + string.ascii_uppercase[j])
            outfile.write(' ' + str((j + 1) % 10))
        outfile.write('\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')
        for i in range(d):
            outfile.write(' ' * (3 if i < 9 else 2) + str(i + 1) + ' | ')
            for j in range(d):
                a = GomokuEnv.coordinate_to_action(board, [i, j])
                if board[2, i, j] == 1:
                    outfile.write('. ')
                elif board[0, i, j] == 1:
                    if self.prev_move == a:
                        outfile.write('X)')
                    else:
                        outfile.write('X ')
                else:
                    if self.prev_move == a:
                        outfile.write('O)')
                    else:
                        outfile.write('O ')
            outfile.write('|\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')

        if mode != 'human':
            return outfile

    @staticmethod
    def resign_move(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def valid_move(board, action):
        coords = GomokuEnv.action_to_coordinate(board, action)
        if board[2, coords[0], coords[1]] == 1:
            return True
        else:
            return False

    @staticmethod
    def make_move(board, action, player):
        coords = GomokuEnv.action_to_coordinate(board, action)
        board[2, coords[0], coords[1]] = 0
        board[player, coords[0], coords[1]] = 1

    @staticmethod
    def revert_move(board, action, player):
        coords = GomokuEnv.action_to_coordinate(board, action)
        board[2, coords[0], coords[1]] = 1
        board[player, coords[0], coords[1]] = 0

    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def get_possible_actions(board):
        free_x, free_y = np.where(board[2, :, :] == 1)
        return [GomokuEnv.coordinate_to_action(board, [x, y]) for x, y in zip(free_x, free_y)]

    @staticmethod
    def test_horizontal(player_board):
        d = player_board.shape[0]
        state = ''
        for i in range(d):
            state += ''.join(map(str, player_board[i])) + '-'

        return re.search('1{5}', state)

    @staticmethod
    def game_finished(board, first_color):
        # Returns 1 if first_color wins, -1 if first_color loses and 0 otherwise
        d = board.shape[1]

        # test horizontal and vertical
        player_board = board[first_color, :, :]
        if GomokuEnv.test_horizontal(player_board) or GomokuEnv.test_horizontal(np.transpose(player_board)):
            return 1

        # test diagonal
        for i in range(d-4):
            forward = np.zeros((d, d), dtype=int)
            backward = np.zeros((d, d), dtype=int)
            for j in range(i, i + 5):
                forward[j] = shift(player_board[j, :], j - i)
                backward[j] = shift(player_board[j, :], i - j)
            if GomokuEnv.test_horizontal(np.transpose(forward)) or GomokuEnv.test_horizontal(np.transpose(backward)):
                return 1

        # test player 2 horizontal and vertical
        player_board = board[1 - first_color, :, :]
        if GomokuEnv.test_horizontal(player_board) or GomokuEnv.test_horizontal(np.transpose(player_board)):
            return -1

        # test diagonal
        for i in range(d-4):
            forward = np.zeros((d, d), dtype=int)
            backward = np.zeros((d, d), dtype=int)
            for j in range(i, i + 5):
                forward[j] = shift(player_board[j, :], j - i)
                backward[j] = shift(player_board[j, :], i - j)
            if GomokuEnv.test_horizontal(np.transpose(forward)) or GomokuEnv.test_horizontal(np.transpose(backward)):
                return -1

        return 0
