import ai.Board as b
import ai.Eval_funcs as ef
import numpy as np
from gomoku import GomokuEnv as Env

'''
    This is the opponent using classic AI written by Oren Finard. The files are in the opponent directory.
    This is simply a wrapper for it.
'''
def make_ai_policy(board_size, tlimit):
    def opponent_policy(curr_state, prev_state, prev_action):
        # check if a new games is started.
        prev_steps = np.count_nonzero(curr_state[2, :, :])
        if prev_steps == board_size ** 2 or prev_steps + 1 == board_size ** 2:
            opponent_policy.board = b.Board(board_size, 5)
            opponent_policy.second_move = True
        board = opponent_policy.board
        if prev_state is None:
            move = ef.firstmove(board)
            opponent_policy.second_move = False
        else:
            coords = Env.action_to_coordinate(curr_state, prev_action)
            board = board.move(coords)
            if opponent_policy.second_move:
                move = ef.secondmove(board)
                opponent_policy.second_move = False
            else:
                move = ef.nextMove(board, tlimit, 1)
        opponent_policy.board = board.move(move)

        return Env.coordinate_to_action(curr_state, move)

    return opponent_policy
