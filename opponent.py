'''
    This is the opponent using classic AI written by Oren Finard. The files are in the opponent directory.
    This is simply a wrapper for it.
'''
import ai.Board as b
import ai.Eval_funcs as ef
from gomoku import GomokuEnv as env

def make_opponent_policy(board_size, tlimit):
    def opponent_policy(curr_state, prev_state, prev_action):
        board = opponent_policy.board
        if prev_state is None:
            move = ef.firstmove(board)
        else:
            coords = env.action_to_coordinate(curr_state, prev_action)
            print 'player action: '  + str(coords[0] + 1) + ',' + str(coords[1] + 1)
            board = board.move(coords)
            if opponent_policy.second_move:
                move = ef.secondmove(board)
                opponent_policy.second_move = False
            else:
                move = ef.nextMove(board, tlimit, 3)
        print 'opponent action:' + str(move[0] + 1) + ',' + str(move[1] + 1)
        opponent_policy.board = board.move(move)

        return env.coordinate_to_action(curr_state, move)
    opponent_policy.board = b.Board(board_size, 5)
    opponent_policy.second_move = True

    return opponent_policy