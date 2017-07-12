#import ai.Board as b
#import ai.Eval_funcs as ef
import numpy as np
import random
from gomoku import GomokuEnv as Env

'''
    This is the opponent using classic AI written by Oren Finard. The files are in the opponent directory.
    This is simply a wrapper for it.
'''
#'''
def get_ai_policy(board_size, tlimit):
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
#'''
'''
    Implements the naive policy. This will be the evaluation metric for the Agent.
    level:  0 do not search for connection
            1 search for connected 3's
            2 search for connected 2's
            3 search for 1's (this is the highest level for the agent)
'''
def get_naive_policy(board_size, level=3):
    '''
        mode(binary):   0100. return start of the pattern
                        1000. return 1 before the start
                        0010. end of the pattern
                        0001. one after the end of the pattern
            these mode can be bitwise combined..
    '''
    def search_move(board, pattern, size, mode):
        d = board.shape[0]
        # print 'searching for pattern ' + pattern
        search = Env.search_board(board, pattern, size)
        if search is not None:
            # print 'found: ' + str(search)
            coord = search[0]
            dir = search[1]
            deltas = []
            if mode & 0b1000:
                deltas.append((-dir[0], -dir[1]))
            if mode & 0b0100:
                deltas.append((0, 0))
            if mode & 0b0010:
                deltas.append(((size - 1) * dir[0], (size - 1) * dir[1]))
            if mode & 0b0001:
                deltas.append((size * dir[0], size * dir[1]))
            for delta in deltas:
                newcoord = [coord[0] + delta[1], coord[1] + delta[0]]
                if newcoord[0] >= 0 and newcoord[0] < d and newcoord[1] >= 0 and newcoord[1] < d \
                        and board[newcoord[0], newcoord[1]] == 2:
                    return newcoord

        return None

    def opponent_policy(curr_state, prev_state, prev_action):
        opponent_policy.second_move = False
        # check if a new games is started.
        prev_steps = np.count_nonzero(curr_state[2, :, :])
        if prev_steps + 1 == board_size ** 2:
            opponent_policy.second_move = True

        coords = Env.action_to_coordinate(curr_state, prev_action) if prev_action is not None else None

        if prev_state is None:
            '''
                First move should be the center of the board.
            '''
            move = (board_size//2, board_size//2)
        elif opponent_policy.second_move:
            '''
                If the AI must go second, it shouldn't think,
                it should just go diagonal adjacent to the first
                placed tile; diagonal into the larger area of the
                board if one exists
            '''
            if coords[1] <= board_size//2:
                dy = 1
            else:
                dy = -1

            if coords[0] <= board_size//2:
                dx = 1
            else:
                dx = -1
            move = (coords[0] + dx, coords[1] + dy)
            opponent_policy.second_move = False
        else:
            free_x, free_y = np.where(curr_state[2, :, :] == 1)
            possible_moves = [(x, y) for x, y in zip(free_x, free_y)]
            if len(possible_moves) == 0:
                # resign if there is no more moves
                return curr_state.shape[-1] ** 2
            '''
                Strategy for the naive agent:
                1. Search if there is a win opportunity.
                2. Search if opponent is winning, if yes, then block
                3. Search if opponent has a open 3, if yes, then block
                3. Try to extend the longest existing trend.
            '''
            if curr_state[0, coords[0], coords[1]] != 0:
                color = 1
            else:
                color = 0

            # 1: opponent position, 2: empty, 3: my position
            my_board = curr_state[color, :, :] - curr_state[1-color, :, :] + 2
            # check if we have 4 connected and empty space to make a win
            move = search_move(my_board, '23{4}', 5, 0b0100)
            if move is None:
                # check if we have 4 connected and empty space to make a win
                move = search_move(my_board, '3{4}2', 5, 0b0010)
            if move is None:
                # check if opponent has 4 connected
                move = search_move(my_board, '21{4}', 5, 0b0100)
            if move is None:
                # check if opponent has 4 connected
                move = search_move(my_board, '1{4}2', 5, 0b0010)
            if move is None:
                # check if opponent has open 3
                move = search_move(my_board, '21{3}2', 5, 0b0110)

            if move is None:
                for i in range(level):
                    # search for connected 3-i stones
                    move = search_move(my_board, '23{%d}' % (3-i), 4-i, 0b0100)
                    if move is None:
                        move = search_move(my_board, '3{%d}2' % (3-i), 4-i, 0b0010)
                    if move is not None:
                        break

            if move is None:
                move = random.choice(possible_moves)

        return Env.coordinate_to_action(curr_state, move)

    return opponent_policy
