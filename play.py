import numpy as np
from game import gobang
from policyGradientNetwork import *
import torch
from MCTS import MCTS


def play(game: gobang, player1: PolicyNetworkAgent, player2: PolicyNetworkAgent, NUM_SIMULATION: int, mct1=None, mct2=None, display=False, time_limit=None):
    board = game.getEmptyBoard()
    if player1 != 'human' and mct1 is None:
        mct1 = MCTS(game, player1)
    if player2 != 'human' and mct2 is None:
        mct2 = MCTS(game, player2)

    player = player1
    mct = mct1
    turn = 1

    if display:
        game.printBoard(board)
    while True:
        winrate = None
        i = j = None
        if player == 'human':
            pos = list(map(int, input('Your turn, enter x y =>').split()))
            if len(pos) != 2:
                print('wrong format')
                continue
            i, j = pos
            i -= 1
            j -= 1
            if not (0 <= i < game.boardsize and 0 <= j < game.boardsize and board[i][j] == 0):
                print('invalid position')
                continue
            board = game.play(board, i, j, turn)
        else: # AI
            probs, v = mct.simulateAndPredict(board * turn, NUM_SIMULATION, get_reward=True, time_limit=time_limit)
            # print(f"debug: probs = {probs}")
            pos = np.argmax(probs)
            winrate = (v + 1) * 50
            # print(f"debug: pos = {pos}")
            i = pos // game.boardsize
            j = pos % game.boardsize
            board = game.play(board, i, j, turn)

        if display:
            game.printBoard(board, (i, j))
            if winrate is not None:
                print(f"winrate:{winrate: .2f}%")
        result = game.evaluate(board)
        if result != 0:
            if display:
                if result == 1:
                    print("Player1 won!")
                elif result == -1:
                    print("Player2 won!")
                else:
                    print("Its a Tie!")
            return result

        if turn == 1:
            player = player2
            mct = mct2
            turn = -1
        else:
            player = player1
            mct = mct1
            turn = 1
        



