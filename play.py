import numpy as np
from game import gobang
from policyGradientNetwork import *
import torch
from MCTS import MCTS


def play(game: gobang, player1: PolicyNetworkAgent, player2: PolicyNetworkAgent, NUM_SIMULATION, display=False):
    game.clearBoard()
    mct1 = None
    mct2 = None
    if player1 != 'human':
        mct1 = MCTS(player1)
    if player2 != 'human':
        mct2 = MCTS(player2)
    curPlayer = player1
    mct = mct1
    turn = 1
    while True:
        result = game.checkWin()
        if result != 0:
            if display:
                game.printBoard()
                if result == 1:
                    print("Player1 wins!")
                elif result == -1:
                    print("Player2 wins!")
                else:
                    print("Tie!")
            return result

        if display:
            game.printBoard()
        if curPlayer == 'human':
            pos = list(map(int, input('x y =>').split()))
            if len(pos) != 2:
                print('wrong format')
                continue
            i, j = pos
            if not (0 <= i < game.boardsize and 0 <= j < game.boardsize and game.board[i][j] == 0):
                print('invalid position')
                continue
            game.play(i, j)

        else: # AI
            probs = mct.simulateAndPredict(game, NUM_SIMULATION)
            pos = np.argmax(probs)
            game.play(pos // game.boardsize, pos % game.boardsize)

        if turn == 1:
            curPlayer = player2
            mct = mct2
            turn = 2
        else:
            curPlayer = player1
            mct = mct1
            turn = 1
        



