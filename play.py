import numpy as np
from game import gobang
from policyGradientNetwork import *
import torch
from MCTS import MCTS
from player import Player


def play(game: gobang, player1: Player, player2: Player, num_simulation: int=100, display=False, time_limit :int=None):
    board = game.getEmptyBoard()
    player1.set_color(1)
    player2.set_color(-1)

    player = player1

    if display:
        game.printBoard(board)
    while True:
        winrate = None
        i = j = None
        if player.player_type == 'HUMAN':
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
            board = game.play(board, i, j, player.color)
        else: # AI
            probs, v, totalSimulation = player.mct.simulateAndPredict(board * player.color, num_simulation, get_reward=True, time_limit=time_limit)
            # print(f"debug: probs = {probs}")
            pos = np.argmax(probs)
            winrate = (v + 1) * 50
            # print(f"debug: pos = {pos}")
            i = pos // game.boardsize
            j = pos % game.boardsize
            board = game.play(board, i, j, player.color)

        if display:
            game.printBoard(board, (i, j))
            if winrate is not None:
                print(f"winrate:{winrate: .2f}%, {totalSimulation} moves simulated")
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

        player = player1 if player == player2 else player2
        



