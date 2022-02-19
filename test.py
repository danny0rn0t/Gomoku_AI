from game import gobang
import numpy as np

game = gobang(9)
board = game.getEmptyBoard()
curPlayer = 1
while True:
    game.printBoard(board)
    res = game.evaluate(board)
    if res != 0:
        if res == 1:
            print("player1 wins!")
        elif res == -1:
            print("player2 wins!")
        else:
            print("Tie!")
        break
    validMoves = game.getValidMoves(board)
    a = np.random.choice(81, p=validMoves / np.sum(validMoves))
    i = a // 9
    j = a % 9
    print(f"Player {curPlayer} place at ({i}, {j})")
    board = game.play(board, i, j, curPlayer)
    curPlayer *= -1
