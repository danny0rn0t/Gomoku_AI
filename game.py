import numpy as np
from copy import deepcopy
class gobang:
    def __init__(self, boardsize):
        # black: 1
        # white: -1
        self.boardsize = boardsize
    def getEmptyBoard(self):
        return np.zeros((self.boardsize, self.boardsize))
    def getValidMoves(self, board):
        res = []
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                if board[i][j] == 0:
                    res.append(1)
                else:
                    res.append(0)
        return np.array(res)
    def play(self, board: np.ndarray, i: int, j: int, player: int = 1) -> int:
        assert (0 <= i < self.boardsize and 0 <= j < self.boardsize), 'play error.'
        # print(f"DEBUG: board[i][j] = {board[i][j]}")
        assert (board[i][j] == 0), 'play error.'
        res = board.copy()
        res[i][j] = player
        return res
    def printBoard(self, board) -> int:
        print('*|', end='')
        for i in range(self.boardsize):
            print(f'{i + 1}', end='')
            if i != self.boardsize - 1:
                print(' ', end='')
        print('|')
        # print(' |', end='')
        # for i in range(self.boardsize * 2 + 1):
        #     print('-', end='')
        # print('|', end='')
        # print()
        for i in range(self.boardsize):
            print(f'{i + 1}|', end='')
            for j in range(self.boardsize):
                if board[i][j] == 0:
                    print('-', end='')
                elif board[i][j] == 1:
                    print('O', end='')
                else:
                    print('X', end='')
                if j != self.boardsize - 1:
                    print(' ', end='')
            print('|', end='')
            print()
        # print(' |', end='')
        # for i in range(self.boardsize * 2 + 1):
        #     print('-' ,end='')
        # print('|')
        return 0
    def evaluate(self, board: np.ndarray) -> int:
        for j in range(self.boardsize): # check column
            res = self._helper(board, 0, j, 1, 0)
            if res != 0:
                return res
        for i in range(self.boardsize): # check row
            res = self._helper(board, i, 0, 0, 1)
            if res != 0:
                return res
        # left diagonal
        for i in range(self.boardsize):
            res = self._helper(board, i, 0, 1, 1)
            if res != 0:
                return res
        for j in range(1, self.boardsize):
            res = self._helper(board, 0, j, 1, 1)
            if res != 0:
                return res
        # right diagnoal
        for i in range(self.boardsize):
            res = self._helper(board, i, 0, -1, 1)
            if res != 0:
                return res
        for j in range(self.boardsize):
            res = self._helper(board, self.boardsize - 1, j, -1, 1)
            if res != 0:
                return res
        if 0 not in board:
            return 2
        return 0           
    def _helper(self, board, si, sj, di, dj):
        cur = 0
        cnt = 0
        i = si
        j = sj
        while 0 <= i < self.boardsize and 0 <= j < self.boardsize:
            if board[i][j] == 0:
                cur = 0
                cnt = 0
            elif cur == board[i][j]:
                cnt += 1
            else:
                cur = board[i][j]
                cnt = 1
            if cnt >= 5:
                return cur
            i += di
            j += dj
        return 0
            
            
 


