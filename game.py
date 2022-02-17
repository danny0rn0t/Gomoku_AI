import numpy as np
class gobang:
    def __init__(self, boardsize):
        # black: 1
        # white: -1
        self.boardsize = boardsize
        self.turn = 1
        self.totalMoves = 0
        self.board = np.zeros((boardsize, boardsize))
        self.lastmove = (0, 0)
    def getBoard(self):
        return (self.board * self.turn)
    def getValidMoves(self):
        res = []
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                if self.board[i][j] == 0:
                    res.append(1)
                else:
                    res.append(0)
        return np.array(res)
    def play(self, i: int, j: int) -> int:
        assert (0 <= i < self.boardsize and 0 <= j < self.boardsize), 'play error.'
        assert (self.board[i][j] == 0), 'play error.'
        
        self.board[i][j] = self.turn
        self.turn = 1 if self.turn == -1 else -1
        self.totalMoves += 1
        self.lastmove = (i, j)
        return 0
    def clearBoard(self) -> int:
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                self.board[i][j] = 0
        self.turn = 1
        self.totalMoves = 0
        return 0
    def printBoard(self) -> int:
        print('|', end='')
        for i in range(self.boardsize):
            print('--', end='')
        print('|', end='')
        print()
        for i in range(self.boardsize):
            print('|', end='')
            for j in range(self.boardsize):
                if self.board[i][j] == 0:
                    print('- ', end='')
                elif self.board[i][j] == 1:
                    print('O ', end='')
                else:
                    print('X ', end='')
            print('|', end='')
            print()
        print('|', end='')
        for i in range(self.boardsize):
            print('--' ,end='')
        print('|')
        return 0
    def checkWin(self, i=None, j=None) -> int:
        assert ((i == None and j == None) or (i != None and j != None))
        if i == None and j == None:
            i, j = self.lastmove
        assert (0 <= i < self.boardsize and 0 <= j < self.boardsize), 'checkWin error.'
        if self.board[i][j] == 0:
            return 0
        if self.totalMoves == self.boardsize * self.boardsize:
            return 2

        cnt = 1
        k = j + 1
        while(k < self.boardsize and self.board[i][k] == self.board[i][j]):
            cnt += 1
            k += 1
        k = j - 1
        while(k >= 0 and self.board[i][k] == self.board[i][j]):
            cnt += 1
            k -= 1
        if cnt >= 5:
            return self.board[i][j]

        cnt = 1
        k = i + 1
        while(k < self.boardsize and self.board[k][j] == self.board[i][j]):
            cnt += 1
            k += 1
        k = i - 1
        while(k >= 0 and self.board[k][j] == self.board[i][j]):
            cnt += 1
            k -= 1
        if cnt >= 5:
            return self.board[i][j]

        cnt = 1
        k = i + 1
        l = j + 1
        while(k < self.boardsize and l < self.boardsize and self.board[k][l] == self.board[i][j]):
            cnt += 1
            k += 1
            l += 1
        k = i - 1
        l = j - 1
        while(k >= 0 and l >= 0 and self.board[k][l] == self.board[i][j]):
            cnt += 1
            k -= 1
            l -= 1
        if cnt >= 5:
            return self.board[i][j]
        
        cnt = 1
        k = i + 1
        l = j - 1
        while(k < self.boardsize and l >= 0 and self.board[k][l] == self.board[i][j]):
            cnt += 1
            k += 1
            l -= 1
        k = i - 1
        l = j + 1
        while(k >= 0 and l < self.boardsize and self.board[k][l] == self.board[i][j]):
            cnt += 1
            k -= 1
            l += 1
        if cnt >= 5:
            return self.board[i][j]        
        return 0


