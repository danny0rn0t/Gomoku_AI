'''
from board import five_in_a_row

game1 = five_in_a_row(13)

game1.play(1, 5, 5)
game1.play(2, 5, 6)
print(game1.checkWin(5, 6))
game1.play(1, 6, 5)
game1.play(2, 6, 6)
print(game1.checkWin(6, 6))
game1.play(1, 7, 5)
game1.play(2, 7, 6)
print(game1.checkWin(7, 6))
game1.play(1, 8, 5)
game1.play(2, 8, 6)
print(game1.checkWin(8, 6))
game1.play(1, 9, 5)
game1.play(2, 9, 6)
print(game1.checkWin(9, 5))
# print(game1.board)
'''
from game import gobang
from play import *


game = gobang(13)
play(game, 'human', 'human', True)
