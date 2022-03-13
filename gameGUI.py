import pygame
from game import gobang
from policyGradientNetwork import *
from MCTS import MCTS
from enum import Enum
import numpy as np
from player import Player


BLACK = 1
WHITE = -1

class Chessboard:
    def __init__(self, game: gobang, args, player1: Player, player2: Player, time_limit=10):
        self.args = args
        self.game = game
        if player1.color is None:
            player1.set_color(1)
        if player2.color is None:
            player2.set_color(-1)
        self.player1 = player1
        self.player2 = player2
        self.player = self.player1
        self.time_limit = time_limit
        
        self.grid_size = 26
        self.start_x, self.start_y = 30, 50
        self.edge_size = self.grid_size / 2
        self.boardsize = args.boardsize
        self.peice = 'b'
        self.win_message = None
        self.game_over = False
        self.last_move = None
        
        self.board = game.getEmptyBoard()
    def play(self, r=None, c=None):
        if self.player.player_type == 'AI':
            probs, v, totalSimulation = self.player.mct.simulateAndPredict(self.board * self.player.color, self.args.num_simulation, get_reward=True, time_limit=self.time_limit)
            pos = np.argmax(probs)
            r, c = pos // self.boardsize, pos % self.boardsize
        assert (r is not None and c is not None)
        if self.board[r][c] != 0:
            return
        self.board = self.game.play(self.board, r, c, self.player.color)
        self.last_move = (r, c)
        self.player = self.player1 if self.player == self.player2 else self.player2
        self.check_win()
    def check_win(self):
        result = self.game.evaluate(self.board)
        if result != 0:
            self.game_over = True
            if result == BLACK:
                self.win_message = 'Black won!'
            elif result == WHITE:
                self.win_message = 'White won!'
            else:
                self.win_message = 'Its a tie!'
        
    def handle_key_event(self, e):
        origin_x = self.start_x - self.edge_size
        origin_y = self.start_y - self.edge_size
        size = (self.boardsize - 1) * self.grid_size + self.edge_size * 2
        pos = e.pos 
        if origin_x <= pos[0] <= origin_x + size and origin_y <= pos[1] <= origin_y + size:
            if not self.game_over:
                x = pos[0] - origin_x
                y = pos[1] - origin_y
                r = int(y // self.grid_size)
                c = int(x // self.grid_size)
                self.play(r, c)
    
    def draw(self, screen):
        pygame.draw.rect(screen, (246, 211, 122),\
            [self.start_x - self.edge_size, self.start_y - self.edge_size,
            (self.boardsize - 1) * self.grid_size + self.edge_size * 2, (self.boardsize - 1) * self.grid_size + self.edge_size * 2], 0)
        for r in range(self.boardsize):
            y = self.start_y + r * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [self.start_x, y], [self.start_x + self.grid_size * (self.boardsize - 1), y], 2)
        for c in range(self.boardsize):
            x = self.start_x + c * self.grid_size
            pygame.draw.line(screen, (0, 0, 0), [x, self.start_y], [x, self.start_y + self.grid_size * (self.boardsize - 1)], 2)
        for r in range(self.boardsize):
            for c in range(self.boardsize):
                piece = self.board[r][c]
                if piece != 0:
                    if piece == BLACK:
                        color = (0, 0, 0)
                    else:
                        color = (255, 255, 255)
                    
                    x = self.start_x + c * self.grid_size
                    y = self.start_y + r * self.grid_size
                    pygame.draw.circle(screen, color, [x, y], self.grid_size // 2)
                    if (r, c) == self.last_move:
                        pygame.draw.circle(screen, (0, 0, 0) if piece == WHITE else (255, 255, 255), [x, y], self.grid_size / 6)


        

class GobangGUI:
    def __init__(self, chessboard: Chessboard):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)
        self.going = True
        self.chessboard = chessboard
    def loop(self):
        while self.going:
            self.draw()
            self.clock.tick(60)
            self.update()
        pygame.quit()
    def update(self):
        if self.chessboard.player.player_type == 'AI':
            self.chessboard.play()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.going = False
                    return
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.going = False
                return
            if e.type == pygame.MOUSEBUTTONDOWN and self.chessboard.player.player_type == 'HUMAN':
                self.chessboard.handle_key_event(e)
    def draw(self):
        self.screen.fill((255, 255, 255))
        # self.screen.blit(self.font.render(""))
        self.chessboard.draw(self.screen)
        if self.chessboard.game_over:
            self.screen.blit(self.font.render(self.chessboard.win_message, True, (0, 0, 0)), (500, 10))
        pygame.display.update()

