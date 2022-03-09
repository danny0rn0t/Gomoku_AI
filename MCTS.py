import torch
import numpy as np
import math
from game import gobang
from policyGradientNetwork import *
from copy import deepcopy
from collections import defaultdict
import time
from torch.distributions.dirichlet import Dirichlet
import random
class MCTS():
    def __init__(self, game: gobang, model: PolicyNetworkAgent, args):
        self.model = model
        self.game = game
        self.args = args

        # adding Dirichlet noise to the root node for additional exploration
        self.Dir_noise = Dirichlet(torch.Tensor([self.alpha for _ in range(self.game.boardsize**2)]))

        self.Nsa = defaultdict(int) # action visit count
        self.Ns = defaultdict(int) # state visit count
        self.Wsa = defaultdict(int) # total action value
        self.Qsa = defaultdict(int) # mean action value
        
        self.Ps = defaultdict(int) # policy output (probs) of neural network from state s
        self.Es = defaultdict(int) # check game result at state s
        self.Vs = defaultdict(int) # valid moves at state s
    
    def simulateAndPredict(self, state: np.ndarray, NUM_SIMULATION: int, get_reward=False, time_limit=None, is_root=False):
        # is_root is True if the board (state) is empty
        s = state.tobytes()
        if time_limit is not None:
            startTime = time.time()
            while time.time() - startTime < time_limit:
                self.run(state, is_root)
        else:
            for i in range(NUM_SIMULATION):
                self.run(state, is_root)
        cnt = []
        for a in range(self.game.boardsize**2):
            if (s, a) not in self.Nsa:
                cnt.append(0)
            else:
                cnt.append(self.Nsa[(s, a)])
        cnt = np.array(cnt)
        #print(self.Nsa.values())
        if get_reward:
            a = np.argmax(cnt)
            v = self.Qsa[(s, a)]
            return cnt / np.sum(cnt), v
        else:
            return cnt / np.sum(cnt)
    def run(self, state: np.ndarray, is_root=False):
        s = state.tobytes()
        if s not in self.Es:
            self.Es[s] = self.game.evaluate(state)
        if self.Es[s] != 0:
            if self.Es[s] == 2: # tie
                return 0
            return -self.Es[s]
        
        # expand
        if s not in self.Ps:
            v = self.expand(s, state, is_root)
            return -v

        # select
        a = self.select(s)
        nxtState = self.game.play(state, a // self.game.boardsize, a % self.game.boardsize)
        nxtState = nxtState * (-1) # switch to the perspective of the other player
        v = self.run(nxtState)

        # backup
        self.Ns[s] += 1
        self.Nsa[(s, a)] += 1
        self.Wsa[(s, a)] += v
        self.Qsa[(s, a)] = self.Wsa[(s, a)] / self.Nsa[(s, a)]

        return -v
    def select(self, s) -> int:
        bestMove = None
        bestScore = None
        for a in range(self.game.boardsize**2):
            if not self.Vs[s][a]: continue
            u = self.Qsa[(s, a)] + self.args.c_puct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            if bestScore is None or u > bestScore:
                bestScore = u
                bestMove = a
        return bestMove
    def expand(self, s, state, is_root=False) -> int:
        # random rotate and flip
        seed = random.randint(0, 15)
        pi, v = self.model.forward(self.rotateAndReflect(state, seed))
        pi = np.reshape(pi, (self.game.boardsize, self.game.boardsize))
        pi = self.rotateAndReflectComplement(pi, seed)
        pi = np.reshape(pi, self.game.boardsize**2)
        
        validMoves = self.game.getValidMoves(state)
        if is_root: # adding Dirichlet noise for additional exploration
            pi = (1 - self.args.epsilon) * pi + self.args.epsilon * (self.Dir_noise.sample().numpy())
        pi *= validMoves
        if np.sum(pi) <= 0:
            pi = pi + validMoves
        pi /= np.sum(pi)
        self.Ps[s] = pi
        self.Vs[s] = validMoves
        return v
    def rotateAndReflect(self, board: np.ndarray, seed: int):
        # generate symmetries of board
        assert (0 <= seed < 16)
        rotate = seed >> 2
        for i in range(rotate):
            board = np.rot90(board)
        if seed & 0b0010:
            board = np.fliplr(board)
        if seed & 0b0001:
            board = np.flipud(board)
        return board
    def rotateAndReflectComplement(self, board: np.ndarray, seed: int):
        assert (0<= seed < 16)
        rotate = 4 - (seed >> 2)
        for i in range(rotate):
            board = np.rot90(board)
        if seed & 0b0010:
            board = np.fliplr(board)
        if seed & 0b0001:
            board = np.flipud(board)
        return board
