import torch
import numpy as np
import math
from game import gobang
from policyGradientNetwork import *
from copy import deepcopy
class MCTS():
    def __init__(self, model: PolicyNetworkAgent):
        self.model = model
        self.Qsa = {} # expected reward for taking action a from state s
        self.Nsa = {} # number of time taking action a from state s
        self.Ns = {} # number of time state s was visited 
        self.Ps = {} # policy output (probs) of neural network from state s
        self.Es = {} # check game result at state s
        self.Vs = {} # valid moves at state s
    
    def simulateAndPredict(self, startGame: gobang, NUM_SIMULATION):
        s = startGame.getBoard().tobytes()

        for i in range(NUM_SIMULATION):
            game = deepcopy(startGame)
            self.search(game)
        cnt = []
        for a in range(startGame.boardsize * startGame.boardsize):
            if (s, a) not in self.Nsa:
                cnt.append(0)
            else:
                cnt.append(self.Nsa[(s, a)])
        cnt = np.array(cnt)
        print(self.Nsa.values())
        return cnt / np.sum(cnt)
    def search(self, game: gobang):
        s = game.getBoard().tobytes()
        if s not in self.Es:
            self.Es[s] = game.checkWin()
        if self.Es[s] != 0:
            return -self.Es[s]
        
        if s not in self.Ps:
            pi, v = self.model.forward(game.getBoard())
            validMoves = game.getValidMoves()
            pi *= validMoves
            assert(np.sum(pi) != 0)
            pi /= np.sum(pi)

            self.Ps[s] = pi
            self.Vs[s] = validMoves
            self.Ns[s] = 0
            return -v
        
        validMoves = self.Vs[s]
        bestScore = -float('inf')
        bestMove = None

        for a in range(game.boardsize * game.boardsize):
            if not validMoves[a]:
                continue
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + 1 * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = 1 * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            if u > bestScore:
                bestScore = u
                bestMove = a
        
        game.play(bestMove // game.boardsize, bestMove % game.boardsize)
        v = self.search(game)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v


            

