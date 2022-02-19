import torch
import numpy as np
import math
from game import gobang
from policyGradientNetwork import *
from copy import deepcopy
class MCTS():
    def __init__(self, game: gobang, model: PolicyNetworkAgent):
        self.model = model
        self.game = game
        self.Qsa = {} # expected reward for taking action a from state s
        self.Nsa = {} # number of time taking action a from state s
        self.Ns = {} # number of time state s was visited 
        self.Ps = {} # policy output (probs) of neural network from state s
        self.Es = {} # check game result at state s
        self.Vs = {} # valid moves at state s
    
    def simulateAndPredict(self, state: np.ndarray, NUM_SIMULATION: int):
        s = state.tobytes()

        for i in range(NUM_SIMULATION):
            self._run(state)
        cnt = []
        for a in range(self.game.boardsize**2):
            if (s, a) not in self.Nsa:
                cnt.append(0)
            else:
                cnt.append(self.Nsa[(s, a)])
        cnt = np.array(cnt)
        #print(self.Nsa.values())
        return cnt / np.sum(cnt)
    def _run(self, state: np.ndarray):
        s = state.tobytes()
        if s not in self.Es:
            self.Es[s] = self.game.evaluate(state)
        if self.Es[s] != 0:
            if self.Es[s] == 2: # tie
                return 0
            return -self.Es[s]
        
        if s not in self.Ps:
            pi, v = self.model.forward(state)
            validMoves = self.game.getValidMoves()
            pi *= validMoves
            if np.sum(pi) <= 0:
                pi = pi + validMoves
            pi /= np.sum(pi)


            self.Ps[s] = pi
            self.Vs[s] = validMoves
            self.Ns[s] = 0
            return -v
        
        validMoves = self.Vs[s]
        bestScore = -float('inf')
        bestMove = None

        for a in range(self.game.boardsize * self.game.boardsize):
            if not validMoves[a]:
                continue
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + 1 * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = 1 * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            if u > bestScore:
                bestScore = u
                bestMove = a
        
        nxtState = self.game.play(state, bestMove // self.game.boardsize, bestMove % self.game.boardsize)
        nxtState = nxtState * (-1) # switch to the perspective of the other player
        v = self._run(nxtState)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v


            

