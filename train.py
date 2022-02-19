import numpy as np
from policyGradientNetwork import *
from game import gobang
from MCTS import MCTS
from tqdm import tqdm
from play import play
from copy import deepcopy

class train:
    def __init__(self, game: gobang, model: PolicyNetworkAgent, args):
        self.oldModel = model
        self.newModel = PolicyNetworkAgent(PolicyNetwork(game), args)
        self.game = game
        self.mcts = MCTS(game, self.oldModel)
        self.args = args
        self.trainData = []
    def executeEpisode(self):
        trainData = [] # [board, action, player{1, -1}]
        board = self.game.getEmptyBoard()
        turn = 1
        while True:
            result = self.game.evaluate(board)
            if result != 0: # game ended
                if result == 2: # tie
                    result = 0
                for item in trainData:
                    item[2] = item[2] * result
                return trainData
            probs = self.mcts.simulateAndPredict(board * turn, self.args.NUM_SIMULATION)
            s = (board * turn).tobytes()
            a = np.random.choice(range(len(probs)), p=probs)
            trainData.append([board * turn, probs, turn])
            board = self.game.play(board, a // self.game.boardsize, a % self.game.boardsize, turn)
            turn *= (-1)
    
    def train(self):
        for i in range(self.args.NUM_ITERATION):
            pass
            data = []
            for _ in tqdm(range(self.args.NUM_EPISODE)):
                data += self.executeEpisode()
            self.oldModel.save(self.args.MODEL_SAVE_PATH)
            self.newModel.load(self.args.MODEL_SAVE_PATH)
            
            self.newModel.learn(data)
            oldWins = newWins = ties = 0
            for j in range(self.args.NUM_GAME_INFERENCE):
                result = play(self.game, self.newModel, self.oldModel, self.args.NUM_SIMULATION)
                if result == 1:
                    newWins += 1
                elif result == -1:
                    oldWins += 1
                else:
                    ties += 1
            winrate = newWins / (newWins + oldWins) if newWins > 0 else 0
            print(f"iteration: {i} | {newWins} win, {oldWins} loss, {ties} tie |")
            if winrate >= 0.55:
                self.newModel.save(self.args.MODEL_SAVE_PATH)
                self.oldModel.load(self.args.MODEL_SAVE_PATH)








