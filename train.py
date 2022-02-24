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
        self.newModel = PolicyNetworkAgent(ResidualPolicyNetwork(game, args.residual_layers), args)
        self.game = game
        self.mcts = MCTS(game, self.oldModel)
        self.args = args
        self.trainData = []
    def selfPlay(self, res: list = None):
        mcts = MCTS(self.game, self.oldModel)
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
                if res is not None:
                    res.extend(trainData)
                return trainData
            probs = mcts.simulateAndPredict(board * turn, self.args.num_simulation)
            s = (board * turn).tobytes()
            a = np.random.choice(range(len(probs)), p=probs)
            trainData.append([board * turn, probs, turn])
            board = self.game.play(board, a // self.game.boardsize, a % self.game.boardsize, turn)
            turn *= (-1)
    
    def train(self):
        for i in range(self.args.num_iteration):
            pass
            data = []
            for _ in tqdm(range(self.args.num_episode)):
                data += self.selfPlay()
            self.oldModel.save(self.args.model_save_path)
            self.newModel.load(self.args.model_save_path)
            
            self.newModel.learn(data)
            oldWins = newWins = ties = 0
            mct_old = MCTS(self.game, self.oldModel)
            mct_new = MCTS(self.game, self.newModel)
            for j in tqdm(range(self.args.num_game_inference)):
                if j % 2 == 0:
                    result = play(self.game, self.newModel, self.oldModel, self.args.num_simulation, mct1=mct_new, mct2=mct_old)
                    if result == 1: newWins += 1
                    elif result == -1: oldWins += 1
                    else: ties += 1
                else:
                    result = play(self.game, self.oldModel, self.newModel, self.args.num_simulation, mct1=mct_old, mct2=mct_new)
                    if result == 1: oldWins += 1
                    elif result == -1: newWins += 1
                    else: ties += 1
            winrate = newWins / (newWins + oldWins) if newWins > 0 else 0
            print(f"iteration: {i} | {newWins} win, {oldWins} loss, {ties} tie |")
            if winrate >= self.args.update_threshold:
                print("Update new model!")
                self.newModel.save(self.args.model_save_path)
                self.oldModel.load(self.args.model_save_path)
            else:
                print("Discard new model!")








