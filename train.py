import numpy as np
from policyGradientNetwork import *
from game import gobang
from MCTS import MCTS
from tqdm import tqdm
from play import play
from copy import deepcopy
import threading


class train:
    def __init__(self, game: gobang, model: PolicyNetworkAgent, args):
        self.oldModel = model
        self.newModel = PolicyNetworkAgent(ResidualPolicyNetwork(game, args.residual_layers), args)
        self.game = game
        self.mcts = MCTS(game, self.oldModel, args)
        self.args = args
        self.trainData = []
        self.lock = threading.Lock()
    def selfPlay(self):
        mcts = MCTS(self.game, self.oldModel, self.args)
        trainData = [] # [board, action, player{1, -1}]
        board = self.game.getEmptyBoard()
        turn = 1
        is_root = True
        moveCnt = 0
        while True:
            result = self.game.evaluate(board)
            if result != 0: # game ended
                if result == 2: # tie
                    result = 0
                for item in trainData:
                    item[2] = item[2] * result
                return trainData
            probs = mcts.simulateAndPredict(board * turn, self.args.num_simulation, is_root=is_root)
            #print(probs)
            is_root = False
            s = (board * turn).tobytes()
            if moveCnt < 10:
                a = np.random.choice(range(len(probs)), p=probs)
            else:
                a = np.argmax(probs)
            trainData.append([board * turn, probs, turn])
            board = self.game.play(board, a // self.game.boardsize, a % self.game.boardsize, turn)
            turn *= (-1)
            moveCnt += 1
    def selfPlayN(self, n, data): # for multithreading
        res = []
        for _ in tqdm(range(n)):
            res.extend(self.selfPlay())
        self.lock.acquire()
        data.append(res)
        self.lock.release()
    def train(self):
        for i in range(self.args.num_iteration):
            data = []
            threads = []
            for t in range(self.args.num_thread):
                threads.append(threading.Thread(target=self.selfPlayN, args=(self.args.num_episode // self.args.num_thread, data)))
                threads[t].start()
            for t in range(self.args.num_thread):
                threads[t].join()
            lsts = []
            for lst in data:
                lsts.extend(lst)
            data = lsts
            # for _ in tqdm(range(self.args.num_episode)):
            #     data += self.selfPlay()
            self.oldModel.save(self.args.model_save_path)
            self.newModel.load(self.args.model_save_path)
            
            self.newModel.learn(data)
            oldWins = newWins = ties = 0
            # mct_old = MCTS(self.game, self.oldModel)
            # mct_new = MCTS(self.game, self.newModel)
            for j in tqdm(range(self.args.num_game_inference)):
                if j % 2 == 0:
                    result = play(self.game, self.newModel, self.oldModel, self.args.num_simulation, self.args, mct1=None, mct2=None)
                    if result == 1: newWins += 1
                    elif result == -1: oldWins += 1
                    else: ties += 1
                else:
                    result = play(self.game, self.oldModel, self.newModel, self.args.num_simulation, self.args, mct1=None, mct2=None)
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








