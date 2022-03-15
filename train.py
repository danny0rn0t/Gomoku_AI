import numpy as np
from policyGradientNetwork import *
from game import gobang
from MCTS import MCTS
from tqdm import tqdm
from play import play
import threading
from player import Player


class train:
    def __init__(self, game: gobang, model: PolicyNetworkAgent, args):
        self.oldModel = model
        self.newModel = PolicyNetworkAgent(args.boardsize, args.num_layer, args.num_feature, args.learning_rate, args.device, args.batchsize, args.num_epoch)
        self.game = game
        self.mcts = MCTS(game, self.oldModel, args.alpha, args.epsilon, args.c_puct)
        self.args = args
        self.trainData = []
        self.lock = threading.Lock()
    def selfPlay(self):
        mcts = MCTS(self.game, self.oldModel, self.args.alpha, self.args.epsilon, self.args.c_puct)
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
            is_root = False
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
        data.extend(res)
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
            # for _ in tqdm(range(self.args.num_episode)):
            #     data += self.selfPlay()
            self.oldModel.save(self.args.model_path)
            self.newModel.load(self.args.model_path)
            
            self.newModel.learn(data)

            # Evaluate
            oldWins = newWins = ties = 0
            for j in tqdm(range(self.args.num_game_inference)):
                new_mct = MCTS(self.game, self.newModel, self.args.alpha, self.args.epsilon, self.args.c_puct)
                old_mct = MCTS(self.game, self.oldModel, self.args.alpha, self.args.epsilon, self.args.c_puct)
                player_new = Player('AI', self.newModel, new_mct)
                player_old = Player('AI', self.oldModel, old_mct)
                if j % 2 == 0:
                    result = play(self.game, player_new, player_old, self.args.num_simulation)
                    if result == 1: newWins += 1
                    elif result == -1: oldWins += 1
                    else: ties += 1
                else:
                    result = play(self.game, player_old, player_new, self.args.num_simulation)
                    if result == 1: oldWins += 1
                    elif result == -1: newWins += 1
                    else: ties += 1
            winrate = newWins / (newWins + oldWins) if newWins > 0 else 0
            print(f"iteration: {i} | {newWins} win, {oldWins} loss, {ties} tie |")
            if winrate >= self.args.update_threshold:
                print("Update new model!")
                self.newModel.save(self.args.model_path)
                self.oldModel.load(self.args.model_path)
            else:
                print("Discard new model!")








