import torch
import torch.nn as nn
import numpy as np
from game import gobang
from policyGradientNetwork import *
import argparse
from MCTS import MCTS
from train import train
from play import play
from gameGUI import *
from player import Player
BLACK = 1
WHITE = -1

parser = argparse.ArgumentParser()
parser.add_argument("--play", action='store_true')
parser.add_argument("--train", action='store_true')
parser.add_argument("--device", type=str)
parser.add_argument("--model_path", type=str)
parser.add_argument("--num_thread", type=int, choices=range(1, 17) ,default=1)

# game parameters:
parser.add_argument("--boardsize", type=int, default=9)

# policy network architecture parameters:
parser.add_argument("--num_layer", type=int, default=5) # paper: 20
parser.add_argument("--num_feature", type=int, default=256)

# training parameters:
parser.add_argument("--num_iteration", type=int, default=1000)
parser.add_argument("--num_episode", type=int, default=100) # paper: 500000
parser.add_argument("--num_simulation", type=int, default=160) # paper: 1600
parser.add_argument("--num_game_inference", type=int, default=40) # paper: 400
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--update_threshold", type=float, default=0.55) # paper: 0.55
parser.add_argument("--learning_rate", type=float, default=0.0001)

# playing parameters:
parser.add_argument("-t", "--time_limit", type=float, choices=range(0, 60),default=5) # time limit for each move

parser.add_argument("--GUI", action="store_true")
parser.add_argument("-p1", "--player_type1", type=str)
parser.add_argument("-l1", "--num_layer1", type=int, default=5)
parser.add_argument("-m1", "--model_path1", type=str, default=None)
parser.add_argument("-p2", "--player_type2", type=str)
parser.add_argument("-l2", "--num_layer2", type=int, default=5)
parser.add_argument("-m2", "--model_path2", type=str, default=None)



# MCTS parameters:
parser.add_argument("--epsilon", type=float, default=0.25) # dirichlet noise
parser.add_argument("--alpha", type=float) # dirichlet noise
parser.add_argument("--c_puct", type=float, default=4) # origin paper := 1

args = parser.parse_args()


if __name__ == '__main__':
    if args.train and args.play:
        print("One work at a time!")
        exit(0)
    if (not args.train) and (not args.play):
        print("No work was assigned!")
        exit(0)
    
    # check args
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_path is None:
        if args.num_feature == 256:
            args.model_path = f"gobang{args.boardsize}x{args.boardsize}_{args.num_layer}L.ckpt"
        else:
            args.model_path = f"gobang{args.boardsize}x{args.boardsize}_{args.num_layer}L_{args.num_feature}F.ckpt"
    if args.model_path1 is None:
        args.model_path1 = f"gobang{args.boardsize}x{args.boardsize}_{args.num_layer1}L.ckpt"
    if args.model_path2 is None:
        args.model_path2 = f"gobang{args.boardsize}x{args.boardsize}_{args.num_layer2}L.ckpt"
    if args.alpha is None:
        args.alpha = 10 / ((args.boardsize**2)/2)
    if args.player_type1 is not None:
        args.player_type1 = args.player_type1.upper()
    if args.player_type2 is not None:
        args.player_type2 = args.player_type2.upper()
    
    
    game = gobang(args.boardsize)
    if args.train:
        model = PolicyNetworkAgent(args.boardsize, args.num_layer, args.num_feature, args.learning_rate, args.device, args.batchsize, args.num_epoch)
        model.load(args.model_path)
        trainer = train(game, model, args)
        trainer.train()
    elif args.play:
        if args.player_type1 == 'AI':
            model1 = PolicyNetworkAgent(args.boardsize, args.num_layer1, 256, args.learning_rate, args.device, args.batchsize, args.num_epoch)
            model1.load(args.model_path1)
            mct1 = MCTS(game, model1, args.alpha, args.epsilon, args.c_puct)
            player1 = Player(args.player_type1, model1, mct1)
        else:
            player1 = Player(args.player_type1)
        if args.player_type2 == 'AI':
            model2 = PolicyNetworkAgent(args.boardsize, args.num_layer2, 256, args.learning_rate, args.device, args.batchsize, args.num_epoch)
            model2.load(args.model_path2)
            mct2 = MCTS(game, model2, args.alpha, args.epsilon, args.c_puct)
            player2 = Player(args.player_type2, model2, mct2)
        else:
            player2 = Player(args.player_type2)

        if args.GUI:
            chessboard = Chessboard(game, args, player1, player2, args.time_limit)
            gobangGUI = playWithGUI(chessboard)
            gobangGUI.loop()
        else:
            play(game, player1, player2, -1, True, args.time_limit)






