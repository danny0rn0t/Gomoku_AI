import torch.nn as nn
import numpy as np
from game import gobang
from policyGradientNetwork import *
import argparse
from MCTS import MCTS
from train import train
from play import play


parser = argparse.ArgumentParser()
parser.add_argument("--play", action='store_true')
parser.add_argument("--train", action='store_true')
parser.add_argument("--order", type=int, default=1)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--NUM_ITERATION", type=int, default=1000)
parser.add_argument("--NUM_EPISODE", type=int, default=100) # paper: 500000
parser.add_argument("--boardsize", type=int, default=9)
parser.add_argument("--NUM_SIMULATION", type=int, default=160) # paper: 1600
parser.add_argument("--NUM_GAME_INFERENCE", type=int, default=40) # paper: 400
parser.add_argument("--MODEL_SAVE_PATH", type=str, default="checkpoint.pt")
parser.add_argument("--NUM_EPOCH", type=int, default=10)
parser.add_argument("--BATCHSIZE", type=int, default=8)
args = parser.parse_args()

if __name__ == '__main__':
    if args.train and args.play:
        print("One work at a time!")
    elif args.train:
        game = gobang(args.boardsize)
        model = ResidualPolicyNetwork(game, num_layers=10)
        model = PolicyNetworkAgent(model, args)
        model.load(args.MODEL_SAVE_PATH)
        trainer = train(game, model, args)
        trainer.train()
    elif args.play:
        assert (args.order == 1 or args.order == 2)
        game = gobang(args.boardsize)
        model = ResidualPolicyNetwork(game, num_layers=10)
        model = PolicyNetworkAgent(model, args)
        model.load(args.MODEL_SAVE_PATH)
        if args.order == 1:
            play(game, 'human', model, args.NUM_SIMULATION, True)
        else:
            play(game, model, 'human', args.NUM_SIMULATION, True)
    else:
        print("No work was assigned!")





