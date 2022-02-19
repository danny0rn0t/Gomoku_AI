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
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--NUM_ITERATION", type=int, default=1000)
parser.add_argument("--NUM_EPISODE", type=int, default=100)
parser.add_argument("--boardsize", type=int, default=9)
parser.add_argument("--NUM_SIMULATION", type=int, default=25)
parser.add_argument("--NUM_GAME_INFERENCE", type=int, default=40)
parser.add_argument("--MODEL_SAVE_PATH", type=str, default="checkpoint.ckpt")
parser.add_argument("--NUM_EPOCH", type=int, default=10)
parser.add_argument("--BATCHSIZE", type=int, default=8)
args = parser.parse_args()

def training():
    game = gobang(args.boardsize)
    model1 = PolicyNetwork(game)
    model1 = PolicyNetworkAgent(model1, args)
    model1.load(args.MODEL_SAVE_PATH)
    trainer = train(game, model1, args)
    trainer.train()
def playing():
    game = gobang(args.boardsize)
    model1 = PolicyNetwork(game)
    model1 = PolicyNetworkAgent(model1, args)
    model1.load(args.MODEL_SAVE_PATH)
    play(game, 'human', model1, args.NUM_SIMULATION, True)

if __name__ == '__main__':
    if args.train and args.play:
        print("One work at a time!")
    elif args.train:
        training()
    elif args.play:
        playing()
    else:
        print("No work was assigned!")





