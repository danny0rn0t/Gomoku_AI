import torch.nn as nn
import numpy as np
from game import gobang
from policyGradientNetwork import *
import argparse
from MCTS import MCTS
from train import train


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=False)
parser.add_argument("--NUM_ITERATION", type=int, default=1000)
parser.add_argument("--NUM_EPISODE", type=int, default=100)
parser.add_argument("--boardsize", type=int, default=9)
parser.add_argument("--NUM_SIMULATION", type=int, default=25)
parser.add_argument("--NUM_GAME_INFERENCE", type=int, default=40)
parser.add_argument("--MODEL_SAVE_PATH", type=str, default="checkpoint.ckpt")
parser.add_argument("--BATCHSIZE", type=int, default=8)

args = parser.parse_args()

game = gobang(args.boardsize)
model = PolicyNetwork(game)
model = PolicyNetworkAgent(model, args)
model.load(args.MODEL_SAVE_PATH)
helper = train(game, model, args)
helper.train()





