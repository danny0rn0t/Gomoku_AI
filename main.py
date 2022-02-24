import torch
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
parser.add_argument("--device", type=str)
parser.add_argument("--model_save_path", type=str, default="checkpoint.ckpt")

# game parameters:
parser.add_argument("--boardsize", type=int, default=9)

# policy network architecture parameters:
parser.add_argument("--residual_layers", type=int, default=5) # paper: 20

# training parameters:
parser.add_argument("--num_iteration", type=int, default=1000)
parser.add_argument("--num_episode", type=int, default=100) # paper: 500000
parser.add_argument("--num_simulation", type=int, default=160) # paper: 1600
parser.add_argument("--num_game_inference", type=int, default=40) # paper: 400
parser.add_argument("--num_epoch", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=8)
parser.add_argument("--update_threshold", type=float, default=0.55) # paper: 0.55
parser.add_argument("--learning_rate", type=int, default=0.0001)

# playing parameters:
parser.add_argument("-o", "--play_order", type=int, default=2)
parser.add_argument("-t", "--time_limit", type=float, choice=range(0, 60),default=None)
parser.add_argument("-s", "--num_simulation_p", type=int, default=1000)
args = parser.parse_args()


if __name__ == '__main__':
    if args.train and args.play:
        print("One work at a time!")
        exit(0)
    if not args.train and not args.play:
        print("No work was assigned!")
        exit(0)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.train:
        game = gobang(args.boardsize)
        model = ResidualPolicyNetwork(game, num_layers=args.residual_layers)
        model = PolicyNetworkAgent(model, args)
        model.load(args.model_save_path)
        trainer = train(game, model, args)
        trainer.train()
    else: # play
        assert (args.play_order == 1 or args.play_order == 2)
        game = gobang(args.boardsize)
        model = ResidualPolicyNetwork(game, num_layers=args.residual_layers)
        model = PolicyNetworkAgent(model, args)
        model.load(args.model_save_path)
        if args.play_order == 1:
            play(game, 'human', model, args.num_simulation_p, display=True)
        else:
            play(game, model, 'human', args.num_simulation_p, display=True)





