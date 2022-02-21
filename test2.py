from policyGradientNetwork import *
from game import *
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.cuda = True
game = gobang(9)
model = ResidualPolicyNetwork(game, num_layers=5)
model = PolicyNetworkAgent(model, args)
model.save('test.ckpt')
model.load('test.ckpt')
