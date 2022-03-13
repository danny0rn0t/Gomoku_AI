import numpy as np
from game import gobang
from play import play
from policyGradientNetwork import *
from MCTS import MCTS
from enum import Enum
import numpy as np
BLACK = 1
WHITE = -1
class Player:
    def __init__(self, game: gobang, args, player_type: str, num_layer: int, num_feature: int=256, model_path: str=None):
        assert (player_type == 'AI' or player_type == 'HUMAN')
        self.player_type = player_type
        if player_type == 'AI':
            self.model = ResidualPolicyNetwork(game, num_layers=num_layer, feature=num_feature)
            self.model = PolicyNetworkAgent(self.model, args)
            if model_path is None:
                model_path = f"gobang{args.boardsize}x{args.boardsize}_{num_layer}L.ckpt"
            self.model.load(model_path)
            self.mct = MCTS(game, self.model, args)
        self.color = None
    def set_color(self, color: int):
        assert (color == BLACK or color == WHITE)
        self.color = color