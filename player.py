import numpy as np
from play import play
from policyGradientNetwork import *
from MCTS import MCTS
import numpy as np
BLACK = 1
WHITE = -1
class Player:
    def __init__(self, player_type: str, model: PolicyNetworkAgent=None, mct: MCTS=None):
        assert (player_type == 'AI' or player_type == 'HUMAN')
        self.player_type = player_type
        if player_type == 'AI':
            self.model = model
            self.mct = mct
        self.color = None
    def set_color(self, color: int):
        assert (color == BLACK or color == WHITE)
        self.color = color