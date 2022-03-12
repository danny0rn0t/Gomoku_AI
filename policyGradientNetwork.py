import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from game import *
from tqdm import tqdm
from dataset import PolicyGradientNetworkDataset
import numpy as np

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, feature, 3, 1, 1)
        # self.bn1 = nn.BatchNorm2d(feature)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature, out_channels, 3, 1, 1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x += residual
        x = self.act2(x)
        return x

class ResidualPolicyNetwork(nn.Module):
    def __init__(self, game: gobang, num_layers=20, feature=256):
        super(ResidualPolicyNetwork, self).__init__()
        self.game = game
        self.convNet1 = nn.Sequential(
            nn.Conv2d(1, feature, 3, 1, 1),
            # nn.BatchNorm2d(feature),
            nn.ReLU()
        )
        layers = []
        for i in range(num_layers):
            layers.append(_ResidualBlock(feature, feature, feature))
        self.residualBlocks = nn.Sequential(*layers)
        self.piHead = nn.Sequential(
            # pi.shape = N * 256 * bs * bs
            nn.Conv2d(feature, 2, 1, 1), # N * 2 * bs * bs
            # nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.game.boardsize * self.game.boardsize, self.game.boardsize * self.game.boardsize),
            nn.LogSoftmax(dim=1)
        )
        self.vHead = nn.Sequential(
            # v.shape = N * 256 * bs * bs
            nn.Conv2d(feature, 1, 1, 1), # N * 1 * bs * bs
            # nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.game.boardsize * self.game.boardsize, feature),
            nn.ReLU(),
            nn.Linear(feature, 1),
            nn.Tanh()
        )
    def forward(self, x):
        # x.shape = N * boardsize * boardsize
        x = x.view(-1, 1, self.game.boardsize, self.game.boardsize) # N * 1 * boardsize * boardsize
        x = self.convNet1(x) # N * 256 * bs * bs
        x = self.residualBlocks(x) # N * 256 * bs * bs
        pi = self.piHead(x)
        v = self.vHead(x)
        print(f"flag1: {pi.shape}, {v.shape}")
        return torch.exp(pi), v

class PolicyNetworkAgent():
    def __init__(self, network: ResidualPolicyNetwork, args):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
        self.args = args
        self.boardsize = network.game.boardsize
        self.device = torch.device(self.args.device)
        self.network.to(self.device)
    def forward(self, board: np.ndarray):
        # board.shape = (boardsize, boardsize)
        board = torch.FloatTensor(board.astype(np.float64)).contiguous()
        board = board.view(1, self.boardsize, self.boardsize)
        board = board.to(self.device)
        #print(f"DEBUG: board.shape = {board.shape}")
        self.network.eval()
        with torch.no_grad():
            pi, v = self.network(board)
        pi = pi.detach().cpu().numpy()[0]
        v = v.detach().cpu().numpy().item()
        return pi, v
    def learn(self, target):
        target = PolicyGradientNetworkDataset(target)
        target = DataLoader(target, batch_size=self.args.batchsize, shuffle=True, drop_last=True)
        for epoch in range(self.args.num_epoch):
            self.network.train()
            total_loss = 0
            n = 0
            for board, y_pi, y_v in tqdm(target):
                board = torch.FloatTensor(np.array(board).astype(np.float64)).contiguous().to(self.device)
                y_pi = torch.FloatTensor(np.array(y_pi).astype(np.float64)).contiguous().to(self.device)
                y_v = torch.FloatTensor(np.array(y_v).astype(np.float64)).contiguous().to(self.device)
                # print(f"debug: board.shape = {board.shape}, y_pi.shape = {y_pi.shape}, y_v.shape = {y_v.shape}")
                X_pi, X_v = self.network(board)
                # print(f"debug: X_pi.shape = {X_pi.shape}, X_v.shape = {X_v.shape}")
                loss = self.calcLoss(X_pi, y_pi, X_v, y_v)
                total_loss += loss
                n += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"epoch {epoch + 1} | loss: {total_loss / n :.5f}")
    def calcLoss(self, X_pi, y_pi, X_v, y_v):
        print(y_v.shape, y_pi.shape)
        print(X_v)
        print(y_v)
        l1 = torch.sum((X_v - y_v) ** 2) / y_v.shape[0]
        l2 = torch.sum(y_pi * torch.log(X_pi)) / y_pi.shape[0]
        print(y_v.shape, y_pi.shape, l1, l2)
        return l1 - l2
    def save(self, PATH):
        Agent_Dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)
    def load(self, PATH) -> int:
        try:
            checkpoint = torch.load(PATH)
            print(f"Loading checkpoint from {PATH} ...")
        except:
            print(f"Checkpoint not found, skip loading.")
            return -1
        # checkpoint = torch.load(PATH, strict=True)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return 0



    



