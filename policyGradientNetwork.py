from email.policy import strict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from game import *
from tqdm import tqdm
from dataset import PolicyGradientNetworkDataset
import numpy as np

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.act2(x)
        return x

class ResidualPolicyNetwork(nn.Module):
    def __init__(self, game: gobang, num_layers=20):
        super(ResidualPolicyNetwork, self).__init__()
        self.game = game
        self.convNet1 = nn.Sequential(
            nn.Conv2d(1, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        layers = []
        for i in range(num_layers):
            layers.append(_ResidualBlock(256, 256))
        self.residualBlocks = nn.Sequential(*layers)
        self.piHead = nn.Sequential(
            # pi.shape = N * 256 * bs * bs
            nn.Conv2d(256, 2, 1, 1), # N * 2 * bs * bs
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.game.boardsize * self.game.boardsize, self.game.boardsize * self.game.boardsize),
            nn.LogSoftmax(dim=1)
        )
        self.vHead = nn.Sequential(
            # v.shape = N * 256 * bs * bs
            nn.Conv2d(256, 1, 1, 1), # N * 1 * bs * bs
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.game.boardsize * self.game.boardsize, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    def forward(self, x):
        # x.shape = N * boardsize * boardsize
        x = x.view(-1, 1, self.game.boardsize, self.game.boardsize) # N * 1 * boardsize * boardsize
        x = self.convNet1(x) # N * 256 * bs * bs
        x = self.residualBlocks(x) # N * 256 * bs * bs
        pi = self.piHead(x)
        v = self.vHead(x)
        return torch.exp(pi), v

class PolicyNetworkAgent():
    def __init__(self, network: ResidualPolicyNetwork, args):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
        self.args = args
        self.boardsize = network.game.boardsize
        if args.cuda:
            self.network.cuda()
    def forward(self, board: np.ndarray):
        # board.shape = (boardsize, boardsize)
        board = torch.FloatTensor(board.astype(np.float64)).contiguous()
        if self.args.cuda:
            board = board.cuda()
        board = board.view(1, self.boardsize, self.boardsize)
        #print(f"DEBUG: board.shape = {board.shape}")
        self.network.eval()
        with torch.no_grad():
            pi, v = self.network(board)
        pi = pi.detach().cpu().numpy()[0]
        v = v.detach().cpu().numpy().item()
        return pi, v
    def learn(self, target):
        target = PolicyGradientNetworkDataset(target)
        target = DataLoader(target, batch_size=self.args.BATCHSIZE, shuffle=True, drop_last=True)
        for epoch in range(self.args.NUM_EPOCH):
            self.network.train()
            for board, y_pi, y_v in tqdm(target):
                board = torch.FloatTensor(np.array(board).astype(np.float64)).contiguous()
                y_pi = torch.FloatTensor(np.array(y_pi).astype(np.float64)).contiguous()
                y_v = torch.FloatTensor(np.array(y_v).astype(np.float64)).contiguous()
                # print(f"debug: board.shape = {board.shape}, y_pi.shape = {y_pi.shape}, y_v.shape = {y_v.shape}")

                if self.args.cuda:
                    board = board.cuda()
                    y_pi = y_pi.cuda()
                    y_v = y_v.cuda()
                X_pi, X_v = self.network(board)
                # print(f"debug: X_pi.shape = {X_pi.shape}, X_v.shape = {X_v.shape}")
                loss = self.calcLoss(X_pi, y_pi, X_v, y_v)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    def calcLoss(self, X_pi, y_pi, X_v, y_v):
        l1 = torch.sum((X_v - y_v) ** 2) / y_v.shape[0]
        l2 = torch.sum(y_pi * torch.log(X_pi)) / y_pi.shape[0]
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


'''
class PolicyNetwork(nn.Module):
    def __init__(self, game: gobang):
        super().__init__()
        self.boardsize = game.boardsize
        self.conv1 = nn.Conv2d(1, 512, 3, 1, 1)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv3 = nn.Conv2d(512, 512, 3, 1)
        self.conv4 = nn.Conv2d(512, 512, 3, 1)
    
        self.bn2d = nn.BatchNorm2d(512)
        self.bn1d1 = nn.BatchNorm1d(1024)
        self.bn1d2 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512 * (self.boardsize - 4) * (self.boardsize - 4), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.boardsize * self.boardsize)
        self.fc4 = nn.Linear(512, 1)    
    def forward(self, x):
        x = x.view(-1, 1, self.boardsize, self.boardsize)
        
        x = self.conv1(x)
        x = self.bn2d(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = self.bn2d(x)
        x = nn.ReLU()(x)

        x = self.conv3(x)
        x = self.bn2d(x)
        x = nn.ReLU()(x)

        x = self.conv4(x)
        x = self.bn2d(x)
        x = nn.ReLU()(x)

        x = x.view(-1, 512 * (self.boardsize - 4) * (self.boardsize - 4))

        x = self.fc1(x)
        x = self.bn1d1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.3, inplace=True)(x)

        x = self.fc2(x)
        x = self.bn1d2(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.3, inplace=True)(x)

        pi = v = x

        pi = self.fc3(pi)
        pi = nn.functional.log_softmax(pi, dim=1)
        #pi = nn.exp(pi)

        v = self.fc4(v)
        v = nn.Tanh()(v)

        return torch.exp(pi), v
'''    
    



