import numpy as np
import torch
from torch import nn
import os
from game2048.game import Game
from game2048.displays import Display, IPythonDisplay

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class NeuralNet(nn.Module):

    def __init__(self, size, n_actions):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 2, 2),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(2, 4, 2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear((size-2)^2 * 4, 64),
            nn.ReLU(True)
        )
        self.layer4 =  nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(True)
        )
        self.layer5 = nn.Linear(16, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class DQN(object):

    def __init__(self, n_lattice=16, n_actions=4, loadpath=None, epsilon=0.9, lr=0.01, decay=0.9, batch_size=32, capacity=2000, freq=100):
        self.n_lattice = n_lattice
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        self.capacity = capacity
        self.freq = freq
        self.prediction_net = NeuralNet(n_lattice, n_actions)
        self.target_net = NeuralNet(n_lattice, n_actions)

        if loadpath is not None:
            net = torch.load(loadpath+'/prediction.pth')
            self.prediction_net.load_state_dict(net.state_dict())
            net = torch.load(loadpath+'/prediction.pth')
            self.target_net.load_state_dict(net.state_dict())

        self.stepcnt = 0
        self.memorycnt = 0
        self.states = np.zeros((capacity, 4, 4))
        self.actions = np.zeros((capacity, 1))
        self.rewards = np.zeros((capacity, 1))
        self.stateps = np.zeros((capacity, 4, 4))
        self.optimizer = torch.optim.SGD(self.prediction_net.parameters(), lr=self.lr)
        self.Lossfunc = torch.nn.CrossEntropyLoss

    def decision(self, state):
        actions = consultant.step()
        return action
    
    '''def store(self, state, action, reward, statep):
        if self.memorycnt <= self.capacity:
            self.memory.append([state, action, reward, statep])
            
        else:
            index = self.memorycnt % self.capacity
            self.memory[index] = [state, action, reward, statep]
        self.memorycnt += 1'''
        
    
    def save(self, savepath='.'):
        torch.save(self.prediction_net, savepath+'/prediction.pth')
        #torch.save(self.target_net, savepath+'/target.pth')

    def learn(self):
        if self.stepcnt == self.freq:
            self.stepcnt = 0
            self.target_net.load_state_dict(self.prediction_net.state_dict())
        self.stepcnt += 1

        n_memory = self.memorycnt if self.memorycnt < self.capacity else self.capacity
        index = np.random.choice(n_memory, self.batch_size)
        
        sts = torch.Tensor(self.states[index, :, :])
        ats = torch.Tensor(self.actions[index]).type(torch.long)
        res = torch.Tensor(self.rewards[index])
        sps = torch.Tensor(self.stateps[index, :, :])

        Q_pred = self.prediction_net(sts).gather(1,ats)
        Q_next = self.target_net(sps).detach()
        Q_target = res + self.lr * Q_next.max(1)[0].reshape((-1,1))
        loss = torch.sum((Q_pred - Q_target) * (Q_pred - Q_target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

net = DQN()
for epoch in range(40000):
    g = Game(4, score_to_win=2048, random=False)
    consultant = ExpectiMaxAgent(g)
    while True:
        s = np.array(consultant.game.board)
        r1 = consultant.game.score
        a = consultant.step()
        consultant.game.move(a)
        s_ = np.array(consultant.game.board)

        r = 1 if consultant.game.score > r1 else 0
        index = net.memorycnt % net.capacity
        net.states[index, :, :] = s
        net.actions[index] = a
        net.rewards[index] = r
        net.stateps[index, :, :] = s_
        net.memorycnt = (net.memorycnt + 1) % net.capacity

        if net.memorycnt == 0:
            net.learn()
        
        if consultant.game.end:
            break

net.save()
print("finish {}".format(consultant.game.score))