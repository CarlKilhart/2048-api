from model import CNN
import numpy as np
import torch
from torch import nn

class DQN(object):

    def __init__(self, n_lattice=16, n_actions=4, loadpath=None, norm=10, lr=0.001, soft=0.1, decay=0.9, batch_size=128, capacity=10000, freq=100):
        self.n_lattice = n_lattice
        self.n_actions = n_actions
        self.norm = norm
        self.lr = lr
        self.soft = soft
        self.decay = decay
        self.batch_size = batch_size
        self.capacity = capacity
        self.freq = freq
        self.prediction_net = CNN()
        self.target_net = CNN()

        if loadpath is not None:
            net = torch.load(loadpath+'/CNN3.pth')
            self.prediction_net.load_state_dict(net.state_dict())
        
        self.target_net.load_state_dict(self.prediction_net.state_dict())

        self.clip = 1
        self.stepcnt = 0
        self.memorycnt = 0
        self.states = np.zeros((capacity, 16, 4, 4))
        self.actions = np.zeros((capacity, 1))
        self.rewards = np.zeros((capacity, 1))
        self.stateps = np.zeros((capacity, 16, 4, 4))
        self.optimizer = torch.optim.SGD(self.prediction_net.parameters(), lr=self.lr)
        from game2048.expectimax import board_to_move
        self.consultant = board_to_move

    def consult(self, state):
        action = self.consultant(state)
        return action
    
    def b2o(board):
        ohc = np.zeros((16, 4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                ohc[self.tf[board[i, j]], i, j] = 1
        return ohc

    def store(self, state, action, reward, statep):
        index = self.memorycnt
        self.states[index, :, :, :] = self.b2o(state)
        self.actions[index] = action
        self.rewards[index] = np.log2(reward)
        self.stateps[index, :, :, :] = self.b2o(statep)
        self.memorycnt = (self.memorycnt + 1) % self.capacity
        
    
    def save(self, savepath='.', filename='CNN1'):
        torch.save(self.prediction_net, savepath+'/'+filename+'.pkl')
    
    def learn(self):
        if self.stepcnt == self.freq:
            for param1, param2 in zip(self.prediction_net.parameters(), self.target_net.parameters()):
                param2.data = self.soft * param1.data + (1 - self.soft) * param2.data
            self.stepcnt = 0
        self.stepcnt += 1

        index = np.random.choice(self.capacity, self.batch_size)
        
        states = torch.Tensor(self.states[index, :, :, :])
        actions = torch.Tensor(self.actions[index]).type(torch.long)
        rewards = torch.Tensor(self.rewards[index])
        stateps = torch.Tensor(self.stateps[index, :, :, :])

        Q_pred = self.prediction_net(states).gather(1, actions)
        Q_next = self.target_net(stateps).detach()
        Q_target = rewards + self.lr * Q_next.max(1)[0].reshape((-1, 1))
        loss = torch.sum((Q_pred - Q_target) * (Q_pred - Q_target))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.prediction_net.parameters(), self.clip)
        self.optimizer.step()