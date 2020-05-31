import numpy as np
import torch
import torch.nn
import os

class QLearning(object):

    def __init__(self, action=[0, 1, 2, 3], lr=0.01, decay=0.9, epsilon=0.9, read=False):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        if read:
            self.table = np.load('Qtable.npy')
        else:
            self.table = np.zeros(1, len(action))


    def learn(self, state, action, reward, state_):
        approx = self.table[state, action]
        
class NeuralNet(nn.Module):

    def __init__(self,n_states, n_actions):
        self.layer1 = nn.Sequential(
            nn.Linear(n_states, 10),
            nn.Relu(True),
        )
        self.layer2 = nn.Linear(10, n_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class DQN(object):

    def __init__(self, n_lattice=10, n_actions=4, loadpath=None, epsilon=0.9, lr=0.01, decay=0.9, batch_size=32, capacity=2000, freq=100):
        self.n_lattice = n_lattice
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        self.capacity = capacity
        self.freq = freq
        self.prediction_net = NeuralNet(n_states, n_actions)
        self.target_net = NeuralNet(n_states, n_actions)

        if loadpath is not None:
            net = torch.load(loadpath+'/prediction.pth')
            self.prediction_net.load(net.state_dict())
            net = torch.load(loadpath+'/target.pth')
            self.target_net.load_state_dict(net.state_dict())

        self.stepcnt = 0
        self.memorycnt = 0
        self.memory = []
        self.optimizer = torch.optim.SGD(self.prediction_net.parameters(), lr=self.lr)
        self.Lossfunc = torch.nn.CrossEntropyLoss()

    def decision(self, state):
        if np.random.uniform() < self.epsilon:
            # state = np.array(state)
            # state = np.reshape(state.size)
            action = np.argmax(self.prediction_net(state))
        else:
            action = np.random.uniform(0, self.n_actions)
        return action
    
    def store(self, state, action, reward, statep):
        index = self.memorycnt % self.capacity
        self.memory.append([state, action, reward, statep])
        self.memorycnt = (self.memorycnt + 1)
    
    def save(self, savepath='.'):
        torch.save(self.prediction_net, savepath+'/prediction.pth')
        torch.save(self.target_net, savepath+'/target.pth')
    def learn(self):
        if self.stepcnt == self.freq:
            self.stepcnt = 0
            self.target_net.load_state_dict(self.prediction_net.state_dict())
        self.stepcnt += 1

        n_memory = self.memorycnt if self.memorycnt < self.capacity else self.capacity
        index = np.random.choice(self.n_memory, self.batch_size)
        batch_memory = self.memory[index]
        