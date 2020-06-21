import numpy as np
from model import CNN
import torch

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


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
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

class RLAgent(Agent):
    def __init__(self, game, display=None):
        super().__init__(game, display)
        CNN1, CNN2, CNN3 = CNN(), CNN(), CNN()
        CNN1.load_state_dict(torch.load('CNN1.pkl'))
        CNN2.load_state_dict(torch.load('CNN2.pkl'))
        CNN3.load_state_dict(torch.load('CNN3.pkl'))
        tf = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 
        2048: 11, 4096: 12}

    def b2o(board):
        ohc = np.zeros((16, 4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                ohc[self.tf[board[i, j]], i, j] = 1
        return ohc

    def step(self):
        state = self.b2o(self.game.board)
        state = np.expand_dims(state, 0)
        state = torch.Tensor(state).to(torch.float32)
        if self.game.score < 256:
            direction = torch.max(self.CNN1(state), 1)[1]
        elif self.game.score < 512:
            direction = torch.max(self.CNN2(state), 1)[1]
        else:
            direction = torch.max(self.CNN2(state), 1)[1]
        return direction


