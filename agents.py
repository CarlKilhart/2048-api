import numpy as np
import copy as cp
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
'''
class ReinforcedAgent(Agent):

    def __init__(self, game, display=None, n_lattice=16, n_actions=4, loadpath=None):
        super().__init__(game, display)
        self.model = NeuralNet(n_lattice, n_actions)
        if not loadpath == None:
            net = torch.load(loadpath+'/prediction.pth')
            self.model.load_state_dict(net.state_dict())

    def step(self):
        state = np.array(self.game.board)
        state = state.reshape((1, -1))
        state = torch.Tensor(state)
        direction = np.argmax(self.model(state).detach())
        return direction'''

class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
