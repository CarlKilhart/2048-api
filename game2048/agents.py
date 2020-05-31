import numpy as np
import copy as cp

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

class SerialAgent(Agent):
    
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        self.cnt = 0

    def step(self):
        self.cnt += 1
        return self.cnt % 4

class ReinforcedAgent(Agent):
    
    def __init__(self, game, display=None):
        super().__init__(game, display)
        self.tmpmax = 0

    def __iterstep(self, game, steps=1):
        if steps == 1:
            for i in range(4):
                tmpgame = cp.deepcopy(game)
                tmpgame.move(i)
                score = tmpgame.score
                if score > self.tmpmax:
                    self.tmpmax = score
                    return 1
                
        else:
            tmpr = 0
            for i in range(4):
                tmpgame = cp.deepcopy(game)
                tmpgame.move(i)

    def step(self):
        max = 0
        for i in range(4):
            tmpgame1 = cp.deepcopy(self.game)
            tmpgame1.move(i)
            for j in range(4):
                tmpgame2 = cp.deepcopy(tmpgame1)
                tmpgame2.move(i)
                for k in range(4):
                    tmpgame3 = cp.deepcopy(tmpgame2)
                    tmpgame3.move(i)
                    tmpscore = tmpgame3.score
                    if tmpscore > max:
                        direction = i
                        max = tmpscore            
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
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction
