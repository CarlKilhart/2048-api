from DQN import DQN
import torch
import numpy as np 
from game import Game

agent = DQN()
rounds = 20000

for i in range(rounds):
    g = Game(score_to_win=1024)
    state = g.board
    while True:
        action = agent.consult(state)
        g.move(action)
        statep = g.board
        reward = g.score
        agent.store(state, action, reward, statep)
        state = statep

        if net.memorycnt == 0:
            net.learn()

        if g.end:
            break
    agent.save(path='../model/', name='CNN3.pkl')