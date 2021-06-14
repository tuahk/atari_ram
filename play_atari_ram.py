from cgp import *
import gym
import time
from atari_ram import play_atari_ram


n_inputs = 128
n_outputs =3 
n_cols = 600
arity = 2
KERN = ['sum', 'diff', 'mul', 'pdiv', 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
kernels = kernel_set_double(KERN)()

games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14]]

game = games[0][0]
n_outputs = games[0][1]

ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
load_genes(ind, 'ind/all_time_Assault-ram-v0_simple')

play_atari_ram(game, ind)
