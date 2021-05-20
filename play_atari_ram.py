from cgp import *
import gym
import time
from atari_ram import play_atari_ram

n_inputs = 128
n_outputs = 7
n_cols = 600
arity = ARITY
kernels = kernel_set_double(KERNELS)()
ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
load_genes(ind, 'ind/Assault-ram-v0_1621414442.3691218')

play_atari_ram('Assault-ram-v0', ind)
