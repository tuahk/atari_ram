
from cgp import *
from atari_ram import *

# best_ind, highest_reward = learn_atari_ram('Asteroids-ram-v0')
games = [['Assault-ram-v0',7], ['Boxing-ram-v0',18], ['Breakout-ram-v0',4], ['Seaquest-ram-v0',18], ['Skiing-ram-v0',3]]
# algorithms = ['original', 'all_time_best', '5-run-mean']

n_inputs = 128 
n_outputs = 7 
n_cols = 600 
arity = ARITY
iterations = 2
kernels = kernel_set_double(KERNELS)()
for game in games:
  n_outputs = game[1]
  # for algorithm in algorithms:
  best_ind, highest_reward = learn_atari_ram(game[0], n_inputs, n_outputs, n_cols, arity, kernels, iterations)
  save_ind(best_ind, 'ind/best_ind')
