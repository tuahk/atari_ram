from cgp import *
import gym
import time
from statistics import mean, median
# from atari_ram import play_atari_ram

n_inputs = 128
n_outputs =3 
n_cols = 600
arity = 2
KERN = ['sum', 'diff', 'mul', 'pdiv', 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
kernels = kernel_set_double(KERN)()

games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14]]
ends = ['_simple', '_only_active', '_5_rounds', '_keep_the_best', '_lambda_9']

# games = [['Bowling-ram-v0',6]]
# ends = ['_simple']
for game_tuple in games:
  for end in ends:
    game = game_tuple[0]
    n_outputs = game_tuple[1]

    ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)

    print('Game:', game, 'Version:', end)

    print('last_one')
    load_genes(ind, 'ind/'+game+end)

    graph = extract_computational_subgraph(ind, kernels)
    visualize(graph, 'img/last_' + game + '_' + end + '.pdf', ind=ind)

    # print('last one')
    # load_genes(ind, 'ind/'+game+end)
    # play_atari_ram(game, ind)
