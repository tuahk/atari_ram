from cgp import *
import gym
import time
from statistics import mean, median
# from atari_ram import play_atari_ram

def play_atari_ram(game, ind):
  env = gym.make(game)
  reward_list = []
  observation = env.reset()
  done = False
  rewards = 0
  while not done:
    env.render()
    outputs = ind(observation)
    action = outputs.index(max(outputs))

    observation, reward, done, info = env.step(action)
    rewards = rewards + reward 
    if done:
      reward_list.append(rewards)
      break
    time.sleep(.02)
  print(mean(reward_list),'&', max(reward_list), '&' ,min(reward_list))
  print('\n-----------------\n')
  env.close()

n_inputs = 128
n_outputs =3 
n_cols = 600
arity = 2
KERN = ['sum', 'diff', 'mul', 'pdiv', 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
kernels = kernel_set_double(KERN)()

games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14]]
ends = ['_simple', '_only_active', '_5_rounds', '_keep_the_best', '_lambda_9']

for game_tuple in games:
  for end in ends:
    game = game_tuple[0]
    n_outputs = game_tuple[1]
# 
    ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)

    print('Game:', game, 'Version:', end)

    print('all_time')
    load_genes(ind, 'ind/all_time_'+game+end)
    play_atari_ram(game, ind)

    print('last one')
    load_genes(ind, 'ind/'+game+end)
    play_atari_ram(game, ind)
