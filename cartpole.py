from dcgpy import *
from dcgpy import expression_double, kernel_set_double
from pyaudi import gdual_double as gdual

from cgp import *
import gym
import functools

# Setup parameters
KERNELS = ['sum', 'diff', 'mul', 'div']
ARITY = 2
SEED = 23
MUTATION_RATE = 0.05



def learn_cart_pole():

  n_inputs = 4
  n_outputs = 2
  n_cols = 50
  arity = ARITY
  kernels = kernel_set_double(KERNELS)()

  pop = generate_population(n_inputs, n_outputs, n_cols, arity, kernels)
  # Tehdään erillinen olio parhaalle yksilölle jottei python viittuakset sotke kaikkea.
  best_ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  best_ind.set(pop[0].get())

  env = gym.make('CartPole-v0')
  highest_reward = 0
  for i_episode in range(100):
    for i, ind in enumerate(pop):
      observation = env.reset()
      done = False
      rewards = 0
      while not done:
        outputs = ind(observation)
        action = outputs.index(max(outputs))

        observation, reward, done, info = env.step(action)
        rewards = rewards + reward 

        if done:
          # Natural drift
          # Edellinen paras yksilö on aina listan ensimmäisenä, joten ei valita sitä jatkoon jos joku muu tuottaa yhtä hyvän tuloksen  
          if i != 0 and rewards >= highest_reward:
            best_ind.set(ind.get())
            highest_reward = rewards
          print("Episode finished after {} timesteps".format(rewards))
          break
    mu_lambda(pop, best_ind)
  env.close()
  return best_ind, highest_reward

def play_cart_pole(ind):
  env = gym.make('CartPole-v0')
  for i_episode in range(10):
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
        print("Episode finished after {} timesteps".format(rewards))
        break
  env.close()


best_ind, high_score = learn_cart_pole()
draw_ind(best_ind)