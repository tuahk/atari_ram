from cgp import *
import gym
import time
from statistics import mean

def save_history(history, f_name):
  with open('results/'+f_name, 'a') as f:
    for event in history:
      f.write('%s\n' % ','.join([str(x) for x in event]))

def learn_atari_ram(game, n_inputs, n_outputs, n_cols, arity, kernels, iterations=200, natural_drift=True):
  start_time = time.time()
  f_name = game + '_' + str(start_time)
  pop = generate_population(n_inputs, n_outputs, n_cols, arity, kernels)

  # Tehdään erillinen olio parhaalle yksilölle jottei python viittuakset sotke kaikkea.
  best_ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  best_ind.set(pop[0].get())

  all_time_best_ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  all_time_best_ind.set(pop[0].get())

  env = gym.make(game)
  highest_reward = 0
  all_time_high = 0
  history = []
  for i_episode in range(iterations):
    for i, ind in enumerate(pop):
      observation = env.reset()
      done = False
      rewards = 0
      if natural_drift:
        highest_reward = 0
      while not done:
        outputs = ind(observation)
        action = outputs.index(max(outputs))

        observation, reward, done, info = env.step(action)
        if done:
          # Natural drift
          # Edellinen paras yksilö on aina listan ensimmäisenä, joten ei valita sitä jatkoon jos joku muu tuottaa yhtä hyvän tuloksen  
          if reward >= highest_reward:
            highest_reward = rewards
            best_ind.set(ind.get())


          # Jos halutaan säilyttä se yksilö, joka sai edellisellä kierroksella korkeammat pisteet kuin kukaan tässä populaatiossa.
          if reward >= highest_reward:
            highest_reward = rewards
            best_ind.set(ind.get())

          print("Episode finished with {} total score.".format(rewards))
          break
    
    # averages = [mean(l) for l in rewards_list]
    # # Hypätään ekan indeksin yli, koska halutaan löytää paras lapsi
    # Kirjanpitoa
    history.append([i_episode, highest_reward, choose_child])
    if i_episode % 100 == 0:
      save_history(history, f_name)
      history = []

    # Evolve
    mu_lambda(pop, best_ind)
    print('Number of episodes:',i_episode, "Highest avg score:",max(averages), "All time high:", highest_reward)
    print('\n------------------------------\n')
  env.close()
  save_history(history, f_name)
  save_ind(best_ind, 'ind/'+f_name)

  # Save stats
  with open('results/' + 'stats_'+f_name, 'a') as f:
      f.write('%s\n' % str(time.time() - start_time))

  print('Aikaa kului:',time.time() - start_time)
  return best_ind, highest_reward


def play_atari_ram(game, ind):
  env = gym.make(game)
  for i_episode in range(5):
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

if __name__ == "__main__":
  n_inputs = 128 
  n_outputs = 7 
  n_cols = 600 
  arity = ARITY
  kernels = kernel_set_double(KERNELS)()
  best_ind, highest_reward = learn_atari_ram('Assault-ram-v0', n_inputs, n_outputs, n_cols, arity, kernels)