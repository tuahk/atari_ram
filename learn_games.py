from cgp import *
import gym
import time
from statistics import mean


def save_history(history, f_name):
  with open('results/'+f_name, 'a') as f:
    for event in history:
      f.write('%s\n' % ','.join([str(x) for x in event]))

def learn_atari_ram(game, fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations=200, rounds=1, only_active=False):
  start_time = time.time()
  f_name = game + '_' + str(fname)
  pop = generate_population(n_inputs, n_outputs, n_cols, arity, kernels, n=10)
  # Tehdään erillinen olio parhaalle yksilölle jottei python viittuakset sotke kaikkea

  best_ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  best_ind.set(pop[0].get())

  all_time_best_ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  all_time_best_ind.set(pop[0].get())

  env = gym.make(game)
  highest_reward = 0
  history = []
  for i_episode in range(iterations):
    rewards_list = [[] for x in range(len(pop))]
    for round in range(rounds):
      for i, ind in enumerate(pop):
        observation = env.reset()
        done = False
        rewards = 0
        while not done:
          outputs = ind(observation)
          action = outputs.index(max(outputs))

          observation, reward, done, info = env.step(action)
          rewards = rewards + reward 
          # # print(action, reward)
          # # print('\n--------------\n')
          if done:
            # Natural drift
            # Edellinen paras yksilö on aina listan ensimmäisenä, joten ei valita sitä jatkoon jos joku muu tuottaa yhtä hyvän tuloksen  

            # Käytetään tätä metodia, koska viime kierroksen paras yksilö ei välttämättä pääse samaan tuloksenn uudestaan.
            rewards_list[i].append(rewards)

            # Jos halutaan säilyttä se yksilö, joka sai edellisellä kierroksella korkeammat pisteet kuin kukaan tässä populaatiossa.
            if rewards >= highest_reward:
              highest_reward = rewards
              all_time_best_ind.set(ind.get())

            break
    
    averages = [mean(l) for l in rewards_list]
    # # Hypätään ekan indeksin yli, koska halutaan löytää paras lapsi
    best_index = averages.index(max(averages[1:]), 1)  
    print("Paras lapsi i:",best_index)
    print("Vanhemman suoritus:",averages[0])

    choose_child = False
    if averages[best_index] >= averages[0]:
      highest_scoring_ind = pop[best_index]
      best_ind.set(highest_scoring_ind.get())
      print("Valitiin lapsi")
      choose_child = True
    else:
      print("Ei muutosta parhaaseen yksilöön eli valitiiin vanhempi.")

    # Kirjanpitoa
    history.append([i_episode, max(averages), highest_reward, choose_child])
    if i_episode % 100 == 0:
      save_history(history, f_name)
      history = []

    # Evolve
    if only_active:
      mu_lambda_active(pop, best_ind)
    else:
      mu_lambda(pop, best_ind)
    print('Number of episodes:',i_episode, "Highest avg score:",max(averages), "All time high:", highest_reward)
    print('\n------------------------------\n')
  env.close()
  save_history(history, f_name)
  save_ind(best_ind, 'ind/'+f_name)
  save_ind(all_time_best_ind, 'ind/'+'all_time_'+f_name)

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

  arity = 2
  KERN = ['sum', 'diff', 'mul', 'pdiv', 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
  kernels = kernel_set_double(KERN)()

  rounds=1
  iterations=5000
  fname = 'lambda_9'
  only_active=False

  games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14]]

  for game in games:
    n_outputs = game[1]
    fname = 'lambda_9' 
    best_ind, highest_reward = learn_atari_ram(game[0], fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations, rounds, only_active)
    save_ind(best_ind, 'ind/' + game[0])

