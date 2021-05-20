from cgp import *
import gym
import time
from statistics import mean


def save_history(history, f_name):
  with open('results/'+f_name, 'a') as f:
    for event in history:
      f.write('%s\n' % ','.join([str(x) for x in event]))

def learn_atari_ram(game, n_inputs, n_outputs, n_cols, arity, kernels, iterations=200):
  start_time = time.time()
  f_name = game + '_' + str(start_time)
  pop = generate_population(n_inputs, n_outputs, n_cols, arity, kernels)
  # Tehdään erillinen olio parhaalle yksilölle jottei python viittuakset sotke kaikkea.
  best_ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  best_ind.set(pop[0].get())

  env = gym.make(game)
  highest_reward = 0
  history = []
  for i_episode in range(iterations):
    rewards_list = [[] for x in range(len(pop))]
    for rounds in range(5):
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
            #   best_ind.set(ind.get())

            # print("Episode finished with {} total score.".format(rewards))
            break
    
    averages = [mean(l) for l in rewards_list]
    # # Hypätään ekan indeksin yli, koska halutaan löytää paras lapsi
    print(rewards_list)
    print(averages)
    best_index = averages.index(max(averages[1:]), 1)  
    print("Paras lapsi i:",best_index)
    print("Vanhemman suoritus:",averages[0])
    # print("Valittaisiin:", "Lapsi" if rewards_list[best_index] >= rewards_list[0] else "Vanhmepi" )

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