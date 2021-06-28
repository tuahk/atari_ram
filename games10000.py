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
  pop = generate_population(n_inputs, n_outputs, n_cols, arity, kernels)
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

            # print("Episode finished with {} total score.".format(rewards))
            break
    
    averages = [mean(l) for l in rewards_list]
    # # Hypätään ekan indeksin yli, koska halutaan löytää paras lapsi
    print(rewards_list)
    print(averages)
    best_index = averages.index(max(averages[1:]), 1)  
    print("Paras lapsi i:",best_index)
    print("Korkein tulos tähän mennessä", highest_reward)
    # print("Valittaisiin:", "Lapsi" if rewards_list[best_index] >= rewards_list[0] else "Vanhmepi" )

    choose_child = False
    # Otetaan vain parempi tulos jatkoon + Natural drift koska yhtähyvä kelpaa
    if averages[best_index] >= highest_reward:
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
  game = 'Assault-ram-v0'
  n_inputs = 128 
  n_outputs = 7 
  n_cols = 600 

  # arity = ARITY
  # kernels = kernel_set_double(KERNELS)()

  arity = 2
  # KERN = ['sum', 'diff', 'mul', 'div'] #, 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
  # KERNELS = ['sum', 'diff', 'mul', 'pdiv', 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
  KERN = ['sum', 'diff', 'mul', 'pdiv', 'sin','cos', 'log', 'exp', 'psqrt', 'sig', 'tanh', 'ReLu']
  kernels = kernel_set_double(KERN)()

  rounds=1
  iterations=5
  fname = '5_times'
  only_active=False

  # games = [['Assault-ram-v0',7], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14], ['Skiing-ram-v0',3]]

  # games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14]]
  games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6]]

  for game in games:
    n_outputs = game[1]
    fname = 'keep_the_best'
      # for algorithm in algorithms:
    best_ind, highest_reward = learn_atari_ram(game[0], fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations, rounds, only_active)
    save_ind(best_ind, 'ind/' + game[0])

  # best_ind, highest_reward = learn_atari_ram(game, fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations, rounds, only_active)

  # for i in range(5):
  #   fname = 'hypoteesi2_rounds5' + str(i)
  #   only_active=False
  #   rounds=5
  #   iterations=2000
  #   best_ind, highest_reward = learn_atari_ram(game, fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations, rounds, only_active)

  # for i in range(5):
  #   fname = 'hypoteesi2_rounds1' + str(i)
  #   only_active=False
  #   rounds=1
  #   iterations=10000
  #   best_ind, highest_reward = learn_atari_ram(game, fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations, rounds, only_active)

  # for i in range(5):
  #   fname = 'hypoteesi_all_nodes_' + str(i)
  #   only_active=False
  #   best_ind, highest_reward = learn_atari_ram(game, fname, n_inputs, n_outputs, n_cols, arity, kernels, iterations, rounds, only_active)

  # graph = extract_computational_subgraph(best_ind, KERNELS)
  # visualize(graph, 'img/' + game + '_' + fname + '.pdf', ind=best_ind)

# Play time!
  # ind = generate_individual(n_inputs, n_outputs, n_cols, arity, kernels)
  # load_genes(ind, 'ind/Assault-ram-v0_hypoteesi_only_active_0')
  # load_genes(ind, 'ind/all_time_Assault-ram-v0_hypoteesi2_rounds50')

  # load_genes(ind, 'ind/all_time_Assault-ram-v0_hypoteesi_all_nodes_0')
  # load_genes(ind, 'ind/Assault-ram-v0_hypoteesi_all_nodes_0')
  # play_atari_ram(game, ind)
