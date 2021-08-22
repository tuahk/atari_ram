import gym
# from dcgpy import expression_gdual_double as expression
# from dcgpy import kernel_set_gdual_double as kernel_set
# from pyaudi import gdual_double as gdual



games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['Pong-ram-v0',6], ['KungFuMaster-ram-v0', 14]]
# games = [['Assault-ram-v0',7], ['Bowling-ram-v0',6], ['Boxing-ram-v0',18], ['KungFuMaster-ram-v0', 14]]

for game_tuple in games:
    game = game_tuple[0]
    env = gym.make(game)
    print('Game:',  game)
    print(env.unwrapped.get_action_meanings())
    print('------------------\n\n')
    env.close()