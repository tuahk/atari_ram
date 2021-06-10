import gym
# from dcgpy import expression_gdual_double as expression
# from dcgpy import kernel_set_gdual_double as kernel_set
# from pyaudi import gdual_double as gdual


env = gym.make('Bowling-ram-v0')
for i_episode in range(2):
    observation = env.reset()
    for t in range(10):
        env.render()
        print(observation)
        print(env.action_space)
        action = env.action_space.sample()
        action = 5
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()