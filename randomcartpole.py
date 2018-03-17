import gym
env = gym.make('CartPole-v0')
env.reset()
for e in range(200):
    steps = 0
    while True:
        steps += 1
        #ienv.render()
        next_state, reward, done, info = env.step(env.action_space.sample()) # take a random action
        if done:
                print("{2} Episode {0} finished after {1} steps"
                      .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
                # episode_durations.append(steps)
                # plot_durations()
                break
